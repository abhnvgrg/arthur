from __future__ import annotations

import array
import asyncio
import io
import os
import wave
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_TTS_VOICE = "daniel"
TTS_VOICES = ("autumn", "diana", "hannah", "austin", "daniel", "troy")
DEFAULT_TIMEOUT_SECONDS = 60.0

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

MAX_SPOKEN_CHARS = 600
MIN_CLIP_SECONDS = 0.3

BLOCK_SECONDS = 0.1
MAX_RECORD_SECONDS = 20.0
HUSH_SECONDS = 1.2
PATIENCE_SECONDS = 6.0
SPEECH_THRESHOLD = 450.0


class VoiceError(Exception):
    pass


class VoiceUnavailable(VoiceError):
    pass


@dataclass(frozen=True)
class Clip:
    audio: bytes
    sample_rate: int = SAMPLE_RATE

    @property
    def seconds(self) -> float:
        frames = len(self.audio) / (SAMPLE_WIDTH * CHANNELS)
        return frames / self.sample_rate if self.sample_rate else 0.0

    def as_wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(SAMPLE_WIDTH)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.audio)
        return buffer.getvalue()

    @classmethod
    def from_wav(cls, payload: bytes) -> "Clip":
        try:
            with wave.open(io.BytesIO(payload), "rb") as handle:
                return cls(
                    handle.readframes(handle.getnframes()), handle.getframerate()
                )
        except wave.Error as error:
            raise VoiceError(f"Not readable as WAV audio: {error}") from error


class Transcriber(Protocol):
    async def transcribe(self, clip: Clip) -> str: ...


class Speaker(Protocol):
    async def speak(self, text: str) -> Clip: ...


@dataclass
class ScriptedTranscriber:
    script: list[str | Exception] = field(default_factory=list)
    heard: list[Clip] = field(default_factory=list)

    async def transcribe(self, clip: Clip) -> str:
        self.heard.append(clip)
        if not self.script:
            raise VoiceError("ScriptedTranscriber ran out of lines")
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class ScriptedSpeaker:
    said: list[str] = field(default_factory=list)
    fail_with: Exception | None = None

    async def speak(self, text: str) -> Clip:
        if self.fail_with is not None:
            raise self.fail_with
        self.said.append(text)
        return Clip(b"\x00\x00" * 100)


def _credentials() -> tuple[str, str]:
    key = os.getenv("ARTHUR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise VoiceUnavailable(
            "No model API key. Set ARTHUR_LLM_API_KEY or OPENAI_API_KEY"
        )
    base = os.getenv("ARTHUR_VOICE_BASE_URL") or os.getenv(
        "ARTHUR_LLM_BASE_URL", DEFAULT_BASE_URL
    )
    return key, base.rstrip("/")


class GroqTranscriber:
    def __init__(
        self,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        self.model = model or os.getenv("ARTHUR_STT_MODEL", DEFAULT_STT_MODEL)
        self.timeout = timeout
        self._transport = transport

    async def transcribe(self, clip: Clip) -> str:
        if clip.seconds < MIN_CLIP_SECONDS:
            return ""

        import httpx

        key, base = _credentials()
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("speech.wav", clip.as_wav(), "audio/wav")},
                data={"model": self.model, "response_format": "json"},
            )
            if response.status_code >= 400:
                raise VoiceError(
                    f"Transcription failed ({response.status_code}): "
                    f"{response.text[:200]}"
                )
            return (response.json().get("text") or "").strip()


class GroqSpeaker:
    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        self.model = model or os.getenv("ARTHUR_TTS_MODEL", DEFAULT_TTS_MODEL)
        self.voice = voice or os.getenv("ARTHUR_TTS_VOICE", DEFAULT_TTS_VOICE)
        self.timeout = timeout
        self._transport = transport

    async def speak(self, text: str) -> Clip:
        import httpx

        key, base = _credentials()
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{base}/audio/speech",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.model,
                    "voice": self.voice,
                    "input": text[:MAX_SPOKEN_CHARS],
                    "response_format": "wav",
                },
            )
            if response.status_code >= 400:
                raise VoiceError(
                    f"Speech failed ({response.status_code}): {response.text[:200]}"
                )
            return Clip.from_wav(response.content)


def loudness(block: bytes) -> float:
    if len(block) < SAMPLE_WIDTH:
        return 0.0
    samples = array.array("h")
    samples.frombytes(block[: len(block) - len(block) % SAMPLE_WIDTH])
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5


def _audio_device() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise VoiceUnavailable(
            "Audio needs sounddevice. pip install sounddevice"
        ) from error
    return sounddevice


@dataclass
class Microphone:
    sample_rate: int = SAMPLE_RATE
    device: Any = None

    def _sounddevice(self) -> Any:
        return self.device or _audio_device()

    def record(self, seconds: float) -> Clip:
        sounddevice = self._sounddevice()
        with sounddevice.RawInputStream(
            samplerate=self.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            data, _ = stream.read(int(seconds * self.sample_rate))
        return Clip(bytes(data), self.sample_rate)

    def record_until_quiet(
        self,
        max_seconds: float = MAX_RECORD_SECONDS,
        hush_seconds: float = HUSH_SECONDS,
        threshold: float = SPEECH_THRESHOLD,
    ) -> Clip:
        sounddevice = self._sounddevice()
        block = int(self.sample_rate * BLOCK_SECONDS)
        collected = bytearray()
        elapsed = 0.0
        quiet = 0.0
        spoke = False

        with sounddevice.RawInputStream(
            samplerate=self.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            while elapsed < max_seconds:
                data, _ = stream.read(block)
                chunk = bytes(data)
                if not chunk:
                    break
                collected.extend(chunk)
                elapsed += BLOCK_SECONDS

                if loudness(chunk) >= threshold:
                    spoke = True
                    quiet = 0.0
                    continue

                quiet += BLOCK_SECONDS
                if spoke and quiet >= hush_seconds:
                    break
                if not spoke and quiet >= PATIENCE_SECONDS:
                    break

        return Clip(bytes(collected) if spoke else b"", self.sample_rate)


@dataclass
class Playback:
    device: Any = None

    def _sounddevice(self) -> Any:
        return self.device or _audio_device()

    def play(self, clip: Clip) -> None:
        if not clip.audio:
            return

        sounddevice = self._sounddevice()
        with sounddevice.RawOutputStream(
            samplerate=clip.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            stream.write(clip.audio)


def spoken_form(answer: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    text = " ".join(answer.split())
    for marker in ("**", "*", "`", "#", "_"):
        text = text.replace(marker, "")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: stop + 1] if stop > limit // 2 else cut


async def say(speaker: Speaker, playback: Playback, text: str) -> bool:
    spoken = spoken_form(text)
    if not spoken:
        return False
    clip = await speaker.speak(spoken)
    await asyncio.to_thread(playback.play, clip)
    return True
