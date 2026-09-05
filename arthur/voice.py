from __future__ import annotations

import array
import asyncio
import io
import os
import threading
import wave
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

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

PLAYBACK_CHUNK_FRAMES = 1024
BARGE_THRESHOLD = 1800.0
BARGE_BLOCKS = 3
MIN_SPOKEN_SENTENCE = 24
DEFAULT_WAKE_WORD = "diana"
WAKE_WINDOW_WORDS = 4


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
        return await self.transcribe_file(clip.as_wav(), "speech.wav", "audio/wav")

    async def transcribe_file(
        self,
        payload: bytes,
        filename: str = "speech.wav",
        content_type: str = "audio/wav",
    ) -> str:
        if not payload:
            return ""

        import httpx

        key, base = _credentials()
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, payload, content_type)},
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

    def listen_forever(
        self,
        stop: threading.Event | None = None,
        max_seconds: float = MAX_RECORD_SECONDS,
        hush_seconds: float = HUSH_SECONDS,
        threshold: float = SPEECH_THRESHOLD,
    ) -> Iterator[Clip]:
        sounddevice = self._sounddevice()
        block = int(self.sample_rate * BLOCK_SECONDS)

        with sounddevice.RawInputStream(
            samplerate=self.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            collected = bytearray()
            elapsed = 0.0
            quiet = 0.0
            spoke = False

            while stop is None or not stop.is_set():
                data, _ = stream.read(block)
                chunk = bytes(data)
                if not chunk:
                    break

                loud = loudness(chunk) >= threshold

                if not spoke and not loud:
                    continue

                collected.extend(chunk)
                elapsed += BLOCK_SECONDS

                if loud:
                    spoke = True
                    quiet = 0.0
                    if elapsed < max_seconds:
                        continue
                else:
                    quiet += BLOCK_SECONDS
                    if quiet < hush_seconds and elapsed < max_seconds:
                        continue

                yield Clip(bytes(collected), self.sample_rate)
                collected = bytearray()
                elapsed = 0.0
                quiet = 0.0
                spoke = False

    def watch_for_speech(
        self,
        stop: threading.Event,
        threshold: float = BARGE_THRESHOLD,
        blocks: int = BARGE_BLOCKS,
    ) -> None:
        sounddevice = self._sounddevice()
        block = int(self.sample_rate * BLOCK_SECONDS)
        loud_run = 0

        with sounddevice.RawInputStream(
            samplerate=self.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            while not stop.is_set():
                data, _ = stream.read(block)
                chunk = bytes(data)
                if not chunk:
                    return
                loud_run = loud_run + 1 if loudness(chunk) >= threshold else 0
                if loud_run >= blocks:
                    stop.set()
                    return


@dataclass
class Playback:
    device: Any = None
    chunk_frames: int = PLAYBACK_CHUNK_FRAMES

    def _sounddevice(self) -> Any:
        return self.device or _audio_device()

    def play(self, clip: Clip, stop: threading.Event | None = None) -> bool:
        if not clip.audio:
            return True

        sounddevice = self._sounddevice()
        step = self.chunk_frames * SAMPLE_WIDTH * CHANNELS

        with sounddevice.RawOutputStream(
            samplerate=clip.sample_rate, channels=CHANNELS, dtype="int16"
        ) as stream:
            for start in range(0, len(clip.audio), step):
                if stop is not None and stop.is_set():
                    return False
                stream.write(clip.audio[start : start + step])
        return True


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


SENTENCE_STOPS = ".!?\n"


def split_sentence(text: str, minimum: int = MIN_SPOKEN_SENTENCE) -> tuple[str | None, str]:
    for index, char in enumerate(text):
        if char not in SENTENCE_STOPS:
            continue
        tail = text[index + 1 :]
        if tail and not tail[0].isspace():
            continue
        if index + 1 < minimum:
            continue
        return text[: index + 1].strip(), tail.lstrip()
    return None, text


@dataclass
class SentenceBuffer:
    minimum: int = MIN_SPOKEN_SENTENCE
    pending: str = ""

    def feed(self, delta: str) -> list[str]:
        self.pending += delta
        ready: list[str] = []
        while True:
            sentence, rest = split_sentence(self.pending, self.minimum)
            if sentence is None:
                break
            self.pending = rest
            if sentence:
                ready.append(sentence)
        return ready

    def drain(self) -> str:
        rest, self.pending = self.pending.strip(), ""
        return rest


def wake_word() -> str:
    return os.getenv("ARTHUR_WAKE_WORD", DEFAULT_WAKE_WORD).strip().lower()


def _letters(text: str) -> str:
    return "".join(c if c.isalpha() or c.isspace() else " " for c in text.lower())


def heard_wake_word(text: str, word: str, window: int = WAKE_WINDOW_WORDS) -> bool:
    if not word:
        return True
    opening = _letters(text).split()[:window]
    return any(word in part or part in word for part in opening if len(part) > 2)


def strip_wake_word(text: str, word: str, window: int = WAKE_WINDOW_WORDS) -> str:
    if not word:
        return text.strip()

    words = text.split()
    for index, part in enumerate(words[:window]):
        bare = "".join(c for c in part.lower() if c.isalpha())
        if bare and (word in bare or bare in word) and len(bare) > 2:
            return " ".join(words[index + 1 :]).lstrip(",.! ").strip()
    return text.strip()


DEFAULT_CONFIRM_WORD = "confirm"

YES_WORDS = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go", "confirm", "approved",
     "do it", "go ahead", "please do", "affirmative", "correct", "right"}
)
NO_WORDS = frozenset(
    {"no", "nope", "nah", "stop", "cancel", "don't", "dont", "negative", "skip",
     "no thanks", "forget it", "never mind", "nevermind"}
)


def affirmative(text: str) -> bool | None:
    cleaned = " ".join(_letters(text).split())
    if not cleaned:
        return None
    if cleaned in YES_WORDS:
        return True
    if cleaned in NO_WORDS:
        return False

    opening = cleaned.split()[0]
    if opening in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm"}:
        return True
    if opening in {"no", "nope", "nah", "cancel", "stop", "dont", "never"}:
        return False
    return None


class SpeechStream:
    def __init__(
        self,
        speaker: Speaker,
        playback: Playback,
        stop: threading.Event | None = None,
        on_error: Any = None,
    ) -> None:
        self.speaker = speaker
        self.playback = playback
        self.stop = stop or threading.Event()
        self.on_error = on_error
        self.spoken: list[str] = []
        self.interrupted = False
        self._text: asyncio.Queue = asyncio.Queue()
        self._clips: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

    def start(self) -> "SpeechStream":
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._synthesise()),
                asyncio.create_task(self._play()),
            ]
        return self

    def feed(self, sentence: str) -> None:
        text = sentence.strip()
        if text and not self.stop.is_set():
            self._text.put_nowait(text)

    async def _synthesise(self) -> None:
        while True:
            sentence = await self._text.get()
            if sentence is None:
                await self._clips.put(None)
                return
            if self.stop.is_set():
                continue
            try:
                clip = await self.speaker.speak(spoken_form(sentence))
            except VoiceError as error:
                if self.on_error is not None:
                    self.on_error(error)
                continue
            await self._clips.put((sentence, clip))

    async def _play(self) -> None:
        while True:
            item = await self._clips.get()
            if item is None:
                return
            sentence, clip = item
            if self.stop.is_set():
                continue
            finished = await asyncio.to_thread(self.playback.play, clip, self.stop)
            if finished:
                self.spoken.append(sentence)
            else:
                self.interrupted = True

    async def close(self) -> None:
        if not self._workers:
            return
        self._text.put_nowait(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def abort(self) -> None:
        self.stop.set()
        self.interrupted = True
        await self.close()


def confirm_word() -> str:
    return os.getenv("ARTHUR_CONFIRM_WORD", DEFAULT_CONFIRM_WORD).strip().lower()


def confirms(text: str, word: str | None = None) -> bool:
    needle = (word or confirm_word()).strip().lower()
    if not needle:
        return False
    spoken = _letters(text).split()
    return any(part == needle for part in spoken)
