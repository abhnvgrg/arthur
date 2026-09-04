from __future__ import annotations

import httpx
import pytest

from arthur.voice import (
    SAMPLE_RATE,
    Clip,
    GroqSpeaker,
    GroqTranscriber,
    Microphone,
    Playback,
    ScriptedSpeaker,
    SPEECH_THRESHOLD,
    ScriptedTranscriber,
    VoiceError,
    VoiceUnavailable,
    loudness,
    say,
    spoken_form,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("ARTHUR_LLM_API_KEY", "gsk-test")
    monkeypatch.delenv("ARTHUR_VOICE_BASE_URL", raising=False)
    monkeypatch.setenv("ARTHUR_LLM_BASE_URL", "https://api.groq.com/openai/v1")


def silence(seconds: float = 1.0) -> Clip:
    return Clip(b"\x00\x00" * int(SAMPLE_RATE * seconds))


async def test_a_clip_reports_its_length():
    assert silence(1.0).seconds == pytest.approx(1.0)
    assert silence(0.5).seconds == pytest.approx(0.5)


async def test_a_clip_survives_a_wav_round_trip():
    original = Clip(b"\x01\x02\x03\x04" * 50)
    assert Clip.from_wav(original.as_wav()) == original


async def test_junk_is_not_mistaken_for_audio():
    with pytest.raises(VoiceError, match="WAV"):
        Clip.from_wav(b"this is not audio")


async def test_transcription_sends_the_clip_as_a_wav_file():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  what time is it  "})

    transcriber = GroqTranscriber(transport=httpx.MockTransport(handler))
    heard = await transcriber.transcribe(silence())

    assert heard == "what time is it"
    assert seen["url"].endswith("/audio/transcriptions")
    assert seen["auth"] == "Bearer gsk-test"
    assert b"RIFF" in seen["body"]
    assert b"whisper-large-v3-turbo" in seen["body"]


async def test_a_clip_too_short_to_be_speech_is_not_uploaded():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have called the API")

    transcriber = GroqTranscriber(transport=httpx.MockTransport(handler))
    assert await transcriber.transcribe(silence(0.1)) == ""


async def test_a_refused_transcription_names_the_status():
    transcriber = GroqTranscriber(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(413, text="file too large")
        )
    )
    with pytest.raises(VoiceError, match="413"):
        await transcriber.transcribe(silence())


async def test_speech_asks_for_wav_and_returns_a_clip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, content=silence(0.2).as_wav())

    speaker = GroqSpeaker(voice="tara", transport=httpx.MockTransport(handler))
    clip = await speaker.speak("Good evening.")

    assert clip.seconds == pytest.approx(0.2)
    assert seen["input"] == "Good evening."
    assert seen["voice"] == "tara"
    assert seen["response_format"] == "wav"


async def test_a_refused_voice_names_the_status():
    speaker = GroqSpeaker(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(400, text="invalid voice")
        )
    )
    with pytest.raises(VoiceError, match="400"):
        await speaker.speak("hello")


async def test_a_missing_key_is_a_clean_error(monkeypatch):
    monkeypatch.delenv("ARTHUR_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(VoiceUnavailable, match="ARTHUR_LLM_API_KEY"):
        await GroqTranscriber().transcribe(silence())


async def test_a_separate_voice_endpoint_can_be_configured(monkeypatch):
    monkeypatch.setenv("ARTHUR_VOICE_BASE_URL", "https://elsewhere.test/v1")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"text": "ok"})

    await GroqTranscriber(transport=httpx.MockTransport(handler)).transcribe(silence())
    assert seen["url"].startswith("https://elsewhere.test/v1/")


async def test_markdown_is_stripped_before_speaking():
    assert spoken_form("**Due:** `13:00`") == "Due: 13:00"
    assert spoken_form("line one\n\nline two") == "line one line two"


async def test_a_long_answer_is_cut_at_a_sentence():
    answer = ("This is a sentence. " * 60).strip()
    spoken = spoken_form(answer, limit=100)

    assert len(spoken) <= 100
    assert spoken.endswith(".")


async def test_a_long_answer_with_no_sentence_break_is_still_cut():
    spoken = spoken_form("x" * 500, limit=100)
    assert len(spoken) == 100


async def test_saying_nothing_is_not_an_error():
    speaker = ScriptedSpeaker()
    assert await say(speaker, Playback(device=object()), "   ") is False
    assert speaker.said == []


async def test_speech_is_played_back():
    played = []

    class Stream:
        def __init__(self, samplerate, channels, dtype):
            self.samplerate = samplerate

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, payload):
            played.append((len(payload), self.samplerate))

    class Device:
        RawOutputStream = Stream

    speaker = ScriptedSpeaker()
    assert await say(speaker, Playback(device=Device()), "**Hello**") is True
    assert speaker.said == ["Hello"]
    assert played


async def test_recording_asks_the_device_for_the_right_shape():
    asked = {}

    class Stream:
        def __init__(self, samplerate, channels, dtype):
            asked.update(samplerate=samplerate, channels=channels, dtype=dtype)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, frames):
            asked["frames"] = frames
            return b"\x00\x00" * frames, False

    class Device:
        RawInputStream = Stream

    clip = Microphone(device=Device()).record(2.0)

    assert asked["frames"] == SAMPLE_RATE * 2
    assert asked["dtype"] == "int16"
    assert clip.seconds == pytest.approx(2.0)


async def test_a_missing_sounddevice_is_a_clean_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("no sounddevice")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(VoiceUnavailable, match="pip install sounddevice"):
        Microphone().record(1.0)


async def test_a_transcriber_can_be_scripted_for_tests():
    transcriber = ScriptedTranscriber(["hello there"])

    assert await transcriber.transcribe(silence()) == "hello there"
    assert len(transcriber.heard) == 1

    with pytest.raises(VoiceError, match="ran out"):
        await transcriber.transcribe(silence())


def blocks(*pattern):
    class Stream:
        def __init__(self, samplerate, channels, dtype):
            self.queue = list(pattern)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, frames):
            loud = self.queue.pop(0) if self.queue else False
            payload = bytes([0, 4] * frames) if loud else bytes(frames * 2)
            return payload, False

    class Device:
        RawInputStream = Stream

    return Device()


async def test_loudness_separates_speech_from_silence():
    assert loudness(bytes(1600)) == 0.0
    assert loudness(bytes([0, 4] * 800)) > SPEECH_THRESHOLD
    assert loudness(b"") == 0.0
    assert loudness(b"\x01") == 0.0


async def test_recording_stops_once_you_stop_talking():
    device = blocks(True, True, True, False, False, False, False, False, False)
    clip = Microphone(device=device).record_until_quiet(
        max_seconds=5.0, hush_seconds=0.3
    )

    assert 0.5 < clip.seconds < 1.0


async def test_recording_gives_up_when_nobody_speaks():
    device = blocks(*([False] * 300))
    clip = Microphone(device=device).record_until_quiet(max_seconds=20.0)

    assert clip.audio == b""
    assert clip.seconds == 0.0


async def test_recording_stops_at_the_ceiling_if_you_never_pause():
    device = blocks(*([True] * 300))
    clip = Microphone(device=device).record_until_quiet(max_seconds=1.0)

    assert clip.seconds == pytest.approx(1.0, abs=0.15)
