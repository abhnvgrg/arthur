from __future__ import annotations

import threading

import pytest

from arthur.voice import (
    Clip,
    Microphone,
    Playback,
    ScriptedSpeaker,
    SentenceBuffer,
    SpeechStream,
    VoiceError,
    affirmative,
    confirm_word,
    confirms,
    heard_wake_word,
    split_sentence,
    strip_wake_word,
)


def test_a_sentence_is_cut_at_the_full_stop():
    sentence, rest = split_sentence("The meeting starts at four. And the second")
    assert sentence == "The meeting starts at four."
    assert rest == "And the second"


def test_a_short_fragment_is_held_back():
    assert split_sentence("Yes. But wait a moment longer")[0] is None


def test_a_decimal_point_does_not_end_a_sentence():
    assert split_sentence("The total came to 3.5 million pounds overall") == (
        None,
        "The total came to 3.5 million pounds overall",
    )


def test_the_buffer_releases_sentences_as_they_complete():
    buffer = SentenceBuffer()
    assert buffer.feed("The meeting is at four") == []
    assert buffer.feed(" o'clock today. ") == ["The meeting is at four o'clock today."]
    assert buffer.feed("Nothing else is") == []
    assert buffer.drain() == "Nothing else is"


def test_draining_twice_yields_nothing_the_second_time():
    buffer = SentenceBuffer()
    buffer.feed("half a thought")
    assert buffer.drain() == "half a thought"
    assert buffer.drain() == ""


def test_the_wake_word_is_heard_at_the_start():
    assert heard_wake_word("Diana, what is the time", "diana") is True
    assert heard_wake_word("Hey diana turn the music down", "diana") is True


def test_the_wake_word_is_ignored_late_in_a_sentence():
    assert heard_wake_word("I was talking to my friend about diana", "diana") is False


def test_a_near_miss_from_the_transcriber_still_wakes_her():
    assert heard_wake_word("Dian, what is the time", "diana") is True


def test_an_empty_wake_word_means_always_listening():
    assert heard_wake_word("anything at all", "") is True


def test_the_wake_word_is_stripped_from_the_request():
    assert strip_wake_word("Diana, what is the time", "diana") == "what is the time"
    assert strip_wake_word("hey diana play some music", "diana") == "play some music"


def test_stripping_leaves_a_request_that_never_named_her():
    assert strip_wake_word("what is the time", "diana") == "what is the time"


def test_yes_and_no_are_recognised():
    assert affirmative("yes") is True
    assert affirmative("go ahead") is True
    assert affirmative("yeah, do it") is True
    assert affirmative("no") is False
    assert affirmative("nope, cancel that") is False


def test_an_unclear_answer_is_neither():
    assert affirmative("what does that mean") is None
    assert affirmative("") is None


class ChunkedOutput:
    def __init__(self):
        self.written = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, data):
        self.written.extend(data)


class RecordingDevice:
    def __init__(self):
        self.stream = ChunkedOutput()

    def RawOutputStream(self, **kwargs):
        return self.stream


def test_playback_writes_the_whole_clip():
    device = RecordingDevice()
    clip = Clip(b"\x01\x02" * 4000)
    assert Playback(device=device).play(clip) is True
    assert len(device.stream.written) == len(clip.audio)


def test_playback_stops_early_when_told_to():
    device = RecordingDevice()
    stop = threading.Event()
    stop.set()

    assert Playback(device=device, chunk_frames=64).play(Clip(b"\x01\x02" * 4000), stop) is False
    assert device.stream.written == b""


def test_playback_of_silence_is_a_no_op():
    assert Playback(device=RecordingDevice()).play(Clip(b"")) is True


@pytest.mark.asyncio
async def test_the_speech_stream_speaks_each_sentence_in_order():
    speaker = ScriptedSpeaker()
    stream = SpeechStream(speaker, Playback(device=RecordingDevice())).start()
    stream.feed("First thing.")
    stream.feed("Second thing.")
    await stream.close()

    assert speaker.said == ["First thing.", "Second thing."]
    assert stream.spoken == ["First thing.", "Second thing."]
    assert stream.interrupted is False


@pytest.mark.asyncio
async def test_a_failing_voice_does_not_stop_the_rest():
    problems = []
    speaker = ScriptedSpeaker(fail_with=VoiceError("no credit"))
    stream = SpeechStream(
        speaker, Playback(device=RecordingDevice()), on_error=problems.append
    ).start()
    stream.feed("Anything at all.")
    await stream.close()

    assert stream.spoken == []
    assert "no credit" in str(problems[0])


@pytest.mark.asyncio
async def test_aborting_marks_the_stream_interrupted():
    stream = SpeechStream(ScriptedSpeaker(), Playback(device=RecordingDevice())).start()
    await stream.abort()
    assert stream.interrupted is True


def test_listening_yields_one_clip_per_utterance():
    loud = (b"\x00\x40" * 800)
    quiet = (b"\x00\x00" * 800)
    blocks = [loud, loud, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, quiet, b""]

    class Blocks:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, frames):
            return (blocks.pop(0) if blocks else b""), False

    class Device:
        def RawInputStream(self, **kwargs):
            return Blocks()

    heard = list(Microphone(device=Device()).listen_forever(threading.Event()))
    assert len(heard) == 1
    assert heard[0].audio.startswith(loud)


def test_listening_stops_when_asked():
    stop = threading.Event()
    stop.set()

    class Blocks:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, frames):
            raise AssertionError("should not have read anything")

    class Device:
        def RawInputStream(self, **kwargs):
            return Blocks()

    assert list(Microphone(device=Device()).listen_forever(stop)) == []


def test_the_confirm_word_must_be_said_exactly():
    assert confirms("confirm", "confirm") is True
    assert confirms("yes, confirm that", "confirm") is True
    assert confirms("yes", "confirm") is False
    assert confirms("confirmation", "confirm") is False
    assert confirms("", "confirm") is False


def test_the_confirm_word_can_be_changed(monkeypatch):
    monkeypatch.setenv("ARTHUR_CONFIRM_WORD", "engage")
    assert confirm_word() == "engage"
    assert confirms("engage", None) is True
    assert confirms("confirm", None) is False
