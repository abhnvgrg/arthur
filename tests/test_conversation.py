from __future__ import annotations

import time

import pytest

from arthur.conversation import Conversation, Ears, describe_call, flag, heard_stop
from arthur.dispatch import Dispatcher
from arthur.llm import Completion, LLMError, ScriptedLLM, ToolCall
from arthur.tools.registry import Risk
from arthur.voice import Clip, Playback, ScriptedSpeaker, ScriptedTranscriber

pytestmark = pytest.mark.asyncio

SPEECH = b"\x00\x40" * 800


PACE_SECONDS = 0.15


class FakeMicrophone:
    def __init__(self, clips: int = 1, pace: float = PACE_SECONDS) -> None:
        self.clips = clips
        self.pace = pace
        self.started = 0

    def listen_forever(self, stop=None, **kwargs):
        self.started += 1
        for index in range(self.clips):
            if stop is not None and stop.is_set():
                return
            if index:
                time.sleep(self.pace)
            yield Clip(SPEECH)


class SilentPlayback(Playback):
    def __init__(self) -> None:
        super().__init__()
        self.played: list[Clip] = []

    def play(self, clip, stop=None):
        self.played.append(clip)
        return True


def build(script, heard, dispatcher, **kwargs):
    speaker = ScriptedSpeaker()
    talker = Conversation(
        ScriptedLLM(script),
        dispatcher,
        ScriptedTranscriber(script=list(heard)),
        speaker,
        FakeMicrophone(clips=len(heard)),
        SilentPlayback(),
        report=None,
        approval_seconds=3.0,
        **kwargs,
    )
    return talker, speaker


def test_stop_words_are_recognised():
    assert heard_stop("stop") is True
    assert heard_stop("Goodbye.") is True
    assert heard_stop("stop the music") is False


def test_a_call_is_described_for_speaking():
    class Spec:
        name = "add_task"

    assert describe_call(Spec(), {"title": "buy milk"}) == "add task with title buy milk"
    assert describe_call(Spec(), {"title": ""}) == "add task"


def test_flags_read_the_environment(monkeypatch):
    monkeypatch.setenv("ARTHUR_TEST_FLAG", "yes")
    assert flag("ARTHUR_TEST_FLAG") is True

    monkeypatch.setenv("ARTHUR_TEST_FLAG", "0")
    assert flag("ARTHUR_TEST_FLAG", True) is False

    monkeypatch.delenv("ARTHUR_TEST_FLAG")
    assert flag("ARTHUR_TEST_FLAG", True) is True


async def test_an_answer_is_spoken_without_pressing_anything(dispatcher):
    talker, speaker = build(
        [Completion(text="It is four o'clock.")],
        ["Diana, what is the time"],
        dispatcher,
    )
    assert await talker.run(turns=1) == 0
    assert speaker.said == ["It is four o'clock."]


async def test_speech_without_the_wake_word_is_ignored(dispatcher):
    talker, speaker = build([], ["the weather is nice today"], dispatcher)
    assert await talker.run() == 0
    assert speaker.said == []


async def test_the_wake_word_alone_gets_an_acknowledgement(dispatcher):
    talker, speaker = build([], ["Diana"], dispatcher)
    await talker.run()
    assert speaker.said == ["Yes?"]


async def test_a_follow_up_needs_no_wake_word(dispatcher):
    talker, speaker = build(
        [Completion(text="Four o'clock."), Completion(text="Tuesday.")],
        ["Diana, the time please", "and the day"],
        dispatcher,
    )
    await talker.run(turns=2)
    assert speaker.said == ["Four o'clock.", "Tuesday."]


async def test_the_wake_word_can_be_turned_off(dispatcher):
    talker, speaker = build(
        [Completion(text="Certainly.")],
        ["play some music"],
        dispatcher,
        require_wake=False,
    )
    await talker.run(turns=1)
    assert speaker.said == ["Certainly."]


async def test_saying_stop_ends_the_conversation(dispatcher):
    talker, speaker = build([], ["Diana, stop"], dispatcher)
    assert await talker.run() == 0
    assert speaker.said == ["Goodbye."]


async def test_silence_is_never_sent_to_the_model(dispatcher):
    talker, speaker = build([], ["   "], dispatcher)
    await talker.run()
    assert speaker.said == []


async def test_a_model_failure_is_spoken_not_raised(dispatcher):
    class Broken:
        async def complete(self, messages, tools):
            raise LLMError("no model")

    speaker = ScriptedSpeaker()
    talker = Conversation(
        Broken(),
        dispatcher,
        ScriptedTranscriber(script=["Diana, what is the time"]),
        speaker,
        FakeMicrophone(),
        SilentPlayback(),
        report=None,
        approval_seconds=3.0,
    )
    await talker.run(turns=1)
    assert speaker.said == ["I could not reach the model just then."]


async def test_history_carries_between_turns(dispatcher):
    talker, _ = build(
        [Completion(text="Four o'clock."), Completion(text="Tuesday.")],
        ["Diana, the time please", "and the day"],
        dispatcher,
    )
    await talker.run(turns=2)

    said = [m for m in talker.history if m.get("role") == "user"]
    assert [m["content"] for m in said] == ["the time please", "and the day"]


async def test_a_writing_call_is_approved_by_voice(dispatcher):
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="add_task", arguments={"title": "buy milk"})]),
            Completion(text="Added."),
        ],
        ["Diana, add buy milk to my list", "yes"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert any("Shall I?" in line for line in speaker.said)
    assert "Right away." in speaker.said
    assert "Added." in speaker.said


async def test_a_spoken_refusal_leaves_the_call_alone(dispatcher, tasks):
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="add_task", arguments={"title": "buy milk"})]),
            Completion(text="Left it."),
        ],
        ["Diana, add buy milk", "no"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert "Leaving it." in speaker.said
    assert tasks.list() == []


async def test_an_unclear_answer_asks_once_more_then_declines(dispatcher, tasks):
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="add_task", arguments={"title": "milk"})]),
            Completion(text="Not done."),
        ],
        ["Diana, add milk", "what do you mean", "who knows"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert "Yes or no?" in speaker.said
    assert tasks.list() == []


async def test_an_irreversible_call_asks_for_the_confirm_word(dispatcher, memory):
    memory.remember("home", "Delhi")
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="forget", arguments={"key": "home"})]),
            Completion(text="Forgotten."),
        ],
        ["Diana, forget where I live", "confirm"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert any("cannot be undone" in line for line in speaker.said)
    assert "Doing it now." in speaker.said
    assert memory.recall("home") is None


async def test_a_plain_yes_is_not_enough_for_an_irreversible_call(dispatcher, memory):
    memory.remember("home", "Delhi")
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="forget", arguments={"key": "home"})]),
            Completion(text="I left it."),
        ],
        ["Diana, forget where I live", "yes", "yes please"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert any("needs the word confirm" in line for line in speaker.said)
    assert memory.recall("home") == "Delhi"


async def test_saying_no_to_an_irreversible_call_stops_it_at_once(dispatcher, memory):
    memory.remember("home", "Delhi")
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="forget", arguments={"key": "home"})]),
            Completion(text="I left it."),
        ],
        ["Diana, forget where I live", "no"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert "Leaving it." in speaker.said
    assert memory.recall("home") == "Delhi"


async def test_silence_never_approves_an_irreversible_call(dispatcher, memory):
    memory.remember("home", "Delhi")
    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="forget", arguments={"key": "home"})]),
            Completion(text="I left it."),
        ],
        ["Diana, forget where I live"],
        dispatcher,
    )
    await talker.run(turns=1)

    assert "I did not catch that, so I have left it alone." in speaker.said
    assert memory.recall("home") == "Delhi"


async def test_typing_can_be_put_back_for_irreversible_calls(dispatcher, monkeypatch):
    asked: list[str] = []

    def refuse(prompt):
        asked.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", refuse)

    talker, speaker = build(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="forget", arguments={"key": "home"})]),
            Completion(text="I left it."),
        ],
        ["Diana, forget where I live"],
        dispatcher,
        typed_irreversible=True,
    )
    await talker.run(turns=1)

    assert asked
    assert not any("cannot be undone" in line for line in speaker.said)


async def test_the_librarian_is_offered_the_exchange(dispatcher):
    absorbed: list[list] = []

    class Recording:
        async def absorb(self, messages):
            absorbed.append(messages)
            return [("home_city", "Delhi")]

    talker, _ = build(
        [Completion(text="Noted.")],
        ["Diana, I live in Delhi"],
        dispatcher,
        librarian=Recording(),
    )
    await talker.run(turns=1)

    assert len(absorbed) == 1
    assert any("Delhi" in str(message) for message in absorbed[0])


async def test_a_broken_librarian_does_not_break_the_reply(dispatcher):
    class Broken:
        async def absorb(self, messages):
            raise RuntimeError("disk full")

    talker, speaker = build(
        [Completion(text="Noted.")],
        ["Diana, I live in Delhi"],
        dispatcher,
        librarian=Broken(),
    )
    await talker.run(turns=1)
    assert speaker.said == ["Noted."]


async def test_ears_report_a_microphone_that_dies(dispatcher):
    class Dying:
        def listen_forever(self, stop=None, **kwargs):
            raise OSError("no input device")
            yield

    ears = Ears(Dying()).start()
    assert await ears.next() is None
    assert isinstance(ears.failure, OSError)


async def test_ears_can_be_drained_of_stale_audio():
    ears = Ears(FakeMicrophone())
    ears.queue.put_nowait(Clip(SPEECH))
    ears.queue.put_nowait(Clip(SPEECH))
    assert ears.drain() == 2
    assert ears.queue.empty()
