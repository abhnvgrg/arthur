from __future__ import annotations

import asyncio

import pytest

from arthur.events import Emitter, Event, EventBus, EventType
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.selection import run_turn

pytestmark = pytest.mark.asyncio


def call(name, arguments, call_id="c1"):
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def drain(bus: EventBus, session_id: str) -> list[Event]:
    return bus.history(session_id)


async def test_an_event_reaches_a_subscriber():
    bus = EventBus()
    queue = await bus.subscribe("s1")

    await bus.emit(Event(type="test", session_id="s1", data={"n": 1}))

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event.data["n"] == 1


async def test_events_do_not_cross_sessions():
    bus = EventBus()
    queue = await bus.subscribe("s1")

    await bus.emit(Event(type="test", session_id="s2"))

    assert queue.empty()


async def test_every_subscriber_gets_the_event():
    bus = EventBus()
    first = await bus.subscribe("s1")
    second = await bus.subscribe("s1")

    await bus.emit(Event(type="test", session_id="s1"))

    assert (await first.get()).type == "test"
    assert (await second.get()).type == "test"


async def test_history_is_kept_per_session():
    bus = EventBus()

    await bus.emit(Event(type="a", session_id="s1"))
    await bus.emit(Event(type="b", session_id="s1"))
    await bus.emit(Event(type="c", session_id="s2"))

    assert [e.type for e in bus.history("s1")] == ["a", "b"]
    assert [e.type for e in bus.history("s2")] == ["c"]


async def test_a_late_subscriber_can_replay():
    bus = EventBus()
    await bus.emit(Event(type="earlier", session_id="s1"))

    queue = await bus.subscribe("s1", replay=True)

    assert (await queue.get()).type == "earlier"


async def test_unsubscribing_stops_delivery():
    bus = EventBus()
    queue = await bus.subscribe("s1")
    await bus.unsubscribe("s1", queue)

    await bus.emit(Event(type="test", session_id="s1"))

    assert queue.empty()
    assert bus.subscriber_count("s1") == 0


async def test_clearing_drops_history():
    bus = EventBus()
    await bus.emit(Event(type="a", session_id="s1"))

    bus.clear("s1")

    assert bus.history("s1") == []


async def test_an_emitter_without_a_bus_is_silent():
    emitter = Emitter(None, "s1")

    await emitter("anything", value=1)


async def test_a_turn_emits_its_lifecycle(dispatcher):
    bus = EventBus()
    llm = ScriptedLLM([Completion(text="Hello.")])

    await run_turn(llm, dispatcher, "hi", bus=bus, session_id="s1")

    types = [event.type for event in bus.history("s1")]
    assert types[0] == EventType.TURN_STARTED
    assert EventType.THINKING in types
    assert EventType.ANSWER in types
    assert types[-1] == EventType.TURN_FINISHED


async def test_a_tool_call_emits_proposal_and_completion(dispatcher):
    bus = EventBus()
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("calculate", {"expression": "2+2"}),)),
            Completion(text="4."),
        ]
    )

    await run_turn(llm, dispatcher, "add", bus=bus, session_id="s1")

    types = [event.type for event in bus.history("s1")]
    assert EventType.TOOL_PROPOSED in types
    assert EventType.TOOL_FINISHED in types

    finished = next(
        e for e in bus.history("s1") if e.type == EventType.TOOL_FINISHED
    )
    assert finished.data["tool"] == "calculate"
    assert finished.data["outcome"] == "ok"


async def test_a_refused_tool_emits_approval_required_then_denied(dispatcher):
    bus = EventBus()
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="Could not."),
        ]
    )

    await run_turn(llm, dispatcher, "remember", bus=bus, session_id="s1")

    types = [event.type for event in bus.history("s1")]
    assert EventType.APPROVAL_REQUIRED in types
    assert EventType.APPROVAL_DENIED in types
    assert EventType.APPROVAL_GRANTED not in types


async def test_an_approved_tool_emits_granted_and_started(dispatcher):
    bus = EventBus()
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="Saved."),
        ]
    )

    await run_turn(
        llm, dispatcher, "remember", approve=lambda s, a: True, bus=bus, session_id="s1"
    )

    types = [event.type for event in bus.history("s1")]
    assert EventType.APPROVAL_GRANTED in types
    assert EventType.TOOL_STARTED in types


async def test_the_approval_event_names_the_risk(dispatcher):
    bus = EventBus()
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("forget", {"key": "a"}),)),
            Completion(text="No."),
        ]
    )

    await run_turn(
        llm, dispatcher, "forget a", bus=bus, session_id="s1", reflect=False
    )

    required = next(
        e for e in bus.history("s1") if e.type == EventType.APPROVAL_REQUIRED
    )
    assert required.data["risk"] == "irreversible"
    assert required.data["tool"] == "forget"


async def test_a_failing_turn_emits_an_error(dispatcher):
    bus = EventBus()

    with pytest.raises(Exception):
        await run_turn(ScriptedLLM([]), dispatcher, "hi", bus=bus, session_id="s1")

    types = [event.type for event in bus.history("s1")]
    assert EventType.ERROR in types


async def test_a_turn_without_a_bus_still_runs(dispatcher):
    llm = ScriptedLLM([Completion(text="fine")])

    turn = await run_turn(llm, dispatcher, "hi")

    assert turn.answer == "fine"
