from __future__ import annotations

import json

import pytest

from arthur.dispatch import Outcome
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.selection import SYSTEM_PROMPT, build_messages, run_turn

pytestmark = pytest.mark.asyncio


def call(name: str, arguments: dict, call_id: str = "call_1", malformed: str | None = None):
    return ToolCall(id=call_id, name=name, arguments=arguments, malformed=malformed)


def answer(text: str) -> Completion:
    return Completion(text=text)


def wants(*calls: ToolCall) -> Completion:
    return Completion(text=None, tool_calls=tuple(calls))


async def test_a_direct_answer_needs_no_tools(dispatcher):
    llm = ScriptedLLM([answer("Delhi is the capital.")])

    turn = await run_turn(llm, dispatcher, "What is the capital of India?")

    assert turn.answer == "Delhi is the capital."
    assert turn.tool_results == []
    assert len(turn.steps) == 1


async def test_a_read_only_tool_runs_and_the_result_is_fed_back(dispatcher):
    llm = ScriptedLLM(
        [wants(call("calculate", {"expression": "6*7"})), answer("That is 42.")]
    )

    turn = await run_turn(llm, dispatcher, "What is six times seven?")

    assert turn.answer == "That is 42."
    assert turn.tools_used == ["calculate"]

    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["result"]["result"] == 42


async def test_the_model_is_given_the_tool_schemas(dispatcher):
    llm = ScriptedLLM([answer("ok")])

    await run_turn(llm, dispatcher, "hello")

    offered = {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    assert offered == set(dispatcher.registry.names())


async def test_risky_tools_can_be_withheld_from_the_model(dispatcher):
    llm = ScriptedLLM([answer("ok")])

    await run_turn(llm, dispatcher, "hello", expose_risky=False)

    offered = {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    assert "forget" not in offered
    assert "remember" not in offered
    assert "calculate" in offered


async def test_a_writing_tool_is_refused_without_an_approval_hook(dispatcher, memory):
    llm = ScriptedLLM(
        [
            wants(call("remember", {"key": "city", "value": "Delhi"})),
            answer("I could not save that."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "Remember my city is Delhi")

    assert turn.tool_results[0].outcome == Outcome.CONFIRMATION_REQUIRED
    assert memory.recall("city") is None


async def test_the_model_is_told_why_a_call_was_refused(dispatcher):
    llm = ScriptedLLM(
        [wants(call("remember", {"key": "a", "value": "1"})), answer("Refused.")]
    )

    await run_turn(llm, dispatcher, "remember a")

    payload = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert payload["ok"] is False
    assert payload["outcome"] == Outcome.CONFIRMATION_REQUIRED


async def test_an_approved_writing_tool_runs(dispatcher, memory):
    llm = ScriptedLLM(
        [wants(call("remember", {"key": "city", "value": "Delhi"})), answer("Saved.")]
    )

    turn = await run_turn(llm, dispatcher, "remember", approve=lambda spec, args: True)

    assert turn.tools_used == ["remember"]
    assert memory.recall("city") == "Delhi"


async def test_a_denied_writing_tool_does_not_run(dispatcher, memory):
    llm = ScriptedLLM(
        [wants(call("remember", {"key": "city", "value": "Delhi"})), answer("Fine.")]
    )

    await run_turn(
        llm, dispatcher, "remember", approve=lambda spec, args: False, reflect=False
    )

    assert memory.recall("city") is None


async def test_the_approval_hook_sees_the_tool_and_validated_arguments(dispatcher):
    seen = {}

    def approve(spec, arguments):
        seen["tool"] = spec.name
        seen["risk"] = spec.risk.value
        seen["arguments"] = arguments
        return True

    llm = ScriptedLLM(
        [wants(call("remember", {"key": "city", "value": "Delhi"})), answer("ok")]
    )
    await run_turn(llm, dispatcher, "remember", approve=approve)

    assert seen["tool"] == "remember"
    assert seen["risk"] == "writes"
    assert seen["arguments"] == {"key": "city", "value": "Delhi"}


async def test_an_async_approval_hook_is_awaited(dispatcher, memory):
    async def approve(spec, arguments):
        return True

    llm = ScriptedLLM(
        [wants(call("remember", {"key": "city", "value": "Delhi"})), answer("ok")]
    )
    await run_turn(llm, dispatcher, "remember", approve=approve)

    assert memory.recall("city") == "Delhi"


async def test_an_unknown_tool_is_reported_back_not_raised(dispatcher):
    llm = ScriptedLLM(
        [wants(call("launch_missiles", {"target": "moon"})), answer("I cannot do that.")]
    )

    turn = await run_turn(llm, dispatcher, "launch the missiles")

    assert turn.answer == "I cannot do that."
    assert turn.tool_results[0].outcome == Outcome.UNKNOWN_TOOL


async def test_malformed_arguments_never_reach_the_dispatcher(dispatcher, audit):
    llm = ScriptedLLM(
        [
            wants(call("calculate", {}, malformed="Arguments were not valid JSON")),
            answer("Sorry."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "calculate something", reflect=False)

    assert turn.tool_results[0].outcome == Outcome.INVALID_ARGUMENTS
    assert list(audit.entries()) == []


async def test_several_tools_in_one_step_all_run(dispatcher):
    llm = ScriptedLLM(
        [
            wants(
                call("calculate", {"expression": "1+1"}, "a"),
                call("calculate", {"expression": "2+2"}, "b"),
                call("current_time", {"timezone_name": "UTC"}, "c"),
            ),
            answer("Done."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "several things")

    assert len(turn.tool_results) == 3
    assert all(result.ok for result in turn.tool_results)

    roles = [message["role"] for message in llm.calls[1]["messages"][-3:]]
    assert roles == ["tool", "tool", "tool"]


async def test_a_tool_calling_loop_is_capped(dispatcher):
    llm = ScriptedLLM([wants(call("calculate", {"expression": "1+1"}))] * 10)

    turn = await run_turn(llm, dispatcher, "loop forever", max_steps=3)

    assert turn.stopped_at_limit is True
    assert len(turn.steps) == 3
    assert len(llm.calls) == 3


async def test_a_turn_that_finishes_early_is_not_flagged_as_capped(dispatcher):
    llm = ScriptedLLM([wants(call("calculate", {"expression": "1+1"})), answer("2.")])

    turn = await run_turn(llm, dispatcher, "add", max_steps=3)

    assert turn.stopped_at_limit is False
    assert turn.answer == "2."


async def test_max_steps_must_be_positive(dispatcher):
    with pytest.raises(ValueError):
        await run_turn(ScriptedLLM([]), dispatcher, "hi", max_steps=0)


async def test_history_is_placed_between_system_and_user():
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    messages = build_messages("new question", history)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "new question"}


async def test_the_conversation_is_returned_for_the_next_turn(dispatcher):
    llm = ScriptedLLM([wants(call("calculate", {"expression": "2+2"})), answer("4.")])

    turn = await run_turn(llm, dispatcher, "two plus two")

    roles = [message["role"] for message in turn.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def test_a_failing_tool_does_not_end_the_turn(dispatcher):
    llm = ScriptedLLM(
        [
            wants(call("current_time", {"timezone_name": "Mars/Olympus"})),
            answer("That timezone does not exist."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "time on mars")

    assert turn.tool_results[0].outcome == Outcome.FAILED
    assert turn.answer == "That timezone does not exist."


async def test_every_executed_call_is_audited(dispatcher, audit):
    llm = ScriptedLLM(
        [
            wants(
                call("calculate", {"expression": "1+1"}, "a"),
                call("nope", {}, "b"),
            ),
            answer("done"),
        ]
    )

    await run_turn(llm, dispatcher, "mixed", reflect=False)

    outcomes = sorted(entry["outcome"] for entry in audit.entries())
    assert outcomes == sorted([Outcome.OK, Outcome.UNKNOWN_TOOL])


async def test_sequential_execution_preserves_call_order(dispatcher, audit):
    llm = ScriptedLLM(
        [
            wants(
                call("calculate", {"expression": "1+1"}, "a"),
                call("nope", {}, "b"),
            ),
            answer("done"),
        ]
    )

    await run_turn(llm, dispatcher, "mixed", max_parallel=1, reflect=False)

    outcomes = [entry["outcome"] for entry in audit.entries()]
    assert outcomes == [Outcome.OK, Outcome.UNKNOWN_TOOL]


async def test_an_approved_call_is_audited_twice(dispatcher, audit):
    llm = ScriptedLLM(
        [wants(call("remember", {"key": "a", "value": "1"})), answer("ok")]
    )

    await run_turn(llm, dispatcher, "remember", approve=lambda spec, args: True)

    outcomes = [entry["outcome"] for entry in audit.entries()]
    assert outcomes == [Outcome.CONFIRMATION_REQUIRED, Outcome.OK]
