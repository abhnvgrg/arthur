from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from arthur.dispatch import Dispatcher, Outcome
from arthur.tools.registry import Risk, ToolRegistry, ToolSpec

pytestmark = pytest.mark.asyncio


class Empty(BaseModel):
    pass


async def test_a_read_only_tool_runs_without_confirmation(dispatcher):
    result = await dispatcher.invoke("current_time", {"timezone_name": "UTC"})

    assert result.ok
    assert result.value["timezone"] == "UTC"


async def test_a_writing_tool_is_held_for_confirmation(dispatcher, memory):
    result = await dispatcher.invoke("remember", {"key": "city", "value": "Delhi"})

    assert result.needs_confirmation
    assert result.outcome == Outcome.CONFIRMATION_REQUIRED
    assert memory.recall("city") is None


async def test_confirming_lets_a_writing_tool_through(dispatcher, memory):
    result = await dispatcher.invoke(
        "remember", {"key": "city", "value": "Delhi"}, confirmed=True
    )

    assert result.ok
    assert memory.recall("city") == "Delhi"


async def test_an_irreversible_tool_is_held_for_confirmation(dispatcher, memory):
    memory.remember("city", "Delhi")

    result = await dispatcher.invoke("forget", {"key": "city"})

    assert result.needs_confirmation
    assert memory.recall("city") == "Delhi"


async def test_confirmation_does_not_carry_to_the_next_call(dispatcher, memory):
    await dispatcher.invoke("remember", {"key": "a", "value": "1"}, confirmed=True)
    second = await dispatcher.invoke("remember", {"key": "b", "value": "2"})

    assert second.needs_confirmation
    assert memory.recall("b") is None


async def test_an_unknown_tool_is_reported_not_raised(dispatcher):
    result = await dispatcher.invoke("launch_missiles", {})

    assert result.outcome == Outcome.UNKNOWN_TOOL
    assert not result.ok


async def test_invalid_arguments_are_reported_with_the_offending_field(dispatcher):
    result = await dispatcher.invoke("calculate", {"expression": ""})

    assert result.outcome == Outcome.INVALID_ARGUMENTS
    assert "expression" in result.error


async def test_missing_arguments_are_reported(dispatcher):
    result = await dispatcher.invoke("remember", {"key": "only-a-key"})

    assert result.outcome == Outcome.INVALID_ARGUMENTS
    assert "value" in result.error


async def test_unexpected_arguments_do_not_reach_the_handler(dispatcher):
    result = await dispatcher.invoke(
        "current_time", {"timezone_name": "UTC", "sudo": True}
    )

    assert result.ok
    assert "sudo" not in result.arguments


async def test_a_raising_tool_is_contained(audit):
    registry = ToolRegistry()

    def explode(_: Empty):
        raise RuntimeError("handler blew up")

    registry.register(
        ToolSpec(
            name="explode",
            description="always fails",
            parameters=Empty,
            handler=explode,
        )
    )

    result = await Dispatcher(registry, audit=audit).invoke("explode", {})

    assert result.outcome == Outcome.FAILED
    assert "handler blew up" in result.error


async def test_a_hanging_tool_is_timed_out(audit):
    registry = ToolRegistry()

    async def hang(_: Empty):
        await asyncio.sleep(5)

    registry.register(
        ToolSpec(
            name="hang",
            description="never returns",
            parameters=Empty,
            handler=hang,
            timeout_seconds=0.05,
        )
    )

    result = await Dispatcher(registry, audit=audit).invoke("hang", {})

    assert result.outcome == Outcome.TIMEOUT
    assert "0.05" in result.error


async def test_a_synchronous_tool_does_not_block_the_loop(audit):
    registry = ToolRegistry()

    def slow(_: Empty):
        import time

        time.sleep(0.2)
        return "done"

    registry.register(
        ToolSpec(
            name="slow", description="blocking", parameters=Empty, handler=slow
        )
    )
    dispatcher = Dispatcher(registry, audit=audit)

    ticks = 0

    async def counter():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(counter())
    result = await dispatcher.invoke("slow", {})
    task.cancel()

    assert result.ok
    assert ticks > 1


async def test_policy_can_allow_writes_without_confirmation(registry, audit, memory):
    dispatcher = Dispatcher(
        registry, audit=audit, auto_confirm=frozenset({Risk.READ_ONLY, Risk.WRITES})
    )

    result = await dispatcher.invoke("remember", {"key": "city", "value": "Delhi"})

    assert result.ok
    assert memory.recall("city") == "Delhi"


async def test_policy_still_holds_irreversible_tools(registry, audit, memory):
    memory.remember("city", "Delhi")
    dispatcher = Dispatcher(
        registry, audit=audit, auto_confirm=frozenset({Risk.READ_ONLY, Risk.WRITES})
    )

    result = await dispatcher.invoke("forget", {"key": "city"})

    assert result.needs_confirmation
    assert memory.recall("city") == "Delhi"


async def test_a_failed_result_is_shaped_for_the_model(dispatcher):
    result = await dispatcher.invoke("nope", {})
    payload = result.for_model()

    assert payload["ok"] is False
    assert payload["outcome"] == Outcome.UNKNOWN_TOOL


async def test_a_successful_result_is_shaped_for_the_model(dispatcher):
    result = await dispatcher.invoke("calculate", {"expression": "2+2"})
    payload = result.for_model()

    assert payload["ok"] is True
    assert payload["result"]["result"] == 4
