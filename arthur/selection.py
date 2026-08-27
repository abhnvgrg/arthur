from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from arthur.dispatch import Dispatcher, Outcome, ToolResult
from arthur.events import Emitter, EventBus, EventType
from arthur.llm import LLM, Completion, ToolCall
from arthur.reflection import MAX_REFLECTIONS, Critique, critique
from arthur.tools.registry import ToolSpec

MAX_STEPS = 4
MAX_PARALLEL_TOOLS = 4

SYSTEM_PROMPT = (
    "You are ARTHUR, a personal assistant.\n"
    "Call a tool when one can answer the request; answer directly when none applies.\n"
    "You may call several tools at once when they do not depend on each other.\n"
    "Tools marked as writing or irreversible need the user's approval, which may be "
    "refused. If a call is refused or fails, tell the user plainly and do not retry "
    "the same call unchanged.\n"
    "Never claim an action succeeded unless a tool result says so."
)

ApprovalHook = Callable[..., Awaitable[bool] | bool]


@dataclass
class Step:
    completion: Completion
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class Turn:
    answer: str | None
    messages: list[dict[str, Any]]
    steps: list[Step] = field(default_factory=list)
    stopped_at_limit: bool = False
    session_id: str = ""
    critique: Critique | None = None
    reflections: int = 0

    @property
    def tool_results(self) -> list[ToolResult]:
        return [result for step in self.steps for result in step.results]

    @property
    def tools_used(self) -> list[str]:
        return [result.tool for result in self.tool_results if result.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "steps": len(self.steps),
            "stopped_at_limit": self.stopped_at_limit,
            "reflections": self.reflections,
            "critique": self.critique.to_dict() if self.critique else None,
            "tool_results": [result.for_model() for result in self.tool_results],
        }


def build_messages(
    user_message: str,
    history: Sequence[dict[str, Any]] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})
    return messages


def _assistant_message(completion: Completion) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": completion.text}
    if completion.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in completion.tool_calls
        ]
    return message


def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(result.for_model(), default=str),
    }


def _accepts_call_id(approve: ApprovalHook) -> bool:
    try:
        parameters = inspect.signature(approve).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in parameters.values()):
        return True
    positional = [
        p
        for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3


async def _approved(
    approve: ApprovalHook | None,
    spec: ToolSpec,
    arguments: dict[str, Any],
    call_id: str = "",
) -> bool:
    if approve is None:
        return False

    if _accepts_call_id(approve):
        outcome = approve(spec, arguments, call_id)
    else:
        outcome = approve(spec, arguments)

    if hasattr(outcome, "__await__"):
        return bool(await outcome)
    return bool(outcome)


async def execute_call(
    dispatcher: Dispatcher,
    call: ToolCall,
    approve: ApprovalHook | None,
    emit: Emitter | None = None,
) -> ToolResult:
    async def announce(event_type: str, **data: Any) -> None:
        if emit is not None:
            await emit(event_type, tool=call.name, call_id=call.id, **data)

    if call.malformed is not None:
        result = ToolResult(
            tool=call.name,
            outcome=Outcome.INVALID_ARGUMENTS,
            error=call.malformed,
        )
        await announce(EventType.TOOL_FINISHED, outcome=result.outcome, error=result.error)
        return result

    await announce(EventType.TOOL_PROPOSED, arguments=call.arguments)

    result = await dispatcher.invoke(call.name, call.arguments)
    if result.outcome != Outcome.CONFIRMATION_REQUIRED:
        await announce(
            EventType.TOOL_FINISHED,
            outcome=result.outcome,
            value=result.value,
            error=result.error,
            duration_ms=round(result.duration_ms, 1),
        )
        return result

    spec = dispatcher.registry.get(call.name)
    await announce(
        EventType.APPROVAL_REQUIRED, risk=spec.risk.value, arguments=result.arguments
    )

    if not await _approved(approve, spec, result.arguments, call.id):
        await announce(EventType.APPROVAL_DENIED)
        await announce(EventType.TOOL_FINISHED, outcome=result.outcome, error=result.error)
        return result

    await announce(EventType.APPROVAL_GRANTED)
    await announce(EventType.TOOL_STARTED, arguments=result.arguments)

    granted = await dispatcher.invoke(call.name, call.arguments, confirmed=True)
    await announce(
        EventType.TOOL_FINISHED,
        outcome=granted.outcome,
        value=granted.value,
        error=granted.error,
        duration_ms=round(granted.duration_ms, 1),
    )
    return granted


async def _execute_step(
    dispatcher: Dispatcher,
    calls: Sequence[ToolCall],
    approve: ApprovalHook | None,
    emit: Emitter | None,
    max_parallel: int,
) -> list[ToolResult]:
    """Run a step's tool calls, bounded by `max_parallel`.

    Calls that need approval are run one at a time regardless: two approval
    prompts racing for the same terminal or the same UI would be unreadable,
    and the user could not tell which one they were answering.
    """
    needs_approval = [
        call
        for call in calls
        if call.malformed is None
        and call.name in dispatcher.registry
        and dispatcher.requires_confirmation(dispatcher.registry.get(call.name))
    ]

    if needs_approval or max_parallel <= 1:
        return [await execute_call(dispatcher, call, approve, emit) for call in calls]

    semaphore = asyncio.Semaphore(max_parallel)

    async def guarded(call: ToolCall) -> ToolResult:
        async with semaphore:
            return await execute_call(dispatcher, call, approve, emit)

    return list(await asyncio.gather(*(guarded(call) for call in calls)))


async def run_turn(
    llm: LLM,
    dispatcher: Dispatcher,
    user_message: str,
    history: Sequence[dict[str, Any]] | None = None,
    approve: ApprovalHook | None = None,
    max_steps: int = MAX_STEPS,
    expose_risky: bool = True,
    bus: EventBus | None = None,
    session_id: str = "",
    max_parallel: int = MAX_PARALLEL_TOOLS,
    reflect: bool = True,
    max_reflections: int = MAX_REFLECTIONS,
) -> Turn:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if max_reflections < 0:
        raise ValueError("max_reflections cannot be negative")

    emit = Emitter(bus, session_id) if bus is not None else None
    messages = build_messages(user_message, history)
    tools = dispatcher.registry.openai_tools(include_risky=expose_risky)
    steps: list[Step] = []
    reflections = 0
    verdict: Critique | None = None

    async def announce(event_type: str, **data: Any) -> None:
        if emit is not None:
            await emit(event_type, **data)

    await announce(EventType.TURN_STARTED, message=user_message, tools=len(tools))

    try:
        for remaining in range(max_steps, 0, -1):
            await announce(EventType.THINKING, step=len(steps) + 1)
            completion = await llm.complete(messages, tools)
            step = Step(completion=completion)
            steps.append(step)

            if not completion.wants_tools:
                messages.append({"role": "assistant", "content": completion.text})

                results = [r for step in steps for r in step.results]
                verdict = critique(completion.text, results) if reflect else None

                if (
                    verdict is not None
                    and not verdict.passed
                    and reflections < max_reflections
                    and remaining > 1
                ):
                    reflections += 1
                    await announce(
                        EventType.REFLECTION,
                        passed=False,
                        issues=[issue.to_dict() for issue in verdict.issues],
                        attempt=reflections,
                    )
                    messages.append({"role": "user", "content": verdict.gap()})
                    continue

                if verdict is not None:
                    await announce(
                        EventType.REFLECTION,
                        passed=verdict.passed,
                        issues=[issue.to_dict() for issue in verdict.issues],
                        attempt=reflections,
                    )

                await announce(EventType.ANSWER, text=completion.text)
                turn = Turn(
                    answer=completion.text,
                    messages=messages,
                    steps=steps,
                    session_id=session_id,
                    critique=verdict,
                    reflections=reflections,
                )
                await announce(EventType.TURN_FINISHED, **turn.to_dict())
                return turn

            messages.append(_assistant_message(completion))

            step.results = await _execute_step(
                dispatcher, completion.tool_calls, approve, emit, max_parallel
            )
            for call, result in zip(completion.tool_calls, step.results):
                messages.append(_tool_message(call, result))

            if remaining == 1:
                results = [r for step in steps for r in step.results]
                turn = Turn(
                    answer=completion.text,
                    messages=messages,
                    steps=steps,
                    stopped_at_limit=True,
                    session_id=session_id,
                    critique=(
                        critique(completion.text, results, stopped_at_limit=True)
                        if reflect
                        else None
                    ),
                    reflections=reflections,
                )
                await announce(EventType.TURN_FINISHED, **turn.to_dict())
                return turn
    except Exception as error:
        await announce(
            EventType.ERROR, error=f"{type(error).__name__}: {error}"
        )
        raise

    raise AssertionError("unreachable")
