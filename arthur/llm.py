from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence

MAX_TOOL_CALLS_PER_STEP = 8

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)


class LLMError(Exception):
    pass


def status_of(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient(error: BaseException) -> bool:
    status = status_of(error)
    if status is not None:
        return status in RETRYABLE_STATUS
    if type(error).__name__ in RETRYABLE_NAMES:
        return True
    return isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError))


def retry_after_of(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


@dataclass
class RetryPolicy:
    attempts: int = 3
    base: float = 0.5
    cap: float = 8.0
    jitter: bool = True
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("A retry policy needs at least one attempt.")

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.cap)
        window = min(self.cap, self.base * (2**attempt))
        return random.uniform(0, window) if self.jitter else window

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        last: BaseException | None = None
        for attempt in range(self.attempts):
            try:
                return await operation()
            except Exception as error:
                if not is_transient(error) or attempt == self.attempts - 1:
                    raise
                last = error
                await self.sleep(self.delay_for(attempt, retry_after_of(error)))
        raise LLMError(f"Retries exhausted: {last}")


def policy_from_environment() -> RetryPolicy:
    return RetryPolicy(
        attempts=int(os.getenv("ARTHUR_LLM_ATTEMPTS", "3")),
        base=float(os.getenv("ARTHUR_LLM_BACKOFF", "0.5")),
        cap=float(os.getenv("ARTHUR_LLM_BACKOFF_CAP", "8")),
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    malformed: str | None = None


@dataclass(frozen=True)
class Completion:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Completion: ...


OnDelta = Callable[[str], Awaitable[None]]


class StreamingLLM(Protocol):
    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_delta: OnDelta,
    ) -> Completion: ...


@dataclass
class Fragments:
    id: str = ""
    name: str = ""
    arguments: str = ""

    def absorb(self, fragment: Any) -> None:
        if getattr(fragment, "id", None):
            self.id = fragment.id
        function = getattr(fragment, "function", None)
        if function is None:
            return
        if getattr(function, "name", None):
            self.name += function.name
        if getattr(function, "arguments", None):
            self.arguments += function.arguments

    def to_call(self) -> ToolCall:
        arguments, malformed = parse_arguments(self.arguments)
        return ToolCall(
            id=self.id,
            name=self.name,
            arguments=arguments,
            malformed=malformed,
        )


def _repair_json(raw: str) -> str:
    stripped = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    return re.sub(r",(\s*[}\]])", r"\1", stripped)


def parse_arguments(raw: str | dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None

    for candidate in (raw, _repair_json(raw)):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed, None
        return {}, f"Arguments must be a JSON object, got {type(parsed).__name__}"

    return {}, "Arguments were not valid JSON"


class ScriptedLLM:
    def __init__(self, script: Sequence[Completion]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Completion:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._script:
            raise LLMError("ScriptedLLM ran out of scripted completions")
        return self._script.pop(0)

    @property
    def exhausted(self) -> bool:
        return not self._script


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self.policy = policy or policy_from_environment()

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise LLMError(
                    "The openai package is not installed. pip install openai"
                ) from error

            if not self._api_key:
                raise LLMError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _request(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
        return request

    async def _send(self, request: dict[str, Any]) -> Any:
        async def call():
            return await self._get_client().chat.completions.create(**request)

        try:
            return await self.policy.run(call)
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(f"The model call failed: {error}") from error

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Completion:
        response = await self._send(self._request(messages, tools))
        message = response.choices[0].message

        calls = []
        for raw in (getattr(message, "tool_calls", None) or [])[:MAX_TOOL_CALLS_PER_STEP]:
            arguments, malformed = parse_arguments(raw.function.arguments)
            calls.append(
                ToolCall(
                    id=raw.id,
                    name=raw.function.name,
                    arguments=arguments,
                    malformed=malformed,
                )
            )

        return Completion(text=message.content, tool_calls=tuple(calls))

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        on_delta: OnDelta,
    ) -> Completion:
        request = self._request(messages, tools)
        request["stream"] = True
        chunks = await self._send(request)

        text: list[str] = []
        fragments: dict[int, Fragments] = {}

        try:
            async for chunk in chunks:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    text.append(content)
                    await on_delta(content)

                for fragment in getattr(delta, "tool_calls", None) or []:
                    index = getattr(fragment, "index", 0)
                    fragments.setdefault(index, Fragments()).absorb(fragment)
        except Exception as error:
            raise LLMError(f"The model stream failed: {error}") from error

        calls = [
            fragments[index].to_call()
            for index in sorted(fragments)[:MAX_TOOL_CALLS_PER_STEP]
        ]
        return Completion(text="".join(text) or None, tool_calls=tuple(calls))
