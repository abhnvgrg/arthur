from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

MAX_TOOL_CALLS_PER_STEP = 8


class LLMError(Exception):
    pass


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
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

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

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Completion:
        request: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"

        response = await self._get_client().chat.completions.create(**request)
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
