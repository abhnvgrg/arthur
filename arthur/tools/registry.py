from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator

from pydantic import BaseModel

DEFAULT_TIMEOUT_SECONDS = 10.0


class Risk(enum.Enum):
    READ_ONLY = "read_only"
    WRITES = "writes"
    IRREVERSIBLE = "irreversible"


class ToolError(Exception):
    pass


class UnknownTool(ToolError):
    pass


class DuplicateTool(ToolError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: type[BaseModel]
    handler: Callable[..., Any] | Callable[..., Awaitable[Any]]
    risk: Risk = Risk.READ_ONLY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def needs_confirmation(self) -> bool:
        return self.risk is not Risk.READ_ONLY

    def json_schema(self) -> dict[str, Any]:
        schema = self.parameters.model_json_schema()
        schema.pop("title", None)
        return schema

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not spec.name or not spec.name.replace("_", "").isalnum():
            raise ToolError(
                f"Tool name must be alphanumeric with underscores: {spec.name!r}"
            )
        if spec.name in self._tools:
            raise DuplicateTool(f"A tool named {spec.name!r} is already registered")
        if spec.timeout_seconds <= 0:
            raise ToolError("timeout_seconds must be positive")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: type[BaseModel],
        risk: Risk = Risk.READ_ONLY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        def decorate(handler):
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    risk=risk,
                    timeout_seconds=timeout_seconds,
                )
            )
            return handler

        return decorate

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownTool(f"No tool named {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(sorted(self._tools.values(), key=lambda spec: spec.name))

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_tools(self, include_risky: bool = True) -> list[dict[str, Any]]:
        return [
            spec.as_openai_tool()
            for spec in self
            if include_risky or spec.risk is Risk.READ_ONLY
        ]
