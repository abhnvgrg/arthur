from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from arthur.audit import AuditLog
from arthur.tools.registry import Risk, ToolRegistry, ToolSpec, UnknownTool


class Outcome:
    OK = "ok"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    CONFIRMATION_REQUIRED = "confirmation_required"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolResult:
    tool: str
    outcome: str
    value: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == Outcome.OK

    @property
    def needs_confirmation(self) -> bool:
        return self.outcome == Outcome.CONFIRMATION_REQUIRED

    def for_model(self) -> dict[str, Any]:
        if self.ok:
            return {"tool": self.tool, "ok": True, "result": self.value}
        return {
            "tool": self.tool,
            "ok": False,
            "outcome": self.outcome,
            "error": self.error,
        }


class Dispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditLog | None = None,
        auto_confirm: frozenset[Risk] = frozenset({Risk.READ_ONLY}),
    ) -> None:
        self.registry = registry
        self.audit = audit or AuditLog()
        self.auto_confirm = auto_confirm

    def _record(self, result: ToolResult) -> ToolResult:
        self.audit.record(
            tool=result.tool,
            outcome=result.outcome,
            arguments=result.arguments,
            detail=result.error,
            duration_ms=round(result.duration_ms, 3),
        )
        return result

    def requires_confirmation(self, spec: ToolSpec) -> bool:
        return spec.risk not in self.auto_confirm

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        confirmed: bool = False,
    ) -> ToolResult:
        arguments = arguments or {}

        try:
            spec = self.registry.get(name)
        except UnknownTool as error:
            return self._record(
                ToolResult(
                    tool=name,
                    outcome=Outcome.UNKNOWN_TOOL,
                    error=str(error),
                    arguments=arguments,
                )
            )

        try:
            parsed = spec.parameters(**arguments)
        except ValidationError as error:
            return self._record(
                ToolResult(
                    tool=name,
                    outcome=Outcome.INVALID_ARGUMENTS,
                    error="; ".join(
                        f"{'.'.join(str(part) for part in item['loc']) or 'body'}: {item['msg']}"
                        for item in error.errors()
                    ),
                    arguments=arguments,
                )
            )

        if self.requires_confirmation(spec) and not confirmed:
            return self._record(
                ToolResult(
                    tool=name,
                    outcome=Outcome.CONFIRMATION_REQUIRED,
                    error=f"{name} is {spec.risk.value} and needs explicit confirmation",
                    arguments=parsed.model_dump(),
                )
            )

        started = time.perf_counter()
        try:
            value = await asyncio.wait_for(
                self._call(spec, parsed), timeout=spec.timeout_seconds
            )
        except asyncio.TimeoutError:
            return self._record(
                ToolResult(
                    tool=name,
                    outcome=Outcome.TIMEOUT,
                    error=f"{name} exceeded {spec.timeout_seconds}s",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    arguments=parsed.model_dump(),
                )
            )
        except Exception as error:
            return self._record(
                ToolResult(
                    tool=name,
                    outcome=Outcome.FAILED,
                    error=f"{type(error).__name__}: {error}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    arguments=parsed.model_dump(),
                )
            )

        return self._record(
            ToolResult(
                tool=name,
                outcome=Outcome.OK,
                value=value,
                duration_ms=(time.perf_counter() - started) * 1000,
                arguments=parsed.model_dump(),
            )
        )

    @staticmethod
    async def _call(spec: ToolSpec, parsed) -> Any:
        if inspect.iscoroutinefunction(spec.handler):
            return await spec.handler(parsed)
        return await asyncio.to_thread(spec.handler, parsed)
