from __future__ import annotations

import ast
import json
import math
import operator
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from arthur.tools.registry import Risk, ToolRegistry

MAX_EXPONENT = 64
MAX_EXPRESSION_LENGTH = 256

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalculationError(ValueError):
    pass


def evaluate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(
            f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculationError(f"Could not parse expression: {error.msg}") from error

    return _evaluate(tree.body)


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("Only numeric literals are allowed")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise CalculationError(f"Unknown name: {node.id}")
        return _CONSTANTS[node.id]

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculationError(f"Unsupported unary operator: {type(node.op).__name__}")
        return handler(_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculationError(f"Unsupported operator: {type(node.op).__name__}")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"Exponent above {MAX_EXPONENT} is not allowed")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculationError("Division by zero")

        return handler(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CalculationError("Only whitelisted maths functions may be called")
        if node.keywords:
            raise CalculationError("Keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])

    raise CalculationError(f"Unsupported expression element: {type(node).__name__}")


class MemoryStore:
    def __init__(self, path: Path | str | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            configured = os.getenv("ARTHUR_MEMORY_FILE")
            self.path = (
                Path(configured)
                if configured
                else Path.home() / ".arthur" / "memory.json"
            )
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.path)

    def remember(self, key: str, value: str) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            replaced = key in data
            data[key] = value
            self._write(data)
        return {"key": key, "replaced": replaced}

    def recall(self, key: str) -> str | None:
        return self._read().get(key)

    def forget(self, key: str) -> bool:
        with self._lock:
            data = self._read()
            if key not in data:
                return False
            del data[key]
            self._write(data)
        return True

    def keys(self) -> list[str]:
        return sorted(self._read())


class CurrentTimeArgs(BaseModel):
    timezone_name: str = Field(
        default="UTC", description="IANA timezone name, for example Asia/Kolkata"
    )


class CalculateArgs(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description="Arithmetic expression, for example (2 + 3) * sqrt(16)",
    )


class RememberArgs(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=4096)


class RecallArgs(BaseModel):
    key: str = Field(min_length=1, max_length=128)


class ForgetArgs(BaseModel):
    key: str = Field(min_length=1, max_length=128)


class NoArgs(BaseModel):
    pass


def build_registry(
    memory: MemoryStore | None = None,
    tasks: Any = None,
    workspace: Any = None,
    include_tasks: bool = True,
    include_files: bool = True,
    include_convert: bool = True,
    research_backend: Any = None,
    jobs: Any = None,
    include_jobs: bool = False,
    include_system: bool = False,
    mailbox: Any = None,
    include_mail: bool = False,
) -> ToolRegistry:
    store = memory or MemoryStore()
    registry = ToolRegistry()

    @registry.tool(
        name="current_time",
        description="Return the current date and time in a given IANA timezone.",
        parameters=CurrentTimeArgs,
        risk=Risk.READ_ONLY,
    )
    def current_time(args: CurrentTimeArgs) -> dict[str, str]:
        try:
            zone = ZoneInfo(args.timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"Unknown timezone: {args.timezone_name}") from error

        now = datetime.now(timezone.utc).astimezone(zone)
        return {
            "timezone": args.timezone_name,
            "iso": now.isoformat(timespec="seconds"),
            "spoken": now.strftime("%A %d %B %Y at %I:%M %p").replace(" 0", " "),
        }

    @registry.tool(
        name="calculate",
        description="Evaluate an arithmetic expression and return the numeric result.",
        parameters=CalculateArgs,
        risk=Risk.READ_ONLY,
    )
    def calculate(args: CalculateArgs) -> dict[str, Any]:
        return {"expression": args.expression, "result": evaluate(args.expression)}

    @registry.tool(
        name="recall",
        description="Look up a value previously stored under a key.",
        parameters=RecallArgs,
        risk=Risk.READ_ONLY,
    )
    def recall(args: RecallArgs) -> dict[str, Any]:
        value = store.recall(args.key)
        return {"key": args.key, "found": value is not None, "value": value}

    @registry.tool(
        name="list_memories",
        description="List the keys of everything currently remembered.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def list_memories(_: NoArgs) -> dict[str, Any]:
        keys = store.keys()
        return {"keys": keys, "count": len(keys)}

    @registry.tool(
        name="remember",
        description="Store a value under a key so it can be recalled later.",
        parameters=RememberArgs,
        risk=Risk.WRITES,
    )
    def remember(args: RememberArgs) -> dict[str, Any]:
        return store.remember(args.key, args.value)

    @registry.tool(
        name="forget",
        description="Permanently delete a remembered value.",
        parameters=ForgetArgs,
        risk=Risk.IRREVERSIBLE,
    )
    def forget(args: ForgetArgs) -> dict[str, Any]:
        return {"key": args.key, "deleted": store.forget(args.key)}

    if include_tasks:
        from arthur.tools import tasks as task_tools

        task_tools.register(registry, tasks or task_tools.TaskStore())

    if include_files:
        from arthur.tools import files as file_tools

        file_tools.register(registry, workspace or file_tools.Workspace())

    if include_convert:
        from arthur.tools import convert as convert_tools

        convert_tools.register(registry)

    if research_backend is not None:
        from arthur.tools import research as research_tools

        research_tools.register(registry, research_backend)

    if include_jobs:
        from arthur.jobs import JobStore
        from arthur.tools import scheduling as scheduling_tools

        scheduling_tools.register(registry, jobs or JobStore())

    if include_system:
        from arthur.tools import system as system_tools

        system_tools.register(registry)

    if include_mail:
        from arthur.tools import mail as mail_tools

        box = mailbox
        if box is None:
            box = mail_tools.MailBox() if mail_tools.configured() else None
        if box is not None:
            mail_tools.register(registry, box)

    return registry


def full_registry(**overrides: Any) -> ToolRegistry:
    settings: dict[str, Any] = {
        "include_jobs": True,
        "include_system": True,
        "include_mail": True,
    }
    settings.update(overrides)
    return build_registry(**settings)
