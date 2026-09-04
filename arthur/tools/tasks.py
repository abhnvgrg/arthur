from __future__ import annotations

import json
import os
import threading
import uuid
import re
from datetime import date, datetime, time, timedelta, timezone
from arthur import clock
from arthur.clock import ClockError
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MAX_TASKS = 500
Priority = Literal["low", "normal", "high"]


class TaskError(ValueError):
    pass


def _default_path() -> Path:
    configured = os.getenv("ARTHUR_TASKS_FILE")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "tasks.json"


TIME_SUFFIX = re.compile(r"\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$")

DUE_FORMS = (
    "Use YYYY-MM-DD, an ISO datetime, 'today', 'tomorrow', 'next week', "
    "or 'in N days', each optionally followed by a time such as '1pm' or '13:00'."
)


def normalise_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def local_zone() -> Any:
    try:
        return clock.local_zone()
    except ClockError as error:
        raise TaskError(str(error)) from error


def now_local() -> datetime:
    return datetime.now(local_zone())


def zone_name() -> str:
    return clock.zone_name()


def _render(day: date, moment: time | None) -> str:
    if moment is None:
        return day.isoformat()
    return datetime.combine(day, moment).isoformat(timespec="minutes")


def _read_time(match: re.Match[str]) -> time:
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        raise TaskError(f"{match.group(0).strip()!r} is not a valid time of day")
    return time(hour, minute)


def _split_time(text: str) -> tuple[str, time | None]:
    match = TIME_SUFFIX.search(text)
    if not match or not (match.group(2) or match.group(3)):
        return text, None
    return text[: match.start()].strip(), _read_time(match)


def parse_due(value: str | None) -> Optional[str]:
    if not value:
        return None

    text = normalise_spaces(value)
    if not text:
        return None

    lowered = text.lower()

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        pass
    else:
        if parsed.tzinfo is not None:
            here = parsed.astimezone(local_zone())
            if parsed.utcoffset() != here.utcoffset():
                raise TaskError(
                    f"Due dates are stored in local time, but {value!r} carries "
                    f"an offset that is not local ({zone_name()} is "
                    f"{here.strftime('%z')} then). Give the time the user said, "
                    "without a 'Z' or an offset: 'tomorrow 1pm' or "
                    "'2026-09-03T13:00'."
                )
            parsed = parsed.replace(tzinfo=None)
        timeless = (parsed.hour, parsed.minute, parsed.second) == (0, 0, 0)
        if timeless and "t" not in lowered:
            return parsed.date().isoformat()
        return parsed.replace(tzinfo=None).isoformat(timespec="minutes")

    stem, moment = _split_time(lowered)
    today = now_local().date()

    relative = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "next week": today + timedelta(days=7),
    }
    if stem in relative:
        return _render(relative[stem], moment)

    if stem.startswith("in ") and stem.endswith(("day", "days")):
        try:
            amount = int(stem.split()[1])
        except (IndexError, ValueError):
            raise TaskError(f"Could not read a number of days from {value!r}")
        return _render(today + timedelta(days=amount), moment)

    try:
        day = date.fromisoformat(stem)
    except ValueError:
        raise TaskError(f"Could not understand the due date {value!r}. {DUE_FORMS}")
    return _render(day, moment)


def as_datetime(due: str, end_of_day: bool = False) -> datetime:
    if "T" in due:
        return datetime.fromisoformat(due)
    day = date.fromisoformat(due)
    return datetime.combine(day, time(23, 59, 59) if end_of_day else time(0, 0))


class TaskStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self._lock = threading.Lock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload.get("tasks", []) if isinstance(payload, dict) else []

    def _write(self, tasks: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"tasks": tasks}, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def add(
        self,
        title: str,
        due: str | None = None,
        priority: Priority = "normal",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        due_date = parse_due(due)
        with self._lock:
            tasks = self._read()
            if len(tasks) >= MAX_TASKS:
                raise TaskError(f"The task list is full ({MAX_TASKS} tasks).")

            task = {
                "id": f"t_{uuid.uuid4().hex[:8]}",
                "title": normalise_spaces(title),
                "due": due_date,
                "priority": priority,
                "tags": sorted({tag.strip().lower() for tag in (tags or []) if tag.strip()}),
                "done": False,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "completed_at": None,
                "reminded_at": None,
            }
            tasks.append(task)
            self._write(tasks)
        return task

    def list(
        self,
        include_done: bool = False,
        tag: str | None = None,
        due_before: str | None = None,
    ) -> list[dict[str, Any]]:
        tasks = self._read()
        if not include_done:
            tasks = [task for task in tasks if not task["done"]]
        if tag:
            needle = tag.strip().lower()
            tasks = [task for task in tasks if needle in task.get("tags", [])]
        if due_before:
            cutoff = parse_due(due_before)
            if cutoff:
                limit = as_datetime(cutoff, end_of_day=True)
                tasks = [
                    task
                    for task in tasks
                    if task.get("due") and as_datetime(task["due"]) <= limit
                ]

        rank = {"high": 0, "normal": 1, "low": 2}
        return sorted(
            tasks,
            key=lambda task: (
                task.get("due") or "9999-12-31",
                rank.get(task.get("priority", "normal"), 1),
                task["created_at"],
            ),
        )

    def complete(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            tasks = self._read()
            for task in tasks:
                if task["id"] != task_id:
                    continue
                if task["done"]:
                    return {**task, "already_done": True}
                task["done"] = True
                task["completed_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                self._write(tasks)
                return {**task, "already_done": False}
        raise TaskError(f"No task with id {task_id!r}")

    def mark_reminded(self, task_id: str, when: str) -> bool:
        with self._lock:
            tasks = self._read()
            for task in tasks:
                if task["id"] != task_id:
                    continue
                task["reminded_at"] = when
                self._write(tasks)
                return True
        return False

    def remove(self, task_id: str) -> bool:
        with self._lock:
            tasks = self._read()
            remaining = [task for task in tasks if task["id"] != task_id]
            if len(remaining) == len(tasks):
                return False
            self._write(remaining)
        return True

    def overdue(self) -> list[dict[str, Any]]:
        now = now_local().replace(tzinfo=None)
        return [
            task
            for task in self.list()
            if task.get("due") and as_datetime(task["due"], end_of_day=True) < now
        ]


class AddTaskArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    due: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "When the task is due, in the user's local timezone. Use a "
            "phrase ('today', 'tomorrow', 'next week', 'in 3 days'), "
            "optionally with a time ('tomorrow 1pm', 'in 3 days at "
            "9:30am'), or 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'. A 'Z' "
            "suffix or UTC offset is rejected: a bare time is already "
            "local. If the user named another timezone, convert it to "
            "local time yourself before calling this."
        ),
    )
    priority: Priority = "normal"
    tags: list[str] = Field(default_factory=list, max_length=10)


class ListTasksArgs(BaseModel):
    include_done: bool = False
    tag: str | None = Field(default=None, max_length=40)
    due_before: str | None = Field(
        default=None,
        max_length=40,
        description="Latest due date to include, in the same forms as add_task's due.",
    )


class TaskIdArgs(BaseModel):
    task_id: str = Field(min_length=3, max_length=40)


class NoArgs(BaseModel):
    pass


def register(registry, store: TaskStore) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="add_task",
        description="Add a task to the to-do list, optionally with a due date and tags.",
        parameters=AddTaskArgs,
        risk=Risk.WRITES,
    )
    def add_task(args: AddTaskArgs) -> dict[str, Any]:
        task = store.add(args.title, args.due, args.priority, args.tags)
        return {**task, "timezone": zone_name()}

    @registry.tool(
        name="list_tasks",
        description="List outstanding tasks, optionally filtered by tag or due date.",
        parameters=ListTasksArgs,
        risk=Risk.READ_ONLY,
    )
    def list_tasks(args: ListTasksArgs) -> dict[str, Any]:
        tasks = store.list(args.include_done, args.tag, args.due_before)
        return {"tasks": tasks, "count": len(tasks), "timezone": zone_name()}

    @registry.tool(
        name="overdue_tasks",
        description="List tasks whose due date has already passed.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def overdue_tasks(_: NoArgs) -> dict[str, Any]:
        tasks = store.overdue()
        return {"tasks": tasks, "count": len(tasks), "timezone": zone_name()}

    @registry.tool(
        name="complete_task",
        description="Mark a task as done.",
        parameters=TaskIdArgs,
        risk=Risk.WRITES,
    )
    def complete_task(args: TaskIdArgs) -> dict[str, Any]:
        return store.complete(args.task_id)

    @registry.tool(
        name="delete_task",
        description="Permanently delete a task.",
        parameters=TaskIdArgs,
        risk=Risk.IRREVERSIBLE,
    )
    def delete_task(args: TaskIdArgs) -> dict[str, Any]:
        return {"task_id": args.task_id, "deleted": store.remove(args.task_id)}
