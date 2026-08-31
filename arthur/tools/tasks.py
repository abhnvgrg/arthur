from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
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


def parse_due(value: str | None) -> Optional[str]:
    if not value:
        return None

    text = value.strip().lower()
    today = datetime.now(timezone.utc).date()

    relative = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "next week": today + timedelta(days=7),
    }
    if text in relative:
        return relative[text].isoformat()

    if text.startswith("in ") and text.endswith(("day", "days")):
        try:
            amount = int(text.split()[1])
        except (IndexError, ValueError):
            raise TaskError(f"Could not read a number of days from {value!r}")
        return (today + timedelta(days=amount)).isoformat()

    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise TaskError(
            f"Could not understand the due date {value!r}. "
            "Use YYYY-MM-DD, 'today', 'tomorrow', 'next week', or 'in N days'."
        )


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
                "title": title.strip(),
                "due": due_date,
                "priority": priority,
                "tags": sorted({tag.strip().lower() for tag in (tags or []) if tag.strip()}),
                "done": False,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "completed_at": None,
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
            tasks = [
                task for task in tasks if task.get("due") and task["due"] <= cutoff
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

    def remove(self, task_id: str) -> bool:
        with self._lock:
            tasks = self._read()
            remaining = [task for task in tasks if task["id"] != task_id]
            if len(remaining) == len(tasks):
                return False
            self._write(remaining)
        return True

    def overdue(self) -> list[dict[str, Any]]:
        today = datetime.now(timezone.utc).date().isoformat()
        return [
            task
            for task in self.list()
            if task.get("due") and task["due"] < today
        ]


class AddTaskArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    due: str | None = Field(default=None, max_length=40)
    priority: Priority = "normal"
    tags: list[str] = Field(default_factory=list, max_length=10)


class ListTasksArgs(BaseModel):
    include_done: bool = False
    tag: str | None = Field(default=None, max_length=40)
    due_before: str | None = Field(default=None, max_length=40)


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
        return store.add(args.title, args.due, args.priority, args.tags)

    @registry.tool(
        name="list_tasks",
        description="List outstanding tasks, optionally filtered by tag or due date.",
        parameters=ListTasksArgs,
        risk=Risk.READ_ONLY,
    )
    def list_tasks(args: ListTasksArgs) -> dict[str, Any]:
        tasks = store.list(args.include_done, args.tag, args.due_before)
        return {"tasks": tasks, "count": len(tasks)}

    @registry.tool(
        name="overdue_tasks",
        description="List tasks whose due date has already passed.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def overdue_tasks(_: NoArgs) -> dict[str, Any]:
        tasks = store.overdue()
        return {"tasks": tasks, "count": len(tasks)}

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
