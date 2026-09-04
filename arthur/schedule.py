from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Callable, Sequence

from arthur.clock import now_local
from arthur.notify import FanOut, Notification, build_notifier
from arthur.tools.tasks import TaskStore, as_datetime

DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_LEAD_SECONDS = 600.0
DEFAULT_GRACE_SECONDS = 3600.0
MAX_PER_TICK = 20


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Reminder:
    task_id: str
    title: str
    due: datetime

    def notification(self) -> Notification:
        return Notification(
            title=self.title,
            body=f"Due {self.due.strftime('%A %d %B, %H:%M')}",
        ).clipped()


def _seconds(name: str, fallback: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        raise ScheduleError(f"{name} must be a number of seconds, got {raw!r}")
    if value < 0:
        raise ScheduleError(f"{name} cannot be negative")
    return value


def day_start() -> time | None:
    raw = os.getenv("ARTHUR_REMIND_DAY_AT", "").strip()
    if not raw:
        return None
    try:
        hour, _, minute = raw.partition(":")
        return time(int(hour), int(minute or 0))
    except ValueError:
        raise ScheduleError(
            f"ARTHUR_REMIND_DAY_AT must look like '09:00', got {raw!r}"
        )


class Scheduler:
    def __init__(
        self,
        store: TaskStore | None = None,
        notifier: FanOut | None = None,
        clock: Callable[[], datetime] = now_local,
        interval: float | None = None,
        lead: float | None = None,
        grace: float | None = None,
    ) -> None:
        self.store = store or TaskStore()
        self.notifier = notifier if notifier is not None else build_notifier()
        self.clock = clock
        self.interval = (
            interval
            if interval is not None
            else _seconds("ARTHUR_REMIND_INTERVAL", DEFAULT_INTERVAL_SECONDS)
        )
        self.lead = (
            lead if lead is not None else _seconds("ARTHUR_REMIND_LEAD", DEFAULT_LEAD_SECONDS)
        )
        self.grace = (
            grace
            if grace is not None
            else _seconds("ARTHUR_REMIND_GRACE", DEFAULT_GRACE_SECONDS)
        )
        self.failures: list[str] = []

    def _moment(self, task: dict[str, Any]) -> datetime | None:
        due = task.get("due")
        if not due:
            return None
        if "T" in due:
            return as_datetime(due)

        opening = day_start()
        if opening is None:
            return None
        return datetime.combine(as_datetime(due).date(), opening)

    def pending(self, now: datetime | None = None) -> list[Reminder]:
        moment = (now or self.clock()).replace(tzinfo=None)
        earliest = moment - timedelta(seconds=self.grace)

        found: list[Reminder] = []
        for task in self.store.list():
            if task.get("reminded_at"):
                continue
            due = self._moment(task)
            if due is None:
                continue
            if due - timedelta(seconds=self.lead) > moment:
                continue
            if due < earliest:
                continue
            found.append(Reminder(task["id"], task["title"], due))

        found.sort(key=lambda reminder: reminder.due)
        return found[:MAX_PER_TICK]

    async def tick(self, now: datetime | None = None) -> list[Reminder]:
        moment = now or self.clock()
        sent: list[Reminder] = []

        for reminder in self.pending(moment):
            failures = await self.notifier.send(reminder.notification())
            self.failures = failures
            if failures and len(failures) == len(self.notifier.notifiers):
                continue
            self.store.mark_reminded(
                reminder.task_id, moment.isoformat(timespec="seconds")
            )
            sent.append(reminder)

        return sent

    async def run(
        self,
        stop: asyncio.Event | None = None,
        on_sent: Callable[[Reminder], None] | None = None,
    ) -> None:
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                for reminder in await self.tick():
                    if on_sent is not None:
                        on_sent(reminder)
            except Exception as error:
                self.failures = [f"tick: {error}"]
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue
