from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from arthur import clock
from arthur.dispatch import Dispatcher
from arthur.llm import LLM, LLMError
from arthur.notify import FanOut, Notification, build_notifier
from arthur.selection import run_turn

MAX_JOBS = 200
MIN_INTERVAL_SECONDS = 60.0
DEFAULT_TICK_SECONDS = 30.0
MAX_PER_TICK = 5
ANSWER_LIMIT = 1500

DAILY = "daily"
INTERVAL = "interval"
ONCE = "once"
EVENT = "event"
TRIGGERS = (DAILY, INTERVAL, ONCE, EVENT)

EVENT_NAMES = ("mail", "file", "webhook", "manual")

CLOCK_TIME = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")
EVERY = re.compile(r"^every\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours)$")


class JobError(ValueError):
    pass


def _read_clock(text: str) -> time:
    match = CLOCK_TIME.match(text.strip().lower())
    if match is None:
        raise JobError(f"Could not read a time of day from {text!r}. Use '08:00' or '8pm'.")

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        raise JobError(f"{text!r} is not a valid time of day")
    return time(hour, minute)


def parse_schedule(when: str) -> tuple[str, dict[str, Any]]:
    text = " ".join(when.strip().lower().split())
    if not text:
        raise JobError("A job needs a schedule")

    if text in EVENT_NAMES or text.startswith("on "):
        name = text[3:].strip() if text.startswith("on ") else text
        if name not in EVENT_NAMES:
            raise JobError(f"Unknown event {name!r}. Use one of: {', '.join(EVENT_NAMES)}")
        return EVENT, {"event": name}

    every = EVERY.match(text)
    if every is not None:
        amount = int(every.group(1))
        unit = every.group(2).rstrip("s")
        seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
        if seconds < MIN_INTERVAL_SECONDS:
            raise JobError(
                f"The shortest interval is {MIN_INTERVAL_SECONDS:.0f} seconds"
            )
        return INTERVAL, {"seconds": float(seconds)}

    if text.startswith("daily at "):
        return DAILY, {"at": _read_clock(text[9:]).isoformat(timespec="minutes")}
    if text.startswith("daily "):
        return DAILY, {"at": _read_clock(text[6:]).isoformat(timespec="minutes")}
    if text.startswith("every day at "):
        return DAILY, {"at": _read_clock(text[13:]).isoformat(timespec="minutes")}

    try:
        moment = datetime.fromisoformat(when.strip())
    except ValueError:
        pass
    else:
        return ONCE, {"at": moment.replace(tzinfo=None).isoformat(timespec="minutes")}

    raise JobError(
        f"Could not understand the schedule {when!r}. Use 'daily at 08:00', "
        "'every 30 minutes', an ISO datetime, or 'on mail'."
    )


@dataclass
class Job:
    id: str
    name: str
    prompt: str
    trigger: str
    settings: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    speak: bool = False
    notify: bool = True
    created_at: str = ""
    last_run: str | None = None
    last_answer: str | None = None
    runs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "trigger": self.trigger,
            "settings": self.settings,
            "enabled": self.enabled,
            "speak": self.speak,
            "notify": self.notify,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "last_answer": self.last_answer,
            "runs": self.runs,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Job":
        return cls(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            prompt=raw.get("prompt", ""),
            trigger=raw.get("trigger", EVENT),
            settings=raw.get("settings", {}),
            enabled=raw.get("enabled", True),
            speak=raw.get("speak", False),
            notify=raw.get("notify", True),
            created_at=raw.get("created_at", ""),
            last_run=raw.get("last_run"),
            last_answer=raw.get("last_answer"),
            runs=raw.get("runs", 0),
        )

    def describe(self) -> str:
        if self.trigger == DAILY:
            return f"daily at {self.settings.get('at', '?')}"
        if self.trigger == INTERVAL:
            return f"every {self.settings.get('seconds', 0) / 60:.0f} min"
        if self.trigger == ONCE:
            return f"once at {self.settings.get('at', '?')}"
        return f"on {self.settings.get('event', '?')}"

    def due_at(self, now: datetime) -> datetime | None:
        if not self.enabled:
            return None

        if self.trigger == EVENT:
            return None

        if self.trigger == ONCE:
            if self.runs:
                return None
            return datetime.fromisoformat(self.settings["at"])

        if self.trigger == INTERVAL:
            seconds = float(self.settings.get("seconds", MIN_INTERVAL_SECONDS))
            if self.last_run is None:
                return now
            return datetime.fromisoformat(self.last_run) + timedelta(seconds=seconds)

        moment = time.fromisoformat(self.settings["at"])
        today = datetime.combine(now.date(), moment)
        if self.last_run is not None:
            last = datetime.fromisoformat(self.last_run)
            if last >= today:
                return today + timedelta(days=1)
        return today

    def is_due(self, now: datetime, grace: float = 3600.0) -> bool:
        moment = self.due_at(now)
        if moment is None:
            return False
        return moment <= now and (now - moment).total_seconds() <= grace


def _default_path() -> Path:
    configured = os.getenv("ARTHUR_JOBS_FILE")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "jobs.json"


class JobStore:
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
        return payload.get("jobs", []) if isinstance(payload, dict) else []

    def _write(self, jobs: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def list(self, include_disabled: bool = True) -> list[Job]:
        jobs = [Job.from_dict(raw) for raw in self._read()]
        if not include_disabled:
            jobs = [job for job in jobs if job.enabled]
        return sorted(jobs, key=lambda job: job.created_at)

    def get(self, job_id: str) -> Job | None:
        for job in self.list():
            if job.id == job_id:
                return job
        return None

    def add(
        self,
        name: str,
        prompt: str,
        when: str,
        speak: bool = False,
        notify: bool = True,
    ) -> Job:
        trigger, settings = parse_schedule(when)
        job = Job(
            id=f"j_{uuid.uuid4().hex[:8]}",
            name=" ".join(name.split())[:80],
            prompt=" ".join(prompt.split()),
            trigger=trigger,
            settings=settings,
            speak=speak,
            notify=notify,
            created_at=clock.now_local().replace(tzinfo=None).isoformat(timespec="seconds"),
        )

        with self._lock:
            raw = self._read()
            if len(raw) >= MAX_JOBS:
                raise JobError(f"The job list is full ({MAX_JOBS} jobs).")
            raw.append(job.to_dict())
            self._write(raw)
        return job

    def save(self, job: Job) -> None:
        with self._lock:
            raw = self._read()
            for index, item in enumerate(raw):
                if item["id"] == job.id:
                    raw[index] = job.to_dict()
                    break
            else:
                raw.append(job.to_dict())
            self._write(raw)

    def set_enabled(self, job_id: str, enabled: bool) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        job.enabled = enabled
        self.save(job)
        return job

    def remove(self, job_id: str) -> bool:
        with self._lock:
            raw = self._read()
            remaining = [item for item in raw if item["id"] != job_id]
            if len(remaining) == len(raw):
                return False
            self._write(remaining)
        return True


@dataclass
class JobRun:
    job: Job
    answer: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def notification(self) -> Notification:
        body = self.answer or self.error or "(no answer)"
        return Notification(title=self.job.name, body=body[:ANSWER_LIMIT]).clipped()


class JobRunner:
    def __init__(
        self,
        llm: LLM,
        dispatcher: Dispatcher,
        store: JobStore | None = None,
        notifier: FanOut | None = None,
        clock_fn: Callable[[], datetime] = clock.now_local,
        interval: float = DEFAULT_TICK_SECONDS,
        speak: Callable[[str], Any] | None = None,
    ) -> None:
        self.llm = llm
        self.dispatcher = dispatcher
        self.store = store or JobStore()
        self.notifier = notifier if notifier is not None else build_notifier()
        self.clock = clock_fn
        self.interval = interval
        self.speak = speak
        self.failures: list[str] = []

    def due(self, now: datetime | None = None) -> list[Job]:
        moment = (now or self.clock()).replace(tzinfo=None)
        return [job for job in self.store.list(include_disabled=False) if job.is_due(moment)][
            :MAX_PER_TICK
        ]

    async def run_job(self, job: Job, context: str = "") -> JobRun:
        prompt = f"{job.prompt}\n\n{context}".strip() if context else job.prompt

        try:
            turn = await run_turn(
                self.llm,
                self.dispatcher,
                prompt,
                approve=lambda spec, arguments: False,
                max_reflections=0,
            )
        except LLMError as error:
            outcome = JobRun(job=job, answer=None, error=str(error))
        except Exception as error:
            outcome = JobRun(job=job, answer=None, error=f"{type(error).__name__}: {error}")
        else:
            outcome = JobRun(job=job, answer=turn.answer)

        job.runs += 1
        job.last_run = (self.clock().replace(tzinfo=None)).isoformat(timespec="seconds")
        job.last_answer = outcome.answer or outcome.error
        self.store.save(job)

        await self.deliver(outcome)
        return outcome

    async def deliver(self, outcome: JobRun) -> None:
        if outcome.job.notify and self.notifier.notifiers:
            self.failures = await self.notifier.send(outcome.notification())

        if outcome.job.speak and self.speak is not None and outcome.answer:
            result = self.speak(outcome.answer)
            if asyncio.iscoroutine(result):
                await result

    async def fire(self, event: str, context: str = "") -> list[JobRun]:
        runs = []
        for job in self.store.list(include_disabled=False):
            if job.trigger == EVENT and job.settings.get("event") == event:
                runs.append(await self.run_job(job, context))
        return runs

    async def tick(self, now: datetime | None = None) -> list[JobRun]:
        return [await self.run_job(job) for job in self.due(now)]

    async def run(
        self,
        stop: asyncio.Event | None = None,
        on_run: Callable[[JobRun], None] | None = None,
    ) -> None:
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                for outcome in await self.tick():
                    if on_run is not None:
                        on_run(outcome)
            except Exception as error:
                self.failures = [f"tick: {error}"]
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue
