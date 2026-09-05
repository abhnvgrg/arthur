from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from arthur.tools.mail import MailBox, MailError

DEFAULT_POLL_SECONDS = 120.0
MAX_REMEMBERED = 500
MAX_PER_POLL = 10


def _default_path() -> Path:
    configured = os.getenv("ARTHUR_MAIL_SEEN_FILE")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "mail_seen.json"


class SeenMail:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()

    def read(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload.get("uids", []) if isinstance(payload, dict) else []

    def remember(self, uids: list[str]) -> None:
        combined = self.read() + [uid for uid in uids if uid]
        trimmed = combined[-MAX_REMEMBERED:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"uids": trimmed}, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def describe(message: dict[str, Any]) -> str:
    return (
        f"From: {message.get('from', '')}\n"
        f"Subject: {message.get('subject', '')}\n"
        f"Uid: {message.get('uid', '')}\n"
        f"Body: {message.get('body', '')}"
    )


@dataclass
class MailWatcher:
    mailbox: MailBox
    fire: Callable[[str, str], Any]
    seen: SeenMail = field(default_factory=SeenMail)
    interval: float = DEFAULT_POLL_SECONDS
    failures: list[str] = field(default_factory=list)

    async def poll(self) -> list[dict[str, Any]]:
        try:
            messages = await asyncio.to_thread(self.mailbox.search, "unread", MAX_PER_POLL)
        except MailError as error:
            self.failures = [f"mail: {error}"]
            return []

        known = set(self.seen.read())
        fresh = [message for message in messages if message["uid"] not in known]
        if not fresh:
            return []

        self.seen.remember([message["uid"] for message in fresh])

        for message in fresh:
            outcome = self.fire("mail", describe(message))
            if asyncio.iscoroutine(outcome):
                await outcome

        return fresh

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                await self.poll()
            except Exception as error:
                self.failures = [f"poll: {error}"]
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue
