from __future__ import annotations

import asyncio
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Protocol, Sequence

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TITLE = 200
MAX_BODY = 2000


class NotifyError(Exception):
    pass


@dataclass(frozen=True)
class Action:
    label: str
    url: str
    method: str = "POST"
    body: str = ""
    clear: bool = True
    headers: tuple[tuple[str, str], ...] = ()

    def as_header(self) -> str:
        parts = [
            "http",
            self.label.replace(",", " ").replace(";", " "),
            self.url,
            f"method={self.method}",
            f"clear={'true' if self.clear else 'false'}",
        ]
        parts.extend(f"headers.{name}={value}" for name, value in self.headers)
        if self.body:
            parts.append(f"body={self.body}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Notification:
    title: str
    body: str = ""
    actions: tuple[Action, ...] = ()
    click: str = ""

    def clipped(self) -> "Notification":
        return Notification(
            self.title[:MAX_TITLE], self.body[:MAX_BODY], self.actions, self.click
        )


class Notifier(Protocol):
    name: str

    async def send(self, note: Notification) -> None: ...


@dataclass
class RecordingNotifier:
    name: str = "recording"
    sent: list[Notification] = field(default_factory=list)
    fail_with: Exception | None = None

    async def send(self, note: Notification) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append(note)


@dataclass
class EventNotifier:
    bus: Any
    session_id: str = "scheduler"
    name: str = "events"

    async def send(self, note: Notification) -> None:
        await self.bus.publish(
            self.session_id,
            "reminder",
            {"title": note.title, "body": note.body},
        )


class NtfyNotifier:
    name = "ntfy"

    def __init__(
        self,
        topic: str,
        server: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        if not topic:
            raise NotifyError("ntfy needs a topic")
        self.topic = topic
        self.server = (server or DEFAULT_NTFY_SERVER).rstrip("/")
        self.timeout = timeout
        self._transport = transport

    async def send(self, note: Notification) -> None:
        import httpx

        headers = {"Title": note.title.encode("ascii", "replace").decode()}
        if note.actions:
            headers["Actions"] = "; ".join(
                action.as_header() for action in note.actions[:3]
            )
        if note.click:
            headers["Click"] = note.click

        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self.server}/{self.topic}",
                content=note.body.encode("utf-8"),
                headers=headers,
            )
            if response.status_code >= 400:
                raise NotifyError(
                    f"ntfy refused the message: {response.status_code}"
                )


class ToastNotifier:
    name = "toast"

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend

    def _notifier(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            from plyer import notification
        except ImportError as error:
            raise NotifyError(
                "Desktop notifications need plyer. pip install plyer"
            ) from error
        return notification

    async def send(self, note: Notification) -> None:
        backend = self._notifier()
        await asyncio.to_thread(
            backend.notify,
            title=note.title,
            message=note.body,
            app_name="ARTHUR",
        )


@dataclass
class EmailNotifier:
    host: str
    port: int
    username: str
    password: str
    recipient: str
    sender: str = ""
    name: str = "email"
    opener: Any = None

    def _send(self, note: Notification) -> None:
        message = EmailMessage()
        message["Subject"] = note.title
        message["From"] = self.sender or self.username
        message["To"] = self.recipient
        message.set_content(note.body or note.title)

        opener = self.opener or smtplib.SMTP_SSL
        with opener(self.host, self.port) as server:
            if self.password:
                server.login(self.username, self.password)
            server.send_message(message)

    async def send(self, note: Notification) -> None:
        await asyncio.to_thread(self._send, note)


@dataclass
class FanOut:
    notifiers: Sequence[Notifier]
    name: str = "fanout"

    async def send(self, note: Notification) -> list[str]:
        if not self.notifiers:
            return []

        results = await asyncio.gather(
            *(one.send(note) for one in self.notifiers),
            return_exceptions=True,
        )
        return [
            f"{one.name}: {outcome}"
            for one, outcome in zip(self.notifiers, results)
            if isinstance(outcome, BaseException)
        ]


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_notifier(bus: Any = None) -> FanOut:
    notifiers: list[Notifier] = []

    if bus is not None:
        notifiers.append(EventNotifier(bus))

    topic = os.getenv("ARTHUR_NTFY_TOPIC", "").strip()
    if topic:
        notifiers.append(
            NtfyNotifier(topic, os.getenv("ARTHUR_NTFY_SERVER") or None)
        )

    if _flag("ARTHUR_TOAST"):
        notifiers.append(ToastNotifier())

    host = os.getenv("ARTHUR_SMTP_HOST", "").strip()
    recipient = os.getenv("ARTHUR_SMTP_TO", "").strip()
    if host and recipient:
        notifiers.append(
            EmailNotifier(
                host=host,
                port=int(os.getenv("ARTHUR_SMTP_PORT", "465")),
                username=os.getenv("ARTHUR_SMTP_USER", ""),
                password=os.getenv("ARTHUR_SMTP_PASSWORD", ""),
                recipient=recipient,
                sender=os.getenv("ARTHUR_SMTP_FROM", ""),
            )
        )

    return FanOut(notifiers)
