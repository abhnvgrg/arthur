from __future__ import annotations

import httpx
import pytest

from arthur.notify import (
    EmailNotifier,
    EventNotifier,
    FanOut,
    Notification,
    NotifyError,
    NtfyNotifier,
    RecordingNotifier,
    ToastNotifier,
    build_notifier,
)

pytestmark = pytest.mark.asyncio


def note(title: str = "Meeting", body: str = "Due Thursday, 13:00") -> Notification:
    return Notification(title, body)


async def test_a_long_notification_is_clipped():
    clipped = Notification("t" * 500, "b" * 5000).clipped()
    assert len(clipped.title) == 200
    assert len(clipped.body) == 2000


async def test_fanout_reports_nothing_when_all_succeed():
    one, two = RecordingNotifier(name="one"), RecordingNotifier(name="two")

    assert await FanOut([one, two]).send(note()) == []
    assert len(one.sent) == 1
    assert len(two.sent) == 1


async def test_fanout_names_the_channel_that_failed():
    good = RecordingNotifier(name="good")
    bad = RecordingNotifier(name="bad", fail_with=RuntimeError("down"))

    failures = await FanOut([bad, good]).send(note())

    assert len(failures) == 1
    assert failures[0].startswith("bad: ")
    assert len(good.sent) == 1


async def test_fanout_with_no_channels_is_not_an_error():
    assert await FanOut([]).send(note()) == []


async def test_ntfy_posts_the_body_to_the_topic():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        seen["title"] = request.headers.get("title")
        return httpx.Response(200)

    notifier = NtfyNotifier(
        "arthur-abc123", transport=httpx.MockTransport(handler)
    )
    await notifier.send(note())

    assert seen["url"] == "https://ntfy.sh/arthur-abc123"
    assert seen["body"] == "Due Thursday, 13:00"
    assert seen["title"] == "Meeting"


async def test_ntfy_reports_a_refusal():
    notifier = NtfyNotifier(
        "topic", transport=httpx.MockTransport(lambda _: httpx.Response(403))
    )
    with pytest.raises(NotifyError, match="403"):
        await notifier.send(note())


async def test_ntfy_needs_a_topic():
    with pytest.raises(NotifyError, match="needs a topic"):
        NtfyNotifier("")


async def test_ntfy_survives_a_non_ascii_title():
    def handler(request: httpx.Request) -> httpx.Response:
        request.headers.get("title").encode("ascii")
        return httpx.Response(200)

    notifier = NtfyNotifier("topic", transport=httpx.MockTransport(handler))
    await notifier.send(Notification("Café meeting", "body"))


class FakeSMTP:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.logged_in = None
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message):
        self.messages.append(message)


async def test_email_sends_a_message_to_the_recipient():
    opened = []

    def opener(host, port):
        server = FakeSMTP(host, port)
        opened.append(server)
        return server

    notifier = EmailNotifier(
        host="smtp.example.com",
        port=465,
        username="me@example.com",
        password="secret",
        recipient="me@example.com",
        opener=opener,
    )
    await notifier.send(note())

    server = opened[0]
    assert server.host == "smtp.example.com"
    assert server.logged_in == ("me@example.com", "secret")
    assert server.messages[0]["Subject"] == "Meeting"
    assert server.messages[0]["To"] == "me@example.com"


async def test_events_are_published_to_the_bus():
    published = []

    class Bus:
        async def publish(self, session_id, kind, payload):
            published.append((session_id, kind, payload))

    await EventNotifier(Bus()).send(note())

    assert published[0][1] == "reminder"
    assert published[0][2]["title"] == "Meeting"


async def test_a_missing_plyer_is_a_clean_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "plyer":
            raise ImportError("no plyer here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(NotifyError, match="pip install plyer"):
        await ToastNotifier().send(note())


async def test_a_toast_goes_to_the_backend():
    seen = {}

    class Backend:
        def notify(self, **kwargs):
            seen.update(kwargs)

    await ToastNotifier(backend=Backend()).send(note())

    assert seen["title"] == "Meeting"
    assert seen["app_name"] == "ARTHUR"


async def test_nothing_is_configured_by_default(monkeypatch):
    for name in (
        "ARTHUR_NTFY_TOPIC",
        "ARTHUR_TOAST",
        "ARTHUR_SMTP_HOST",
        "ARTHUR_SMTP_TO",
    ):
        monkeypatch.delenv(name, raising=False)

    assert build_notifier().notifiers == []


async def test_channels_are_built_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARTHUR_NTFY_TOPIC", "arthur-abc123")
    monkeypatch.setenv("ARTHUR_TOAST", "1")
    monkeypatch.setenv("ARTHUR_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ARTHUR_SMTP_TO", "me@example.com")

    names = [one.name for one in build_notifier().notifiers]

    assert names == ["ntfy", "toast", "email"]


async def test_an_empty_topic_does_not_configure_ntfy(monkeypatch):
    monkeypatch.setenv("ARTHUR_NTFY_TOPIC", "   ")
    monkeypatch.delenv("ARTHUR_SMTP_HOST", raising=False)
    monkeypatch.delenv("ARTHUR_TOAST", raising=False)

    assert build_notifier().notifiers == []


async def test_email_needs_both_a_host_and_a_recipient(monkeypatch):
    monkeypatch.delenv("ARTHUR_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("ARTHUR_TOAST", raising=False)
    monkeypatch.setenv("ARTHUR_SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("ARTHUR_SMTP_TO", raising=False)

    assert build_notifier().notifiers == []
