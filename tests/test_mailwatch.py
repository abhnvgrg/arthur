from __future__ import annotations

import pytest

from arthur.mailwatch import MailWatcher, SeenMail, describe

pytestmark = pytest.mark.asyncio


class FakeBox:
    def __init__(self, batches):
        self.batches = list(batches)
        self.searched = 0

    def search(self, query="unread", limit=10):
        self.searched += 1
        return self.batches.pop(0) if self.batches else []


def message(uid: str, subject: str = "Invoice") -> dict:
    return {
        "uid": uid,
        "from": "bank@example.com",
        "subject": subject,
        "body": "Please pay by Friday",
    }


def watcher(tmp_path, batches, fired):
    return MailWatcher(
        FakeBox(batches),
        fire=lambda event, context: fired.append((event, context)),
        seen=SeenMail(tmp_path / "seen.json"),
    )


def test_a_message_is_described_for_the_model():
    text = describe(message("11"))
    assert "From: bank@example.com" in text
    assert "Subject: Invoice" in text
    assert "Uid: 11" in text


async def test_new_mail_fires_the_trigger(tmp_path):
    fired = []
    found = await watcher(tmp_path, [[message("11")]], fired).poll()

    assert len(found) == 1
    assert fired[0][0] == "mail"
    assert "Invoice" in fired[0][1]


async def test_the_same_message_only_fires_once(tmp_path):
    fired = []
    watching = watcher(tmp_path, [[message("11")], [message("11")]], fired)

    await watching.poll()
    await watching.poll()

    assert len(fired) == 1


async def test_a_second_message_fires_again(tmp_path):
    fired = []
    watching = watcher(
        tmp_path, [[message("11")], [message("11"), message("12", "Receipt")]], fired
    )

    await watching.poll()
    await watching.poll()

    assert [context.count("Uid:") for _, context in fired] == [1, 1]
    assert "Receipt" in fired[1][1]


async def test_an_empty_inbox_fires_nothing(tmp_path):
    fired = []
    assert await watcher(tmp_path, [[]], fired).poll() == []
    assert fired == []


async def test_what_was_seen_survives_a_restart(tmp_path):
    fired = []
    await watcher(tmp_path, [[message("11")]], fired).poll()
    await watcher(tmp_path, [[message("11")]], fired).poll()

    assert len(fired) == 1


async def test_a_mailbox_that_fails_is_reported_not_raised(tmp_path):
    from arthur.tools.mail import MailError

    class Broken:
        def search(self, query="unread", limit=10):
            raise MailError("connection refused")

    watching = MailWatcher(Broken(), fire=lambda *a: None, seen=SeenMail(tmp_path / "s.json"))
    assert await watching.poll() == []
    assert "connection refused" in watching.failures[0]


async def test_an_async_trigger_is_awaited(tmp_path):
    fired = []

    async def fire(event, context):
        fired.append(event)

    watching = MailWatcher(
        FakeBox([[message("11")]]), fire=fire, seen=SeenMail(tmp_path / "s.json")
    )
    await watching.poll()
    assert fired == ["mail"]
