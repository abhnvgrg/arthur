from __future__ import annotations

from email.message import EmailMessage

import pytest

from arthur.dispatch import Dispatcher
from arthur.tools.builtins import build_registry
from arthur.tools.mail import (
    MailAccount,
    MailBox,
    MailError,
    MailUnavailable,
    Postman,
    build_criteria,
    compose,
    register,
    summarise,
)

ACCOUNT = MailAccount(
    host="imap.example.com",
    username="me@example.com",
    password="secret",
    smtp_host="smtp.example.com",
    sender="me@example.com",
)


def letter(subject: str = "Invoice", sender: str = "bank@example.com", body: str = "Please pay") -> bytes:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "me@example.com"
    message["Subject"] = subject
    message["Message-ID"] = "<abc@example.com>"
    message.set_content(body)
    return message.as_bytes()


class FakeIMAP:
    def __init__(self, host, port, messages=None, fail_login=False):
        self.host = host
        self.port = port
        self.messages = messages or {b"11": letter()}
        self.fail_login = fail_login
        self.selected = None
        self.readonly = None
        self.stored: list[tuple] = []
        self.copied: list[tuple] = []
        self.appended: list[tuple] = []
        self.expunged = False
        self.logged_out = False

    def login(self, username, password):
        if self.fail_login:
            raise OSError("bad credentials")
        return "OK", []

    def select(self, folder, readonly=True):
        self.selected = folder
        self.readonly = readonly
        return "OK", [b"1"]

    def uid(self, command, *args):
        if command == "SEARCH":
            return "OK", [b" ".join(self.messages)]
        if command == "FETCH":
            uid = args[0]
            if uid not in self.messages:
                return "NO", [None]
            return "OK", [(b"1 (RFC822 {})", self.messages[uid])]
        if command == "STORE":
            self.stored.append(args)
            return "OK", []
        if command == "COPY":
            self.copied.append(args)
            return "OK", []
        return "NO", []

    def append(self, folder, flags, when, payload):
        self.appended.append((folder, payload))
        return "OK", []

    def expunge(self):
        self.expunged = True
        return "OK", []

    def logout(self):
        self.logged_out = True


def opener(messages=None, **kwargs):
    made = {}

    def build(host, port):
        client = FakeIMAP(host, port, messages, **kwargs)
        made["client"] = client
        return client

    build.made = made
    return build


def test_keywords_map_to_imap_flags():
    assert build_criteria("unread") == "UNSEEN"
    assert build_criteria("all") == "ALL"
    assert build_criteria("flagged") == "FLAGGED"


def test_field_searches_are_quoted():
    assert build_criteria("from:bank") == 'FROM "bank"'
    assert build_criteria("subject:invoice") == 'SUBJECT "invoice"'


def test_free_text_searches_the_whole_message():
    assert build_criteria("tax return") == 'TEXT "tax return"'


def test_quotes_cannot_break_out_of_a_search():
    assert '"' not in build_criteria('pay" OR ALL')[6:-1]


def test_an_empty_query_means_everything():
    assert build_criteria("   ") == "ALL"


def test_a_message_is_summarised_into_plain_fields():
    summary = summarise("11", letter())
    assert summary["uid"] == "11"
    assert summary["from"] == "bank@example.com"
    assert summary["subject"] == "Invoice"
    assert "Please pay" in summary["body"]
    assert summary["truncated"] is False


def test_long_bodies_are_marked_as_truncated():
    summary = summarise("11", letter(body="x" * 900), body_limit=100)
    assert len(summary["body"]) == 100
    assert summary["truncated"] is True


def test_encoded_subjects_are_decoded():
    message = EmailMessage()
    message["Subject"] = "=?utf-8?q?Caf=C3=A9?="
    message.set_content("hi")
    assert summarise("1", message.as_bytes())["subject"] == "Café"


def test_missing_credentials_are_reported_clearly(monkeypatch):
    for name in ("ARTHUR_IMAP_HOST", "ARTHUR_IMAP_USER", "ARTHUR_IMAP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MailUnavailable, match="ARTHUR_IMAP_HOST"):
        MailAccount.from_environment()


def test_a_failed_sign_in_names_the_host():
    box = MailBox(ACCOUNT, opener=opener(fail_login=True))
    with pytest.raises(MailError, match="imap.example.com"):
        box.search()


def test_searching_returns_summaries_and_logs_out():
    build = opener()
    box = MailBox(ACCOUNT, opener=build)
    found = box.search("unread", limit=5)

    assert [message["subject"] for message in found] == ["Invoice"]
    assert build.made["client"].logged_out is True


def test_searching_opens_the_mailbox_read_only():
    build = opener()
    MailBox(ACCOUNT, opener=build).search()
    assert build.made["client"].readonly is True


def test_reading_an_unknown_uid_is_an_error():
    box = MailBox(ACCOUNT, opener=opener())
    with pytest.raises(MailError, match="uid"):
        box.read("999")


def test_marking_read_stores_the_seen_flag():
    build = opener()
    assert MailBox(ACCOUNT, opener=build).flag("11", "(\\Seen)") is True
    assert build.made["client"].stored[0][1:] == ("+FLAGS", "(\\Seen)")


def test_moving_copies_then_deletes_and_expunges():
    build = opener()
    assert MailBox(ACCOUNT, opener=build).move("11", "Archive") is True

    client = build.made["client"]
    assert client.copied[0][1] == "Archive"
    assert client.stored[0][1:] == ("+FLAGS", "(\\Deleted)")
    assert client.expunged is True


def test_a_draft_is_appended_to_the_drafts_folder():
    build = opener()
    box = MailBox(ACCOUNT, opener=build)
    assert box.save_draft(compose(ACCOUNT, "you@example.com", "Hi", "there")) is True
    assert build.made["client"].appended[0][0] == ACCOUNT.drafts_folder


def test_composing_rejects_an_address_that_is_not_one():
    with pytest.raises(MailError, match="not an email address"):
        compose(ACCOUNT, "not-an-address", "Hi", "there")


def test_composing_refuses_a_crowd():
    crowd = ",".join(f"person{n}@example.com" for n in range(12))
    with pytest.raises(MailError, match="At most"):
        compose(ACCOUNT, crowd, "Hi", "there")


def test_a_reply_threads_onto_the_original():
    message = compose(ACCOUNT, "you@example.com", "Re: Hi", "yes", in_reply_to="<abc@x>")
    assert message["In-Reply-To"] == "<abc@x>"
    assert message["References"] == "<abc@x>"


class FakeSMTP:
    sent: list[EmailMessage] = []

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def login(self, username, password):
        return None

    def send_message(self, message):
        FakeSMTP.sent.append(message)


def test_sending_reports_the_recipient():
    FakeSMTP.sent = []
    outcome = Postman(ACCOUNT, opener=FakeSMTP).send(
        compose(ACCOUNT, "you@example.com", "Hi", "there")
    )
    assert outcome == {"to": "you@example.com", "subject": "Hi", "sent": True}
    assert FakeSMTP.sent[0]["To"] == "you@example.com"


def test_a_refused_send_is_wrapped():
    class Refusing(FakeSMTP):
        def send_message(self, message):
            raise OSError("relay denied")

    with pytest.raises(MailError, match="relay denied"):
        Postman(ACCOUNT, opener=Refusing).send(
            compose(ACCOUNT, "you@example.com", "Hi", "there")
        )


def mail_dispatcher(audit):
    registry = build_registry(include_tasks=False, include_files=False, include_convert=False)
    box = MailBox(ACCOUNT, opener=opener())
    register(registry, box, Postman(ACCOUNT, opener=FakeSMTP))
    return Dispatcher(registry, audit=audit)


async def test_reading_mail_needs_no_approval(audit):
    result = await mail_dispatcher(audit).invoke("search_mail", {"query": "unread"})
    assert result.ok
    assert result.value["count"] == 1


async def test_archiving_needs_approval(audit):
    result = await mail_dispatcher(audit).invoke("archive_mail", {"uid": "11"})
    assert result.needs_confirmation


async def test_sending_is_irreversible_and_gated(audit):
    dispatcher = mail_dispatcher(audit)
    spec = dispatcher.registry.get("send_mail")

    assert spec.risk.value == "irreversible"
    result = await dispatcher.invoke(
        "send_mail", {"to": "you@example.com", "subject": "Hi", "body": "there"}
    )
    assert result.needs_confirmation


async def test_a_draft_reply_quotes_the_original_subject(audit):
    result = await mail_dispatcher(audit).invoke(
        "draft_reply", {"uid": "11", "body": "Paid, thanks"}, confirmed=True
    )
    assert result.ok
    assert result.value["subject"] == "Re: Invoice"
    assert result.value["to"] == "bank@example.com"
