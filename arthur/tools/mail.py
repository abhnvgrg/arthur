from __future__ import annotations

import email
import imaplib
import os
import smtplib
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_IMAP_PORT = 993
DEFAULT_SMTP_PORT = 465
DEFAULT_FOLDER = "INBOX"
DEFAULT_ARCHIVE = "[Gmail]/All Mail"
DEFAULT_DRAFTS = "[Gmail]/Drafts"

MAX_FETCH = 25
BODY_LIMIT = 4000
PREVIEW_LIMIT = 300
MAX_RECIPIENTS = 10


class MailError(Exception):
    pass


class MailUnavailable(MailError):
    pass


@dataclass(frozen=True)
class MailAccount:
    host: str
    username: str
    password: str
    port: int = DEFAULT_IMAP_PORT
    folder: str = DEFAULT_FOLDER
    archive_folder: str = DEFAULT_ARCHIVE
    drafts_folder: str = DEFAULT_DRAFTS
    smtp_host: str = ""
    smtp_port: int = DEFAULT_SMTP_PORT
    sender: str = ""

    @classmethod
    def from_environment(cls) -> "MailAccount":
        host = os.getenv("ARTHUR_IMAP_HOST", "").strip()
        username = os.getenv("ARTHUR_IMAP_USER", "").strip()
        password = os.getenv("ARTHUR_IMAP_PASSWORD", "")

        if not (host and username and password):
            raise MailUnavailable(
                "Mail needs ARTHUR_IMAP_HOST, ARTHUR_IMAP_USER and "
                "ARTHUR_IMAP_PASSWORD to be set"
            )

        return cls(
            host=host,
            username=username,
            password=password,
            port=int(os.getenv("ARTHUR_IMAP_PORT", str(DEFAULT_IMAP_PORT))),
            folder=os.getenv("ARTHUR_IMAP_FOLDER", DEFAULT_FOLDER),
            archive_folder=os.getenv("ARTHUR_IMAP_ARCHIVE", DEFAULT_ARCHIVE),
            drafts_folder=os.getenv("ARTHUR_IMAP_DRAFTS", DEFAULT_DRAFTS),
            smtp_host=os.getenv("ARTHUR_SMTP_HOST", "").strip(),
            smtp_port=int(os.getenv("ARTHUR_SMTP_PORT", str(DEFAULT_SMTP_PORT))),
            sender=os.getenv("ARTHUR_SMTP_FROM", "").strip() or username,
        )


def configured() -> bool:
    return bool(
        os.getenv("ARTHUR_IMAP_HOST", "").strip()
        and os.getenv("ARTHUR_IMAP_USER", "").strip()
        and os.getenv("ARTHUR_IMAP_PASSWORD", "")
    )


def decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (UnicodeDecodeError, LookupError, ValueError):
        return value


def plain_text(message: email.message.Message) -> str:
    if not message.is_multipart():
        payload = message.get_payload(decode=True)
        if payload is None:
            return str(message.get_payload())
        charset = message.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")

    for part in message.walk():
        if part.get_content_type() != "text/plain":
            continue
        if "attachment" in str(part.get("Content-Disposition", "")):
            continue
        payload = part.get_payload(decode=True)
        if payload:
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def summarise(uid: str, raw: bytes, body_limit: int = PREVIEW_LIMIT) -> dict[str, Any]:
    message = email.message_from_bytes(raw)
    body = " ".join(plain_text(message).split())
    return {
        "uid": uid,
        "from": decode(message.get("From")),
        "to": decode(message.get("To")),
        "subject": decode(message.get("Subject")),
        "date": decode(message.get("Date")),
        "message_id": message.get("Message-ID", ""),
        "body": body[:body_limit],
        "truncated": len(body) > body_limit,
    }


IMAP_KEYWORDS = {
    "unread": "UNSEEN",
    "unseen": "UNSEEN",
    "read": "SEEN",
    "flagged": "FLAGGED",
    "all": "ALL",
}


def build_criteria(query: str) -> str:
    text = " ".join(query.strip().split())
    if not text:
        return "ALL"

    lowered = text.lower()
    if lowered in IMAP_KEYWORDS:
        return IMAP_KEYWORDS[lowered]

    for prefix, key in (("from:", "FROM"), ("to:", "TO"), ("subject:", "SUBJECT")):
        if lowered.startswith(prefix):
            value = text[len(prefix) :].strip().replace('"', "")
            return f'{key} "{value}"'

    return f'TEXT "{text.replace(chr(34), "")}"'


class MailBox:
    def __init__(self, account: MailAccount | None = None, opener: Any = None) -> None:
        self.account = account or MailAccount.from_environment()
        self.opener = opener or imaplib.IMAP4_SSL

    def _connect(self):
        try:
            client = self.opener(self.account.host, self.account.port)
            client.login(self.account.username, self.account.password)
        except Exception as error:
            raise MailError(f"Could not sign in to {self.account.host}: {error}") from error
        return client

    def _select(self, client, folder: str | None = None, readonly: bool = True) -> None:
        status, _ = client.select(folder or self.account.folder, readonly=readonly)
        if status != "OK":
            raise MailError(f"Could not open the folder {folder or self.account.folder!r}")

    def search(self, query: str = "unread", limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(limit, MAX_FETCH))
        client = self._connect()
        try:
            self._select(client)
            status, data = client.uid("SEARCH", None, build_criteria(query))
            if status != "OK":
                raise MailError(f"The mail search failed: {query!r}")

            uids = (data[0] or b"").split()[-limit:]
            found = []
            for uid in reversed(uids):
                status, payload = client.uid("FETCH", uid, "(RFC822)")
                if status != "OK" or not payload or not isinstance(payload[0], tuple):
                    continue
                found.append(summarise(uid.decode(), payload[0][1]))
            return found
        finally:
            self._close(client)

    def read(self, uid: str) -> dict[str, Any]:
        client = self._connect()
        try:
            self._select(client)
            status, payload = client.uid("FETCH", uid.encode(), "(RFC822)")
            if status != "OK" or not payload or not isinstance(payload[0], tuple):
                raise MailError(f"No message with uid {uid!r}")
            return summarise(uid, payload[0][1], body_limit=BODY_LIMIT)
        finally:
            self._close(client)

    def flag(self, uid: str, flags: str, remove: bool = False) -> bool:
        client = self._connect()
        try:
            self._select(client, readonly=False)
            status, _ = client.uid(
                "STORE", uid.encode(), "-FLAGS" if remove else "+FLAGS", flags
            )
            return status == "OK"
        finally:
            self._close(client)

    def move(self, uid: str, folder: str) -> bool:
        client = self._connect()
        try:
            self._select(client, readonly=False)
            status, _ = client.uid("COPY", uid.encode(), folder)
            if status != "OK":
                return False
            client.uid("STORE", uid.encode(), "+FLAGS", "(\\Deleted)")
            client.expunge()
            return True
        finally:
            self._close(client)

    def save_draft(self, message: EmailMessage) -> bool:
        client = self._connect()
        try:
            status, _ = client.append(
                self.account.drafts_folder, "", None, message.as_bytes()
            )
            return status == "OK"
        finally:
            self._close(client)

    @staticmethod
    def _close(client) -> None:
        try:
            client.logout()
        except Exception:
            pass


def compose(
    account: MailAccount,
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
) -> EmailMessage:
    recipients = [part.strip() for part in to.split(",") if part.strip()]
    if not recipients:
        raise MailError("A message needs at least one recipient")
    if len(recipients) > MAX_RECIPIENTS:
        raise MailError(f"At most {MAX_RECIPIENTS} recipients at a time")
    for address in recipients:
        if "@" not in parseaddr(address)[1]:
            raise MailError(f"{address!r} is not an email address")

    message = EmailMessage()
    message["From"] = account.sender or account.username
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
        message["References"] = in_reply_to
    message.set_content(body)
    return message


class Postman:
    def __init__(self, account: MailAccount | None = None, opener: Any = None) -> None:
        self.account = account or MailAccount.from_environment()
        self.opener = opener or smtplib.SMTP_SSL

    def send(self, message: EmailMessage) -> dict[str, Any]:
        host = self.account.smtp_host or self.account.host.replace("imap", "smtp")
        try:
            with self.opener(host, self.account.smtp_port) as server:
                if self.account.password:
                    server.login(self.account.username, self.account.password)
                server.send_message(message)
        except Exception as error:
            raise MailError(f"Could not send the message: {error}") from error
        return {"to": message["To"], "subject": message["Subject"], "sent": True}


class SearchMailArgs(BaseModel):
    query: str = Field(
        default="unread",
        max_length=200,
        description=(
            "'unread', 'all', 'flagged', or a search such as 'from:bank', "
            "'subject:invoice', or free text to look for anywhere in the message"
        ),
    )
    limit: int = Field(default=10, ge=1, le=MAX_FETCH)


class MailUidArgs(BaseModel):
    uid: str = Field(min_length=1, max_length=40, description="The uid from search_mail")


class MoveMailArgs(BaseModel):
    uid: str = Field(min_length=1, max_length=40)
    folder: str = Field(default="", max_length=120, description="Leave empty to use the archive folder")


class DraftReplyArgs(BaseModel):
    uid: str = Field(min_length=1, max_length=40, description="The message being replied to")
    body: str = Field(min_length=1, max_length=BODY_LIMIT)


class SendMailArgs(BaseModel):
    to: str = Field(min_length=3, max_length=400, description="Comma separated addresses")
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=BODY_LIMIT)


def register(registry, mailbox: MailBox, postman: Postman | None = None) -> None:
    from arthur.tools.registry import Risk

    post = postman or Postman(mailbox.account)

    @registry.tool(
        name="search_mail",
        description="Search the mailbox and return matching messages, newest first.",
        parameters=SearchMailArgs,
        risk=Risk.READ_ONLY,
        timeout_seconds=30.0,
    )
    def search_mail(args: SearchMailArgs) -> dict[str, Any]:
        messages = mailbox.search(args.query, args.limit)
        return {"query": args.query, "count": len(messages), "messages": messages}

    @registry.tool(
        name="read_mail",
        description="Read one message in full by its uid.",
        parameters=MailUidArgs,
        risk=Risk.READ_ONLY,
        timeout_seconds=30.0,
    )
    def read_mail(args: MailUidArgs) -> dict[str, Any]:
        return mailbox.read(args.uid)

    @registry.tool(
        name="mark_mail_read",
        description="Mark a message as read.",
        parameters=MailUidArgs,
        risk=Risk.WRITES,
        timeout_seconds=30.0,
    )
    def mark_mail_read(args: MailUidArgs) -> dict[str, Any]:
        return {"uid": args.uid, "marked": mailbox.flag(args.uid, "(\\Seen)")}

    @registry.tool(
        name="archive_mail",
        description="Move a message out of the inbox into the archive folder.",
        parameters=MoveMailArgs,
        risk=Risk.WRITES,
        timeout_seconds=30.0,
    )
    def archive_mail(args: MoveMailArgs) -> dict[str, Any]:
        folder = args.folder or mailbox.account.archive_folder
        return {"uid": args.uid, "folder": folder, "moved": mailbox.move(args.uid, folder)}

    @registry.tool(
        name="draft_reply",
        description=(
            "Write a reply to a message and save it in the drafts folder without "
            "sending it. Prefer this over send_mail unless the user asks to send."
        ),
        parameters=DraftReplyArgs,
        risk=Risk.WRITES,
        timeout_seconds=30.0,
    )
    def draft_reply(args: DraftReplyArgs) -> dict[str, Any]:
        original = mailbox.read(args.uid)
        subject = original["subject"]
        reply = compose(
            mailbox.account,
            to=original["from"],
            subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
            body=args.body,
            in_reply_to=original.get("message_id", ""),
        )
        return {
            "uid": args.uid,
            "to": reply["To"],
            "subject": reply["Subject"],
            "saved": mailbox.save_draft(reply),
        }

    @registry.tool(
        name="send_mail",
        description="Send an email immediately. This cannot be undone.",
        parameters=SendMailArgs,
        risk=Risk.IRREVERSIBLE,
        timeout_seconds=30.0,
    )
    def send_mail(args: SendMailArgs) -> dict[str, Any]:
        return post.send(compose(mailbox.account, args.to, args.subject, args.body))
