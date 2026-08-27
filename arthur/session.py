from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_CHARACTERS = 24_000


def new_session_id() -> str:
    return f"s_{uuid.uuid4().hex[:12]}"


def _characters(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(message, default=str)) for message in messages)


def trim_history(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_characters: int = MAX_HISTORY_CHARACTERS,
) -> list[dict[str, Any]]:
    """Drop the oldest exchanges until the history fits.

    Trimming happens from the front and never splits an assistant message from
    the tool results that answer it: a `tool` message whose matching
    `tool_calls` has been dropped is rejected by the API, so an orphan is worse
    than a shorter history.
    """
    trimmed = list(messages)

    while trimmed and (
        len(trimmed) > max_messages or _characters(trimmed) > max_characters
    ):
        trimmed.pop(0)
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed.pop(0)

    return trimmed


@dataclass
class Session:
    id: str = field(default_factory=new_session_id)
    title: str = "New conversation"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: int = 0
    tool_calls: int = 0

    def record_turn(self, messages: list[dict[str, Any]], tool_calls: int = 0) -> None:
        self.messages = trim_history([m for m in messages if m.get("role") != "system"])
        self.turns += 1
        self.tool_calls += tool_calls
        self.updated_at = time.time()

    def rename_from(self, text: str) -> None:
        if self.title != "New conversation" or not text:
            return
        collapsed = " ".join(text.split())
        self.title = collapsed[:60] + ("..." if len(collapsed) > 60 else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Session":
        return cls(
            id=payload["id"],
            title=payload.get("title", "New conversation"),
            messages=payload.get("messages", []),
            created_at=payload.get("created_at", time.time()),
            updated_at=payload.get("updated_at", time.time()),
            turns=payload.get("turns", 0),
            tool_calls=payload.get("tool_calls", 0),
        )


class SessionStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for record in payload.get("sessions", []):
            session = Session.from_dict(record)
            self._sessions[session.id] = session

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [s.to_dict() for s in self._sessions.values()]}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def create(self, title: str | None = None) -> Session:
        session = Session()
        if title:
            session.title = title
        with self._lock:
            self._sessions[session.id] = session
            self._persist()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._persist()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
            if removed:
                self._persist()
        return removed

    def list(self) -> list[Session]:
        return sorted(
            self._sessions.values(), key=lambda s: s.updated_at, reverse=True
        )

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[Session]:
        return iter(self.list())
