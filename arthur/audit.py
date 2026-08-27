from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

GENESIS = "GENESIS"

_REDACTED_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "private_key",
}


def default_log_path() -> Path:
    configured = os.getenv("ARTHUR_AUDIT_LOG")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "audit.jsonl"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key.lower() in _REDACTED_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + f"...<truncated {len(value) - 512} chars>"
    return value


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_log_path()
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        previous = GENESIS
        for entry in self.entries():
            previous = entry["entry_hash"]
        return previous

    def record(
        self,
        tool: str,
        outcome: str,
        arguments: dict[str, Any] | None = None,
        detail: str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        body = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tool": tool,
            "outcome": outcome,
            "arguments": redact(arguments or {}),
            "detail": detail,
            "duration_ms": duration_ms,
        }

        with self._lock:
            previous_hash = self._last_hash()
            entry = dict(body)
            entry["previous_hash"] = previous_hash
            entry["entry_hash"] = hashlib.sha256(
                (previous_hash + _canonical(body)).encode("utf-8")
            ).hexdigest()

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")

        return entry

    def entries(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())

        def read() -> Iterator[dict[str, Any]]:
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        return read()

    def verify(self) -> dict[str, Any]:
        previous_hash = GENESIS
        checked = 0

        for position, entry in enumerate(self.entries(), start=1):
            if entry.get("previous_hash") != previous_hash:
                return {
                    "status": "TAMPERED",
                    "broken_at": position,
                    "reason": "previous_hash mismatch",
                }

            body = {
                key: entry[key]
                for key in (
                    "timestamp",
                    "tool",
                    "outcome",
                    "arguments",
                    "detail",
                    "duration_ms",
                )
            }
            recomputed = hashlib.sha256(
                (previous_hash + _canonical(body)).encode("utf-8")
            ).hexdigest()
            if recomputed != entry.get("entry_hash"):
                return {
                    "status": "TAMPERED",
                    "broken_at": position,
                    "reason": "entry_hash mismatch",
                }

            previous_hash = entry["entry_hash"]
            checked += 1

        return {"status": "VERIFIED", "entries_checked": checked}
