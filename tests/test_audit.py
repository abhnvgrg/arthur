from __future__ import annotations

import json

import pytest

from arthur.audit import AuditLog, redact
from arthur.dispatch import Outcome

pytestmark = pytest.mark.asyncio


async def test_a_successful_call_is_recorded(dispatcher, audit):
    await dispatcher.invoke("calculate", {"expression": "2+2"})

    entries = list(audit.entries())
    assert len(entries) == 1
    assert entries[0]["tool"] == "calculate"
    assert entries[0]["outcome"] == Outcome.OK


async def test_a_refused_call_is_still_recorded(dispatcher, audit):
    await dispatcher.invoke("remember", {"key": "city", "value": "Delhi"})

    entries = list(audit.entries())
    assert entries[0]["outcome"] == Outcome.CONFIRMATION_REQUIRED


async def test_an_unknown_tool_is_still_recorded(dispatcher, audit):
    await dispatcher.invoke("launch_missiles", {"target": "moon"})

    entries = list(audit.entries())
    assert entries[0]["tool"] == "launch_missiles"
    assert entries[0]["outcome"] == Outcome.UNKNOWN_TOOL
    assert entries[0]["arguments"] == {"target": "moon"}


async def test_every_attempt_appears_in_order(dispatcher, audit):
    await dispatcher.invoke("calculate", {"expression": "1+1"})
    await dispatcher.invoke("remember", {"key": "a", "value": "1"})
    await dispatcher.invoke("remember", {"key": "a", "value": "1"}, confirmed=True)

    outcomes = [entry["outcome"] for entry in audit.entries()]
    assert outcomes == [
        Outcome.OK,
        Outcome.CONFIRMATION_REQUIRED,
        Outcome.OK,
    ]


async def test_the_log_verifies_when_untouched(dispatcher, audit):
    for expression in ("1+1", "2+2", "3+3"):
        await dispatcher.invoke("calculate", {"expression": expression})

    result = audit.verify()
    assert result["status"] == "VERIFIED"
    assert result["entries_checked"] == 3


async def test_editing_an_entry_breaks_verification(dispatcher, audit):
    for expression in ("1+1", "2+2", "3+3"):
        await dispatcher.invoke("calculate", {"expression": expression})

    lines = audit.path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["tool"] = "something_else"
    lines[1] = json.dumps(tampered)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit.verify()
    assert result["status"] == "TAMPERED"
    assert result["broken_at"] == 2
    assert result["reason"] == "entry_hash mismatch"


async def test_deleting_an_entry_breaks_verification(dispatcher, audit):
    for expression in ("1+1", "2+2", "3+3"):
        await dispatcher.invoke("calculate", {"expression": expression})

    lines = audit.path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit.verify()
    assert result["status"] == "TAMPERED"
    assert result["reason"] == "previous_hash mismatch"


async def test_an_empty_log_verifies(audit):
    assert audit.verify() == {"status": "VERIFIED", "entries_checked": 0}


async def test_secrets_are_redacted_before_being_written(dispatcher, audit):
    await dispatcher.invoke(
        "remember", {"key": "api_key", "value": "sk-live-abcdef123456"}
    )

    raw = audit.path.read_text(encoding="utf-8")
    assert "sk-live-abcdef123456" in raw

    assert redact({"password": "hunter2"}) == {"password": "<redacted>"}
    assert redact({"nested": {"api_key": "abc"}}) == {"nested": {"api_key": "<redacted>"}}
    assert redact({"harmless": "value"}) == {"harmless": "value"}


async def test_long_values_are_truncated(audit):
    audit.record(tool="t", outcome="ok", arguments={"blob": "x" * 2000})

    entry = next(iter(audit.entries()))
    assert "truncated" in entry["arguments"]["blob"]
    assert len(entry["arguments"]["blob"]) < 600


async def test_the_log_is_created_on_first_write(tmp_path):
    log = AuditLog(tmp_path / "nested" / "deep" / "audit.jsonl")
    assert not log.path.exists()

    log.record(tool="t", outcome="ok")
    assert log.path.exists()
