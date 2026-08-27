from __future__ import annotations

import pytest

from arthur.session import Session, SessionStore, new_session_id, trim_history


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_calling(call_id: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ],
    }


def tool_reply(call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "t", "content": "{}"}


def test_session_ids_are_unique():
    assert new_session_id() != new_session_id()


def test_a_short_history_is_left_alone():
    messages = [user("one"), user("two")]

    assert trim_history(messages) == messages


def test_history_is_trimmed_from_the_front():
    messages = [user(f"message {n}") for n in range(60)]

    trimmed = trim_history(messages, max_messages=10)

    assert len(trimmed) <= 10
    assert trimmed[-1]["content"] == "message 59"


def test_trimming_never_leaves_an_orphan_tool_message():
    messages = [user("one"), assistant_calling(), tool_reply(), user("two")]

    trimmed = trim_history(messages, max_messages=2)

    assert not trimmed or trimmed[0]["role"] != "tool"


def test_trimming_respects_a_character_budget():
    messages = [user("x" * 1000) for _ in range(50)]

    trimmed = trim_history(messages, max_messages=1000, max_characters=5000)

    assert len(trimmed) < 50


def test_an_empty_history_trims_to_nothing():
    assert trim_history([]) == []


def test_recording_a_turn_drops_the_system_prompt():
    session = Session()

    session.record_turn(
        [{"role": "system", "content": "prompt"}, user("hi")], tool_calls=2
    )

    assert all(m["role"] != "system" for m in session.messages)
    assert session.turns == 1
    assert session.tool_calls == 2


def test_turns_and_tool_calls_accumulate():
    session = Session()

    session.record_turn([user("one")], tool_calls=1)
    session.record_turn([user("two")], tool_calls=3)

    assert session.turns == 2
    assert session.tool_calls == 4


def test_the_title_is_taken_from_the_first_message():
    session = Session()

    session.rename_from("What is the capital of India?")

    assert session.title == "What is the capital of India?"


def test_the_title_is_not_replaced_later():
    session = Session()
    session.rename_from("first question")
    session.rename_from("second question")

    assert session.title == "first question"


def test_a_long_title_is_shortened():
    session = Session()

    session.rename_from("word " * 60)

    assert len(session.title) <= 63
    assert session.title.endswith("...")


def test_an_empty_message_does_not_set_a_title():
    session = Session()

    session.rename_from("")

    assert session.title == "New conversation"


def test_a_session_round_trips_through_a_dict():
    session = Session(title="Test")
    session.record_turn([user("hello")], tool_calls=1)

    restored = Session.from_dict(session.to_dict())

    assert restored.id == session.id
    assert restored.title == session.title
    assert restored.messages == session.messages
    assert restored.tool_calls == 1


def test_the_store_creates_and_finds_sessions(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")

    created = store.create()

    assert store.get(created.id) is created
    assert len(store) == 1


def test_get_or_create_reuses_a_known_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    created = store.create()

    assert store.get_or_create(created.id) is created


def test_get_or_create_makes_a_new_one_for_an_unknown_id(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")

    session = store.get_or_create("s_unknown")

    assert session.id != "s_unknown"
    assert len(store) == 1


def test_sessions_persist_across_stores(tmp_path):
    path = tmp_path / "sessions.json"
    created = SessionStore(path).create(title="Persisted")

    reopened = SessionStore(path)

    assert reopened.get(created.id) is not None
    assert reopened.get(created.id).title == "Persisted"


def test_deleting_removes_a_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    created = store.create()

    assert store.delete(created.id) is True
    assert store.get(created.id) is None


def test_deleting_an_unknown_session_reports_false(tmp_path):
    assert SessionStore(tmp_path / "s.json").delete("s_nope") is False


def test_listing_puts_the_most_recent_first(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    first = store.create(title="older")
    second = store.create(title="newer")
    second.record_turn([user("hi")])
    store.save(second)

    assert [s.id for s in store.list()][0] == second.id


def test_a_store_without_a_path_keeps_sessions_in_memory():
    store = SessionStore()
    created = store.create()

    assert store.get(created.id) is created
    assert store.path is None


def test_a_corrupt_session_file_does_not_crash(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("{not json", encoding="utf-8")

    store = SessionStore(path)

    assert len(store) == 0
    store.create()
    assert len(store) == 1
