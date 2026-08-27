from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arthur.tools.tasks import TaskError, TaskStore, parse_due


@pytest.fixture
def tasks(tmp_path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.json")


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def days_from_now(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


def test_an_iso_date_is_accepted():
    assert parse_due("2026-03-15") == "2026-03-15"


@pytest.mark.parametrize(
    "phrase,offset",
    [("today", 0), ("tomorrow", 1), ("next week", 7), ("in 3 days", 3)],
)
def test_natural_phrases_are_understood(phrase, offset):
    assert parse_due(phrase) == days_from_now(offset)


def test_phrases_are_case_insensitive():
    assert parse_due("TOMORROW") == days_from_now(1)


def test_no_due_date_stays_empty():
    assert parse_due(None) is None
    assert parse_due("") is None


@pytest.mark.parametrize("bad", ["someday", "next fortnight", "2026-13-45", "in many days"])
def test_an_unreadable_due_date_is_refused(bad):
    with pytest.raises(TaskError):
        parse_due(bad)


def test_a_task_is_added_with_an_id(tasks):
    task = tasks.add("Write the report")

    assert task["id"].startswith("t_")
    assert task["title"] == "Write the report"
    assert task["done"] is False
    assert task["priority"] == "normal"


def test_a_task_survives_a_new_store(tmp_path):
    TaskStore(tmp_path / "t.json").add("Persisted")

    assert len(TaskStore(tmp_path / "t.json").list()) == 1


def test_tags_are_normalised(tasks):
    task = tasks.add("Tagged", tags=["  Work ", "WORK", "home"])

    assert task["tags"] == ["home", "work"]


def test_listing_hides_completed_tasks_by_default(tasks):
    done = tasks.add("Finished")
    tasks.add("Outstanding")
    tasks.complete(done["id"])

    titles = [task["title"] for task in tasks.list()]
    assert titles == ["Outstanding"]


def test_listing_can_include_completed_tasks(tasks):
    done = tasks.add("Finished")
    tasks.complete(done["id"])

    assert len(tasks.list(include_done=True)) == 1


def test_listing_can_filter_by_tag(tasks):
    tasks.add("Work item", tags=["work"])
    tasks.add("Home item", tags=["home"])

    titles = [task["title"] for task in tasks.list(tag="work")]
    assert titles == ["Work item"]


def test_listing_can_filter_by_due_date(tasks):
    tasks.add("Soon", due="today")
    tasks.add("Later", due="next week")

    titles = [task["title"] for task in tasks.list(due_before="tomorrow")]
    assert titles == ["Soon"]


def test_tasks_are_ordered_by_due_date_then_priority(tasks):
    tasks.add("Later low", due="next week", priority="low")
    tasks.add("Today low", due="today", priority="low")
    tasks.add("Today high", due="today", priority="high")

    titles = [task["title"] for task in tasks.list()]
    assert titles == ["Today high", "Today low", "Later low"]


def test_undated_tasks_sort_last(tasks):
    tasks.add("Whenever")
    tasks.add("Dated", due="next week")

    titles = [task["title"] for task in tasks.list()]
    assert titles == ["Dated", "Whenever"]


def test_completing_a_task_marks_it_done(tasks):
    task = tasks.add("Do it")

    completed = tasks.complete(task["id"])

    assert completed["done"] is True
    assert completed["completed_at"] is not None
    assert completed["already_done"] is False


def test_completing_twice_is_reported_not_an_error(tasks):
    task = tasks.add("Do it")
    tasks.complete(task["id"])

    assert tasks.complete(task["id"])["already_done"] is True


def test_completing_an_unknown_task_is_an_error(tasks):
    with pytest.raises(TaskError, match="No task"):
        tasks.complete("t_nonexistent")


def test_deleting_removes_a_task(tasks):
    task = tasks.add("Temporary")

    assert tasks.remove(task["id"]) is True
    assert tasks.list() == []


def test_deleting_an_unknown_task_reports_false(tasks):
    assert tasks.remove("t_nonexistent") is False


def test_overdue_lists_only_past_due_tasks(tasks):
    tasks.add("Past", due="2020-01-01")
    tasks.add("Future", due="next week")
    tasks.add("Undated")

    titles = [task["title"] for task in tasks.overdue()]
    assert titles == ["Past"]


def test_a_completed_task_is_not_overdue(tasks):
    task = tasks.add("Past", due="2020-01-01")
    tasks.complete(task["id"])

    assert tasks.overdue() == []


def test_a_corrupt_task_file_does_not_crash(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{not json", encoding="utf-8")

    store = TaskStore(path)
    assert store.list() == []
    store.add("Recovered")
    assert len(store.list()) == 1


def test_the_task_tools_are_registered_with_the_right_risk(registry):
    assert registry.get("list_tasks").risk.value == "read_only"
    assert registry.get("add_task").risk.value == "writes"
    assert registry.get("complete_task").risk.value == "writes"
    assert registry.get("delete_task").risk.value == "irreversible"
