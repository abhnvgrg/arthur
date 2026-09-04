from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import arthur.tools.tasks as tasks_module
from arthur.tools.tasks import TaskError, TaskStore, now_local, parse_due


@pytest.fixture
def tasks(tmp_path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.json")


def today() -> str:
    return now_local().date().isoformat()


def days_from_now(days: int) -> str:
    return (now_local().date() + timedelta(days=days)).isoformat()


IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def at_night(monkeypatch):
    frozen = datetime(2026, 9, 3, 2, 0, tzinfo=IST)
    monkeypatch.setattr(tasks_module, "now_local", lambda: frozen)
    return frozen


def test_relative_dates_follow_the_local_day_not_utc(at_night):
    assert at_night.astimezone(timezone.utc).date().isoformat() == "2026-09-02"
    assert parse_due("today") == "2026-09-03"
    assert parse_due("tomorrow") == "2026-09-04"
    assert parse_due("in 2 days") == "2026-09-05"


def test_a_configured_timezone_is_used(monkeypatch):
    monkeypatch.setenv("ARTHUR_TIMEZONE", "Asia/Tokyo")
    assert str(tasks_module.local_zone()) == "Asia/Tokyo"


def test_the_clock_is_not_utc(monkeypatch):
    monkeypatch.setenv("ARTHUR_TIMEZONE", "Asia/Tokyo")
    moment = tasks_module.now_local()
    assert moment.utcoffset() == timedelta(hours=9)
    assert moment.date() == datetime.now(timezone(timedelta(hours=9))).date()


def test_an_unknown_timezone_is_refused(monkeypatch):
    monkeypatch.setenv("ARTHUR_TIMEZONE", "Mars/Olympus_Mons")
    with pytest.raises(TaskError, match="not a known timezone"):
        tasks_module.local_zone()


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("tomorrow 1pm", "2026-09-04T13:00"),
        ("tomorrow at 1 PM", "2026-09-04T13:00"),
        ("today 18:00", "2026-09-03T18:00"),
        ("today 12am", "2026-09-03T00:00"),
        ("today 12pm", "2026-09-03T12:00"),
        ("in 3 days at 9:30am", "2026-09-06T09:30"),
        ("2026-12-25 7pm", "2026-12-25T19:00"),
    ],
)
def test_a_time_of_day_is_kept(at_night, phrase, expected):
    assert parse_due(phrase) == expected


def test_a_full_iso_datetime_is_accepted():
    assert parse_due("2026-09-04T13:00:00") == "2026-09-04T13:00"


@pytest.mark.parametrize(
    "value",
    ["2026-09-04T13:00:00+09:00", "2026-09-03T13:00:00Z", "2026-09-03T13:00+00:00"],
)
def test_an_offset_that_is_not_local_is_refused(monkeypatch, value):
    monkeypatch.setenv("ARTHUR_TIMEZONE", "Asia/Kolkata")
    with pytest.raises(TaskError, match="not local"):
        parse_due(value)


@pytest.mark.parametrize(
    "zone,value",
    [
        ("Asia/Kolkata", "2026-09-03T13:00:00+05:30"),
        ("Asia/Tokyo", "2026-09-03T13:00:00+09:00"),
        ("UTC", "2026-09-03T13:00:00Z"),
    ],
)
def test_an_offset_that_matches_local_is_kept_as_written(monkeypatch, zone, value):
    monkeypatch.setenv("ARTHUR_TIMEZONE", zone)
    assert parse_due(value) == "2026-09-03T13:00"


def test_a_summer_offset_is_judged_against_that_date_not_today(monkeypatch):
    monkeypatch.setenv("ARTHUR_TIMEZONE", "Europe/London")
    assert parse_due("2026-07-01T13:00:00+01:00") == "2026-07-01T13:00"
    with pytest.raises(TaskError, match="not local"):
        parse_due("2026-01-01T13:00:00+01:00")


@pytest.mark.parametrize("value", ["2026-09-03 13:00", "2026-09-03T13:00"])
def test_the_local_iso_forms_are_accepted(value):
    assert parse_due(value) == "2026-09-03T13:00"


def test_the_due_field_tells_the_model_the_format():
    from arthur.tools.builtins import build_registry

    schema = build_registry().get("add_task").json_schema()
    described = schema["properties"]["due"]["description"]
    assert "local timezone" in described
    assert "tomorrow 1pm" in described
    assert "rejected" in described


def test_a_bare_date_keeps_no_time():
    assert parse_due("2026-09-04") == "2026-09-04"


def test_unicode_spaces_do_not_break_parsing(at_night):
    assert parse_due("tomorrow 1 pm") == "2026-09-04T13:00"


def test_a_bad_time_of_day_is_refused(at_night):
    with pytest.raises(TaskError, match="not a valid time"):
        parse_due("tomorrow 25:00")


def test_a_dated_task_is_not_overdue_until_the_day_ends(tasks, monkeypatch):
    monkeypatch.setattr(
        tasks_module, "now_local", lambda: datetime(2026, 9, 3, 14, 0, tzinfo=IST)
    )
    tasks.add("all day", due="2026-09-03")
    assert tasks.overdue() == []

    monkeypatch.setattr(
        tasks_module, "now_local", lambda: datetime(2026, 9, 4, 0, 30, tzinfo=IST)
    )
    assert [task["title"] for task in tasks.overdue()] == ["all day"]


def test_a_timed_task_goes_overdue_at_its_time(tasks, monkeypatch):
    monkeypatch.setattr(
        tasks_module, "now_local", lambda: datetime(2026, 9, 3, 12, 0, tzinfo=IST)
    )
    tasks.add("standup", due="2026-09-03T13:00")
    assert tasks.overdue() == []

    monkeypatch.setattr(
        tasks_module, "now_local", lambda: datetime(2026, 9, 3, 13, 30, tzinfo=IST)
    )
    assert [task["title"] for task in tasks.overdue()] == ["standup"]


def test_due_before_a_bare_date_includes_that_whole_day(tasks):
    tasks.add("late meeting", due="2026-09-03T23:00")
    tasks.add("next day", due="2026-09-04")
    found = [task["title"] for task in tasks.list(due_before="2026-09-03")]
    assert found == ["late meeting"]


def test_unicode_spaces_are_stripped_from_titles(tasks):
    task = tasks.add("Meeting with client at 1 PM")
    assert task["title"] == "Meeting with client at 1 PM"


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
