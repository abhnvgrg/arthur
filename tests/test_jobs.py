from __future__ import annotations

from datetime import datetime

import pytest

from arthur.dispatch import Dispatcher
from arthur.jobs import (
    DAILY,
    EVENT,
    INTERVAL,
    ONCE,
    Job,
    JobError,
    JobRunner,
    JobStore,
    parse_schedule,
)
from arthur.llm import Completion, LLMError, ScriptedLLM
from arthur.notify import FanOut, RecordingNotifier


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.json")


def test_daily_schedules_are_read_in_several_shapes():
    assert parse_schedule("daily at 08:00") == (DAILY, {"at": "08:00"})
    assert parse_schedule("daily 8pm") == (DAILY, {"at": "20:00"})
    assert parse_schedule("every day at 7:30am") == (DAILY, {"at": "07:30"})


def test_intervals_are_converted_to_seconds():
    assert parse_schedule("every 30 minutes") == (INTERVAL, {"seconds": 1800.0})
    assert parse_schedule("every 2 hours") == (INTERVAL, {"seconds": 7200.0})


def test_an_interval_under_a_minute_is_refused():
    with pytest.raises(JobError, match="shortest interval"):
        parse_schedule("every 5 seconds")


def test_event_triggers_are_named():
    assert parse_schedule("on mail") == (EVENT, {"event": "mail"})
    assert parse_schedule("webhook") == (EVENT, {"event": "webhook"})


def test_an_unknown_event_is_refused():
    with pytest.raises(JobError, match="Unknown event"):
        parse_schedule("on pigeon")


def test_an_iso_datetime_becomes_a_one_off():
    trigger, settings = parse_schedule("2026-09-06T18:30")
    assert trigger == ONCE
    assert settings == {"at": "2026-09-06T18:30"}


def test_nonsense_schedules_explain_the_options():
    with pytest.raises(JobError, match="daily at 08:00"):
        parse_schedule("whenever I feel like it")


def test_a_daily_job_is_due_once_the_hour_arrives(store):
    job = store.add("brief", "brief me", "daily at 08:00")
    assert job.is_due(datetime(2026, 9, 5, 7, 59)) is False
    assert job.is_due(datetime(2026, 9, 5, 8, 1)) is True


def test_a_daily_job_that_already_ran_today_waits_for_tomorrow(store):
    job = store.add("brief", "brief me", "daily at 08:00")
    job.last_run = "2026-09-05T08:00:10"
    assert job.is_due(datetime(2026, 9, 5, 9, 0)) is False
    assert job.due_at(datetime(2026, 9, 5, 9, 0)).date() == datetime(2026, 9, 6).date()


def test_a_long_overdue_job_is_not_fired_late(store):
    job = store.add("brief", "brief me", "daily at 08:00")
    assert job.is_due(datetime(2026, 9, 5, 23, 0)) is False


def test_an_interval_job_runs_immediately_then_waits(store):
    job = store.add("check", "check things", "every 30 minutes")
    now = datetime(2026, 9, 5, 12, 0)
    assert job.is_due(now) is True

    job.last_run = now.isoformat(timespec="seconds")
    assert job.is_due(datetime(2026, 9, 5, 12, 10)) is False
    assert job.is_due(datetime(2026, 9, 5, 12, 31)) is True


def test_a_one_off_job_never_runs_twice(store):
    job = store.add("once", "do it", "2026-09-05T09:00")
    assert job.is_due(datetime(2026, 9, 5, 9, 1)) is True

    job.runs = 1
    assert job.is_due(datetime(2026, 9, 5, 9, 1)) is False


def test_event_jobs_are_never_time_due(store):
    job = store.add("triage", "triage it", "on mail")
    assert job.due_at(datetime(2026, 9, 5, 9, 0)) is None
    assert job.is_due(datetime(2026, 9, 5, 9, 0)) is False


def test_a_paused_job_is_not_due(store):
    job = store.add("brief", "brief me", "every 30 minutes")
    store.set_enabled(job.id, False)
    assert store.get(job.id).is_due(datetime(2026, 9, 5, 12, 0)) is False


def test_jobs_survive_a_restart(store, tmp_path):
    store.add("brief", "brief me", "daily at 08:00")
    assert len(JobStore(tmp_path / "jobs.json").list()) == 1


def test_removing_a_job_reports_whether_it_existed(store):
    job = store.add("brief", "brief me", "daily at 08:00")
    assert store.remove(job.id) is True
    assert store.remove(job.id) is False


def runner(store, script, notifier=None):
    return JobRunner(
        ScriptedLLM(script),
        Dispatcher(__import__("arthur.tools.builtins", fromlist=["x"]).build_registry()),
        store=store,
        notifier=notifier or FanOut([RecordingNotifier()]),
    )


async def test_running_a_job_records_the_answer_and_notifies(store):
    recorder = RecordingNotifier()
    job = store.add("brief", "what is on today", "daily at 08:00")
    outcome = await runner(store, [Completion(text="Two meetings.")], FanOut([recorder])).run_job(job)

    assert outcome.ok
    assert outcome.answer == "Two meetings."
    assert recorder.sent[0].title == "brief"
    assert "Two meetings." in recorder.sent[0].body

    saved = store.get(job.id)
    assert saved.runs == 1
    assert saved.last_answer == "Two meetings."
    assert saved.last_run is not None


async def test_a_failing_job_is_reported_not_raised(store):
    class Broken:
        async def complete(self, messages, tools):
            raise LLMError("no model")

    job = store.add("brief", "brief me", "daily at 08:00")
    engine = JobRunner(
        Broken(),
        Dispatcher(__import__("arthur.tools.builtins", fromlist=["x"]).build_registry()),
        store=store,
        notifier=FanOut([]),
    )
    outcome = await engine.run_job(job)

    assert outcome.ok is False
    assert "no model" in outcome.error
    assert store.get(job.id).runs == 1


async def test_a_job_that_is_not_notified_stays_quiet(store):
    recorder = RecordingNotifier()
    job = store.add("quiet", "think", "daily at 08:00", notify=False)
    await runner(store, [Completion(text="done")], FanOut([recorder])).run_job(job)
    assert recorder.sent == []


async def test_firing_an_event_runs_only_the_matching_jobs(store):
    store.add("mail triage", "triage", "on mail")
    store.add("morning", "brief", "daily at 08:00")

    engine = runner(store, [Completion(text="handled")])
    runs = await engine.fire("mail", "From: bank")

    assert [run.job.name for run in runs] == ["mail triage"]
    assert runs[0].answer == "handled"


async def test_a_job_prompt_carries_the_trigger_context(store):
    store.add("mail triage", "triage this", "on mail")
    llm = ScriptedLLM([Completion(text="ok")])
    engine = JobRunner(
        llm,
        Dispatcher(__import__("arthur.tools.builtins", fromlist=["x"]).build_registry()),
        store=store,
        notifier=FanOut([]),
    )
    await engine.fire("mail", "From: the bank")

    asked = llm.calls[0]["messages"][-1]["content"]
    assert "triage this" in asked
    assert "From: the bank" in asked


async def test_spoken_jobs_reach_the_speaker(store):
    said: list[str] = []
    job = store.add("brief", "brief me", "daily at 08:00", speak=True, notify=False)

    engine = JobRunner(
        ScriptedLLM([Completion(text="All clear.")]),
        Dispatcher(__import__("arthur.tools.builtins", fromlist=["x"]).build_registry()),
        store=store,
        notifier=FanOut([]),
        speak=said.append,
    )
    await engine.run_job(job)

    assert said == ["All clear."]


async def test_tick_runs_every_due_job(store):
    store.add("a", "one", "every 30 minutes")
    store.add("b", "two", "every 30 minutes")

    engine = runner(store, [Completion(text="x"), Completion(text="y")])
    runs = await engine.tick(datetime(2026, 9, 5, 12, 0))

    assert len(runs) == 2
    assert all(run.ok for run in runs)


async def test_jobs_never_approve_their_own_risky_calls(store):
    from arthur.llm import ToolCall

    job = store.add("tidy", "delete task t_1", "daily at 08:00")
    engine = runner(
        store,
        [
            Completion(tool_calls=[ToolCall(id="c1", name="delete_task", arguments={"task_id": "t_1"})]),
            Completion(text="I could not delete it without approval."),
        ],
    )
    outcome = await engine.run_job(job)

    assert outcome.ok
    assert "approval" in outcome.answer
