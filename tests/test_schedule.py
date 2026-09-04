from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from arthur.notify import FanOut, Notification, RecordingNotifier
from arthur.schedule import Reminder, ScheduleError, Scheduler
from arthur.tools.tasks import TaskStore

pytestmark = pytest.mark.asyncio

IST = timezone(timedelta(hours=5, minutes=30))


def at(hour: int, minute: int = 0, day: int = 3) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=IST)


@pytest.fixture
def store(tmp_path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.json")


@pytest.fixture
def recorder() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def scheduler(store, recorder) -> Scheduler:
    return Scheduler(
        store=store,
        notifier=FanOut([recorder]),
        clock=lambda: at(12, 0),
        interval=0.01,
        lead=600,
        grace=3600,
    )


async def test_a_task_with_no_due_date_is_never_reminded(scheduler, store):
    store.add("someday")
    assert scheduler.pending(at(12, 0)) == []


async def test_a_dated_task_is_skipped_without_a_day_start(scheduler, store):
    store.add("all day", due="2026-09-03")
    assert scheduler.pending(at(12, 0)) == []


async def test_a_dated_task_fires_at_the_configured_hour(scheduler, store, monkeypatch):
    monkeypatch.setenv("ARTHUR_REMIND_DAY_AT", "09:00")
    store.add("all day", due="2026-09-03")

    assert scheduler.pending(at(8, 0)) == []
    assert [r.title for r in scheduler.pending(at(9, 0))] == ["all day"]


async def test_a_timed_task_fires_within_the_lead_window(scheduler, store):
    store.add("standup", due="2026-09-03T13:00")

    assert scheduler.pending(at(12, 40)) == []
    assert [r.title for r in scheduler.pending(at(12, 50))] == ["standup"]
    assert [r.title for r in scheduler.pending(at(13, 0))] == ["standup"]


async def test_a_long_missed_task_is_not_shouted_about(scheduler, store):
    store.add("yesterday", due="2026-09-02T13:00")
    assert scheduler.pending(at(12, 0)) == []


async def test_a_recently_missed_task_still_fires(scheduler, store):
    store.add("just missed", due="2026-09-03T11:30")
    assert [r.title for r in scheduler.pending(at(12, 0))] == ["just missed"]


async def test_a_completed_task_is_not_reminded(scheduler, store):
    task = store.add("done thing", due="2026-09-03T13:00")
    store.complete(task["id"])
    assert scheduler.pending(at(12, 55)) == []


async def test_a_reminder_fires_once(scheduler, store, recorder):
    store.add("standup", due="2026-09-03T13:00")

    first = await scheduler.tick(at(12, 55))
    second = await scheduler.tick(at(12, 56))

    assert [r.title for r in first] == ["standup"]
    assert second == []
    assert len(recorder.sent) == 1


async def test_the_notification_names_the_task_and_its_time(scheduler, store, recorder):
    store.add("Meeting with client", due="2026-09-03T13:00")

    await scheduler.tick(at(12, 55))

    note = recorder.sent[0]
    assert note.title == "Meeting with client"
    assert "13:00" in note.body
    assert "Thursday" in note.body


async def test_a_task_is_not_marked_when_every_channel_fails(store):
    broken = RecordingNotifier(name="broken", fail_with=RuntimeError("no network"))
    scheduler = Scheduler(
        store=store,
        notifier=FanOut([broken]),
        clock=lambda: at(12, 55),
        lead=600,
        grace=3600,
    )
    store.add("standup", due="2026-09-03T13:00")

    assert await scheduler.tick() == []
    assert store.list()[0]["reminded_at"] is None
    assert scheduler.failures


async def test_one_working_channel_is_enough_to_mark_it_sent(store, recorder):
    broken = RecordingNotifier(name="broken", fail_with=RuntimeError("no network"))
    scheduler = Scheduler(
        store=store,
        notifier=FanOut([broken, recorder]),
        clock=lambda: at(12, 55),
        lead=600,
        grace=3600,
    )
    store.add("standup", due="2026-09-03T13:00")

    assert [r.title for r in await scheduler.tick()] == ["standup"]
    assert store.list()[0]["reminded_at"] is not None
    assert len(recorder.sent) == 1


async def test_reminders_come_out_in_due_order(scheduler, store):
    store.add("second", due="2026-09-03T12:50")
    store.add("first", due="2026-09-03T12:10")

    assert [r.title for r in scheduler.pending(at(12, 55))] == ["first", "second"]


async def test_a_bad_interval_is_refused(monkeypatch, store):
    monkeypatch.setenv("ARTHUR_REMIND_INTERVAL", "soon")
    with pytest.raises(ScheduleError, match="number of seconds"):
        Scheduler(store=store, notifier=FanOut([]))


async def test_a_bad_day_start_is_refused(monkeypatch, scheduler, store):
    monkeypatch.setenv("ARTHUR_REMIND_DAY_AT", "breakfast")
    store.add("all day", due="2026-09-03")
    with pytest.raises(ScheduleError, match="09:00"):
        scheduler.pending(at(12, 0))


async def test_the_loop_stops_when_asked(store, recorder):
    scheduler = Scheduler(
        store=store,
        notifier=FanOut([recorder]),
        clock=lambda: at(12, 55),
        interval=0.01,
        lead=600,
        grace=3600,
    )
    store.add("standup", due="2026-09-03T13:00")
    stop = asyncio.Event()

    async def runner():
        await scheduler.run(stop)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert len(recorder.sent) == 1


async def test_a_failing_tick_does_not_kill_the_loop(store, recorder, monkeypatch):
    scheduler = Scheduler(
        store=store,
        notifier=FanOut([recorder]),
        clock=lambda: at(12, 55),
        interval=0.01,
    )

    calls = {"n": 0}
    original = scheduler.pending

    def explode(now=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk on fire")
        return original(now)

    monkeypatch.setattr(scheduler, "pending", explode)
    store.add("standup", due="2026-09-03T13:00")

    stop = asyncio.Event()
    task = asyncio.create_task(scheduler.run(stop))
    await asyncio.sleep(0.08)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert calls["n"] > 1
    assert len(recorder.sent) == 1
