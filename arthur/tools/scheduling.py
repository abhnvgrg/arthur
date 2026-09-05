from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from arthur.jobs import JobStore

SCHEDULE_HELP = (
    "When to run it: 'daily at 08:00', 'every 30 minutes', an ISO datetime for a "
    "one-off ('2026-09-06T18:30'), or an event trigger - 'on mail', 'on file', "
    "'on webhook'."
)


class ScheduleJobArgs(BaseModel):
    name: str = Field(min_length=1, max_length=80, description="Short label for the job")
    prompt: str = Field(
        min_length=1,
        max_length=1000,
        description="The instruction to carry out each time the job runs, written as if the user said it",
    )
    when: str = Field(min_length=1, max_length=60, description=SCHEDULE_HELP)
    speak: bool = Field(default=False, description="Read the result aloud when voice is running")
    notify: bool = Field(default=True, description="Push the result to the notification channels")


class JobIdArgs(BaseModel):
    job_id: str = Field(min_length=3, max_length=40)


class PauseJobArgs(BaseModel):
    job_id: str = Field(min_length=3, max_length=40)
    enabled: bool = Field(description="False pauses the job, True resumes it")


class NoArgs(BaseModel):
    pass


def register(registry, store: JobStore) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="schedule_job",
        description=(
            "Schedule an instruction to run later, on a repeating schedule, or when "
            "something happens. Use this whenever the user asks for something to "
            "happen regularly or in future."
        ),
        parameters=ScheduleJobArgs,
        risk=Risk.WRITES,
    )
    def schedule_job(args: ScheduleJobArgs) -> dict[str, Any]:
        job = store.add(args.name, args.prompt, args.when, args.speak, args.notify)
        return {**job.to_dict(), "schedule": job.describe()}

    @registry.tool(
        name="list_jobs",
        description="List the scheduled jobs and when each one runs.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def list_jobs(_: NoArgs) -> dict[str, Any]:
        jobs = store.list()
        return {
            "count": len(jobs),
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "schedule": job.describe(),
                    "enabled": job.enabled,
                    "runs": job.runs,
                    "last_run": job.last_run,
                }
                for job in jobs
            ],
        }

    @registry.tool(
        name="pause_job",
        description="Pause or resume a scheduled job without deleting it.",
        parameters=PauseJobArgs,
        risk=Risk.WRITES,
    )
    def pause_job(args: PauseJobArgs) -> dict[str, Any]:
        job = store.set_enabled(args.job_id, args.enabled)
        if job is None:
            return {"job_id": args.job_id, "found": False}
        return {"job_id": job.id, "found": True, "enabled": job.enabled}

    @registry.tool(
        name="cancel_job",
        description="Permanently delete a scheduled job.",
        parameters=JobIdArgs,
        risk=Risk.IRREVERSIBLE,
    )
    def cancel_job(args: JobIdArgs) -> dict[str, Any]:
        return {"job_id": args.job_id, "deleted": store.remove(args.job_id)}
