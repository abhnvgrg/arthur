from __future__ import annotations

import asyncio
import json
import shlex
import sys

from arthur.audit import AuditLog
from arthur.conversation import heard_stop
from arthur.dispatch import Dispatcher, Outcome
from arthur.llm import LLMError, OpenAILLM
from arthur.recall import Librarian
from arthur.reflection import critic_from_environment
from arthur.selection import run_turn
from arthur.tools.builtins import MemoryStore, build_registry
from arthur.tools.registry import Risk

RISK_LABEL = {
    Risk.READ_ONLY: "read-only",
    Risk.WRITES: "writes",
    Risk.IRREVERSIBLE: "irreversible",
}


def parse(line: str) -> tuple[str, dict[str, str]]:
    parts = shlex.split(line)
    if not parts:
        return "", {}

    name, arguments = parts[0], {}
    for token in parts[1:]:
        key, separator, value = token.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, got {token!r}")
        arguments[key] = value
    return name, arguments


def show_tools(dispatcher: Dispatcher) -> None:
    for spec in dispatcher.registry:
        fields = ", ".join(spec.parameters.model_fields)
        gate = "" if not dispatcher.requires_confirmation(spec) else "  [confirm]"
        print(f"  {spec.name:<14} {RISK_LABEL[spec.risk]:<13} ({fields}){gate}")
        print(f"  {'':<14} {spec.description}")


def render(result) -> None:
    if result.ok:
        print(json.dumps(result.value, indent=2, default=str))
        print(f"  ({result.duration_ms:.1f} ms)")
    else:
        print(f"  {result.outcome}: {result.error}")


COMMANDS = frozenset({"quit", "exit", "reset", "tools", "verify"})


def route(line: str, registry) -> tuple[str, str]:
    text = line.strip()
    if not text:
        return "none", ""
    if text in COMMANDS:
        return "command", text
    if text.startswith("say "):
        return "chat", text[4:].strip()
    if text.split(maxsplit=1)[0] in registry:
        return "tool", text
    return "chat", text


YES = frozenset({"y", "yes"})
NO = frozenset({"n", "no", ""})


def read_answer(ask, prompt: str) -> tuple[bool, str | None]:
    reply = ask(prompt).strip()
    lowered = reply.lower()
    if lowered in YES:
        return True, None
    if lowered in NO:
        return False, None
    return False, reply


async def run_line(dispatcher: Dispatcher, line: str, ask) -> str | None:
    try:
        name, arguments = parse(line)
    except ValueError as error:
        print(f"  {error}")
        return None

    if not name:
        return None

    result = await dispatcher.invoke(name, arguments)

    if result.outcome == Outcome.CONFIRMATION_REQUIRED:
        spec = dispatcher.registry.get(name)
        print(f"  {name} is {RISK_LABEL[spec.risk]}.")
        print(f"  arguments: {json.dumps(result.arguments, default=str)}")
        approved, unread = read_answer(ask, "  run it? [y/N] ")
        if unread is not None:
            print("  skipped - approval needs y or n, nothing else")
            return unread
        if not approved:
            print("  skipped")
            return None
        result = await dispatcher.invoke(name, arguments, confirmed=True)

    render(result)


def approve_at_prompt(spec, arguments) -> bool:
    print(f"  {spec.name} is {RISK_LABEL[spec.risk]}.")
    print(f"  arguments: {json.dumps(arguments, default=str)}")
    return input("  allow it? [y/N] ").strip().lower() in {"y", "yes"}


async def chat(
    dispatcher: Dispatcher, message: str, history: list
) -> tuple[list, str | None]:
    llm = OpenAILLM()
    try:
        turn = await run_turn(
            llm,
            dispatcher,
            message,
            history=history,
            approve=approve_at_prompt,
            critic=critic_from_environment(llm),
        )
    except LLMError as error:
        print(f"  {error}")
        return history, None

    for result in turn.tool_results:
        marker = "ok" if result.ok else result.outcome
        print(f"  [{result.tool}: {marker}]")
    if turn.stopped_at_limit:
        print("  [stopped at the step limit]")

    print()
    print(turn.answer or "(no answer)")
    print()
    return turn.messages[1:], turn.answer


async def repl() -> int:
    audit = AuditLog()
    from arthur.server import default_research_backend

    dispatcher = Dispatcher(
        build_registry(research_backend=default_research_backend()), audit=audit
    )

    print("arthur tool layer")
    print(f"audit log: {audit.path}")
    print("commands: tools, verify, reset, quit")
    print("web ui:   python -m arthur serve")
    print("chat:     just type                (needs ARTHUR_LLM_API_KEY)")
    print("direct:   <tool> key=value ...\n")
    show_tools(dispatcher)
    print()

    history: list = []

    pending: str | None = None

    while True:
        if pending is not None:
            line, pending = pending, None
        else:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

        kind, payload = route(line, dispatcher.registry)

        if kind == "none":
            continue

        if kind == "command":
            if payload in {"quit", "exit"}:
                return 0
            if payload == "reset":
                history = []
                print("  conversation cleared")
            elif payload == "tools":
                show_tools(dispatcher)
            else:
                print(f"  {json.dumps(audit.verify())}")
            continue

        if kind == "chat":
            history, _ = await chat(dispatcher, payload, history)
            continue

        pending = await run_line(dispatcher, payload, input)


def serve() -> int:
    from arthur.server import main as serve_main

    return serve_main()


def build_dispatcher() -> Dispatcher:
    from arthur.server import default_research_backend
    from arthur.tools.builtins import full_registry

    return Dispatcher(
        full_registry(research_backend=default_research_backend()), audit=AuditLog()
    )


async def conversation() -> int:
    from arthur.conversation import Conversation, flag
    from arthur.voice import (
        GroqSpeaker,
        GroqTranscriber,
        Microphone,
        Playback,
        VoiceUnavailable,
        wake_word,
    )

    dispatcher = build_dispatcher()
    transcriber = GroqTranscriber()
    speaker = GroqSpeaker()
    wake = wake_word()
    require_wake = flag("ARTHUR_REQUIRE_WAKE", True)
    barge = flag("ARTHUR_BARGE_IN", False)

    llm = OpenAILLM()
    talker = Conversation(
        llm,
        dispatcher,
        transcriber,
        speaker,
        Microphone(),
        Playback(),
        wake=wake,
        require_wake=require_wake,
        barge_in=barge,
        speak_approvals=flag("ARTHUR_SPOKEN_APPROVALS", True),
        typed_irreversible=flag("ARTHUR_TYPED_IRREVERSIBLE", False),
        critic=critic_from_environment(llm),
        librarian=Librarian(llm, MemoryStore()),
    )

    print("arthur talk - always listening")
    print(f"listening: {transcriber.model}")
    print(f"speaking:  {speaker.model} / {speaker.voice}")
    print(f"wake word: {wake!r}" if require_wake else "wake word: off, speak any time")
    print(f"barge-in:  {'on (use headphones)' if barge else 'off'}")
    print(f"approvals:  spoken - say {talker.confirm!r} for anything irreversible")
    print("say 'stop', or ctrl-c, to finish\n")

    try:
        return await talker.run()
    except VoiceUnavailable as error:
        print(f"  {error}")
        return 1


def talk() -> int:
    try:
        return asyncio.run(conversation())
    except KeyboardInterrupt:
        print()
        return 0


async def watching() -> int:
    from arthur.jobs import JobRunner, JobStore
    from arthur.mailwatch import MailWatcher
    from arthur.schedule import Scheduler
    from arthur.tools import mail as mail_tools

    scheduler = Scheduler()
    channels = [one.name for one in scheduler.notifier.notifiers]

    if not channels:
        print("no notification channels are configured.")
        print("set at least one of:")
        print("  ARTHUR_NTFY_TOPIC       push to your phone via ntfy.sh")
        print("  ARTHUR_TOAST=1          desktop notification (needs plyer)")
        print("  ARTHUR_SMTP_HOST + _TO  email")
        return 1

    dispatcher = build_dispatcher()
    jobs = JobStore()
    runner = JobRunner(OpenAILLM(), dispatcher, store=jobs, notifier=scheduler.notifier)

    print(f"arthur watch - notifying: {', '.join(channels)}")
    print(
        f"reminders every {scheduler.interval:.0f}s, "
        f"{scheduler.lead / 60:.0f} min before a task is due"
    )
    print(f"jobs: {len(jobs.list(include_disabled=False))} active, "
          f"checked every {runner.interval:.0f}s")

    def announce(reminder) -> None:
        print(f"  reminded: {reminder.title} (due {reminder.due:%H:%M})")
        for failure in scheduler.failures:
            print(f"    channel failed - {failure}")

    def ran(outcome) -> None:
        marker = "ok" if outcome.ok else "failed"
        print(f"  job {outcome.job.name}: {marker}")
        if outcome.error:
            print(f"    {outcome.error}")

    stop = asyncio.Event()
    running = [
        asyncio.create_task(scheduler.run(stop=stop, on_sent=announce)),
        asyncio.create_task(runner.run(stop=stop, on_run=ran)),
    ]

    if mail_tools.configured():
        watcher = MailWatcher(mail_tools.MailBox(), fire=runner.fire)
        print(f"mail: watching {watcher.mailbox.account.username} "
              f"every {watcher.interval:.0f}s")
        running.append(asyncio.create_task(watcher.run(stop=stop)))
    else:
        print("mail: not configured (set ARTHUR_IMAP_HOST, _USER, _PASSWORD)")

    print("ctrl-c to stop\n")

    try:
        await asyncio.gather(*running)
    except asyncio.CancelledError:
        pass
    return 0


def watch() -> int:
    try:
        return asyncio.run(watching())
    except KeyboardInterrupt:
        print()
        return 0


def show_jobs() -> int:
    from arthur.jobs import JobStore

    jobs = JobStore().list()
    if not jobs:
        print("no jobs scheduled. ask in chat: 'every morning at 8, brief me'")
        return 0

    print(f"{len(jobs)} job(s):")
    for job in jobs:
        state = "" if job.enabled else "  [paused]"
        print(f"  {job.id}  {job.name:<28} {job.describe():<22} runs={job.runs}{state}")
        print(f"  {'':<10}  {job.prompt[:90]}")
    return 0


USAGE = """arthur - a personal assistant

  python -m arthur              chat in the terminal
  python -m arthur talk         always-on voice
  python -m arthur serve        web UI and API for phone and laptop
  python -m arthur watch        reminders, scheduled jobs and mail triage
  python -m arthur jobs         list scheduled jobs
"""


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    if command == "serve":
        return serve()
    if command == "watch":
        return watch()
    if command == "talk":
        return talk()
    if command == "jobs":
        return show_jobs()
    if command in {"help", "--help", "-h"}:
        print(USAGE)
        return 0
    return asyncio.run(repl())


if __name__ == "__main__":
    sys.exit(main())
