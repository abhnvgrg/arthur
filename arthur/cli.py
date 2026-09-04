from __future__ import annotations

import asyncio
import json
import shlex
import sys

from arthur.audit import AuditLog
from arthur.dispatch import Dispatcher, Outcome
from arthur.llm import LLMError, OpenAILLM
from arthur.reflection import critic_from_environment
from arthur.selection import run_turn
from arthur.tools.builtins import build_registry
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


STOP_WORDS = frozenset({"quit", "exit", "stop", "goodbye", "good bye"})


def heard_stop(text: str) -> bool:
    cleaned = "".join(c for c in text.lower() if c.isalpha() or c.isspace()).strip()
    return cleaned in STOP_WORDS


async def conversation() -> int:
    from arthur.voice import (
        GroqSpeaker,
        GroqTranscriber,
        Microphone,
        Playback,
        VoiceError,
        say,
    )

    audit = AuditLog()
    from arthur.server import default_research_backend

    dispatcher = Dispatcher(
        build_registry(research_backend=default_research_backend()), audit=audit
    )
    transcriber = GroqTranscriber()
    speaker = GroqSpeaker()
    microphone = Microphone()
    playback = Playback()

    print("arthur talk")
    print(f"listening: {transcriber.model}")
    print(f"speaking:  {speaker.model} / {speaker.voice}")
    print("approvals are typed, never spoken")
    print("say 'stop', or ctrl-c, to finish\n")

    history: list = []

    while True:
        try:
            input("press enter, then speak > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        try:
            clip = await asyncio.to_thread(microphone.record_until_quiet)
        except VoiceError as error:
            print(f"  {error}")
            return 1

        if not clip.audio:
            print("  heard nothing")
            continue

        try:
            heard = await transcriber.transcribe(clip)
        except VoiceError as error:
            print(f"  {error}")
            continue

        if not heard:
            print("  heard nothing")
            continue

        print(f"  you: {heard}")
        if heard_stop(heard):
            return 0

        history, answer = await chat(dispatcher, heard, history)
        if not answer:
            continue

        try:
            await say(speaker, playback, answer)
        except VoiceError as error:
            print(f"  could not speak: {error}")


def talk() -> int:
    return asyncio.run(conversation())


def watch() -> int:
    from arthur.schedule import Scheduler

    scheduler = Scheduler()
    channels = [one.name for one in scheduler.notifier.notifiers]

    if not channels:
        print("no notification channels are configured.")
        print("set at least one of:")
        print("  ARTHUR_NTFY_TOPIC       push to your phone via ntfy.sh")
        print("  ARTHUR_TOAST=1          desktop notification (needs plyer)")
        print("  ARTHUR_SMTP_HOST + _TO  email")
        return 1

    print(f"arthur watch - notifying: {', '.join(channels)}")
    print(
        f"checking every {scheduler.interval:.0f}s, "
        f"{scheduler.lead / 60:.0f} min before a task is due"
    )
    print("ctrl-c to stop\n")

    def announce(reminder) -> None:
        print(f"  reminded: {reminder.title} (due {reminder.due:%H:%M})")
        for failure in scheduler.failures:
            print(f"    channel failed - {failure}")

    try:
        asyncio.run(scheduler.run(on_sent=announce))
    except KeyboardInterrupt:
        print()
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return serve()
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        return watch()
    if len(sys.argv) > 1 and sys.argv[1] == "talk":
        return talk()
    return asyncio.run(repl())


if __name__ == "__main__":
    sys.exit(main())
