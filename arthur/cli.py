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


async def run_line(dispatcher: Dispatcher, line: str, ask) -> None:
    try:
        name, arguments = parse(line)
    except ValueError as error:
        print(f"  {error}")
        return

    if not name:
        return

    result = await dispatcher.invoke(name, arguments)

    if result.outcome == Outcome.CONFIRMATION_REQUIRED:
        spec = dispatcher.registry.get(name)
        print(f"  {name} is {RISK_LABEL[spec.risk]}.")
        print(f"  arguments: {json.dumps(result.arguments, default=str)}")
        if ask("  run it? [y/N] ").strip().lower() in {"y", "yes"}:
            result = await dispatcher.invoke(name, arguments, confirmed=True)
        else:
            print("  skipped")
            return

    render(result)


def approve_at_prompt(spec, arguments) -> bool:
    print(f"  {spec.name} is {RISK_LABEL[spec.risk]}.")
    print(f"  arguments: {json.dumps(arguments, default=str)}")
    return input("  allow it? [y/N] ").strip().lower() in {"y", "yes"}


async def chat(dispatcher: Dispatcher, message: str, history: list) -> list:
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
        return history

    for result in turn.tool_results:
        marker = "ok" if result.ok else result.outcome
        print(f"  [{result.tool}: {marker}]")
    if turn.stopped_at_limit:
        print("  [stopped at the step limit]")

    print()
    print(turn.answer or "(no answer)")
    print()
    return turn.messages[1:]


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
    print("chat:     say <message>          (needs OPENAI_API_KEY)")
    print("direct:   <tool> key=value ...\n")
    show_tools(dispatcher)
    print()

    history: list = []

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if line in {"quit", "exit"}:
            return 0
        if line == "reset":
            history = []
            print("  conversation cleared")
            continue
        if line.startswith("say "):
            history = await chat(dispatcher, line[4:].strip(), history)
            continue
        if line == "tools":
            show_tools(dispatcher)
            continue
        if line == "verify":
            print(f"  {json.dumps(audit.verify())}")
            continue

        await run_line(dispatcher, line, input)


def serve() -> int:
    from arthur.server import main as serve_main

    return serve_main()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return serve()
    return asyncio.run(repl())


if __name__ == "__main__":
    sys.exit(main())
