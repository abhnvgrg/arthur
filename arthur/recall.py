from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from arthur.llm import LLM, LLMError
from arthur.tools.builtins import MemoryStore

MAX_RECALLED = 12
MAX_VALUE = 200
MAX_FACTS_PER_TURN = 3
KEY_LIMIT = 60

EXTRACT_PROMPT = (
    "You keep a personal assistant's long term memory.\n"
    "Read the exchange and pull out facts about the user that will still matter "
    "next week: their name, where they live, who they live with, their work, "
    "their preferences, their routines, their tools.\n"
    "Ignore anything transient - the weather, one-off questions, task text, "
    "what a tool returned, anything the assistant said about itself.\n"
    "Reply with JSON only: {\"facts\": [{\"key\": \"...\", \"value\": \"...\"}]}\n"
    "Keys are short snake_case labels such as home_city or coffee_order. "
    f"At most {MAX_FACTS_PER_TURN} facts, and an empty list when nothing lasting "
    "was said."
)


def enabled() -> bool:
    raw = os.getenv("ARTHUR_AMBIENT_MEMORY", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def clean_key(raw: str) -> str:
    kept = "".join(c if c.isalnum() or c in "_ -" else "" for c in raw.strip().lower())
    return "_".join(kept.split())[:KEY_LIMIT]


def parse_facts(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return []

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        _, _, text = text.partition("\n")

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    facts = []
    for item in payload.get("facts", [])[:MAX_FACTS_PER_TURN]:
        if not isinstance(item, dict):
            continue
        key = clean_key(str(item.get("key", "")))
        value = " ".join(str(item.get("value", "")).split())[:MAX_VALUE]
        if key and value:
            facts.append((key, value))
    return facts


def transcript(messages: Sequence[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not content:
            continue
        lines.append(f"{role}: {' '.join(str(content).split())[:600]}")
    return "\n".join(lines[-6:])


@dataclass
class Librarian:
    llm: LLM
    store: MemoryStore

    async def absorb(self, messages: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
        if not enabled():
            return []

        exchange = transcript(messages)
        if not exchange:
            return []

        try:
            completion = await self.llm.complete(
                [
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": exchange},
                ],
                [],
            )
        except LLMError:
            return []

        kept = []
        for key, value in parse_facts(completion.text):
            if self.store.recall(key) == value:
                continue
            self.store.remember(key, value)
            kept.append((key, value))
        return kept


def remembered(store: MemoryStore | None = None, limit: int = MAX_RECALLED) -> list[tuple[str, str]]:
    if not enabled():
        return []
    memory = store or MemoryStore()
    facts = []
    for key in memory.keys()[:limit]:
        value = memory.recall(key)
        if value:
            facts.append((key, value[:MAX_VALUE]))
    return facts


def recall_block(store: MemoryStore | None = None, limit: int = MAX_RECALLED) -> str:
    facts = remembered(store, limit)
    if not facts:
        return ""
    lines = "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in facts)
    return f"What you already know about the user:\n{lines}"
