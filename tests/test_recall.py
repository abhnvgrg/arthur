from __future__ import annotations

import pytest

from arthur.llm import Completion, LLMError, ScriptedLLM
from arthur.recall import (
    Librarian,
    clean_key,
    parse_facts,
    recall_block,
    remembered,
    transcript,
)
from arthur.selection import SYSTEM_PROMPT, build_system_prompt


@pytest.fixture(autouse=True)
def ambient_on(monkeypatch):
    monkeypatch.setenv("ARTHUR_AMBIENT_MEMORY", "1")


def test_keys_are_normalised():
    assert clean_key("Home City") == "home_city"
    assert clean_key("  coffee/order!  ") == "coffeeorder"


def test_facts_are_read_from_plain_json():
    raw = '{"facts": [{"key": "home_city", "value": "Delhi"}]}'
    assert parse_facts(raw) == [("home_city", "Delhi")]


def test_facts_are_read_from_a_fenced_block():
    raw = '```json\n{"facts": [{"key": "job", "value": "engineer"}]}\n```'
    assert parse_facts(raw) == [("job", "engineer")]


def test_prose_around_the_json_is_ignored():
    raw = 'Sure! {"facts": [{"key": "pet", "value": "a cat"}]} Hope that helps.'
    assert parse_facts(raw) == [("pet", "a cat")]


def test_broken_json_yields_nothing():
    assert parse_facts("{not json at all") == []
    assert parse_facts("") == []
    assert parse_facts(None) == []


def test_an_empty_fact_list_is_fine():
    assert parse_facts('{"facts": []}') == []


def test_facts_without_a_key_or_value_are_dropped():
    raw = '{"facts": [{"key": "", "value": "x"}, {"key": "y", "value": ""}, "junk"]}'
    assert parse_facts(raw) == []


def test_only_a_few_facts_are_taken_from_one_turn():
    items = ", ".join(f'{{"key": "k{n}", "value": "v{n}"}}' for n in range(10))
    assert len(parse_facts('{"facts": [' + items + "]}")) == 3


def test_the_transcript_keeps_only_what_was_said():
    messages = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "I live in Delhi"},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "Noted."},
    ]
    assert transcript(messages) == "user: I live in Delhi\nassistant: Noted."


def test_an_exchange_with_nothing_said_is_empty():
    assert transcript([{"role": "tool", "content": "{}"}]) == ""


async def test_a_fact_is_learned_and_stored(memory):
    llm = ScriptedLLM([Completion(text='{"facts": [{"key": "home_city", "value": "Delhi"}]}')])
    learned = await Librarian(llm, memory).absorb(
        [{"role": "user", "content": "I live in Delhi"}]
    )

    assert learned == [("home_city", "Delhi")]
    assert memory.recall("home_city") == "Delhi"


async def test_a_fact_already_known_is_not_rewritten(memory):
    memory.remember("home_city", "Delhi")
    llm = ScriptedLLM([Completion(text='{"facts": [{"key": "home_city", "value": "Delhi"}]}')])

    assert await Librarian(llm, memory).absorb([{"role": "user", "content": "Delhi"}]) == []


async def test_a_changed_fact_replaces_the_old_one(memory):
    memory.remember("home_city", "Delhi")
    llm = ScriptedLLM([Completion(text='{"facts": [{"key": "home_city", "value": "Pune"}]}')])

    await Librarian(llm, memory).absorb([{"role": "user", "content": "I moved to Pune"}])
    assert memory.recall("home_city") == "Pune"


async def test_a_model_failure_loses_no_memories(memory):
    class Broken:
        async def complete(self, messages, tools):
            raise LLMError("no model")

    assert await Librarian(Broken(), memory).absorb([{"role": "user", "content": "hi"}]) == []
    assert memory.keys() == []


async def test_nothing_is_learned_from_an_empty_exchange(memory):
    llm = ScriptedLLM([])
    assert await Librarian(llm, memory).absorb([{"role": "tool", "content": "{}"}]) == []


async def test_memory_can_be_switched_off(memory, monkeypatch):
    monkeypatch.setenv("ARTHUR_AMBIENT_MEMORY", "0")
    llm = ScriptedLLM([Completion(text='{"facts": [{"key": "a", "value": "b"}]}')])

    assert await Librarian(llm, memory).absorb([{"role": "user", "content": "hi"}]) == []
    assert memory.keys() == []


def test_what_is_remembered_is_listed(memory):
    memory.remember("home_city", "Delhi")
    memory.remember("job", "engineer")
    assert remembered(memory) == [("home_city", "Delhi"), ("job", "engineer")]


def test_the_recall_block_reads_as_prose(memory):
    memory.remember("home_city", "Delhi")
    block = recall_block(memory)

    assert "What you already know about the user" in block
    assert "- home city: Delhi" in block


def test_an_empty_memory_adds_nothing(memory):
    assert recall_block(memory) == ""


def test_the_system_prompt_still_opens_the_same_way():
    assert build_system_prompt(memories="").startswith(SYSTEM_PROMPT)


def test_memories_are_appended_to_the_system_prompt():
    prompt = build_system_prompt(memories="What you already know: he lives in Delhi")
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.endswith("he lives in Delhi")
