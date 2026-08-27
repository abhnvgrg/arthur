from __future__ import annotations

import math

import pytest

from arthur.tools.builtins import CalculationError, MemoryStore, evaluate

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 3", 5),
        ("(2 + 3) * 4", 20),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 3", 1),
        ("2 ** 10", 1024),
        ("-5 + 2", -3),
        ("sqrt(16)", 4),
        ("round(3.14159, 2)", 3.14),
        ("max(1, 7, 3)", 7),
        ("abs(-8)", 8),
    ],
)
async def test_arithmetic_is_evaluated(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


async def test_constants_are_available():
    assert evaluate("pi") == pytest.approx(math.pi)


@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "exec('x=1')",
        "eval('1+1')",
        "lambda: 1",
        "[i for i in range(10)]",
        "globals()",
        "1 if True else 2",
        "'a' * 100",
    ],
)
async def test_code_execution_is_refused(attack):
    with pytest.raises(CalculationError):
        evaluate(attack)


async def test_a_huge_exponent_is_refused():
    with pytest.raises(CalculationError):
        evaluate("2 ** 999999999")


async def test_division_by_zero_is_a_clean_error():
    with pytest.raises(CalculationError):
        evaluate("1 / 0")


async def test_an_overlong_expression_is_refused():
    with pytest.raises(CalculationError):
        evaluate("1+" * 500 + "1")


async def test_an_unknown_function_is_refused():
    with pytest.raises(CalculationError):
        evaluate("system('ls')")


async def test_booleans_are_not_numbers():
    with pytest.raises(CalculationError):
        evaluate("True + True")


async def test_memory_round_trips(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")

    assert store.recall("city") is None
    assert store.remember("city", "Delhi")["replaced"] is False
    assert store.recall("city") == "Delhi"


async def test_remembering_twice_reports_a_replacement(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember("city", "Delhi")

    assert store.remember("city", "Mumbai")["replaced"] is True
    assert store.recall("city") == "Mumbai"


async def test_forgetting_reports_whether_anything_was_there(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember("city", "Delhi")

    assert store.forget("city") is True
    assert store.forget("city") is False
    assert store.recall("city") is None


async def test_memory_survives_a_new_instance(tmp_path):
    MemoryStore(tmp_path / "memory.json").remember("city", "Delhi")
    assert MemoryStore(tmp_path / "memory.json").recall("city") == "Delhi"


async def test_a_corrupt_memory_file_does_not_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")

    store = MemoryStore(path)
    assert store.keys() == []
    store.remember("city", "Delhi")
    assert store.recall("city") == "Delhi"


async def test_current_time_rejects_an_unknown_timezone(dispatcher):
    result = await dispatcher.invoke("current_time", {"timezone_name": "Mars/Olympus"})

    assert not result.ok
    assert "Unknown timezone" in result.error


async def test_current_time_honours_a_real_timezone(dispatcher):
    result = await dispatcher.invoke("current_time", {"timezone_name": "Asia/Kolkata"})

    assert result.ok
    assert result.value["timezone"] == "Asia/Kolkata"


async def test_listing_memories_reflects_what_was_stored(dispatcher, memory):
    memory.remember("b", "2")
    memory.remember("a", "1")

    result = await dispatcher.invoke("list_memories", {})

    assert result.value["keys"] == ["a", "b"]
    assert result.value["count"] == 2
