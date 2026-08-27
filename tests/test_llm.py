from __future__ import annotations

import pytest

from arthur.llm import Completion, LLMError, ScriptedLLM, ToolCall, parse_arguments

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"city": "Delhi"}', {"city": "Delhi"}),
        ("{}", {}),
        ("", {}),
        (None, {}),
        ({"already": "parsed"}, {"already": "parsed"}),
        ('```json\n{"city": "Delhi"}\n```', {"city": "Delhi"}),
        ('```\n{"city": "Delhi"}\n```', {"city": "Delhi"}),
        ('{"city": "Delhi",}', {"city": "Delhi"}),
    ],
)
async def test_arguments_are_parsed_and_repaired(raw, expected):
    parsed, malformed = parse_arguments(raw)
    assert parsed == expected
    assert malformed is None


@pytest.mark.parametrize("raw", ["not json at all", "{unclosed", "{'single': 'quotes'}"])
async def test_unparseable_arguments_are_reported(raw):
    parsed, malformed = parse_arguments(raw)
    assert parsed == {}
    assert malformed == "Arguments were not valid JSON"


@pytest.mark.parametrize("raw", ["[1, 2, 3]", '"a string"', "42"])
async def test_non_object_arguments_are_reported(raw):
    parsed, malformed = parse_arguments(raw)
    assert parsed == {}
    assert "must be a JSON object" in malformed


async def test_a_completion_without_tool_calls_does_not_want_tools():
    assert Completion(text="hello").wants_tools is False


async def test_a_completion_with_tool_calls_wants_tools():
    completion = Completion(tool_calls=(ToolCall(id="a", name="t", arguments={}),))
    assert completion.wants_tools is True


async def test_the_scripted_llm_returns_completions_in_order():
    first, second = Completion(text="one"), Completion(text="two")
    llm = ScriptedLLM([first, second])

    assert await llm.complete([], []) is first
    assert await llm.complete([], []) is second
    assert llm.exhausted


async def test_the_scripted_llm_records_what_it_was_asked():
    llm = ScriptedLLM([Completion(text="ok")])

    await llm.complete([{"role": "user", "content": "hi"}], [{"name": "t"}])

    assert llm.calls[0]["messages"][0]["content"] == "hi"
    assert llm.calls[0]["tools"] == [{"name": "t"}]


async def test_running_past_the_script_is_an_error():
    llm = ScriptedLLM([])
    with pytest.raises(LLMError):
        await llm.complete([], [])


async def test_a_missing_openai_package_is_a_clean_error(monkeypatch):
    import builtins

    from arthur.llm import OpenAILLM

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(LLMError, match="openai package is not installed"):
        OpenAILLM(api_key="sk-test")._get_client()


async def test_a_missing_api_key_is_a_clean_error(monkeypatch):
    from arthur.llm import OpenAILLM

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLM()._get_client()
