from __future__ import annotations

import os

import httpx
import pytest

from arthur.dispatch import Dispatcher, Outcome
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.selection import run_turn
from arthur.tools.builtins import build_registry
from arthur.tools.research import (
    HttpResearchBackend,
    ResearchError,
    ResearchResult,
    ResearchUnavailable,
    StubResearchBackend,
)

pytestmark = pytest.mark.asyncio


def answer(text: str = "Delhi is the capital.") -> ResearchResult:
    return ResearchResult(
        answer=text,
        citations={"1": "https://example.test/a"},
        cycles=2,
        run_id="run-1",
    )


def routed(handler) -> HttpResearchBackend:
    return HttpResearchBackend(
        base_url="http://research.test",
        token="tok",
        timeout=2.0,
        poll_interval=0.01,
        transport=httpx.MockTransport(handler),
    )


def registry_with(backend, memory, tasks, workspace):
    return build_registry(
        memory=memory, tasks=tasks, workspace=workspace, research_backend=backend
    )


async def test_the_stub_returns_what_it_was_given():
    backend = StubResearchBackend([answer()])

    result = await backend.research("What is the capital of India?")

    assert result.answer == "Delhi is the capital."
    assert backend.queries == ["What is the capital of India?"]


async def test_the_stub_raises_what_it_was_given():
    backend = StubResearchBackend([ResearchError("no index")])

    with pytest.raises(ResearchError, match="no index"):
        await backend.research("anything at all")


async def test_a_result_serialises_for_the_model():
    payload = answer().to_dict()

    assert payload["answer"] == "Delhi is the capital."
    assert payload["sources"] == 1
    assert payload["reflection_cycles"] == 2


async def test_a_completed_run_is_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1", "status": "running"})
        return httpx.Response(
            200,
            json={
                "run_id": "r1",
                "status": "complete",
                "answer": "Delhi.",
                "citations": {"1": "https://example.test"},
                "cycle_count": 1,
            },
        )

    result = await routed(handler).research("What is the capital of India?")

    assert result.answer == "Delhi."
    assert result.run_id == "r1"
    assert result.cycles == 1


async def test_the_backend_polls_until_the_run_finishes():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json={"run_id": "r1", "status": "running"})
        return httpx.Response(
            200, json={"run_id": "r1", "status": "complete", "answer": "Done."}
        )

    result = await routed(handler).research("a question worth asking")

    assert result.answer == "Done."
    assert calls["n"] == 3


async def test_the_query_is_sent_in_the_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            import json

            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(
            200, json={"run_id": "r1", "status": "complete", "answer": "x"}
        )

    await routed(handler).research("What is the capital of India?")

    assert seen["body"] == {"query": "What is the capital of India?"}
    assert seen["auth"] == "Bearer tok"


async def test_no_token_sends_no_authorization_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("auth", request.headers.get("Authorization"))
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(
            200, json={"run_id": "r1", "status": "complete", "answer": "x"}
        )

    backend = HttpResearchBackend(
        base_url="http://research.test",
        token="",
        timeout=2.0,
        poll_interval=0.01,
        transport=httpx.MockTransport(handler),
    )
    await backend.research("a question worth asking")

    assert seen["auth"] is None


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_credentials_are_named_as_such(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "nope"})

    with pytest.raises(ResearchUnavailable, match="ARTHUR_RESEARCH_TOKEN"):
        await routed(handler).research("a question worth asking")


async def test_an_unreachable_service_is_named_as_such():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ResearchUnavailable, match="Could not reach"):
        await routed(handler).research("a question worth asking")


async def test_a_refused_query_reports_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="query too short")

    with pytest.raises(ResearchError, match="422"):
        await routed(handler).research("a question worth asking")


async def test_a_missing_run_id_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"status": "running"})

    with pytest.raises(ResearchError, match="no run id"):
        await routed(handler).research("a question worth asking")


async def test_a_failed_run_surfaces_its_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(
            200,
            json={"run_id": "r1", "status": "failed", "error": "retrieval timed out"},
        )

    with pytest.raises(ResearchError, match="retrieval timed out"):
        await routed(handler).research("a question worth asking")


async def test_a_run_that_finishes_with_no_answer_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(200, json={"run_id": "r1", "status": "complete"})

    with pytest.raises(ResearchError, match="without an answer"):
        await routed(handler).research("a question worth asking")


async def test_a_run_that_never_finishes_times_out():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(200, json={"run_id": "r1", "status": "running"})

    backend = HttpResearchBackend(
        base_url="http://research.test",
        token="tok",
        timeout=0.15,
        poll_interval=0.01,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ResearchError, match="did not finish within"):
        await backend.research("a question worth asking")


async def test_an_unreadable_poll_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/research/query":
            return httpx.Response(202, json={"run_id": "r1"})
        return httpx.Response(500, text="boom")

    with pytest.raises(ResearchError, match="500"):
        await routed(handler).research("a question worth asking")


async def test_the_tool_is_read_only(memory, tasks, workspace):
    registry = registry_with(StubResearchBackend([]), memory, tasks, workspace)

    assert registry.get("research").risk.value == "read_only"


async def test_the_tool_has_a_generous_timeout(memory, tasks, workspace):
    registry = registry_with(StubResearchBackend([]), memory, tasks, workspace)

    assert registry.get("research").timeout_seconds > 60


async def test_the_tool_is_absent_without_a_backend(registry):
    assert "research" not in registry


async def test_a_short_query_is_rejected_before_the_backend(
    memory, tasks, workspace, audit
):
    backend = StubResearchBackend([answer()])
    dispatcher = Dispatcher(
        registry_with(backend, memory, tasks, workspace), audit=audit
    )

    result = await dispatcher.invoke("research", {"query": "hi"})

    assert result.outcome == Outcome.INVALID_ARGUMENTS
    assert backend.queries == []


async def test_the_tool_returns_the_answer_and_citations(
    memory, tasks, workspace, audit
):
    dispatcher = Dispatcher(
        registry_with(StubResearchBackend([answer()]), memory, tasks, workspace),
        audit=audit,
    )

    result = await dispatcher.invoke(
        "research", {"query": "What is the capital of India?"}
    )

    assert result.ok
    assert result.value["answer"] == "Delhi is the capital."
    assert result.value["sources"] == 1


async def test_a_backend_failure_is_contained_not_raised(
    memory, tasks, workspace, audit
):
    backend = StubResearchBackend([ResearchUnavailable("service is down")])
    dispatcher = Dispatcher(
        registry_with(backend, memory, tasks, workspace), audit=audit
    )

    result = await dispatcher.invoke("research", {"query": "a question worth asking"})

    assert result.outcome == Outcome.FAILED
    assert "service is down" in result.error


async def test_the_assistant_can_use_research_in_a_turn(
    memory, tasks, workspace, audit
):
    dispatcher = Dispatcher(
        registry_with(StubResearchBackend([answer()]), memory, tasks, workspace),
        audit=audit,
    )
    llm = ScriptedLLM(
        [
            Completion(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="research",
                        arguments={"query": "What is the capital of India?"},
                    ),
                )
            ),
            Completion(text="Delhi is the capital of India [1]."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "what is the capital of india")

    assert turn.tools_used == ["research"]
    assert turn.critique.passed
    assert "Delhi" in turn.answer


async def test_a_research_failure_must_be_reported_to_the_user(
    memory, tasks, workspace, audit
):
    backend = StubResearchBackend([ResearchUnavailable("service is down")])
    dispatcher = Dispatcher(
        registry_with(backend, memory, tasks, workspace), audit=audit
    )
    llm = ScriptedLLM(
        [
            Completion(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="research",
                        arguments={"query": "What is the capital of India?"},
                    ),
                )
            ),
            Completion(text="The capital is Delhi."),
            Completion(text="I could not reach the research service."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "capital of india", max_steps=4)

    assert turn.reflections == 1
    assert turn.answer.startswith("I could not reach")


async def test_every_research_call_is_audited(memory, tasks, workspace, audit):
    dispatcher = Dispatcher(
        registry_with(StubResearchBackend([answer()]), memory, tasks, workspace),
        audit=audit,
    )

    await dispatcher.invoke("research", {"query": "a question worth asking"})

    entries = list(audit.entries())
    assert entries[-1]["tool"] == "research"
    assert entries[-1]["outcome"] == Outcome.OK


LIVE_URL = os.getenv("ARTHUR_RESEARCH_URL")

live = pytest.mark.skipif(
    not LIVE_URL,
    reason="Set ARTHUR_RESEARCH_URL (and ARTHUR_RESEARCH_TOKEN) to run against a live service",
)


@live
async def test_live_service_rejects_a_missing_token():
    backend = HttpResearchBackend(
        base_url=LIVE_URL, token="", timeout=15, poll_interval=0.5
    )

    with pytest.raises(ResearchUnavailable):
        await backend.research("What is the capital of India?")


@live
async def test_live_service_rejects_a_malformed_token():
    backend = HttpResearchBackend(
        base_url=LIVE_URL, token="not-a-jwt", timeout=15, poll_interval=0.5
    )

    with pytest.raises(ResearchUnavailable):
        await backend.research("What is the capital of India?")


@live
async def test_live_service_accepts_a_valid_token_and_starts_a_run():
    token = os.getenv("ARTHUR_RESEARCH_TOKEN")
    if not token:
        pytest.skip("ARTHUR_RESEARCH_TOKEN is not set")

    backend = HttpResearchBackend(
        base_url=LIVE_URL, token=token, timeout=60, poll_interval=1.0
    )

    try:
        result = await backend.research("What is the capital of India?")
    except ResearchUnavailable:
        raise
    except ResearchError as error:
        assert "rejected the credentials" not in str(error)
        return

    assert result.answer
    assert result.run_id
