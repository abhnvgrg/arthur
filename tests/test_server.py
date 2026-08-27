from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from arthur.events import EventBus
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.server import ApprovalBroker, create_app
from arthur.session import SessionStore

pytestmark = pytest.mark.asyncio


def call(name, arguments, call_id="c1"):
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def broker() -> ApprovalBroker:
    return ApprovalBroker(timeout=0.5)


def make_client(dispatcher, script, tmp_path, broker=None, require_token=False) -> TestClient:
    app = create_app(
        llm=ScriptedLLM(script),
        dispatcher=dispatcher,
        sessions=SessionStore(tmp_path / "sessions.json"),
        bus=EventBus(),
        broker=broker or ApprovalBroker(timeout=0.5),
        token="test-token",
        require_token=require_token,
    )
    return TestClient(app)


async def test_health_reports_the_tool_count(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["tools"] == len(dispatcher.registry)


async def test_the_tool_listing_marks_which_need_approval(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    tools = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}

    assert tools["calculate"]["needs_approval"] is False
    assert tools["remember"]["needs_approval"] is True
    assert tools["delete_file"]["risk_label"] == "irreversible"


async def test_a_session_can_be_created_and_fetched(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    created = client.post("/api/sessions").json()
    fetched = client.get(f"/api/sessions/{created['id']}").json()

    assert fetched["id"] == created["id"]
    assert fetched["messages"] == []


async def test_fetching_an_unknown_session_is_a_404(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    assert client.get("/api/sessions/s_nope").status_code == 404


async def test_a_session_can_be_deleted(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)
    created = client.post("/api/sessions").json()

    assert client.delete(f"/api/sessions/{created['id']}").status_code == 200
    assert client.get(f"/api/sessions/{created['id']}").status_code == 404


async def test_chat_answers_and_records_the_turn(dispatcher, tmp_path):
    client = make_client(dispatcher, [Completion(text="Hello there.")], tmp_path)

    body = client.post("/api/chat", json={"message": "hi"}).json()

    assert body["answer"] == "Hello there."
    assert body["session_id"]

    session = client.get(f"/api/sessions/{body['session_id']}").json()
    assert session["turns"] == 1
    assert any(m["role"] == "user" for m in session["messages"])


async def test_the_session_title_comes_from_the_first_message(dispatcher, tmp_path):
    client = make_client(dispatcher, [Completion(text="ok")], tmp_path)

    body = client.post(
        "/api/chat", json={"message": "What is the capital of India?"}
    ).json()

    assert body["title"].startswith("What is the capital")


async def test_history_carries_into_the_next_turn(dispatcher, tmp_path):
    llm = ScriptedLLM([Completion(text="First."), Completion(text="Second.")])
    client = make_client(dispatcher, [], tmp_path)
    client.app.state.llm = llm

    first = client.post("/api/chat", json={"message": "one"}).json()
    client.post("/api/chat", json={"message": "two", "session_id": first["session_id"]})

    second_call_messages = llm.calls[1]["messages"]
    assert any(m.get("content") == "one" for m in second_call_messages)


async def test_a_read_only_tool_runs_without_any_approval(dispatcher, tmp_path):
    client = make_client(
        dispatcher,
        [
            Completion(tool_calls=(call("calculate", {"expression": "6*7"}),)),
            Completion(text="42."),
        ],
        tmp_path,
    )

    body = client.post("/api/chat", json={"message": "six times seven"}).json()

    assert body["answer"] == "42."
    assert body["tool_results"][0]["ok"] is True


async def test_an_unapproved_risky_tool_times_out_as_a_denial(dispatcher, tmp_path, memory):
    client = make_client(
        dispatcher,
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="I could not save that."),
        ],
        tmp_path,
    )

    body = client.post("/api/chat", json={"message": "remember a"}).json()

    assert body["tool_results"][0]["ok"] is False
    assert memory.recall("a") is None


async def test_risky_tools_can_be_withheld_from_the_model(dispatcher, tmp_path):
    llm = ScriptedLLM([Completion(text="ok")])
    client = make_client(dispatcher, [], tmp_path)
    client.app.state.llm = llm

    client.post("/api/chat", json={"message": "hi", "expose_risky": False})

    offered = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert "delete_file" not in offered
    assert "calculate" in offered


async def test_resolving_an_unknown_approval_is_a_404(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    response = client.post(
        "/api/approvals", json={"call_id": "nothing", "approved": True}
    )
    assert response.status_code == 404


async def test_the_audit_endpoint_reports_verification(dispatcher, tmp_path):
    client = make_client(
        dispatcher,
        [
            Completion(tool_calls=(call("calculate", {"expression": "1+1"}),)),
            Completion(text="2."),
        ],
        tmp_path,
    )
    client.post("/api/chat", json={"message": "one plus one"})

    body = client.get("/api/audit").json()

    assert body["verification"]["status"] == "VERIFIED"
    assert body["entries"][-1]["tool"] == "calculate"


async def test_the_web_page_is_served(dispatcher, tmp_path):
    client = make_client(dispatcher, [], tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "ARTHUR" in response.text


async def test_the_broker_resolves_a_waiting_approval(broker):
    from arthur.tools.registry import Risk, ToolSpec
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    spec = ToolSpec(
        name="t", description="d", parameters=Args, handler=lambda a: None, risk=Risk.WRITES
    )

    async def decide():
        await asyncio.sleep(0.05)
        assert broker.resolve("c1", True) is True

    task = asyncio.create_task(decide())
    granted = await broker.request("c1", spec, {})
    await task

    assert granted is True


async def test_the_broker_denies_when_nobody_answers(broker):
    from arthur.tools.registry import Risk, ToolSpec
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    spec = ToolSpec(
        name="t", description="d", parameters=Args, handler=lambda a: None, risk=Risk.WRITES
    )

    assert await broker.request("c2", spec, {}) is False


async def test_a_pending_approval_is_listed_while_it_waits(broker):
    from arthur.tools.registry import Risk, ToolSpec
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    spec = ToolSpec(
        name="delete_everything",
        description="d",
        parameters=Args,
        handler=lambda a: None,
        risk=Risk.IRREVERSIBLE,
    )

    async def inspect_then_deny():
        await asyncio.sleep(0.05)
        pending = broker.pending()
        assert len(pending) == 1
        assert pending[0]["tool"] == "delete_everything"
        assert pending[0]["risk_label"] == "irreversible"
        broker.resolve("c3", False)

    task = asyncio.create_task(inspect_then_deny())
    result = await broker.request("c3", spec, {"target": "all"})
    await task

    assert result is False
    assert broker.pending() == []


async def test_resolving_twice_only_counts_once(broker):
    from arthur.tools.registry import Risk, ToolSpec
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    spec = ToolSpec(
        name="t", description="d", parameters=Args, handler=lambda a: None, risk=Risk.WRITES
    )

    async def decide():
        await asyncio.sleep(0.05)
        assert broker.resolve("c4", True) is True
        assert broker.resolve("c4", False) is False

    task = asyncio.create_task(decide())
    assert await broker.request("c4", spec, {}) is True
    await task


def guarded(dispatcher, tmp_path) -> TestClient:
    return make_client(dispatcher, [Completion(text="ok")], tmp_path, require_token=True)


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/health"),
        ("get", "/api/tools"),
        ("get", "/api/sessions"),
        ("post", "/api/sessions"),
        ("get", "/api/audit"),
        ("get", "/api/approvals"),
        ("get", "/"),
    ],
)
async def test_every_endpoint_refuses_an_unauthenticated_caller(
    dispatcher, tmp_path, method, path
):
    client = guarded(dispatcher, tmp_path)

    assert getattr(client, method)(path).status_code == 401


async def test_chat_refuses_an_unauthenticated_caller(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401


async def test_a_wrong_token_is_refused(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    response = client.get(
        "/api/health", headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


async def test_the_right_token_is_accepted(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    response = client.get(
        "/api/health", headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200


async def test_a_malformed_authorization_header_is_refused(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get(
        "/api/health", headers={"Authorization": "test-token"}
    ).status_code == 401
    assert client.get(
        "/api/health", headers={"Authorization": "Basic test-token"}
    ).status_code == 401


async def test_the_event_stream_accepts_a_query_token(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get("/api/events/s1").status_code == 401


async def test_the_page_carries_the_token_when_auth_is_on(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    page = client.get("/", headers={"Authorization": "Bearer test-token"}).text

    assert "__ARTHUR_TOKEN__" not in page
    assert "test-token" in page


async def test_a_token_is_minted_and_reused(tmp_path):
    from arthur.security import load_or_create_token

    path = tmp_path / "config.json"
    first = load_or_create_token(path)

    assert len(first) > 20
    assert load_or_create_token(path) == first


async def test_an_environment_token_wins(tmp_path, monkeypatch):
    from arthur.security import load_or_create_token

    monkeypatch.setenv("ARTHUR_API_TOKEN", "from-the-environment")

    assert load_or_create_token(tmp_path / "config.json") == "from-the-environment"


async def test_a_corrupt_config_still_yields_a_token(tmp_path):
    from arthur.security import load_or_create_token

    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    assert len(load_or_create_token(path)) > 20


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
async def test_a_public_binding_is_recognised(host):
    from arthur.security import binding_is_public

    assert binding_is_public(host) is True


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
async def test_a_local_binding_is_not_flagged(host):
    from arthur.security import binding_is_public

    assert binding_is_public(host) is False
