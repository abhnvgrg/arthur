from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from arthur.events import EventBus
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.security import (
    ALL_SCOPES,
    Principal,
    RateLimiter,
    Scope,
    StreamTickets,
    TokenStore,
)
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


AUTH = {"Authorization": "Bearer test-token"}
PUBLIC_PATHS = {"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def guarded(dispatcher, tmp_path, store=None) -> TestClient:
    app = create_app(
        llm=ScriptedLLM([Completion(text="ok")]),
        dispatcher=dispatcher,
        sessions=SessionStore(tmp_path / "sessions.json"),
        bus=EventBus(),
        broker=ApprovalBroker(timeout=0.5),
        token=None if store is not None else "test-token",
        store=store,
        require_token=True,
    )
    return TestClient(app)


def store_with(secret, scopes, tmp_path):
    store = TokenStore(tmp_path / "config.json", load=False)
    store.adopt("scoped", secret, scopes)
    return store


async def test_no_api_route_is_reachable_without_a_token(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)
    checked = 0

    for route in client.app.routes:
        path = getattr(route, "path", "")
        if not path or path in PUBLIC_PATHS:
            continue
        concrete = path.replace("{session_id}", "s1").replace("{token_id}", "t1")
        for method in sorted(getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}):
            response = client.request(method, concrete)
            assert response.status_code == 401, f"{method} {concrete}"
            checked += 1

    assert checked >= 12


async def test_the_page_is_public_and_carries_no_secret(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    page = client.get("/")

    assert page.status_code == 200
    assert "__ARTHUR_AUTH__" not in page.text
    assert "test-token" not in page.text


async def test_a_wrong_token_is_refused(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    response = client.get("/api/health", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401


async def test_the_right_token_is_accepted(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get("/api/health", headers=AUTH).status_code == 200


async def test_a_malformed_authorization_header_is_refused(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get(
        "/api/health", headers={"Authorization": "test-token"}
    ).status_code == 401
    assert client.get(
        "/api/health", headers={"Authorization": "Basic test-token"}
    ).status_code == 401


async def test_a_token_in_the_query_string_is_no_longer_accepted(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get("/api/health?token=test-token").status_code == 401


async def test_the_event_stream_refuses_a_missing_or_bad_ticket(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.get("/api/events/s1").status_code == 401
    assert client.get("/api/events/s1?ticket=invented").status_code == 401


async def test_a_stream_ticket_carries_the_identity_and_is_single_use(
    dispatcher, tmp_path
):
    client = guarded(dispatcher, tmp_path)
    issued = client.post("/api/events/ticket", headers=AUTH).json()
    tickets = client.app.state.auth.tickets

    principal = tickets.redeem(issued["ticket"])

    assert principal.name == "owner"
    assert issued["expires_in"] > 0
    assert client.get(f"/api/events/s1?ticket={issued['ticket']}").status_code == 401


async def test_a_stream_ticket_needs_the_read_scope(dispatcher, tmp_path):
    store = store_with("writer", [Scope.CHAT], tmp_path)
    client = guarded(dispatcher, tmp_path, store=store)

    response = client.post(
        "/api/events/ticket", headers={"Authorization": "Bearer writer"}
    )

    assert response.status_code == 403


async def test_whoami_reports_the_identity_behind_the_token(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    body = client.get("/api/whoami", headers=AUTH).json()

    assert body["name"] == "owner"
    assert set(body["scopes"]) == {"read", "chat", "approve", "admin"}


async def test_a_read_only_token_cannot_chat_or_approve(dispatcher, tmp_path):
    store = store_with("reader", [Scope.READ], tmp_path)
    client = guarded(dispatcher, tmp_path, store=store)
    headers = {"Authorization": "Bearer reader"}

    assert client.get("/api/health", headers=headers).status_code == 200
    assert client.post(
        "/api/chat", json={"message": "hi"}, headers=headers
    ).status_code == 403
    assert client.post(
        "/api/approvals", json={"call_id": "c1", "approved": True}, headers=headers
    ).status_code == 403
    assert client.get("/api/tokens", headers=headers).status_code == 403


async def test_an_approve_token_cannot_mint_tokens(dispatcher, tmp_path):
    store = store_with("approver", [Scope.READ, Scope.APPROVE], tmp_path)
    client = guarded(dispatcher, tmp_path, store=store)
    headers = {"Authorization": "Bearer approver"}

    assert client.post(
        "/api/tokens", json={"name": "sneaky"}, headers=headers
    ).status_code == 403


async def test_an_approval_names_the_principal_that_decided_it(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)
    spec = dispatcher.registry.get("remember")
    app_broker = client.app.state.broker
    decided = {}

    def post_decision():
        return client.post(
            "/api/approvals",
            json={"call_id": "c9", "approved": True},
            headers=AUTH,
        ).json()

    async def decide():
        await asyncio.sleep(0.05)
        decided.update(await asyncio.to_thread(post_decision))

    task = asyncio.create_task(decide())
    granted = await app_broker.request("c9", spec, {"key": "k", "value": "v"})
    await task

    assert granted is True
    assert decided["decided_by"] == "owner"


async def test_a_minted_token_works_and_can_be_revoked(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    minted = client.post(
        "/api/tokens", json={"name": "laptop", "scopes": ["read"]}, headers=AUTH
    ).json()
    headers = {"Authorization": "Bearer " + minted["token"]}

    assert client.get("/api/health", headers=headers).status_code == 200
    assert client.delete(f"/api/tokens/{minted['id']}", headers=AUTH).status_code == 200
    assert client.get("/api/health", headers=headers).status_code == 401


async def test_revoking_an_unknown_token_is_a_404(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    assert client.delete("/api/tokens/nope", headers=AUTH).status_code == 404


async def test_a_token_cannot_be_minted_with_an_unknown_scope(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)

    response = client.post(
        "/api/tokens", json={"name": "bad", "scopes": ["superuser"]}, headers=AUTH
    )

    assert response.status_code == 400


async def test_listed_tokens_never_expose_a_secret(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)
    minted = client.post("/api/tokens", json={"name": "phone"}, headers=AUTH).json()

    listed = client.get("/api/tokens", headers=AUTH).json()["tokens"]

    assert minted["token"] not in json.dumps(listed)
    assert all("hashed" not in record for record in listed)


async def test_too_many_requests_are_refused_with_a_retry_after(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)
    client.app.state.auth.limiter = RateLimiter(3, 60.0)

    codes = [client.get("/api/health", headers=AUTH).status_code for _ in range(5)]

    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]
    assert client.get("/api/health", headers=AUTH).headers["Retry-After"]


async def test_chat_scope_has_its_own_stricter_limit(dispatcher, tmp_path):
    client = guarded(dispatcher, tmp_path)
    client.app.state.auth.chat_limiter = RateLimiter(1, 60.0)

    assert client.post("/api/sessions", headers=AUTH).status_code == 200
    assert client.post("/api/sessions", headers=AUTH).status_code == 429
    assert client.get("/api/health", headers=AUTH).status_code == 200


async def test_a_token_is_stored_only_as_a_hash(tmp_path):
    path = tmp_path / "config.json"
    store = TokenStore(path)

    secret = store.ensure_owner()

    assert secret is not None
    assert secret not in path.read_text(encoding="utf-8")
    assert TokenStore(path).resolve(secret) is not None


async def test_an_owner_is_minted_only_once(tmp_path):
    path = tmp_path / "config.json"

    first = TokenStore(path).ensure_owner()

    assert first is not None
    assert TokenStore(path).ensure_owner() is None


async def test_a_revoked_token_stops_resolving_across_a_restart(tmp_path):
    path = tmp_path / "config.json"
    store = TokenStore(path)
    record, secret = store.issue("phone", [Scope.READ])

    store.revoke(record.id)

    assert TokenStore(path).resolve(secret) is None


async def test_a_legacy_config_keeps_working_and_is_upgraded(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"api_token": "old-secret"}', encoding="utf-8")

    store = TokenStore(path)

    assert store.resolve("old-secret") is not None
    assert "old-secret" not in path.read_text(encoding="utf-8")
    assert TokenStore(path).resolve("old-secret") is not None


async def test_an_environment_token_is_accepted_without_touching_disk(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARTHUR_API_TOKEN", "from-the-environment")
    path = tmp_path / "config.json"

    store = TokenStore(path)

    assert store.resolve("from-the-environment").scopes == ALL_SCOPES
    assert store.ensure_owner() is None
    assert not path.exists()


async def test_a_corrupt_config_does_not_crash_the_store(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    store = TokenStore(path)

    assert store.records() == []
    assert store.ensure_owner() is not None


async def test_a_ticket_expires(dispatcher, tmp_path):
    now = [0.0]
    tickets = StreamTickets(ttl=30.0, clock=lambda: now[0])
    principal = Principal(id="p", name="p", scopes=ALL_SCOPES)
    value = tickets.issue(principal)

    now[0] = 31.0

    assert tickets.redeem(value) is None
    assert len(tickets) == 0


async def test_the_rate_limiter_forgets_an_old_window():
    now = [0.0]
    limiter = RateLimiter(2, 10.0, clock=lambda: now[0])

    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(10.0)

    now[0] = 11.0

    assert limiter.check("a") is None


async def test_rate_limits_are_per_principal():
    limiter = RateLimiter(1, 60.0)

    assert limiter.check("alice") is None
    assert limiter.check("bob") is None
    assert limiter.check("alice") is not None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
async def test_a_public_binding_is_recognised(host):
    from arthur.security import binding_is_public

    assert binding_is_public(host) is True


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
async def test_a_local_binding_is_not_flagged(host):
    from arthur.security import binding_is_public

    assert binding_is_public(host) is False
