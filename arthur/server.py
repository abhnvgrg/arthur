from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from arthur.audit import AuditLog
from arthur.dispatch import Dispatcher
from arthur.events import Event, EventBus, EventType
from arthur.llm import LLM, LLMError, OpenAILLM
from arthur.security import (
    ALL_SCOPES,
    OWNER_NAME,
    Authenticator,
    Principal,
    RateLimiter,
    Scope,
    StreamTickets,
    TokenStore,
    binding_is_public,
    parse_scopes,
)
from arthur.reflection import Critic, critic_from_environment
from arthur.selection import MAX_STEPS, Turn, run_turn
from arthur.session import SessionStore
from arthur.tools.builtins import build_registry
from arthur.tools.research import HttpResearchBackend
from arthur.tools.registry import Risk, ToolSpec

APPROVAL_TIMEOUT_SECONDS = float(os.getenv("ARTHUR_APPROVAL_TIMEOUT", "120"))
RATE_LIMIT = int(os.getenv("ARTHUR_RATE_LIMIT", "120"))
CHAT_RATE_LIMIT = int(os.getenv("ARTHUR_CHAT_RATE_LIMIT", "20"))
RATE_WINDOW_SECONDS = float(os.getenv("ARTHUR_RATE_WINDOW", "60"))
WEB_ROOT = Path(__file__).parent / "web"

RISK_LABEL = {
    Risk.READ_ONLY: "read-only",
    Risk.WRITES: "writes",
    Risk.IRREVERSIBLE: "irreversible",
}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = None
    max_steps: int = Field(default=MAX_STEPS, ge=1, le=8)
    expose_risky: bool = True


class ApprovalRequest(BaseModel):
    call_id: str = Field(min_length=1, max_length=128)
    approved: bool


class TokenRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    scopes: list[str] = Field(default_factory=lambda: sorted(s.value for s in ALL_SCOPES))


class ApprovalBroker:
    """Holds a turn open while a human decides.

    A pending approval is a future keyed by the model's own tool-call id. The
    HTTP handler resolves it, the turn resumes. A decision that never arrives
    times out as a denial rather than holding the turn forever.
    """

    def __init__(self, timeout: float = APPROVAL_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._pending: dict[str, asyncio.Future] = {}
        self._details: dict[str, dict[str, Any]] = {}

    def pending(self) -> list[dict[str, Any]]:
        return list(self._details.values())

    async def request(
        self, call_id: str, spec: ToolSpec, arguments: dict[str, Any]
    ) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[call_id] = future
        self._details[call_id] = {
            "call_id": call_id,
            "tool": spec.name,
            "risk": spec.risk.value,
            "risk_label": RISK_LABEL[spec.risk],
            "description": spec.description,
            "arguments": arguments,
        }
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(call_id, None)
            self._details.pop(call_id, None)

    def resolve(self, call_id: str, approved: bool) -> bool:
        future = self._pending.get(call_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def deny_all(self) -> int:
        denied = 0
        for call_id in list(self._pending):
            if self.resolve(call_id, False):
                denied += 1
        return denied


def default_research_backend() -> HttpResearchBackend | None:
    if os.getenv("ARTHUR_RESEARCH_URL"):
        return HttpResearchBackend()
    return None


def build_store(token: str | None) -> TokenStore:
    if token is None:
        return TokenStore()
    store = TokenStore(load=False)
    store.adopt(OWNER_NAME, token, ALL_SCOPES)
    return store


def create_app(
    llm: LLM | None = None,
    dispatcher: Dispatcher | None = None,
    sessions: SessionStore | None = None,
    bus: EventBus | None = None,
    broker: ApprovalBroker | None = None,
    token: str | None = None,
    store: TokenStore | None = None,
    critic: Critic | None = None,
    require_token: bool = True,
) -> FastAPI:
    tokens = store if store is not None else build_store(token)
    auth = Authenticator(
        tokens,
        tickets=StreamTickets(),
        limiter=RateLimiter(RATE_LIMIT, RATE_WINDOW_SECONDS),
        chat_limiter=RateLimiter(CHAT_RATE_LIMIT, RATE_WINDOW_SECONDS),
        enabled=require_token,
    )
    read = auth.guard(Scope.READ)
    chat_scope = auth.guard(Scope.CHAT)
    approve_scope = auth.guard(Scope.APPROVE)
    admin = auth.guard(Scope.ADMIN)
    streaming = Depends(auth.requires_ticket())

    app = FastAPI(title="ARTHUR", version="0.5.0")
    app.state.tokens = tokens
    app.state.auth = auth

    app.state.llm = llm or OpenAILLM()
    app.state.dispatcher = dispatcher or Dispatcher(
        build_registry(research_backend=default_research_backend()), audit=AuditLog()
    )
    app.state.sessions = sessions or SessionStore()
    app.state.bus = bus or EventBus()
    app.state.broker = broker or ApprovalBroker()
    app.state.critic = critic or critic_from_environment(app.state.llm)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = WEB_ROOT / "index.html"
        if not page.exists():
            return "<h1>ARTHUR</h1><p>The web UI is not installed.</p>"
        return page.read_text(encoding="utf-8").replace(
            "__ARTHUR_AUTH__", "required" if require_token else "off"
        )

    @app.get("/api/tools", dependencies=[read])
    async def list_tools() -> dict[str, Any]:
        registry = app.state.dispatcher.registry
        return {
            "count": len(registry),
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "risk": spec.risk.value,
                    "risk_label": RISK_LABEL[spec.risk],
                    "needs_approval": app.state.dispatcher.requires_confirmation(spec),
                    "parameters": sorted(spec.parameters.model_fields),
                }
                for spec in registry
            ],
        }

    @app.get("/api/sessions", dependencies=[read])
    async def list_sessions() -> dict[str, Any]:
        return {
            "sessions": [
                {
                    "id": session.id,
                    "title": session.title,
                    "turns": session.turns,
                    "tool_calls": session.tool_calls,
                    "updated_at": session.updated_at,
                }
                for session in app.state.sessions.list()
            ]
        }

    @app.post("/api/sessions", dependencies=[chat_scope])
    async def create_session() -> dict[str, Any]:
        session = app.state.sessions.create()
        return {"id": session.id, "title": session.title}

    @app.get("/api/sessions/{session_id}", dependencies=[read])
    async def get_session(session_id: str) -> dict[str, Any]:
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "No such session")
        return session.to_dict()

    @app.delete("/api/sessions/{session_id}", dependencies=[chat_scope])
    async def delete_session(session_id: str) -> dict[str, Any]:
        if not app.state.sessions.delete(session_id):
            raise HTTPException(404, "No such session")
        app.state.bus.clear(session_id)
        return {"deleted": session_id}

    @app.get("/api/events/{session_id}", dependencies=[streaming])
    async def events(session_id: str) -> StreamingResponse:
        async def stream():
            queue = await app.state.bus.subscribe(session_id)
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        event: Event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    payload = json.dumps(event.to_dict(), default=str)
                    yield f"event: {event.type}\ndata: {payload}\n\n"
            finally:
                await app.state.bus.unsubscribe(session_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/approvals", dependencies=[read])
    async def pending_approvals() -> dict[str, Any]:
        return {"pending": app.state.broker.pending()}

    @app.post("/api/approvals")
    async def resolve_approval(
        request: ApprovalRequest,
        principal: Principal = approve_scope,
    ) -> dict[str, Any]:
        resolved = app.state.broker.resolve(request.call_id, request.approved)
        if not resolved:
            raise HTTPException(404, "No approval is waiting on that call")
        return {
            "call_id": request.call_id,
            "approved": request.approved,
            "decided_by": principal.name,
        }

    @app.post("/api/events/ticket")
    async def stream_ticket(principal: Principal = read) -> dict[str, Any]:
        return {
            "ticket": auth.tickets.issue(principal),
            "expires_in": auth.tickets.ttl,
        }

    @app.get("/api/whoami")
    async def whoami(principal: Principal = read) -> dict[str, Any]:
        return principal.to_dict()

    @app.get("/api/tokens", dependencies=[admin])
    async def list_tokens() -> dict[str, Any]:
        return {"tokens": [record.to_public() for record in tokens.records()]}

    @app.post("/api/tokens", dependencies=[admin])
    async def create_token(request: TokenRequest) -> dict[str, Any]:
        try:
            scopes = parse_scopes(request.scopes)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not scopes:
            raise HTTPException(400, "A token needs at least one scope")
        record, secret = tokens.issue(request.name, scopes)
        return {**record.to_public(), "token": secret}

    @app.delete("/api/tokens/{token_id}", dependencies=[admin])
    async def revoke_token(token_id: str) -> dict[str, Any]:
        if not tokens.revoke(token_id):
            raise HTTPException(404, "No such active token")
        return {"revoked": token_id}

    @app.post("/api/chat")
    async def chat(
        request: ChatRequest, principal: Principal = chat_scope
    ) -> dict[str, Any]:
        session = app.state.sessions.get_or_create(request.session_id)
        session.rename_from(request.message)

        async def approve(
            spec: ToolSpec, arguments: dict[str, Any], call_id: str
        ) -> bool:
            return await app.state.broker.request(call_id, spec, arguments)

        try:
            turn: Turn = await run_turn(
                app.state.llm,
                app.state.dispatcher,
                request.message,
                history=session.messages,
                approve=approve,
                max_steps=request.max_steps,
                expose_risky=request.expose_risky,
                bus=app.state.bus,
                session_id=session.id,
                critic=app.state.critic,
            )
        except LLMError as error:
            raise HTTPException(503, str(error)) from error

        session.record_turn(turn.messages, tool_calls=len(turn.tool_results))
        app.state.sessions.save(session)

        return {
            **turn.to_dict(),
            "session_id": session.id,
            "title": session.title,
        }

    @app.get("/api/audit", dependencies=[read])
    async def audit_tail(limit: int = 50) -> dict[str, Any]:
        entries = list(app.state.dispatcher.audit.entries())
        return {
            "verification": app.state.dispatcher.audit.verify(),
            "entries": entries[-limit:],
        }

    @app.get("/api/health", dependencies=[read])
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tools": len(app.state.dispatcher.registry),
            "sessions": len(app.state.sessions),
        }

    return app


app = None


def main() -> int:
    import uvicorn

    global app
    host = os.getenv("ARTHUR_HOST", "127.0.0.1")
    port = int(os.getenv("ARTHUR_PORT", "8765"))

    tokens = TokenStore()
    minted = tokens.ensure_owner()
    app = create_app(store=tokens)

    print(f"ARTHUR on http://{host}:{port}")
    if minted:
        print(f"owner token: {minted}")
        print("This is the only time it is shown. Save it now.")
    else:
        print("Using the tokens already on file. Paste yours into the web UI.")
    if binding_is_public(host):
        print(
            f"WARNING: bound to {host}, which is reachable from the network. "
            "A token with the approve scope can authorise destructive calls."
        )

    uvicorn.run(app, host=host, port=port)
    return 0
