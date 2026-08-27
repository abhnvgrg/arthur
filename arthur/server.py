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
from arthur.security import TokenGuard, binding_is_public, load_or_create_token
from arthur.selection import MAX_STEPS, Turn, run_turn
from arthur.session import SessionStore
from arthur.tools.builtins import build_registry
from arthur.tools.research import HttpResearchBackend
from arthur.tools.registry import Risk, ToolSpec

APPROVAL_TIMEOUT_SECONDS = float(os.getenv("ARTHUR_APPROVAL_TIMEOUT", "120"))
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


def create_app(
    llm: LLM | None = None,
    dispatcher: Dispatcher | None = None,
    sessions: SessionStore | None = None,
    bus: EventBus | None = None,
    broker: ApprovalBroker | None = None,
    token: str | None = None,
    require_token: bool = True,
) -> FastAPI:
    api_token = token if token is not None else load_or_create_token()
    guard = TokenGuard(api_token, enabled=require_token)
    app = FastAPI(title="ARTHUR", version="0.4.0", dependencies=[Depends(guard)])
    app.state.api_token = api_token
    app.state.guard = guard

    app.state.llm = llm or OpenAILLM()
    app.state.dispatcher = dispatcher or Dispatcher(
        build_registry(research_backend=default_research_backend()), audit=AuditLog()
    )
    app.state.sessions = sessions or SessionStore()
    app.state.bus = bus or EventBus()
    app.state.broker = broker or ApprovalBroker()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = WEB_ROOT / "index.html"
        if not page.exists():
            return "<h1>ARTHUR</h1><p>The web UI is not installed.</p>"
        return page.read_text(encoding="utf-8").replace(
            "__ARTHUR_TOKEN__", api_token if require_token else ""
        )

    @app.get("/api/tools")
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

    @app.get("/api/sessions")
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

    @app.post("/api/sessions")
    async def create_session() -> dict[str, Any]:
        session = app.state.sessions.create()
        return {"id": session.id, "title": session.title}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "No such session")
        return session.to_dict()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        if not app.state.sessions.delete(session_id):
            raise HTTPException(404, "No such session")
        app.state.bus.clear(session_id)
        return {"deleted": session_id}

    @app.get("/api/events/{session_id}")
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

    @app.get("/api/approvals")
    async def pending_approvals() -> dict[str, Any]:
        return {"pending": app.state.broker.pending()}

    @app.post("/api/approvals")
    async def resolve_approval(request: ApprovalRequest) -> dict[str, Any]:
        resolved = app.state.broker.resolve(request.call_id, request.approved)
        if not resolved:
            raise HTTPException(404, "No approval is waiting on that call")
        return {"call_id": request.call_id, "approved": request.approved}

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
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

    @app.get("/api/audit")
    async def audit_tail(limit: int = 50) -> dict[str, Any]:
        entries = list(app.state.dispatcher.audit.entries())
        return {
            "verification": app.state.dispatcher.audit.verify(),
            "entries": entries[-limit:],
        }

    @app.get("/api/health")
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
    api_token = load_or_create_token()
    app = create_app(token=api_token)

    print(f"ARTHUR on http://{host}:{port}")
    print(f"token: {api_token}")
    if binding_is_public(host):
        print(
            f"WARNING: bound to {host}, which is reachable from the network. "
            "Anyone with the token above can approve destructive tool calls."
        )

    uvicorn.run(app, host=host, port=port)
    return 0
