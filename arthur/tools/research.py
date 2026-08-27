from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_SECONDS = 1.0
MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 1000

TERMINAL_STATUSES = {"complete", "completed", "done", "succeeded", "failed", "error"}
FAILED_STATUSES = {"failed", "error"}


class ResearchError(Exception):
    pass


class ResearchUnavailable(ResearchError):
    pass


@dataclass(frozen=True)
class ResearchResult:
    answer: str
    citations: dict[str, str] = field(default_factory=dict)
    cycles: int | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "sources": len(self.citations),
            "reflection_cycles": self.cycles,
            "run_id": self.run_id,
        }


class ResearchBackend(Protocol):
    async def research(self, query: str) -> ResearchResult: ...


class StubResearchBackend:
    def __init__(self, results: Sequence[ResearchResult | Exception]) -> None:
        self._results = list(results)
        self.queries: list[str] = []

    async def research(self, query: str) -> ResearchResult:
        self.queries.append(query)
        if not self._results:
            raise ResearchUnavailable("StubResearchBackend ran out of results")

        outcome = self._results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HttpResearchBackend:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        poll_interval: float | None = None,
        transport: Any = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("ARTHUR_RESEARCH_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.token = token if token is not None else os.getenv("ARTHUR_RESEARCH_TOKEN", "")
        self.timeout = timeout or float(
            os.getenv("ARTHUR_RESEARCH_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        )
        self.poll_interval = poll_interval or float(
            os.getenv("ARTHUR_RESEARCH_POLL", DEFAULT_POLL_SECONDS)
        )
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def research(self, query: str) -> ResearchResult:
        try:
            import httpx
        except ImportError as error:
            raise ResearchUnavailable(
                "The httpx package is required to reach the research service."
            ) from error

        deadline = time.monotonic() + self.timeout

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=30.0,
            transport=self._transport,
        ) as client:
            try:
                started = await client.post(
                    "/research/query", json={"query": query}
                )
            except httpx.HTTPError as error:
                raise ResearchUnavailable(
                    f"Could not reach the research service at {self.base_url}: {error}"
                ) from error

            if started.status_code in (401, 403):
                raise ResearchUnavailable(
                    "The research service rejected the credentials. "
                    "Set ARTHUR_RESEARCH_TOKEN."
                )
            if started.status_code >= 400:
                raise ResearchError(
                    f"The research service refused the query "
                    f"({started.status_code}): {started.text[:200]}"
                )

            run_id = started.json().get("run_id")
            if not run_id:
                raise ResearchError("The research service returned no run id.")

            while True:
                if time.monotonic() >= deadline:
                    raise ResearchError(
                        f"The research run did not finish within {self.timeout:.0f}s "
                        f"(run {run_id})."
                    )

                await asyncio.sleep(self.poll_interval)

                try:
                    polled = await client.get(f"/research/{run_id}/result")
                except httpx.HTTPError as error:
                    raise ResearchUnavailable(
                        f"Lost contact with the research service: {error}"
                    ) from error

                if polled.status_code >= 400:
                    raise ResearchError(
                        f"Could not read the research result "
                        f"({polled.status_code})."
                    )

                body = polled.json()
                status = str(body.get("status", "")).lower()

                if status in FAILED_STATUSES:
                    raise ResearchError(
                        body.get("error") or "The research run failed."
                    )

                if status in TERMINAL_STATUSES:
                    answer = body.get("answer")
                    if not answer:
                        raise ResearchError(
                            "The research run finished without an answer."
                        )
                    return ResearchResult(
                        answer=answer,
                        citations=body.get("citations") or {},
                        cycles=body.get("cycle_count"),
                        run_id=run_id,
                    )


class ResearchArgs(BaseModel):
    query: str = Field(
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description=(
            "A full research question, phrased as a sentence. Not a keyword "
            "search."
        ),
    )


def register(registry, backend: ResearchBackend, timeout_seconds: float | None = None) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="research",
        description=(
            "Answer a question that needs looking up, using a retrieval "
            "pipeline over indexed documents and the web. Returns a synthesised "
            "answer with citations. Use it for questions of fact you cannot "
            "answer directly; it is slow, so do not use it for arithmetic, the "
            "time, or anything already in memory."
        ),
        parameters=ResearchArgs,
        risk=Risk.READ_ONLY,
        timeout_seconds=timeout_seconds
        or float(os.getenv("ARTHUR_RESEARCH_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)) + 15.0,
    )
    async def research(args: ResearchArgs) -> dict[str, Any]:
        return (await backend.research(args.query)).to_dict()
