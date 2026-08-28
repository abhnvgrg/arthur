from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional


class EventType:
    TURN_STARTED = "turn_started"
    THINKING = "thinking"
    TOOL_PROPOSED = "tool_proposed"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    REFLECTION = "reflection"
    ANSWER_DELTA = "answer_delta"
    ANSWER = "answer"
    TURN_FINISHED = "turn_finished"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Listener = Callable[[Event], Awaitable[None] | None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._history: dict[str, list[Event]] = {}
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> None:
        async with self._lock:
            self._history.setdefault(event.session_id, []).append(event)
            queues = list(self._subscribers.get(event.session_id, ()))

        for queue in queues:
            queue.put_nowait(event)

    async def subscribe(self, session_id: str, replay: bool = False) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(session_id, []).append(queue)
            backlog = list(self._history.get(session_id, ())) if replay else []

        for event in backlog:
            queue.put_nowait(event)
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            listeners = self._subscribers.get(session_id)
            if not listeners:
                return
            if queue in listeners:
                listeners.remove(queue)
            if not listeners:
                self._subscribers.pop(session_id, None)

    async def stream(
        self, session_id: str, replay: bool = False
    ) -> AsyncIterator[Event]:
        queue = await self.subscribe(session_id, replay=replay)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.type == EventType.TURN_FINISHED:
                    continue
        finally:
            await self.unsubscribe(session_id, queue)

    def history(self, session_id: str) -> list[Event]:
        return list(self._history.get(session_id, ()))

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._history.clear()
        else:
            self._history.pop(session_id, None)

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, ()))


class Emitter:
    def __init__(self, bus: EventBus | None, session_id: str) -> None:
        self.bus = bus
        self.session_id = session_id

    async def __call__(self, event_type: str, **data: Any) -> None:
        if self.bus is None:
            return
        await self.bus.emit(
            Event(type=event_type, session_id=self.session_id, data=data)
        )
