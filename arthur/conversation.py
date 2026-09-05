from __future__ import annotations

import asyncio
import os
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from arthur.dispatch import Dispatcher
from arthur.events import EventBus, EventType
from arthur.llm import LLM, LLMError
from arthur.selection import run_turn
from arthur.tools.registry import Risk, ToolSpec
from arthur.voice import (
    Clip,
    Microphone,
    Playback,
    SentenceBuffer,
    Speaker,
    SpeechStream,
    Transcriber,
    VoiceError,
    affirmative,
    confirm_word,
    confirms,
    heard_wake_word,
    loudness,
    strip_wake_word,
    wake_word,
)

VOICE_SESSION = "voice"
FOLLOW_UP_SECONDS = 25.0
APPROVAL_SECONDS = 20.0
BARGE_LOUDNESS = 2200.0

FILLERS = (
    "One moment.",
    "Let me check that.",
    "Working on it.",
    "Just a second.",
)

STOP_WORDS = frozenset(
    {"quit", "exit", "stop", "goodbye", "good bye", "thats all", "that is all"}
)


def flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def heard_stop(text: str) -> bool:
    cleaned = "".join(c for c in text.lower() if c.isalpha() or c.isspace()).strip()
    return cleaned in STOP_WORDS


def describe_call(spec: ToolSpec, arguments: dict[str, Any]) -> str:
    detail = ", ".join(
        f"{key} {value}"
        for key, value in list(arguments.items())[:3]
        if value not in (None, "", [])
    )
    action = spec.name.replace("_", " ")
    return f"{action} with {detail}" if detail else action


@dataclass
class Ears:
    microphone: Microphone
    stop: threading.Event = field(default_factory=threading.Event)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    failure: BaseException | None = None
    thread: threading.Thread | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> "Ears":
        if self.thread is not None:
            return self
        running = loop or asyncio.get_running_loop()

        def pump() -> None:
            try:
                for clip in self.microphone.listen_forever(self.stop):
                    running.call_soon_threadsafe(self.queue.put_nowait, clip)
            except BaseException as error:
                self.failure = error
            finally:
                running.call_soon_threadsafe(self.queue.put_nowait, None)

        self.thread = threading.Thread(target=pump, daemon=True, name="arthur-ears")
        self.thread.start()
        return self

    async def next(self, timeout: float | None = None) -> Clip | None:
        if timeout is None:
            return await self.queue.get()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def drain(self) -> int:
        dropped = 0
        ended = False
        while not self.queue.empty():
            if self.queue.get_nowait() is None:
                ended = True
                continue
            dropped += 1
        if ended:
            self.queue.put_nowait(None)
        return dropped

    def close(self) -> None:
        self.stop.set()


class Conversation:
    def __init__(
        self,
        llm: LLM,
        dispatcher: Dispatcher,
        transcriber: Transcriber,
        speaker: Speaker,
        microphone: Microphone,
        playback: Playback,
        wake: str | None = None,
        require_wake: bool = True,
        barge_in: bool = False,
        speak_approvals: bool = True,
        typed_irreversible: bool = False,
        confirm: str | None = None,
        critic: Any = None,
        librarian: Any = None,
        approval_seconds: float = APPROVAL_SECONDS,
        report: Any = print,
    ) -> None:
        self.llm = llm
        self.dispatcher = dispatcher
        self.transcriber = transcriber
        self.speaker = speaker
        self.microphone = microphone
        self.playback = playback
        self.wake = wake_word() if wake is None else wake.strip().lower()
        self.require_wake = require_wake
        self.barge_in = barge_in
        self.speak_approvals = speak_approvals
        self.typed_irreversible = typed_irreversible
        self.confirm = confirm_word() if confirm is None else confirm.strip().lower()
        self.critic = critic
        self.librarian = librarian
        self.approval_seconds = approval_seconds
        self.report = report
        self.history: list[dict[str, Any]] = []
        self.ears: Ears | None = None
        self.speaking: SpeechStream | None = None
        self.awake_until = 0.0

    def log(self, line: str) -> None:
        if self.report is not None:
            self.report(line)

    async def hear(self, clip: Clip) -> str:
        try:
            return await self.transcriber.transcribe(clip)
        except VoiceError as error:
            self.log(f"  could not transcribe: {error}")
            return ""

    async def next_words(self, timeout: float | None = None) -> str | None:
        if self.ears is None:
            return None
        clip = await self.ears.next(timeout)
        if clip is None:
            return None
        return await self.hear(clip)

    async def say(self, text: str) -> None:
        stream = SpeechStream(
            self.speaker,
            self.playback,
            on_error=lambda error: self.log(f"  could not speak: {error}"),
        ).start()
        self.speaking = stream
        stream.feed(text)
        await stream.close()
        self.speaking = None
        if self.ears is not None and not self.barge_in:
            self.ears.drain()

    async def typed_approval(self, spec: ToolSpec, arguments: dict[str, Any]) -> bool:
        self.log(f"  {spec.name} is {spec.risk.value} - approve by typing")
        self.log(f"  arguments: {arguments}")
        reply = await asyncio.to_thread(input, "  allow it? [y/N] ")
        return reply.strip().lower() in {"y", "yes"}

    async def spoken_approval(
        self, spec: ToolSpec, arguments: dict[str, Any], strict: bool = False
    ) -> bool:
        action = describe_call(spec, arguments)

        if strict:
            await self.say(
                f"I am about to {action}. That cannot be undone. "
                f"Say {self.confirm} if you want me to."
            )
        else:
            await self.say(f"You want me to {action}. Shall I?")

        if self.ears is not None:
            self.ears.drain()

        for attempt in range(2):
            heard = await self.next_words(self.approval_seconds)
            if heard is None:
                break
            if not heard.strip():
                continue

            self.log(f"  you: {heard}")

            if strict:
                if confirms(heard, self.confirm):
                    await self.say("Doing it now.")
                    return True
                if affirmative(heard) is False:
                    await self.say("Leaving it.")
                    return False
                if attempt == 0:
                    await self.say(f"That one needs the word {self.confirm}.")
                continue

            verdict = affirmative(heard)
            if verdict is not None:
                await self.say("Right away." if verdict else "Leaving it.")
                return verdict
            if attempt == 0:
                await self.say("Yes or no?")

        await self.say("I did not catch that, so I have left it alone.")
        return False

    async def approve(self, spec: ToolSpec, arguments: dict[str, Any]) -> bool:
        irreversible = spec.risk is Risk.IRREVERSIBLE

        if not self.speak_approvals or (irreversible and self.typed_irreversible):
            return await self.typed_approval(spec, arguments)

        return await self.spoken_approval(spec, arguments, strict=irreversible)

    async def consume(self, queue: asyncio.Queue, stream: SpeechStream) -> bool:
        buffer = SentenceBuffer()
        filled = False
        answered = False

        while True:
            event = await queue.get()

            if event.type == EventType.ANSWER_DELTA:
                text = event.data.get("text", "")
                if text.strip():
                    answered = True
                for sentence in buffer.feed(text):
                    stream.feed(sentence)
            elif event.type == EventType.THINKING:
                tail = buffer.drain()
                if tail:
                    stream.feed(tail)
                answered = False
            elif event.type in (EventType.TOOL_PROPOSED, EventType.TOOL_STARTED):
                if not filled:
                    filled = True
                    stream.feed(random.choice(FILLERS))
            elif event.type in (EventType.TURN_FINISHED, EventType.ERROR):
                break

        tail = buffer.drain()
        if tail:
            stream.feed(tail)
        return answered

    async def watch_for_barge(self, stop: threading.Event) -> None:
        if not self.barge_in or self.ears is None:
            return
        while not stop.is_set():
            clip = await self.ears.next(timeout=0.4)
            if clip is None:
                continue
            if loudness(clip.audio) >= BARGE_LOUDNESS:
                stop.set()
                self.log("  [you cut in]")
                return

    async def respond(self, message: str) -> str | None:
        bus = EventBus()
        queue = await bus.subscribe(VOICE_SESSION)
        stop = threading.Event()
        stream = SpeechStream(
            self.speaker,
            self.playback,
            stop=stop,
            on_error=lambda error: self.log(f"  could not speak: {error}"),
        ).start()
        self.speaking = stream

        turn_task = asyncio.create_task(
            run_turn(
                self.llm,
                self.dispatcher,
                message,
                history=self.history,
                approve=self.approve,
                bus=bus,
                session_id=VOICE_SESSION,
                critic=self.critic,
                max_reflections=0,
            )
        )
        watcher = asyncio.create_task(self.watch_for_barge(stop))
        consumer = asyncio.create_task(self.consume(queue, stream))

        try:
            turn = await turn_task
        except LLMError as error:
            stop.set()
            consumer.cancel()
            watcher.cancel()
            await stream.close()
            self.speaking = None
            self.log(f"  {error}")
            await self.say("I could not reach the model just then.")
            return None
        finally:
            await bus.unsubscribe(VOICE_SESSION, queue)

        answered = await consumer

        if not answered and turn.answer:
            stream.feed(turn.answer)

        await stream.close()
        watcher.cancel()
        self.speaking = None

        for result in turn.tool_results:
            self.log(f"  [{result.tool}: {'ok' if result.ok else result.outcome}]")
        if turn.answer:
            self.log(f"  diana: {turn.answer}")
        if stream.interrupted:
            self.log("  [interrupted]")

        self.history = turn.messages[1:]

        if self.librarian is not None:
            try:
                for key, value in await self.librarian.absorb(turn.messages):
                    self.log(f"  [remembered {key}: {value}]")
            except Exception as error:
                self.log(f"  [memory skipped: {error}]")

        if self.ears is not None and not self.barge_in:
            self.ears.drain()
        return turn.answer

    def is_awake(self, now: float) -> bool:
        return now < self.awake_until

    async def run(self, turns: int | None = None) -> int:
        loop = asyncio.get_running_loop()
        self.ears = Ears(self.microphone).start(loop)
        handled = 0

        try:
            while turns is None or handled < turns:
                clip = await self.ears.next()
                if clip is None:
                    if self.ears.failure is not None:
                        self.log(f"  microphone stopped: {self.ears.failure}")
                        return 1
                    return 0

                heard = await self.hear(clip)
                if not heard.strip():
                    continue

                awake = self.is_awake(loop.time())

                if self.require_wake and not awake:
                    if not heard_wake_word(heard, self.wake):
                        continue
                    heard = strip_wake_word(heard, self.wake)
                    if not heard.strip():
                        self.awake_until = loop.time() + FOLLOW_UP_SECONDS
                        await self.say("Yes?")
                        continue

                self.log(f"  you: {heard}")

                if heard_stop(heard):
                    await self.say("Goodbye.")
                    return 0

                await self.respond(heard)
                handled += 1
                self.awake_until = loop.time() + FOLLOW_UP_SECONDS
        finally:
            self.ears.close()

        return 0
