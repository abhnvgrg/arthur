# ARTHUR

A personal assistant built on the agent architecture from
[Research Assistant](https://github.com/abhnvgrg/Research-Assistant) — a
LangGraph loop with query decomposition, source routing, relevance grading,
synthesis, and reflection.

That engine reads. This project is about letting it **act**, safely.

```
545 tests passing — no API key, no network, no Docker
```

## What it does

- **Chats and acts** — a tool-calling loop that plans, calls tools, reads the
  results, and answers.
- **19 tools** — memory, tasks, sandboxed file access, unit conversion, time,
  a calculator that does not use `eval`, and `research`, which delegates to the
  Research Assistant's retrieval graph.
- **Asks before it acts** — anything that writes or destroys waits for your
  approval, in the terminal or in the browser.
- **Shows its work** — live events for every step, tool call, and decision,
  and the answer itself as it is written.
- **Remembers conversations** — persistent sessions with history trimming.
- **Logs everything** — a hash-chained audit of every attempt, including the
  refused ones.
- **Knows who is asking** — named, scoped, revocable API tokens, stored as
  hashes and rate limited per caller. A token that can watch cannot approve.
- **Checks its own answer** — a turn that claims success for something that
  did not happen is sent back for a rewrite. Deterministic by default; a model
  critic can be switched on for what patterns cannot see.
- **Survives a bad minute** — model calls retry transient failures with
  jittered backoff, and never retry a request that was simply wrong.
- **Speaks first** — a reminder loop that watches the task list and reaches
  you on your phone, your desktop, or by email when something is due.
- **Listens and answers aloud** - push to talk in the terminal, transcribed
  by Whisper and spoken back, on the same key as the model.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt

.venv/Scripts/python -m arthur serve      # web UI at http://127.0.0.1:8765
.venv/Scripts/python -m arthur watch      # reminder daemon
.venv/Scripts/python -m arthur talk       # speak to it, hear it answer
.venv/Scripts/python -m arthur            # terminal REPL
```

`watch` is the only part that speaks first. It polls the task list, and when
something is due it notifies whichever channels are configured — ntfy, a
desktop toast, email — and marks the task so it never fires twice. It refuses
to start with no channel configured, rather than running silently forever.

It deliberately cannot call tools or the model. An unattended loop that could
reach the tool layer would need an approval path with nobody there to answer,
so it reads tasks and sends messages, and that is all it can do.

On first run the server prints the owner token **once** — save it, because
only its hash is kept. Paste it into the web UI when it asks; it is remembered
in `localStorage` after that. `ARTHUR_API_TOKEN` supplies your own instead and
is never written to disk.

The web UI needs a model API key. The terminal REPL can drive tools directly
without one:

```
> calculate expression=(2+3)*sqrt(16)
{ "expression": "(2+3)*sqrt(16)", "result": 20.0 }

> calculate expression=__import__('os').system('echo pwned')
  failed: CalculationError: Could not parse expression

> add_task title="Write the report" due=tomorrow priority=high
  add_task is writes.
  arguments: {"title": "Write the report", "due": "tomorrow", ...}
  run it? [y/N] y

> verify
  {"status": "VERIFIED", "entries_checked": 7}
```

## The idea

An LLM decides which tool to call. An LLM can be wrong, or steered wrong by
something it just read. So the boundary between "the model suggested this" and
"this actually happened" is enforced in code, not in a prompt:

- Every tool declares a **risk tier**. Anything that writes or destroys is held
  for approval, and approval applies to exactly one call — it never carries over.
- Arguments are **validated before the handler runs**. Unexpected fields are
  dropped, not forwarded.
- The dispatcher **never raises**. Every outcome comes back as a result the
  model can read and correct from.
- File tools **cannot leave the workspace**, because paths are resolved before
  they are checked rather than filtered as strings.
- The turn loop **stops at a step cap**, and says so when it does.
- Every attempt is **audited**, including the ones that were refused.

## Layout

| Module | Responsibility |
|---|---|
| `arthur/tools/registry.py` | Tool definitions, risk tiers, OpenAI schemas |
| `arthur/dispatch.py` | Validation, approval policy, timeouts, containment |
| `arthur/audit.py` | Hash-chained, secret-redacting log |
| `arthur/llm.py` | LLM protocol, OpenAI client, scripted fake |
| `arthur/selection.py` | Tool-selection node and the turn loop |
| `arthur/reflection.py` | Self-check on the answer, and the retry loop |
| `arthur/events.py` | Per-session event bus |
| `arthur/clock.py` | Local timezone and the current moment |
| `arthur/schedule.py` | The reminder loop: what is due, and when to say so |
| `arthur/notify.py` | Notification channels: ntfy, toast, email, events |
| `arthur/voice.py` | Speech in and out, and the microphone |
| `arthur/session.py` | Conversations, persistence, history trimming |
| `arthur/server.py` | HTTP API, SSE, approval broker |
| `arthur/web/index.html` | Web UI, no build step |
| `arthur/security.py` | Principals, scopes, hashed tokens, stream tickets, rate limits |
| `arthur/cli.py` | Terminal REPL |
| `arthur/tools/` | tasks · files · convert · memory · time · calculator · research |

## Tests

```bash
.venv/Scripts/python -m pytest
```

Mutation-checked, not just green:

| Guard removed | Tests that fail |
|---|---|
| confirmation gate | 6 |
| sandboxed calculator (replaced with `eval`) | 17 |
| approval enforcement in the turn loop | 4 |
| step cap | 1 |
| unreported-failure check | 10 |
| success-claim check | 2 |
| malformed-argument guard | 1 |

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit, and the
  relationship to the Research Assistant
- [docs/tool-layer.md](docs/tool-layer.md) — risk tiers, the dispatcher, the
  audit log
- [docs/orchestration.md](docs/orchestration.md) — the turn loop, events,
  sessions, parallelism
- [docs/interfaces.md](docs/interfaces.md) — HTTP API, web UI, CLI, and the
  approval broker
- [docs/reflection.md](docs/reflection.md) — the self-check, why it is a
  heuristic, and what it deliberately does not do
- [docs/research.md](docs/research.md) — delegating to the Research Assistant,
  and why the tool polls
- [docs/voice.md](docs/voice.md) - speech in and out, and why approvals
  are never spoken
- [docs/schedule.md](docs/schedule.md) — the reminder loop, what it is allowed
  to do, and why it fires only once
- [docs/skills.md](docs/skills.md) — the tools, and how to add one

## Configuration

| Variable | Default |
|---|---|
| `ARTHUR_LLM_API_KEY` | — (required for chat; falls back to `OPENAI_API_KEY`) |
| `ARTHUR_LLM_BASE_URL` | unset — OpenAI. Any OpenAI-compatible endpoint works |
| `ARTHUR_LLM_MODEL` | `gpt-4o-mini` |
| `ARTHUR_API_TOKEN` | unset — an all-scope token, never stored |
| `ARTHUR_CONFIG_FILE` | `~/.arthur/config.json` |
| `ARTHUR_HOST` / `ARTHUR_PORT` | `127.0.0.1` / `8765` |
| `ARTHUR_APPROVAL_TIMEOUT` | `120` seconds |
| `ARTHUR_RATE_LIMIT` / `ARTHUR_CHAT_RATE_LIMIT` | `120` / `20` per window |
| `ARTHUR_RATE_WINDOW` | `60` seconds |
| `ARTHUR_LLM_ATTEMPTS` | `3` |
| `ARTHUR_LLM_BACKOFF` / `ARTHUR_LLM_BACKOFF_CAP` | `0.5` / `8` seconds |
| `ARTHUR_LLM_CRITIC` | unset — set to `1` for the model critic (a call per turn) |
| `ARTHUR_AUDIT_LOG` | `~/.arthur/audit.jsonl` |
| `ARTHUR_MEMORY_FILE` | `~/.arthur/memory.json` |
| `ARTHUR_TASKS_FILE` | `~/.arthur/tasks.json` |
| `ARTHUR_TIMEZONE` | unset — the system timezone |
| `ARTHUR_STT_MODEL` | `whisper-large-v3-turbo` |
| `ARTHUR_TTS_MODEL` | `canopylabs/orpheus-v1-english` |
| `ARTHUR_TTS_VOICE` | `daniel` - also autumn, diana, hannah, austin, troy |
| `ARTHUR_VOICE_BASE_URL` | unset - falls back to `ARTHUR_LLM_BASE_URL` |
| `ARTHUR_REMIND_INTERVAL` | `60` seconds between checks |
| `ARTHUR_REMIND_LEAD` | `600` seconds — how far ahead of a task to warn |
| `ARTHUR_REMIND_GRACE` | `3600` seconds — how stale a missed task may be |
| `ARTHUR_REMIND_DAY_AT` | unset — dated tasks with no time are not reminded |
| `ARTHUR_NTFY_TOPIC` | unset — no push. Use a long random string |
| `ARTHUR_NTFY_SERVER` | `https://ntfy.sh` |
| `ARTHUR_TOAST` | unset — set to `1` for desktop toasts (needs `plyer`) |
| `ARTHUR_SMTP_HOST` / `_PORT` | unset / `465` |
| `ARTHUR_SMTP_USER` / `_PASSWORD` | — |
| `ARTHUR_SMTP_TO` / `_FROM` | — |
| `ARTHUR_WORKSPACE` | `~/.arthur/workspace` |
| `ARTHUR_RESEARCH_URL` | unset — no research tool without it |
| `ARTHUR_RESEARCH_TOKEN` | empty |

## Not done yet

Scopes are coarse — `read` sees every conversation and the whole audit log,
and nothing records *which* principal approved a call in the hash chain.
Sessions, tickets, and rate-limit counters live in one process's memory, so a
second worker would break all three. Answers are not streamed token by token.
Reflection's default layer is heuristic and English-only, and its optional
model critic is a model grading a model — untested against a real provider. A
streamed answer that the self-check then rejects has already been seen; the UI
withdraws it and streams the rewrite. Only `research` and the reminder
channels reach the network, and only when configured — an ntfy topic is a
shared secret on a public server, so a guessable one publishes your task
titles to anyone who guesses it. Reminders fire only for tasks that carry a
time of day, and `watch` holds its state in one process, so two copies would
both notify. Each document's *Known limits* section is specific about its own
layer.
