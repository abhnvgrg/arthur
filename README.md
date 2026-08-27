# ARTHUR

A personal assistant built on the agent architecture from
[Research Assistant](https://github.com/abhnvgrg/Research-Assistant) — a
LangGraph loop with query decomposition, source routing, relevance grading,
synthesis, and reflection.

That engine reads. This project is about letting it **act**, safely.

```
362 tests passing — no API key, no network, no Docker
```

## What it does

- **Chats and acts** — a tool-calling loop that plans, calls tools, reads the
  results, and answers.
- **19 tools** — memory, tasks, sandboxed file access, unit conversion, time,
  a calculator that does not use `eval`, and `research`, which delegates to the
  Research Assistant's retrieval graph.
- **Asks before it acts** — anything that writes or destroys waits for your
  approval, in the terminal or in the browser.
- **Shows its work** — live events for every step, tool call, and decision.
- **Remembers conversations** — persistent sessions with history trimming.
- **Logs everything** — a hash-chained audit of every attempt, including the
  refused ones.
- **Requires a token** — every endpoint, minted on first run and printed at
  startup.
- **Checks its own answer** — a turn that claims success for something that
  did not happen is sent back for a rewrite.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt

.venv/Scripts/python -m arthur serve      # web UI at http://127.0.0.1:8765
.venv/Scripts/python -m arthur            # terminal REPL
```

The server prints its API token at startup and the web UI receives it
automatically. Set `ARTHUR_API_TOKEN` to supply your own.

The web UI needs `OPENAI_API_KEY`. The terminal REPL can drive tools directly
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
| `arthur/session.py` | Conversations, persistence, history trimming |
| `arthur/server.py` | HTTP API, SSE, approval broker |
| `arthur/web/index.html` | Web UI, no build step |
| `arthur/security.py` | API token, constant-time check, bind warning |
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
- [docs/skills.md](docs/skills.md) — the tools, and how to add one

## Configuration

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | — (required for chat) |
| `ARTHUR_API_TOKEN` | generated on first run |
| `ARTHUR_CONFIG_FILE` | `~/.arthur/config.json` |
| `ARTHUR_HOST` / `ARTHUR_PORT` | `127.0.0.1` / `8765` |
| `ARTHUR_APPROVAL_TIMEOUT` | `120` seconds |
| `ARTHUR_AUDIT_LOG` | `~/.arthur/audit.jsonl` |
| `ARTHUR_MEMORY_FILE` | `~/.arthur/memory.json` |
| `ARTHUR_TASKS_FILE` | `~/.arthur/tasks.json` |
| `ARTHUR_WORKSPACE` | `~/.arthur/workspace` |
| `ARTHUR_RESEARCH_URL` | unset — no research tool without it |
| `ARTHUR_RESEARCH_TOKEN` | empty |

## Not done yet

The API uses **one shared token with no user identity**, so anyone holding it
can approve destructive calls; there is no scope and no rate limiting. Answers
are not streamed token by token. Reflection is heuristic and English-only. No
tool reaches the network. Each
document's *Known limits* section is specific about its own layer.
