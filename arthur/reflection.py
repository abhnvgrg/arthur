from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from arthur.dispatch import Outcome, ToolResult
from arthur.llm import LLM, LLMError, parse_arguments

MAX_REFLECTIONS = 1
MAX_MODEL_ISSUES = 3
MAX_VALUE_CHARACTERS = 400

SUCCESS_CLAIMS = (
    r"\b(saved|stored|remembered|created|added|written|wrote|deleted|removed"
    r"|completed|marked (?:it )?(?:as )?done|updated|scheduled|set up|done)\b"
)

FAILURE_ACKNOWLEDGEMENTS = (
    r"\b(could ?n[o']t|cannot|can ?not|unable|failed|failure|denied|refused"
    r"|rejected|did ?n[o']t|was ?n[o']t able|were ?n[o']t able|no permission"
    r"|not approved|needs? (?:your )?approval|awaiting|blocked|error|problem"
    r"|did not|invalid|unknown|does ?n[o']t exist|not found)\b"
)

_success = re.compile(SUCCESS_CLAIMS, re.IGNORECASE)
_acknowledged = re.compile(FAILURE_ACKNOWLEDGEMENTS, re.IGNORECASE)


class IssueKind:
    NO_ANSWER = "no_answer"
    UNREPORTED_FAILURE = "unreported_failure"
    UNSUPPORTED_SUCCESS_CLAIM = "unsupported_success_claim"
    STOPPED_AT_LIMIT = "stopped_at_limit"
    CONTRADICTS_TOOL_RESULT = "contradicts_tool_result"
    ANSWERS_A_DIFFERENT_QUESTION = "answers_a_different_question"
    FABRICATED_DETAIL = "fabricated_detail"
    MODEL_FLAGGED = "model_flagged"


MODEL_KINDS = frozenset(
    {
        IssueKind.UNREPORTED_FAILURE,
        IssueKind.UNSUPPORTED_SUCCESS_CLAIM,
        IssueKind.CONTRADICTS_TOOL_RESULT,
        IssueKind.ANSWERS_A_DIFFERENT_QUESTION,
        IssueKind.FABRICATED_DETAIL,
    }
)

RULES = "rules"
MODEL = "model"


@dataclass(frozen=True)
class Issue:
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class Critique:
    passed: bool
    issues: tuple[Issue, ...] = field(default_factory=tuple)
    source: str = RULES

    @property
    def kinds(self) -> set[str]:
        return {issue.kind for issue in self.issues}

    def gap(self) -> str:
        if self.passed:
            return ""
        problems = "\n".join(f"- {issue.detail}" for issue in self.issues)
        return (
            "Your previous answer has a problem:\n"
            f"{problems}\n"
            "Rewrite the answer so it states plainly what did and did not happen. "
            "Do not claim any action succeeded unless a tool result says so. "
            "Do not repeat a tool call that was refused."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "source": self.source,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def failed_results(results: Sequence[ToolResult]) -> list[ToolResult]:
    return [result for result in results if not result.ok]


def mentions_failure(answer: str) -> bool:
    return bool(_acknowledged.search(answer))


def claims_success(answer: str) -> bool:
    return bool(_success.search(answer))


def names_any(answer: str, results: Sequence[ToolResult]) -> bool:
    lowered = answer.lower()
    return any(result.tool.replace("_", " ") in lowered for result in results) or any(
        result.tool in lowered for result in results
    )


def critique(
    answer: str | None,
    results: Sequence[ToolResult],
    stopped_at_limit: bool = False,
) -> Critique:
    issues: list[Issue] = []

    if stopped_at_limit:
        issues.append(
            Issue(
                IssueKind.STOPPED_AT_LIMIT,
                "The turn hit the step limit before finishing, so the answer may "
                "be based on incomplete work.",
            )
        )

    if answer is None or not answer.strip():
        issues.append(
            Issue(
                IssueKind.NO_ANSWER,
                "The turn produced no answer text for the user.",
            )
        )
        return Critique(passed=False, issues=tuple(issues))

    failures = failed_results(results)
    if failures:
        acknowledged = mentions_failure(answer)
        named = names_any(answer, failures)

        if not acknowledged and not named:
            listed = ", ".join(
                f"{result.tool} ({result.outcome})" for result in failures
            )
            issues.append(
                Issue(
                    IssueKind.UNREPORTED_FAILURE,
                    f"These tool calls did not succeed and the answer does not "
                    f"mention it: {listed}.",
                )
            )

        if claims_success(answer) and not acknowledged:
            refused = ", ".join(
                result.tool
                for result in failures
                if result.outcome
                in (Outcome.CONFIRMATION_REQUIRED, Outcome.FAILED, Outcome.TIMEOUT)
            )
            if refused:
                issues.append(
                    Issue(
                        IssueKind.UNSUPPORTED_SUCCESS_CLAIM,
                        f"The answer reads as though an action succeeded, but "
                        f"{refused} did not run successfully.",
                    )
                )

    return Critique(passed=not issues, issues=tuple(issues))


CRITIC_SYSTEM = (
    "You are checking whether an assistant's answer honestly describes what "
    "its tools actually did. You are given the answer and the exact record of "
    "every tool call. Judge only consistency between the two, never whether a "
    "fact about the world is correct.\n\n"
    "Report a problem when the answer claims an action succeeded that the "
    "record does not support, omits a failure the user needs to know about, "
    "contradicts a tool result, invents a detail no tool returned, or answers "
    "a question that was not asked.\n\n"
    "An answer that does all of this correctly passes. Saying so is the "
    "expected outcome; do not invent a problem to seem useful.\n\n"
    "Reply with JSON only: "
    '{"passed": true} or '
    '{"passed": false, "issues": [{"kind": "...", "detail": "..."}]}. '
    "Use these kinds: unreported_failure, unsupported_success_claim, "
    "contradicts_tool_result, fabricated_detail, "
    "answers_a_different_question. Each detail is one sentence naming what is "
    "wrong, addressed to the assistant that wrote the answer."
)


def truncate(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= MAX_VALUE_CHARACTERS:
        return value
    return text[:MAX_VALUE_CHARACTERS] + "… (truncated)"


def describe(results: Sequence[ToolResult]) -> str:
    """The tool record as the critic sees it, with long values cut down."""
    if not results:
        return "No tools were called."
    lines = []
    for result in results:
        record = dict(result.for_model())
        if "result" in record:
            record["result"] = truncate(record["result"])
        record["arguments"] = {
            key: truncate(value) for key, value in result.arguments.items()
        }
        lines.append(json.dumps(record, default=str))
    return "\n".join(lines)


def parse_verdict(raw: str | None) -> Critique:
    """Read the critic's reply, treating anything unreadable as no opinion.

    A critic that cannot be understood must not be able to fail a turn, so
    every unparseable shape resolves to a pass. The deterministic layer has
    already run by this point; this one can only add findings, never remove
    the ones that were earned.
    """
    parsed, malformed = parse_arguments(raw)
    if malformed or not parsed:
        return Critique(passed=True, source=MODEL)

    if parsed.get("passed") is True:
        return Critique(passed=True, source=MODEL)

    issues = []
    for entry in parsed.get("issues", [])[:MAX_MODEL_ISSUES]:
        if not isinstance(entry, dict):
            continue
        detail = str(entry.get("detail", "")).strip()
        if not detail:
            continue
        kind = str(entry.get("kind", "")).strip()
        issues.append(
            Issue(kind if kind in MODEL_KINDS else IssueKind.MODEL_FLAGGED, detail)
        )

    if not issues:
        return Critique(passed=True, source=MODEL)
    return Critique(passed=False, issues=tuple(issues), source=MODEL)


class Critic(Protocol):
    async def review(
        self, answer: str, results: Sequence[ToolResult]
    ) -> Critique: ...


@dataclass
class LLMCritic:
    """A second opinion from a model, for what regular expressions cannot see.

    The pattern layer catches the shape of a dishonest answer: a success verb
    with no acknowledged failure. It cannot catch an answer that is fluent,
    contains no trigger word, and is still wrong about what happened. This can,
    at the cost of one extra completion per turn.

    It runs only when the deterministic layer has already passed, so the cheap
    check spends nothing and the expensive one is never asked a question that
    is already answered.
    """

    llm: LLM

    async def review(self, answer: str, results: Sequence[ToolResult]) -> Critique:
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Tool record:\n{describe(results)}\n\n"
                    f"Answer given to the user:\n{answer}"
                ),
            },
        ]
        try:
            completion = await self.llm.complete(messages, [])
        except LLMError:
            return Critique(passed=True, source=MODEL)
        return parse_verdict(completion.text)


def critic_from_environment(llm: LLM) -> Critic | None:
    if os.getenv("ARTHUR_LLM_CRITIC", "").strip().lower() in {"1", "true", "yes", "on"}:
        return LLMCritic(llm)
    return None
