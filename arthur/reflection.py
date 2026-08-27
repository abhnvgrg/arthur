from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from arthur.dispatch import Outcome, ToolResult

MAX_REFLECTIONS = 1

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
