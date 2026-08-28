from __future__ import annotations

import pytest

from arthur.dispatch import Outcome, ToolResult
from arthur.llm import Completion, ScriptedLLM, ToolCall
from arthur.reflection import IssueKind, critique
from arthur.selection import run_turn

pytestmark = pytest.mark.asyncio


def ok(tool: str = "calculate", value=None) -> ToolResult:
    return ToolResult(tool=tool, outcome=Outcome.OK, value=value or {"result": 4})


def refused(tool: str = "remember") -> ToolResult:
    return ToolResult(
        tool=tool,
        outcome=Outcome.CONFIRMATION_REQUIRED,
        error=f"{tool} is writes and needs explicit confirmation",
    )


def broke(tool: str = "current_time") -> ToolResult:
    return ToolResult(tool=tool, outcome=Outcome.FAILED, error="ValueError: nope")


def call(name, arguments, call_id="c1"):
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def test_a_clean_turn_passes():
    verdict = critique("Six times seven is 42.", [ok()])

    assert verdict.passed
    assert verdict.issues == ()


async def test_a_turn_with_no_tools_passes():
    assert critique("Delhi is the capital of India.", []).passed


async def test_an_empty_answer_fails():
    verdict = critique("", [ok()])

    assert not verdict.passed
    assert IssueKind.NO_ANSWER in verdict.kinds


async def test_a_missing_answer_fails():
    assert IssueKind.NO_ANSWER in critique(None, []).kinds


async def test_a_whitespace_answer_fails():
    assert IssueKind.NO_ANSWER in critique("   \n  ", []).kinds


async def test_no_answer_short_circuits_the_text_checks():
    verdict = critique(None, [refused()])

    assert verdict.kinds == {IssueKind.NO_ANSWER}


async def test_a_capped_turn_with_no_answer_reports_both():
    verdict = critique(None, [refused()], stopped_at_limit=True)

    assert verdict.kinds == {IssueKind.NO_ANSWER, IssueKind.STOPPED_AT_LIMIT}


async def test_an_unreported_refusal_fails():
    verdict = critique("All set.", [refused()])

    assert not verdict.passed
    assert IssueKind.UNREPORTED_FAILURE in verdict.kinds


async def test_the_issue_names_the_tool_and_outcome():
    verdict = critique("All set.", [refused("add_task")])

    detail = verdict.issues[0].detail
    assert "add_task" in detail
    assert Outcome.CONFIRMATION_REQUIRED in detail


@pytest.mark.parametrize(
    "answer",
    [
        "I could not save that because you did not approve it.",
        "I wasn't able to store it.",
        "That was denied, so nothing changed.",
        "The call failed.",
        "That needs your approval first.",
        "It was refused.",
    ],
)
async def test_an_acknowledged_failure_passes(answer):
    assert critique(answer, [refused()]).passed


async def test_naming_the_tool_counts_as_acknowledgement():
    assert critique("The remember step is still pending.", [refused()]).passed


async def test_a_tool_named_with_underscores_is_matched_in_prose():
    assert critique("The add task step did not go through.", [refused("add_task")]).passed


async def test_claiming_success_after_a_refusal_fails():
    verdict = critique("Saved it for you.", [refused()])

    assert not verdict.passed
    assert IssueKind.UNSUPPORTED_SUCCESS_CLAIM in verdict.kinds


async def test_claiming_success_after_a_failure_fails():
    verdict = critique("Done, that's updated.", [broke()])

    assert IssueKind.UNSUPPORTED_SUCCESS_CLAIM in verdict.kinds


async def test_claiming_success_when_everything_worked_passes():
    assert critique("Saved it for you.", [ok("remember")]).passed


async def test_a_partial_failure_must_still_be_reported():
    verdict = critique("Here is the time.", [ok("current_time"), refused("remember")])

    assert not verdict.passed
    assert IssueKind.UNREPORTED_FAILURE in verdict.kinds


async def test_hitting_the_step_limit_is_flagged():
    verdict = critique("Here is what I found.", [ok()], stopped_at_limit=True)

    assert not verdict.passed
    assert IssueKind.STOPPED_AT_LIMIT in verdict.kinds


async def test_the_gap_tells_the_model_what_to_fix():
    gap = critique("All set.", [refused("add_task")]).gap()

    assert "add_task" in gap
    assert "Do not claim any action succeeded" in gap
    assert "refused" in gap


async def test_a_passing_critique_has_no_gap():
    assert critique("All good.", [ok()]).gap() == ""


async def test_the_critique_serialises():
    payload = critique("All set.", [refused()]).to_dict()

    assert payload["passed"] is False
    assert payload["issues"][0]["kind"] == IssueKind.UNREPORTED_FAILURE


async def test_a_turn_records_a_passing_critique(dispatcher):
    llm = ScriptedLLM([Completion(text="Delhi is the capital.")])

    turn = await run_turn(llm, dispatcher, "capital of India")

    assert turn.critique is not None
    assert turn.critique.passed
    assert turn.reflections == 0


async def test_a_bad_answer_is_sent_back_for_a_rewrite(dispatcher):
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
            Completion(text="I could not save that; you did not approve it."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "remember a", max_steps=4)

    assert turn.reflections == 1
    assert turn.critique.passed
    assert turn.answer.startswith("I could not save")


async def test_the_rewrite_prompt_reaches_the_model(dispatcher):
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
            Completion(text="I could not save that."),
        ]
    )

    await run_turn(llm, dispatcher, "remember a", max_steps=4)

    last_prompt = llm.calls[2]["messages"][-1]
    assert last_prompt["role"] == "user"
    assert "Do not claim any action succeeded" in last_prompt["content"]


async def test_reflection_gives_up_after_its_budget(dispatcher):
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
            Completion(text="Still all set."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "remember a", max_steps=4, max_reflections=1)

    assert turn.reflections == 1
    assert not turn.critique.passed
    assert turn.answer == "Still all set."


async def test_reflection_can_be_turned_off(dispatcher):
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "remember a", reflect=False)

    assert turn.critique is None
    assert turn.reflections == 0
    assert turn.answer == "All set."


async def test_a_negative_reflection_budget_is_refused(dispatcher):
    with pytest.raises(ValueError):
        await run_turn(ScriptedLLM([]), dispatcher, "hi", max_reflections=-1)


async def test_reflection_does_not_fire_with_no_budget(dispatcher):
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
        ]
    )

    turn = await run_turn(llm, dispatcher, "remember a", max_reflections=0)

    assert turn.reflections == 0
    assert not turn.critique.passed


async def test_a_capped_turn_is_critiqued_as_incomplete(dispatcher):
    llm = ScriptedLLM([Completion(tool_calls=(call("calculate", {"expression": "1+1"}),))] * 4)

    turn = await run_turn(llm, dispatcher, "loop", max_steps=2)

    assert turn.stopped_at_limit
    assert IssueKind.STOPPED_AT_LIMIT in turn.critique.kinds


async def test_the_reflection_event_is_emitted(dispatcher):
    from arthur.events import EventBus, EventType

    bus = EventBus()
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(call("remember", {"key": "a", "value": "1"}),)),
            Completion(text="All set."),
            Completion(text="I could not save that."),
        ]
    )

    await run_turn(llm, dispatcher, "remember a", max_steps=4, bus=bus, session_id="s1")

    reflections = [
        event for event in bus.history("s1") if event.type == EventType.REFLECTION
    ]
    assert len(reflections) == 2
    assert reflections[0].data["passed"] is False
    assert reflections[-1].data["passed"] is True


from arthur.llm import LLMError
from arthur.reflection import (
    MODEL,
    RULES,
    Critique,
    Issue,
    LLMCritic,
    critic_from_environment,
    describe,
    parse_verdict,
)


class Critic:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def complete(self, messages, tools):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply)


class Always:
    def __init__(self, critique):
        self.critique = critique
        self.seen = []

    @property
    def calls(self):
        return len(self.seen)

    async def review(self, answer, results):
        self.seen.append(answer)
        return self.critique


async def test_a_clean_verdict_passes():
    verdict = parse_verdict('{"passed": true}')

    assert verdict.passed is True
    assert verdict.source == MODEL


@pytest.mark.parametrize(
    "raw",
    [None, "", "not json at all", "{unclosed", '{"passed": "maybe"}', "[]"],
)
async def test_an_unreadable_verdict_cannot_fail_a_turn(raw):
    assert parse_verdict(raw).passed is True


async def test_a_fenced_verdict_is_still_read():
    raw = '```json\n{"passed": false, "issues": [{"kind": "fabricated_detail", "detail": "It invented a filename."}]}\n```'

    verdict = parse_verdict(raw)

    assert verdict.passed is False
    assert verdict.issues[0].kind == IssueKind.FABRICATED_DETAIL


async def test_an_unknown_kind_is_kept_but_relabelled():
    raw = '{"passed": false, "issues": [{"kind": "vibes", "detail": "It felt wrong."}]}'

    verdict = parse_verdict(raw)

    assert verdict.passed is False
    assert verdict.issues[0].kind == IssueKind.MODEL_FLAGGED
    assert verdict.issues[0].detail == "It felt wrong."


async def test_issues_without_a_detail_are_dropped_and_a_failure_becomes_a_pass():
    raw = '{"passed": false, "issues": [{"kind": "fabricated_detail", "detail": "  "}]}'

    assert parse_verdict(raw).passed is True


async def test_the_number_of_reported_issues_is_capped():
    entries = ", ".join(
        '{"kind": "fabricated_detail", "detail": "issue %d"}' % n for n in range(8)
    )
    raw = '{"passed": false, "issues": [%s]}' % entries

    assert len(parse_verdict(raw).issues) == 3


async def test_the_tool_record_names_every_call_and_truncates_a_long_value():
    long_value = "x" * 900
    record = describe([ok(), ToolResult(tool="read_file", outcome="ok", value=long_value)])

    assert "read_file" in record
    assert "truncated" in record
    assert long_value not in record


async def test_an_empty_tool_record_says_so():
    assert describe([]) == "No tools were called."


async def test_the_critic_sends_the_answer_and_the_record():
    llm = Critic(['{"passed": true}'])

    verdict = await LLMCritic(llm).review("All set.", [refused()])

    assert verdict.passed is True
    prompt = llm.seen[0][1]["content"]
    assert "All set." in prompt
    assert "remember" in prompt


async def test_a_critic_that_fails_does_not_fail_the_turn():
    llm = Critic([LLMError("provider is down")])

    verdict = await LLMCritic(llm).review("All set.", [])

    assert verdict.passed is True
    assert verdict.source == MODEL


async def test_the_critic_runs_only_when_the_rules_already_passed(dispatcher):
    critic = Always(Critique(passed=True, source=MODEL))
    llm = ScriptedLLM([Completion(text="Delhi is the capital.")])

    await run_turn(llm, dispatcher, "capital of India", critic=critic)

    assert critic.calls == 1


async def test_an_answer_the_rules_rejected_is_never_sent_to_the_critic(dispatcher):
    critic = Always(Critique(passed=True, source=MODEL))
    llm = ScriptedLLM(
        [
            Completion(tool_calls=(ToolCall(id="c1", name="forget", arguments={"key": "k"}),)),
            Completion(text="Deleted it."),
            Completion(text="It was not deleted; the call was refused."),
        ]
    )

    await run_turn(llm, dispatcher, "forget k", approve=lambda *a, **k: False, critic=critic)

    assert "Deleted it." not in critic.seen
    assert critic.seen == ["It was not deleted; the call was refused."]


async def test_a_critic_failure_sends_the_answer_back_for_a_rewrite(dispatcher):
    critic = Always(
        Critique(
            passed=False,
            issues=(Issue(IssueKind.FABRICATED_DETAIL, "You invented a filename."),),
            source=MODEL,
        )
    )
    llm = ScriptedLLM(
        [Completion(text="I read notes.txt."), Completion(text="I did not read any file.")]
    )

    turn = await run_turn(llm, dispatcher, "read something", critic=critic)

    assert turn.answer == "I did not read any file."
    assert critic.calls == 2


async def test_the_critic_is_off_unless_the_environment_asks_for_it(monkeypatch):
    llm = ScriptedLLM([])

    monkeypatch.delenv("ARTHUR_LLM_CRITIC", raising=False)
    assert critic_from_environment(llm) is None

    monkeypatch.setenv("ARTHUR_LLM_CRITIC", "0")
    assert critic_from_environment(llm) is None

    monkeypatch.setenv("ARTHUR_LLM_CRITIC", "true")
    assert isinstance(critic_from_environment(llm), LLMCritic)


async def test_a_rules_critique_says_where_it_came_from():
    assert critique("All good.", [ok()]).source == RULES
