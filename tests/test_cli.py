from __future__ import annotations

import pytest

from arthur.cli import heard_stop, read_answer, route
from arthur.tools.builtins import build_registry


@pytest.fixture
def registry():
    return build_registry()


@pytest.mark.parametrize("line", ["quit", "exit", "reset", "tools", "verify"])
def test_commands_are_recognised(registry, line):
    assert route(line, registry) == ("command", line)


def test_a_blank_line_does_nothing(registry):
    assert route("   ", registry) == ("none", "")


def test_a_tool_name_dispatches_directly(registry):
    assert route("list_tasks", registry) == ("tool", "list_tasks")
    assert route('add_task title="x"', registry) == ("tool", 'add_task title="x"')


def test_anything_else_is_chat(registry):
    assert route("add a meeting for tomorrow at 1pm", registry) == (
        "chat",
        "add a meeting for tomorrow at 1pm",
    )


def test_a_bare_reply_is_chat_not_an_unknown_tool(registry):
    assert route("yes", registry) == ("chat", "yes")
    assert route("y", registry) == ("chat", "y")
    assert route("no, cancel that", registry) == ("chat", "no, cancel that")


def test_say_still_forces_chat(registry):
    assert route("say list_tasks", registry) == ("chat", "list_tasks")


def test_a_command_word_inside_a_sentence_is_chat(registry):
    assert route("reset the meeting to 4pm", registry) == (
        "chat",
        "reset the meeting to 4pm",
    )


def test_a_plain_yes_approves():
    assert read_answer(lambda _: "y", "? ") == (True, None)
    assert read_answer(lambda _: "  YES  ", "? ") == (True, None)


def test_a_plain_no_refuses():
    assert read_answer(lambda _: "n", "? ") == (False, None)
    assert read_answer(lambda _: "", "? ") == (False, None)


def test_anything_else_refuses_and_is_handed_back():
    approved, unread = read_answer(lambda _: "delete_task task_id=t_1", "? ")
    assert approved is False
    assert unread == "delete_task task_id=t_1"


def test_a_pasted_command_starting_with_y_does_not_approve():
    approved, unread = read_answer(lambda _: "yes_man title=x", "? ")
    assert approved is False
    assert unread == "yes_man title=x"


@pytest.mark.parametrize("said", ["stop", "Stop.", "  QUIT  ", "goodbye!", "Good bye."])
def test_a_spoken_stop_ends_the_conversation(said):
    assert heard_stop(said) is True


@pytest.mark.parametrize(
    "said",
    ["stop the timer", "don't quit", "say goodbye to the client", "what time is it"],
)
def test_a_stop_word_inside_a_sentence_keeps_going(said):
    assert heard_stop(said) is False
