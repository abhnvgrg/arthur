from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from arthur.tools.registry import (
    DuplicateTool,
    Risk,
    ToolError,
    ToolRegistry,
    ToolSpec,
    UnknownTool,
)


class Args(BaseModel):
    city: str = Field(description="City name")


def handler(args: Args) -> str:
    return args.city


def _spec(name: str = "lookup", risk: Risk = Risk.READ_ONLY) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Look something up",
        parameters=Args,
        handler=handler,
        risk=risk,
    )


def test_a_registered_tool_can_be_retrieved():
    registry = ToolRegistry()
    registry.register(_spec())

    assert "lookup" in registry
    assert registry.get("lookup").name == "lookup"
    assert len(registry) == 1


def test_an_unregistered_tool_raises():
    with pytest.raises(UnknownTool):
        ToolRegistry().get("nothing")


def test_registering_the_same_name_twice_raises():
    registry = ToolRegistry()
    registry.register(_spec())

    with pytest.raises(DuplicateTool):
        registry.register(_spec())


@pytest.mark.parametrize("name", ["", "has space", "has-dash", "has.dot", "has/slash"])
def test_an_invalid_tool_name_raises(name):
    with pytest.raises(ToolError):
        ToolRegistry().register(_spec(name=name))


def test_a_non_positive_timeout_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.register(
            ToolSpec(
                name="x",
                description="d",
                parameters=Args,
                handler=handler,
                timeout_seconds=0,
            )
        )


def test_read_only_tools_do_not_need_confirmation():
    assert _spec(risk=Risk.READ_ONLY).needs_confirmation is False


@pytest.mark.parametrize("risk", [Risk.WRITES, Risk.IRREVERSIBLE])
def test_side_effecting_tools_need_confirmation(risk):
    assert _spec(risk=risk).needs_confirmation is True


def test_the_openai_schema_carries_name_description_and_parameters():
    schema = _spec().as_openai_tool()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "lookup"
    assert schema["function"]["description"] == "Look something up"
    assert "city" in schema["function"]["parameters"]["properties"]


def test_risky_tools_can_be_withheld_from_the_model():
    registry = ToolRegistry()
    registry.register(_spec(name="safe", risk=Risk.READ_ONLY))
    registry.register(_spec(name="risky", risk=Risk.IRREVERSIBLE))

    exposed = [tool["function"]["name"] for tool in registry.openai_tools(include_risky=False)]
    assert exposed == ["safe"]
    assert len(registry.openai_tools()) == 2


def test_iteration_is_ordered_by_name():
    registry = ToolRegistry()
    for name in ("zebra", "alpha", "middle"):
        registry.register(_spec(name=name))

    assert [spec.name for spec in registry] == ["alpha", "middle", "zebra"]
    assert registry.names() == ["alpha", "middle", "zebra"]


def test_the_builtin_registry_exposes_the_expected_tools(registry):
    assert registry.names() == [
        "add_task",
        "calculate",
        "complete_task",
        "convert_units",
        "current_time",
        "delete_file",
        "delete_task",
        "forget",
        "list_files",
        "list_memories",
        "list_tasks",
        "list_units",
        "overdue_tasks",
        "read_file",
        "recall",
        "remember",
        "search_files",
        "write_file",
    ]


def test_skill_packs_can_be_left_out(memory):
    from arthur.tools.builtins import build_registry

    core = build_registry(
        memory=memory,
        include_tasks=False,
        include_files=False,
        include_convert=False,
    )

    assert core.names() == [
        "calculate",
        "current_time",
        "forget",
        "list_memories",
        "recall",
        "remember",
    ]


def test_no_read_only_tool_can_change_anything(registry):
    from arthur.tools.registry import Risk

    mutating = {
        "remember", "forget", "add_task", "complete_task", "delete_task",
        "write_file", "delete_file",
    }
    for spec in registry:
        if spec.risk is Risk.READ_ONLY:
            assert spec.name not in mutating, f"{spec.name} mutates but is read-only"


def test_builtin_risk_levels_are_assigned_deliberately(registry):
    assert registry.get("calculate").risk is Risk.READ_ONLY
    assert registry.get("current_time").risk is Risk.READ_ONLY
    assert registry.get("recall").risk is Risk.READ_ONLY
    assert registry.get("list_memories").risk is Risk.READ_ONLY
    assert registry.get("remember").risk is Risk.WRITES
    assert registry.get("forget").risk is Risk.IRREVERSIBLE


def test_every_builtin_tool_has_a_description(registry):
    for spec in registry:
        assert spec.description.strip()
        assert spec.description.endswith(".")
