from __future__ import annotations

import pytest

from arthur.dispatch import Dispatcher
from arthur.tools.builtins import build_registry
from arthur.tools.registry import Risk
from arthur.tools.system import (
    MEDIA_KEYS,
    SystemError_,
    register,
    resolve_app,
    system_stats,
)


@pytest.fixture
def system_dispatcher(audit, tmp_path):
    registry = build_registry(
        include_tasks=False, include_files=False, include_convert=False
    )
    register(registry, screenshot_dir=tmp_path / "shots")
    return Dispatcher(registry, audit=audit)


def test_known_apps_resolve_to_their_executables():
    assert resolve_app("notepad") == "notepad.exe"
    assert resolve_app("File Explorer") == "explorer.exe"
    assert resolve_app("settings") == "ms-settings:"


def test_an_unknown_app_gets_an_exe_suffix():
    assert resolve_app("obsidian") == "obsidian.exe"


def test_an_app_that_already_names_its_extension_is_left_alone():
    assert resolve_app("mything.bat") == "mything.bat"


def test_an_empty_app_name_is_refused():
    with pytest.raises(SystemError_, match="Name an application"):
        resolve_app("   ")


def test_every_media_action_has_a_key():
    for action in ("play", "pause", "next", "previous", "stop", "mute"):
        assert action in MEDIA_KEYS


def test_stats_describe_the_machine():
    stats = system_stats()
    assert stats["system"]
    assert stats["disk_free_gb"] > 0


def test_reading_the_machine_needs_no_approval(system_dispatcher):
    for name in ("system_stats", "read_clipboard"):
        assert system_dispatcher.registry.get(name).risk is Risk.READ_ONLY


def test_acting_on_the_machine_needs_approval(system_dispatcher):
    for name in ("open_app", "open_url", "focus_window", "media_control", "write_clipboard"):
        spec = system_dispatcher.registry.get(name)
        assert spec.risk is Risk.WRITES
        assert system_dispatcher.requires_confirmation(spec)


async def test_stats_can_be_read_through_the_dispatcher(system_dispatcher):
    result = await system_dispatcher.invoke("system_stats", {})
    assert result.ok
    assert "system" in result.value


async def test_opening_an_app_is_gated(system_dispatcher):
    result = await system_dispatcher.invoke("open_app", {"name": "notepad"})
    assert result.needs_confirmation


async def test_a_non_web_url_is_refused(system_dispatcher):
    result = await system_dispatcher.invoke(
        "open_url", {"url": "file:///C:/Windows/System32"}, confirmed=True
    )
    assert result.ok is False
    assert "http" in result.error


async def test_an_unknown_media_action_lists_the_real_ones(system_dispatcher):
    result = await system_dispatcher.invoke(
        "media_control", {"action": "rewind"}, confirmed=True
    )
    assert result.ok is False
    assert "volume_up" in result.error
