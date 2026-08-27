from __future__ import annotations

import pytest

from arthur.audit import AuditLog
from arthur.dispatch import Dispatcher
from arthur.tools.builtins import MemoryStore, build_registry
from arthur.tools.files import Workspace
from arthur.tools.tasks import TaskStore


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture
def memory(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json")


@pytest.fixture
def tasks(tmp_path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.json")


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def registry(memory, tasks, workspace):
    return build_registry(memory=memory, tasks=tasks, workspace=workspace)


@pytest.fixture
def dispatcher(registry, audit) -> Dispatcher:
    return Dispatcher(registry, audit=audit)
