"""Regression guard for Phase 2a Task 7 — orchestrator must not import
snapshot_workspace directly. Pre-task snapshot routing is gone post-Task-13;
workspace_snapshot is only invoked as an explicit mechanical workflow step
(routed through salako.run like any other mechanical executor)."""

import inspect

import src.core.orchestrator as orch_mod


def test_orchestrator_does_not_import_snapshot_workspace_directly():
    src = inspect.getsource(orch_mod)
    assert "from .mechanical.workspace_snapshot import snapshot_workspace" not in src
    assert "from src.core.mechanical.workspace_snapshot import snapshot_workspace" not in src
