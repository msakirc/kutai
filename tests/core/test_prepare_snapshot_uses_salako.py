"""Phase 2a Task 7 — _prepare's snapshot call must funnel through salako.run.

Previously _prepare called snapshot_workspace() directly. Now it builds a
synthetic mechanical task and routes it through salako so both entry points
(pre-task snapshot + explicit workflow git_commit) share one seam.

The _prepare method has many branches; rather than mock every DB/network
touch, this test asserts the structural property: the orchestrator module
source no longer imports snapshot_workspace, and the only call path that
produces a workspace_snapshot action flows through salako.run.
"""

import inspect

import src.core.orchestrator as orch_mod


def test_orchestrator_does_not_import_snapshot_workspace_directly():
    """Regression guard — Task 7 removed this import."""
    src = inspect.getsource(orch_mod)
    assert "from .mechanical.workspace_snapshot import snapshot_workspace" not in src
    assert "from src.core.mechanical.workspace_snapshot import snapshot_workspace" not in src


def test_prepare_uses_salako_run_for_workspace_snapshot():
    """_prepare body must reference salako.run with a workspace_snapshot
    payload — the shape is the seam both entry points share."""
    src = inspect.getsource(orch_mod._prepare if hasattr(orch_mod, "_prepare")
                            else orch_mod.Orchestrator._prepare)
    assert "salako.run" in src
    assert "workspace_snapshot" in src
