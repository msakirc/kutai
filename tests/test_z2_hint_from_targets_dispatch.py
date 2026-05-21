"""Z2 fix — hint-from-targets strip moved to dispatch time.

Tests:
  (a) Bug pinned: expand_steps_to_tasks on a fresh mission (no workspace)
      leaves tools_hint unchanged — write_file NOT stripped at expansion.
  (b) Fix: _apply_hint_from_targets_runtime strips write_file when the produce
      file already exists in the mission workspace.
  (c) Negative: same context but file absent → write_file retained.
  (d) force_write override honored at runtime → no strip.
"""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# (a) Bug pinned — expansion does NOT strip write_file (workspace absent)
# ---------------------------------------------------------------------------

def test_expansion_does_not_strip_write_file_on_fresh_mission(tmp_path, monkeypatch):
    """expand_steps_to_tasks must NOT strip write_file at expansion time.

    The workspace dir for a fresh mission does not yet exist, so the strip
    must be deferred to dispatch time.  This test documents that responsibility
    has moved: expansion always preserves the original tools_hint.
    """
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))
    # Do NOT create the mission dir — simulates a fresh mission.

    from src.workflows.engine.expander import expand_steps_to_tasks

    step = {
        "id": "1.1",
        "phase": "phase_1",
        "name": "write_module",
        "agent": "coder",
        "instruction": "Write the module.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
        "produces": ["src/app.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    tasks = expand_steps_to_tasks([step], mission_id=42, initial_context={})
    hint = tasks[0]["context"].get("tools_hint", [])
    # Expansion must NOT strip — workspace didn't exist at expansion time.
    assert "write_file" in hint, (
        "write_file was stripped at expansion time — it must only be stripped "
        "at dispatch time (coulson.execute) when the workspace exists."
    )


# ---------------------------------------------------------------------------
# (b) Fix — runtime helper strips write_file when produce file exists
# ---------------------------------------------------------------------------

def test_runtime_helper_strips_write_file_when_file_exists(tmp_path, monkeypatch):
    """_apply_hint_from_targets_runtime strips write_file if file exists."""
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))

    mission_id = 99
    ws = tmp_path / f"mission_{mission_id}"
    ws.mkdir()
    (ws / "foo.py").write_text("# existing file")

    task_ctx = {
        "mission_id": mission_id,
        "produces": ["foo.py"],
        "tools_hint": ["write_file", "edit_file"],
    }

    from packages.coulson.src.coulson import _apply_hint_from_targets_runtime
    _apply_hint_from_targets_runtime(task_ctx)

    assert "write_file" not in task_ctx["tools_hint"], (
        "write_file should have been stripped because foo.py exists in workspace"
    )
    assert "edit_file" in task_ctx["tools_hint"]


# ---------------------------------------------------------------------------
# (c) Negative — file absent → write_file retained
# ---------------------------------------------------------------------------

def test_runtime_helper_retains_write_file_when_file_absent(tmp_path, monkeypatch):
    """_apply_hint_from_targets_runtime keeps write_file if produce is missing."""
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))

    mission_id = 100
    ws = tmp_path / f"mission_{mission_id}"
    ws.mkdir()
    # foo.py is NOT created

    task_ctx = {
        "mission_id": mission_id,
        "produces": ["foo.py"],
        "tools_hint": ["write_file", "edit_file"],
    }

    from packages.coulson.src.coulson import _apply_hint_from_targets_runtime
    _apply_hint_from_targets_runtime(task_ctx)

    assert "write_file" in task_ctx["tools_hint"], (
        "write_file must be retained when the produce file does not exist"
    )


# ---------------------------------------------------------------------------
# (d) force_write override → no strip even when file exists
# ---------------------------------------------------------------------------

def test_runtime_helper_honors_force_write_override(tmp_path, monkeypatch):
    """force_write=True must prevent stripping even if file exists."""
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))

    mission_id = 101
    ws = tmp_path / f"mission_{mission_id}"
    ws.mkdir()
    (ws / "bar.py").write_text("# existing")

    task_ctx = {
        "mission_id": mission_id,
        "produces": ["bar.py"],
        "tools_hint": ["write_file", "edit_file"],
        "force_write": True,
    }

    from packages.coulson.src.coulson import _apply_hint_from_targets_runtime
    _apply_hint_from_targets_runtime(task_ctx)

    assert "write_file" in task_ctx["tools_hint"], (
        "force_write=True must prevent write_file from being stripped"
    )


# ---------------------------------------------------------------------------
# Edge cases — no mission_id, no workspace dir, no tools_hint
# ---------------------------------------------------------------------------

def test_runtime_helper_no_mission_id_is_noop(tmp_path, monkeypatch):
    """Missing mission_id → no-op, no error."""
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))

    task_ctx = {
        "produces": ["foo.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    from packages.coulson.src.coulson import _apply_hint_from_targets_runtime
    _apply_hint_from_targets_runtime(task_ctx)
    assert "write_file" in task_ctx["tools_hint"]


def test_runtime_helper_missing_workspace_dir_is_noop(tmp_path, monkeypatch):
    """If mission workspace dir doesn't exist → no-op, no error."""
    import src.tools.workspace as _ws_mod
    monkeypatch.setattr(_ws_mod, "WORKSPACE_DIR", str(tmp_path))
    # mission dir is NOT created

    task_ctx = {
        "mission_id": 102,
        "produces": ["foo.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    from packages.coulson.src.coulson import _apply_hint_from_targets_runtime
    _apply_hint_from_targets_runtime(task_ctx)
    assert "write_file" in task_ctx["tools_hint"]
