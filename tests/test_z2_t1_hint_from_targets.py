"""Z2 T1C — Hint-from-targets expander pass.

Verifies:
- Produce file exists in tmp workspace → write_file stripped from tools_hint.
- Produce file doesn't exist → write_file preserved.
- force_write=True → write_file preserved even when file exists.
- tools_hint absent → no-op (no error).
- any_of produce: if any alternative exists → write_file stripped.
"""
from __future__ import annotations

import os
import pytest

from src.workflows.engine.expander import _apply_hint_from_targets


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

def test_write_file_stripped_when_produce_exists(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("# existing")

    context = {
        "produces": ["src/app.py"],
        "tools_hint": ["write_file", "edit_file", "patch_file"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "write_file" not in context["tools_hint"]
    assert "edit_file" in context["tools_hint"]
    assert "patch_file" in context["tools_hint"]


def test_write_file_preserved_when_produce_missing(tmp_path):
    context = {
        "produces": ["src/new_file.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "write_file" in context["tools_hint"]


def test_force_write_skips_strip(tmp_path):
    (tmp_path / "existing.py").write_text("exists")

    context = {
        "produces": ["existing.py"],
        "tools_hint": ["write_file", "edit_file"],
        "force_write": True,
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "write_file" in context["tools_hint"]


def test_no_tools_hint_is_noop(tmp_path):
    (tmp_path / "x.py").write_text("x")
    context = {
        "produces": ["x.py"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "tools_hint" not in context


def test_tools_hint_without_write_file_unmodified(tmp_path):
    (tmp_path / "x.py").write_text("x")
    original = ["edit_file", "patch_file", "apply_diff"]
    context = {
        "produces": ["x.py"],
        "tools_hint": list(original),
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert context["tools_hint"] == original


def test_no_workspace_path_is_noop():
    """When workspace_path is None (default production behavior), no strip occurs."""
    context = {
        "produces": ["some/file.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    _apply_hint_from_targets(context, workspace_path=None)
    assert "write_file" in context["tools_hint"]


# ---------------------------------------------------------------------------
# any_of alternatives
# ---------------------------------------------------------------------------

def test_any_of_one_exists_strips_write_file(tmp_path):
    (tmp_path / "option_b.py").write_text("b")

    context = {
        "produces": [["option_a.py", "option_b.py"]],
        "tools_hint": ["write_file", "edit_file"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "write_file" not in context["tools_hint"]


def test_any_of_none_exists_preserves_write_file(tmp_path):
    context = {
        "produces": [["option_a.py", "option_b.py"]],
        "tools_hint": ["write_file", "edit_file"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert "write_file" in context["tools_hint"]


# ---------------------------------------------------------------------------
# Idempotent
# ---------------------------------------------------------------------------

def test_idempotent_double_call_produces_exists(tmp_path):
    (tmp_path / "mod.py").write_text("mod")
    context = {
        "produces": ["mod.py"],
        "tools_hint": ["write_file", "edit_file"],
    }
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    after_first = list(context["tools_hint"])
    _apply_hint_from_targets(context, workspace_path=str(tmp_path))
    assert context["tools_hint"] == after_first
