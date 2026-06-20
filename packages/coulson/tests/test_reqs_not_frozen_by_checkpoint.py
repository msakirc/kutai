"""Checkpoint must not freeze selection constraints (2026-06-20 root cause).

A task checkpoint (tasks.task_state) snapshots the full ModelRequirements at
an earlier attempt. Restoring that `reqs` verbatim on resume freezes
selection constraints — local_only, token estimates, ctx — so a value captured
under different conditions becomes permanent and bypasses every
requirements-layer fix.

Live failure: analyst task 459220 checkpointed local_only=True (a stale
classifier verdict). Under the user's intentional Minimal load mode
(cloud-only), local_only=True forbids cloud while Minimal forbids local →
empty candidate set → "No model candidates available" on every retry, forever,
surviving restarts and DLQ-recovery (the checkpoint lives in the DB).

The fix: conversation state (messages/iteration/cost) is restored from the
checkpoint, but the selection requirements are ALWAYS re-derived fresh from the
current task state. `reqs_for_run` encodes that decision.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coulson.react import reqs_for_run
from fatih_hoca.requirements import ModelRequirements


def _fresh():
    return ModelRequirements(
        task="analyst", local_only=False, estimated_input_tokens=8000,
    )


def _poisoned():
    return ModelRequirements(
        task="analyst", local_only=True, estimated_input_tokens=164750,
    )


def test_ignores_checkpoint_frozen_reqs_object():
    """A checkpoint whose saved reqs is a ModelRequirements with poisoned
    local_only / estimate must NOT override the fresh build."""
    chosen = reqs_for_run(_fresh(), {"reqs": _poisoned()})
    assert chosen.local_only is False
    assert chosen.estimated_input_tokens == 8000


def test_ignores_checkpoint_frozen_reqs_dict():
    """Old checkpoints store reqs as a dict (dataclasses.asdict). Still ignored."""
    chosen = reqs_for_run(
        _fresh(),
        {"reqs": {"task": "analyst", "local_only": True,
                  "estimated_input_tokens": 164750}},
    )
    assert chosen.local_only is False
    assert chosen.estimated_input_tokens == 8000


def test_no_checkpoint_returns_fresh():
    """No checkpoint (fresh dispatch) — fresh reqs flow through unchanged."""
    fresh = _fresh()
    assert reqs_for_run(fresh, None) is fresh


def test_checkpoint_without_reqs_returns_fresh():
    """Checkpoint present but no saved reqs key — fresh reqs unchanged."""
    fresh = _fresh()
    assert reqs_for_run(fresh, {"iteration": 2, "messages": []}) is fresh
