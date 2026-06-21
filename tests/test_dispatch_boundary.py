"""Tests for the attempt-stamped checkpoint + bounded conversation reset.

Phase 2 of the rejection-ledger / conversation-reset work
(spec ``docs/superpowers/specs/2026-06-22-conversation-ledger-design.md``).

The dispatch discriminator is ``worker_attempts`` (plan-review M1):

  * The checkpoint records the ``worker_attempts`` it was saved under
    (``saved_attempts``).
  * On ``run()`` entry, if ``saved_attempts < current worker_attempts`` the
    checkpoint is from a COMPLETED prior dispatch (a quality re-dispatch
    bumped the counter) → do NOT restore the bloated ``messages`` array.
  * If ``saved_attempts == current`` the SAME attempt was interrupted
    mid-loop (a crash / heartbeat-timeout resume does NOT bump
    ``worker_attempts``; ``sweep.py`` flips ``processing`` → ``pending``
    without touching the column) → restore (crash-resume, C4).

These are PURE tests: T5 round-trips the serialized checkpoint dict; T6
exercises the pure ``should_restore_messages`` helper. No ``run()``, no DB.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest


# ── T5: checkpoint stamps saved_attempts ───────────────────────────────────


@dataclasses.dataclass
class _FakeReqs:
    """Minimal stand-in for ModelRequirements (only needs asdict())."""
    local_only: bool = False
    estimated_tokens: int = 100


def test_save_checkpoint_stamps_saved_attempts(monkeypatch):
    """save_checkpoint serializes the current worker_attempts as saved_attempts.

    Pure: stub save_task_checkpoint so no DB write happens; capture the
    state dict it would persist and assert the field is present and
    round-trips through json (task_state is a JSON column).
    """
    import general_beckman
    from coulson import checkpoint as ckpt_mod

    captured: dict = {}

    async def _fake_save(task_id, state):
        captured["task_id"] = task_id
        captured["state"] = state

    monkeypatch.setattr(general_beckman, "save_task_checkpoint", _fake_save)

    asyncio.run(
        ckpt_mod.save_checkpoint(
            task_id=42,
            next_iteration=3,
            messages=[{"role": "user", "content": "hi"}],
            total_cost=0.1,
            used_model="gemini",
            reqs=_FakeReqs(),
            tools_used=False,
            validation_retried=False,
            worker_attempts=7,
        )
    )

    state = captured["state"]
    assert state["saved_attempts"] == 7
    # JSON column round-trip — survives serialize/parse intact.
    parsed = json.loads(json.dumps(state))
    assert parsed["saved_attempts"] == 7


def test_save_checkpoint_defaults_saved_attempts_zero(monkeypatch):
    """Omitting worker_attempts defaults saved_attempts to 0 (safe: < any real attempt)."""
    import general_beckman
    from coulson import checkpoint as ckpt_mod

    captured: dict = {}

    async def _fake_save(task_id, state):
        captured["state"] = state

    monkeypatch.setattr(general_beckman, "save_task_checkpoint", _fake_save)

    asyncio.run(
        ckpt_mod.save_checkpoint(
            task_id=42,
            next_iteration=1,
            messages=[],
            total_cost=0.0,
            used_model="m",
            reqs=_FakeReqs(),
            tools_used=False,
            validation_retried=False,
        )
    )
    assert captured["state"]["saved_attempts"] == 0


# ── T6: should_restore_messages discriminator ──────────────────────────────


def test_restore_when_same_attempt_interrupted():
    """saved_attempts == current → SAME attempt interrupted mid-loop → restore (C4)."""
    from coulson.react import should_restore_messages
    ckpt = {"saved_attempts": 4, "messages": [{"role": "user", "content": "x"}]}
    assert should_restore_messages(ckpt, current_attempts=4) is True


def test_no_restore_when_prior_dispatch_completed():
    """saved_attempts < current → checkpoint from a COMPLETED prior dispatch → fresh."""
    from coulson.react import should_restore_messages
    ckpt = {"saved_attempts": 3, "messages": [{"role": "user", "content": "x"}]}
    # A quality re-dispatch bumped worker_attempts 3 -> 4; do NOT restore the
    # accumulated messages array (this is the 775k-bloat kill path).
    assert should_restore_messages(ckpt, current_attempts=4) is False


def test_no_restore_without_checkpoint():
    """No checkpoint → nothing to restore."""
    from coulson.react import should_restore_messages
    assert should_restore_messages(None, current_attempts=0) is False
    assert should_restore_messages({}, current_attempts=0) is False


def test_missing_saved_attempts_defaults_zero():
    """A legacy checkpoint with no saved_attempts → treated as attempt 0.

    Any real re-dispatch has worker_attempts >= 1 > 0 → False (fresh). A
    first-ever interrupted dispatch (current_attempts == 0) → 0 >= 0 → True.
    """
    from coulson.react import should_restore_messages
    legacy = {"messages": [{"role": "user", "content": "x"}]}
    assert should_restore_messages(legacy, current_attempts=4) is False
    assert should_restore_messages(legacy, current_attempts=0) is True


@pytest.mark.parametrize(
    "saved, current",
    [
        # F1 bloat re-entry shapes: degenerate / empty-completed /
        # post-availability re-dispatch. Each is a NEW attempt, so the live
        # worker_attempts is strictly greater than the checkpoint's stamp.
        (1, 2),   # degenerate rejection re-dispatch
        (5, 6),   # empty-result re-dispatch
        (8, 9),   # dispatch following an availability bounce
        (0, 1),   # first quality re-dispatch
    ],
)
def test_f1_trigger_shapes_never_restore(saved, current):
    """All F1 bloat-accumulation shapes (incremented worker_attempts) → fresh."""
    from coulson.react import should_restore_messages
    ckpt = {"saved_attempts": saved, "messages": [{"role": "user", "content": "x"}]}
    assert should_restore_messages(ckpt, current_attempts=current) is False
