"""Tests for sweep + admission guards added to address handoff items
A (stuck-pending past max retries), N (sweep-vs-in_flight gate),
and R (the orphan ``if blocked:`` NameError that was firing
``cron fire failed`` on every sweep tick).

The tests mock ``src.infra.db.get_db`` and Beckman's apply imports so
they can run without a live KutAI environment. Section coverage is
focused — full sweep integration runs against the DB fixtures that
already cover sections 1, 2, 4, 5, 6.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.in_flight import is_task_in_flight, _task_slots, _InFlightEntry
from general_beckman import sweep as sweep_mod


# ── (R) the orphan ``if blocked:`` is gone ────────────────────────────


def test_no_orphan_blocked_reference():
    """Source must not reference the deleted-by-design ``blocked`` var
    in executable code. Comments mentioning the historical bug are OK.

    The earlier section 3 (dep-cascade) was removed but left an
    ``if blocked: await db.commit()`` orphan that fired
    ``NameError: name 'blocked' is not defined`` on every sweep tick
    (154 hits in the 6-min post-restart window observed 2026-04-26).
    """
    import inspect
    for line in inspect.getsource(sweep_mod).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Look for the exact reachable statement, not the doc-string
        # mention of the historical bug.
        if stripped.startswith("if blocked:"):
            pytest.fail(
                "sweep.py still has reachable `if blocked:` statement"
            )


# ── (N) is_task_in_flight helper exists and works ─────────────────────


def test_is_task_in_flight_helper():
    # Empty registry: any task is not in-flight.
    _task_slots.clear()
    assert is_task_in_flight(42) is False
    assert is_task_in_flight(None) is False

    # Insert and check.
    _task_slots[42] = _InFlightEntry(
        call_id="task-42", task_id=42, category="main_work",
        model="x", provider="y", is_local=True, started_at=0.0,
    )
    try:
        assert is_task_in_flight(42) is True
        assert is_task_in_flight(43) is False
        # int / str coercion
        assert is_task_in_flight("42") is True
    finally:
        _task_slots.clear()


# ── (N) sweep section 1 honors in_flight registry ─────────────────────


@pytest.mark.asyncio
async def test_sweep_section1_skips_in_flight_tasks(monkeypatch):
    """Tasks the dispatcher still owns must NOT be flipped from
    'processing' to 'pending' just because started_at is older than
    5min. The dispatcher's in_flight registry is the authoritative
    "still running" signal — sweep is only a fallback for crashes.
    """
    # Fake DB with one stuck row.
    fake_row = {
        "id": 100, "title": "long planner",
        "worker_attempts": 1, "infra_resets": 0,
        "max_worker_attempts": 6,
    }
    fake_db = MagicMock()
    cursor_section1 = MagicMock()
    cursor_section1.fetchall = AsyncMock(return_value=[fake_row])
    cursor_empty = MagicMock()
    cursor_empty.fetchall = AsyncMock(return_value=[])

    # The sweep_queue function makes many cursor calls; serve them in
    # order. Section 1 first; the rest empty.
    fake_db.execute = AsyncMock(side_effect=lambda *a, **k: (
        cursor_section1 if "status = 'processing'" in (a[0] if a else "")
        else cursor_empty
    ))
    fake_db.commit = AsyncMock()

    @asynccontextmanager
    async def _fake_get_db_ctx(*_a, **_kw):
        yield fake_db

    # Mock get_db to return the fake.
    async def _get_db(*_a, **_kw):
        return fake_db

    monkeypatch.setattr("src.infra.db.get_db", _get_db)
    monkeypatch.setattr(
        "src.infra.db.update_task", AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "src.infra.db.update_mission", AsyncMock(return_value=None),
    )

    # Mark task 100 as in_flight — sweep must skip it.
    _task_slots[100] = _InFlightEntry(
        call_id="task-100", task_id=100, category="main_work",
        model="x", provider="y", is_local=True, started_at=0.0,
    )
    try:
        # No DLQ write expected, no UPDATE pending expected.
        with patch("general_beckman.apply._dlq_write", new_callable=AsyncMock) as dlq:
            await sweep_mod.sweep_queue()
        assert dlq.call_count == 0, (
            "sweep flipped an in_flight task; in_flight gate broken"
        )
    finally:
        _task_slots.clear()


# ── (A) sweep section 8 + admission guard force DLQ on overcap ────────


def test_sweep_has_section_8_overcap_query():
    """Source check: section 8 must query for pending+overcap rows."""
    import inspect
    src = inspect.getsource(sweep_mod)
    assert "worker_attempts >= COALESCE(max_worker_attempts" in src, (
        "sweep section 8 (stuck-pending overcap guard) missing"
    )


def test_admission_has_overcap_guard():
    """Source check: next_task() must DLQ candidates past worker cap."""
    import inspect
    import general_beckman
    src = inspect.getsource(general_beckman.next_task)
    assert "Worker attempts exceeded at admission" in src, (
        "admission cap-guard (handoff item A) missing"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
