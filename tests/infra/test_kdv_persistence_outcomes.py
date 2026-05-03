"""Bridge-wiring round-trip for KDV outcomes persistence (Step 5c).

Confirms the save→sqlite→load_sync path emits and restores per-model
outcome deques under scope='outcomes'. Other scopes (model/provider/
breaker/meta) round-trip via the same code path; this test focuses on
outcomes specifically because that's what 5c added.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from kuleden_donen_var import KuledenConfig, KuledenDonenVar
from src.infra import kdv_persistence


def _create_kdv_state_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kdv_state (
                scope TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                last_persisted REAL NOT NULL,
                PRIMARY KEY (scope, scope_key)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "kdv_test.db")
    _create_kdv_state_table(p)
    return p


def test_outcomes_round_trip_via_save_load_sync(db_path):
    """save() writes scope='outcomes' rows; load_sync() restores them
    onto a fresh KDV so recent_success_rate survives restart."""
    src = KuledenDonenVar(KuledenConfig())
    src.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    for _ in range(7):
        src.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(3):
        src.record_failure("groq/llama-8b", "groq", "server_error")
    expected_rate = src.recent_success_rate("groq/llama-8b")
    expected_n = src.recent_samples_n("groq/llama-8b")
    assert expected_n == 10  # sanity

    asyncio.run(kdv_persistence.save(src, db_path))

    # Confirm an outcomes row landed.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT scope_key, snapshot_json FROM kdv_state WHERE scope='outcomes'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "groq/llama-8b"

    dst = KuledenDonenVar(KuledenConfig())
    dst.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    report = kdv_persistence.load_sync(dst, db_path)

    assert report["outcomes"] == 1
    assert dst.recent_samples_n("groq/llama-8b") == expected_n
    assert dst.recent_success_rate("groq/llama-8b") == pytest.approx(expected_rate)


def test_load_sync_drops_stale_outcomes_rows(db_path):
    """The stale-row filter in load_sync (last_persisted < cutoff) must
    apply to the outcomes scope just like every other scope."""
    src = KuledenDonenVar(KuledenConfig())
    src.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    for _ in range(6):
        src.post_call("groq/llama-8b", "groq", headers={}, token_count=10)

    asyncio.run(kdv_persistence.save(src, db_path))

    # Backdate the row so the 24h stale filter drops it.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE kdv_state SET last_persisted = ? WHERE scope='outcomes'",
            (time.time() - 48 * 3600,),
        )
        conn.commit()
    finally:
        conn.close()

    dst = KuledenDonenVar(KuledenConfig())
    dst.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    report = kdv_persistence.load_sync(dst, db_path)

    assert report["outcomes"] == 0
    assert report["skipped_stale"] >= 1
    assert dst.recent_samples_n("groq/llama-8b") == 0
