"""Unit tests for intersect.telemetry — yalayut_usage writes."""
import json

import pytest

from intersect import telemetry


@pytest.mark.asyncio
async def test_records_exposed_row(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 7, "exposure_class": "inject",
                  "bind_args": None}],
        conflict_losers=[],
    )
    cur = await intersect_db.execute(
        "SELECT artifact_id, task_id, exposure_class, exposed, "
        "conflict_loser FROM yalayut_usage")
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 7
    assert rows[0][1] == "4101"
    assert rows[0][2] == "inject"
    assert rows[0][3] == 1
    assert rows[0][4] == 0


@pytest.mark.asyncio
async def test_records_conflict_loser(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 1, "exposure_class": "inject",
                  "bind_args": None}],
        conflict_losers=[{"artifact_id": 2, "exposure_class": "inject"}],
    )
    cur = await intersect_db.execute(
        "SELECT artifact_id, conflict_loser, exposed FROM yalayut_usage "
        "ORDER BY artifact_id")
    rows = await cur.fetchall()
    assert rows[0] == (1, 0, 1)
    assert rows[1] == (2, 1, 0)


@pytest.mark.asyncio
async def test_records_bind_args_json(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 9, "exposure_class": "preempt",
                  "bind_args": {"name": "wt"}}],
        conflict_losers=[],
    )
    cur = await intersect_db.execute(
        "SELECT bind_args_json FROM yalayut_usage WHERE artifact_id = 9")
    row = await cur.fetchone()
    assert json.loads(row[0]) == {"name": "wt"}


@pytest.mark.asyncio
async def test_record_usage_never_raises(monkeypatch):
    # DB unavailable → telemetry must swallow, never propagate.
    import src.infra.db as _db

    async def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(_db, "get_db", _boom)
    # Must not raise.
    await telemetry.record_usage(task_id="x", exposed=[], conflict_losers=[])
