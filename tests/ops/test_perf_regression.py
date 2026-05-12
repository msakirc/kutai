"""Z8 T5F — perf_baselines + synthetic_check regression diff."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "perf.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


# ── Pure helpers ─────────────────────────────────────────────────────────────


def test_regression_pct_basic():
    from src.ops.perf_baselines import regression_pct

    assert regression_pct(100.0, 110.0) == pytest.approx(10.0)
    assert regression_pct(100.0, 90.0) == pytest.approx(-10.0)


def test_regression_pct_handles_missing():
    from src.ops.perf_baselines import regression_pct

    assert regression_pct(None, 100.0) is None
    assert regression_pct(100.0, None) is None
    assert regression_pct(0.0, 100.0) is None


def test_has_regression_true():
    from src.ops.perf_baselines import Baseline, has_regression

    b = Baseline(mission_id=1, release_tag="v1", metric="api",
                 p50=100, p95=200, p99=300)
    cur = {"p50": 110, "p95": 250, "p99": 320}  # p95 jumped 25%
    assert has_regression(b, cur, threshold_pct=10.0) is True


def test_has_regression_false():
    from src.ops.perf_baselines import Baseline, has_regression

    b = Baseline(mission_id=1, release_tag="v1", metric="api",
                 p50=100, p95=200, p99=300)
    cur = {"p50": 105, "p95": 208, "p99": 305}  # all <5%
    assert has_regression(b, cur, threshold_pct=10.0) is False


def test_has_regression_no_baseline():
    from src.ops.perf_baselines import has_regression

    assert has_regression(None, {"p50": 100, "p95": 200, "p99": 300}) is False


# ── DB roundtrip ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_perf_baselines_table_exists(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='perf_baselines'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_record_and_latest_roundtrip(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.perf_baselines import latest_green_baseline, record_baseline

    await record_baseline(1, "v1", "api", p50=100, p95=200, p99=300)
    await record_baseline(1, "v2", "api", p50=110, p95=210, p99=310)
    await record_baseline(1, "v1", "ui", p50=50, p95=60, p99=70)

    latest = await latest_green_baseline(1, "api")
    assert latest is not None
    assert latest.release_tag == "v2"
    assert latest.p50 == 110

    none = await latest_green_baseline(1, "no_such_metric")
    assert none is None


# ── Executor: backend skip when binary missing ───────────────────────────────


@pytest.mark.asyncio
async def test_executor_skips_when_lighthouse_missing(monkeypatch):
    from packages.mr_roboto.src.mr_roboto.executors import synthetic_check
    monkeypatch.setattr(synthetic_check.shutil, "which", lambda _: None)

    res = await synthetic_check.run({
        "payload": {
            "action": "synthetic_check",
            "backend": "lighthouse",
            "target_url": "http://example.com",
        },
    })
    assert res["skipped"] is True
    assert res["backend"] == "lighthouse"


@pytest.mark.asyncio
async def test_executor_missing_url_fails():
    from packages.mr_roboto.src.mr_roboto.executors import synthetic_check

    res = await synthetic_check.run({
        "payload": {"backend": "lighthouse"},
    })
    assert res["ok"] is False
    assert "target_url" in res["reason"]


@pytest.mark.asyncio
async def test_executor_detects_regression(tmp_path, monkeypatch):
    """Seed a baseline, monkey-patch the lighthouse subprocess to return a slow sample,
    confirm regression_detected=True."""
    await _setup(tmp_path, monkeypatch)
    from src.ops.perf_baselines import record_baseline
    from packages.mr_roboto.src.mr_roboto.executors import synthetic_check

    await record_baseline(7, "v1", "lighthouse", p50=1000, p95=1000, p99=1000)

    # Pretend lighthouse exists and emits 50% slower than baseline.
    monkeypatch.setattr(synthetic_check.shutil, "which", lambda _: "/usr/bin/lighthouse")

    def fake_run_lh(target_url, payload):
        return {
            "ok": True, "backend": "lighthouse",
            "p50": 1500.0, "p95": 1500.0, "p99": 1500.0,
            "skipped": False,
        }

    monkeypatch.setattr(synthetic_check, "_run_lighthouse", fake_run_lh)

    res = await synthetic_check.run({
        "mission_id": 7,
        "payload": {
            "backend": "lighthouse",
            "target_url": "http://example.com",
            "release_tag": "v2",
            "regression_threshold_pct": 10.0,
        },
    })
    assert res["regression_detected"] is True
    assert res["ok"] is False
    assert res["delta_pct"]["p95"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_executor_records_baseline_on_pass(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from src.ops.perf_baselines import latest_green_baseline
    from packages.mr_roboto.src.mr_roboto.executors import synthetic_check

    monkeypatch.setattr(synthetic_check.shutil, "which", lambda _: "/usr/bin/lighthouse")

    def fake_run_lh(target_url, payload):
        return {"ok": True, "backend": "lighthouse",
                "p50": 800.0, "p95": 800.0, "p99": 800.0, "skipped": False}

    monkeypatch.setattr(synthetic_check, "_run_lighthouse", fake_run_lh)

    res = await synthetic_check.run({
        "mission_id": 11,
        "payload": {
            "backend": "lighthouse",
            "target_url": "http://example.com",
            "release_tag": "v9",
        },
    })
    assert res["regression_detected"] is False
    assert res["ok"] is True

    latest = await latest_green_baseline(11, "lighthouse")
    assert latest is not None
    assert latest.release_tag == "v9"
    assert latest.p95 == 800.0
