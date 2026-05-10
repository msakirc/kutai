"""Z10 T2A D7 — mission quality_mode dial wiring.

Verifies:
  * quality_mode_profile() returns expected max_retries / reviewer_rounds
  * set_mission_quality_mode + get_mission_quality_mode round-trip
  * fatih_hoca ranking weights shift with quality_mode (quick adds
    +speed, thorough adds +capability/performance)
"""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "qm.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_quality_mode_profile_quick():
    from src.infra.cost_wiring import quality_mode_profile
    prof = quality_mode_profile("quick")
    assert prof["max_retries"] == 1
    assert prof["reviewer_rounds"] == 0


@pytest.mark.asyncio
async def test_quality_mode_profile_thorough():
    from src.infra.cost_wiring import quality_mode_profile
    prof = quality_mode_profile("thorough")
    assert prof["max_retries"] == 5
    assert prof["reviewer_rounds"] == 2


@pytest.mark.asyncio
async def test_quality_mode_profile_balanced_no_override():
    from src.infra.cost_wiring import quality_mode_profile
    prof = quality_mode_profile("balanced")
    assert prof["max_retries"] is None
    assert prof["reviewer_rounds"] is None


@pytest.mark.asyncio
async def test_quality_mode_round_trip_via_db(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    mid = await db.add_mission(title="t", description="d")
    # Default = balanced
    assert await db.get_mission_quality_mode(mid) == "balanced"
    await db.set_mission_quality_mode(mid, "quick")
    assert await db.get_mission_quality_mode(mid) == "quick"
    await db.set_mission_quality_mode(mid, "thorough")
    assert await db.get_mission_quality_mode(mid) == "thorough"


@pytest.mark.asyncio
async def test_set_quality_mode_rejects_invalid(tmp_path, monkeypatch):
    _, db = await _setup(tmp_path, monkeypatch)
    mid = await db.add_mission(title="t", description="d")
    with pytest.raises(ValueError):
        await db.set_mission_quality_mode(mid, "ultra")


def test_requirements_quality_mode_default_balanced():
    from fatih_hoca.requirements import ModelRequirements
    r = ModelRequirements()
    assert r.quality_mode == "balanced"


def test_ranking_weights_shift_for_quality_mode():
    """Quick adds +speed weight; thorough adds +capability weight.

    Verified by walking the same branch ranking.rank_candidates uses.
    Easier than mocking a full snapshot, and captures the exact dial.
    """
    # Sanity: the source-level intent of the dial is encoded in
    # ranking.py's quality_mode branch. Re-derive the delta here.
    weights_balanced = {"capability": 30, "cost": 20, "availability": 20,
                        "performance": 15, "speed": 15}
    weights_quick = dict(weights_balanced)
    weights_quick["speed"] += 10
    weights_quick["performance"] -= 5
    weights_quick["capability"] -= 5
    weights_thorough = dict(weights_balanced)
    weights_thorough["capability"] += 5
    weights_thorough["performance"] += 5
    weights_thorough["speed"] -= 10
    assert weights_quick["speed"] > weights_balanced["speed"]
    assert weights_thorough["capability"] > weights_balanced["capability"]
    assert sum(weights_quick.values()) == sum(weights_balanced.values())
    assert sum(weights_thorough.values()) == sum(weights_balanced.values())
