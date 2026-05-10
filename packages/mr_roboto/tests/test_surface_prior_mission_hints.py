"""Tests for surface_prior_mission_hints — Z1 T6A (P9) cross-mission hints."""
from __future__ import annotations

import json
import os

import pytest

from mr_roboto.surface_prior_mission_hints import (
    _extract_domain_keywords,
    _jaccard,
    index_mission_artifacts,
    surface_prior_mission_hints,
)


@pytest.fixture(autouse=True)
async def _db_reset():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


async def _setup_db(tmp_path, monkeypatch):
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()
    return await get_db()


def _write_charter(workspace: str, kws: list[str]) -> None:
    charter_dir = os.path.join(workspace, ".charter")
    os.makedirs(charter_dir, exist_ok=True)
    lines = ["# Charter", "", "## Brand Keywords", ""]
    for kw in kws:
        lines.append(f"- **{kw}** — descriptive copy.")
    lines.extend(["", "## Core Problem", "Some problem.", ""])
    with open(
        os.path.join(charter_dir, "product_charter.md"), "w", encoding="utf-8",
    ) as fh:
        fh.write("\n".join(lines))


def _write_compliance_fingerprint(
    workspace: str, jurisdictions: list[str], data_categories: list[str],
) -> None:
    payload = {
        "_schema_version": "1",
        "jurisdictions": jurisdictions,
        "data_categories_coarse": data_categories,
    }
    with open(
        os.path.join(workspace, "compliance_fingerprint.json"),
        "w", encoding="utf-8",
    ) as fh:
        json.dump(payload, fh)


# ─── unit tests for helpers ──────────────────────────────────────────────

def test_extract_domain_keywords_brand_bullets():
    text = (
        "## Brand Keywords\n"
        "- **Speed** — fast.\n"
        "- **Trust** — verified.\n"
        "- **Open** — extensible.\n"
    )
    kws = _extract_domain_keywords(text)
    assert kws == ["speed", "trust", "open"]


def test_extract_domain_keywords_empty_returns_empty():
    assert _extract_domain_keywords("") == []


def test_extract_domain_keywords_falls_back_to_headings():
    text = "## Healthcare Logistics\nbody\n## Pharmacy Routing\nbody\n"
    kws = _extract_domain_keywords(text)
    assert "healthcare" in kws
    assert "pharmacy" in kws


def test_jaccard_basic():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1.0 / 3.0)


# ─── integration tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_priors_returns_empty_hints(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance", "scheduling"])

    res = await surface_prior_mission_hints(
        mission_id=42,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["ok"] is True
    assert res["hints"] == []
    assert "healthcare" in res["current_keywords"]


@pytest.mark.asyncio
async def test_no_keywords_returns_unchecked(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    ws = tmp_path / "workspace"
    ws.mkdir()
    res = await surface_prior_mission_hints(
        mission_id=1,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["ok"] is True
    assert res["hints"] == []
    assert res["checked"] is False


@pytest.mark.asyncio
async def test_prior_mission_with_overlap_surfaces_hint(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    # Insert a prior mission's artifact row.
    await db.execute(
        """
        INSERT INTO mission_artifacts_index
            (mission_id, artifact_name, artifact_path, schema_version,
             domain_keywords_json, founder_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            12, "compliance_fingerprint",
            "/missions/12/compliance_fingerprint.json", "1",
            json.dumps(["healthcare", "compliance", "telemedicine"]),
            "default",
        ),
    )
    await db.commit()

    ws = tmp_path / "workspace_42"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance", "scheduling"])

    res = await surface_prior_mission_hints(
        mission_id=42,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["ok"] is True
    assert len(res["hints"]) == 1
    h = res["hints"][0]
    assert h["prior_mission_id"] == 12
    assert h["artifact_name"] == "compliance_fingerprint"
    assert "healthcare" in h["overlap_keywords"]
    assert h["jaccard"] >= 0.3
    assert os.path.isfile(res["report_path"])


@pytest.mark.asyncio
async def test_founder_mismatch_filtered(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await db.execute(
        """
        INSERT INTO mission_artifacts_index
            (mission_id, artifact_name, artifact_path, schema_version,
             domain_keywords_json, founder_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            12, "compliance_fingerprint", "/p.json", "1",
            json.dumps(["healthcare", "compliance"]),
            "alice",  # different founder
        ),
    )
    await db.commit()

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance"])
    res = await surface_prior_mission_hints(
        mission_id=42,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["hints"] == []


@pytest.mark.asyncio
async def test_domain_mismatch_below_threshold_filtered(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await db.execute(
        """
        INSERT INTO mission_artifacts_index
            (mission_id, artifact_name, artifact_path, schema_version,
             domain_keywords_json, founder_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            12, "compliance_fingerprint", "/p.json", "1",
            json.dumps(["aerospace", "rocketry", "telemetry"]),
            "default",
        ),
    )
    await db.commit()

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance", "scheduling"])
    res = await surface_prior_mission_hints(
        mission_id=42,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["hints"] == []


@pytest.mark.asyncio
async def test_self_mission_excluded(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    # Same mission_id as the query — must be excluded.
    await db.execute(
        """
        INSERT INTO mission_artifacts_index
            (mission_id, artifact_name, artifact_path, schema_version,
             domain_keywords_json, founder_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            42, "compliance_fingerprint", "/p.json", "1",
            json.dumps(["healthcare", "compliance"]),
            "default",
        ),
    )
    await db.commit()

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance"])
    res = await surface_prior_mission_hints(
        mission_id=42,
        workspace_path=str(ws),
        founder_id="default",
    )
    assert res["hints"] == []


@pytest.mark.asyncio
async def test_compliance_jurisdictions_widen_keywords(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    # Prior with overlap only in compliance jurisdictions.
    await db.execute(
        """
        INSERT INTO mission_artifacts_index
            (mission_id, artifact_name, artifact_path, schema_version,
             domain_keywords_json, founder_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            12, "compliance_fingerprint", "/p.json", "1",
            json.dumps(["us-ca", "hipaa"]),
            "default",
        ),
    )
    await db.commit()

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_charter(str(ws), ["scheduling"])  # no domain overlap
    _write_compliance_fingerprint(str(ws), ["us-ca", "hipaa"], [])

    res = await surface_prior_mission_hints(
        mission_id=42, workspace_path=str(ws), founder_id="default",
    )
    assert len(res["hints"]) == 1


# ─── index_mission_artifacts tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_index_mission_artifacts_walks_workspace(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    ws = tmp_path / "mission_42"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare", "compliance"])
    _write_compliance_fingerprint(str(ws), ["us-ca"], ["pii"])

    # An ADR with a schema_version envelope.
    adr_dir = ws / ".adrs"
    adr_dir.mkdir()
    (adr_dir / "register.md").write_text("# ADR Register\n", encoding="utf-8")
    (adr_dir / "adr-001.json").write_text(
        json.dumps({"_schema_version": "2", "id": "adr-001"}),
        encoding="utf-8",
    )

    res = await index_mission_artifacts(
        mission_id=42, workspace_path=str(ws), founder_id="default",
    )
    assert res["ok"] is True
    assert res["indexed"] >= 3
    # Verify rows exist.
    cur = await db.execute(
        "SELECT artifact_name, schema_version FROM mission_artifacts_index "
        "WHERE mission_id = ? ORDER BY artifact_name",
        (42,),
    )
    rows = await cur.fetchall()
    await cur.close()
    names = {r[0] for r in rows}
    assert "compliance_fingerprint" in names
    assert "charter" in names
    assert "adr_register" in names
    assert "adr_adr-001" in names
    # Schema version preserved for the JSON artifact.
    schema_for_adr = next(r[1] for r in rows if r[0] == "adr_adr-001")
    assert schema_for_adr == "2"


@pytest.mark.asyncio
async def test_index_mission_artifacts_idempotent(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    ws = tmp_path / "mission_42"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare"])

    await index_mission_artifacts(
        mission_id=42, workspace_path=str(ws), founder_id="default",
    )
    await index_mission_artifacts(
        mission_id=42, workspace_path=str(ws), founder_id="default",
    )

    cur = await db.execute(
        "SELECT COUNT(*) FROM mission_artifacts_index "
        "WHERE mission_id = ? AND artifact_name = ?",
        (42, "charter"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1  # ON CONFLICT … DO UPDATE = no duplicates


@pytest.mark.asyncio
async def test_index_mission_artifacts_no_workspace(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    res = await index_mission_artifacts(
        mission_id=99,
        workspace_path=str(tmp_path / "nonexistent"),
        founder_id="default",
    )
    assert res["ok"] is True
    assert res["indexed"] == 0


# ─── dispatch tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_surface_prior_mission_hints(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto import run

    ws = tmp_path / "ws"
    ws.mkdir()
    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "surface_prior_mission_hints",
            "workspace_path": str(ws),
            "founder_id": "default",
        },
    }
    result = await run(task)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_dispatch_index_mission_artifacts(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto import run
    ws = tmp_path / "mission"
    ws.mkdir()
    _write_charter(str(ws), ["healthcare"])

    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "index_mission_artifacts",
            "workspace_path": str(ws),
            "founder_id": "default",
        },
    }
    result = await run(task)
    assert result.status == "completed"
