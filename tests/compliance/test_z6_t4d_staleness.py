"""Z6 T4D — weekly staleness check tests."""
from __future__ import annotations

import datetime
import json
import os

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "stale.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


def _write_template(
    root, jurisdiction, lang, doc_type, last_reviewed, version="1",
):
    d = os.path.join(root, jurisdiction, lang)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{doc_type}.md.j2"), "w", encoding="utf-8") as fh:
        fh.write(f"# {doc_type} template\n")
    meta = {
        "version": version,
        "last_reviewed": last_reviewed,
        "doc_type": doc_type,
        "jurisdiction": jurisdiction,
        "lang": lang,
    }
    with open(
        os.path.join(d, f"{doc_type}.meta.json"), "w", encoding="utf-8",
    ) as fh:
        json.dump(meta, fh)


@pytest.mark.asyncio
async def test_stale_template_emits_legal_counsel_action(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    # 200 days ago - stale.
    stale_date = (datetime.date.today() - datetime.timedelta(days=200)).isoformat()
    _write_template(str(root), "default", "en", "privacy_policy", stale_date)

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res = await compliance_template_staleness(template_root=str(root))
    assert res["ok"]
    assert res["scanned"] == 1
    assert res["stale"] == 1
    assert len(res["emitted"]) == 1
    assert res["skipped_duplicate"] == 0

    actions = await fa.list_pending()
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == "legal_counsel"
    assert "privacy_policy" in a.title
    assert "default" in a.title


@pytest.mark.asyncio
async def test_fresh_template_does_not_emit(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    fresh_date = datetime.date.today().isoformat()
    _write_template(str(root), "default", "en", "privacy_policy", fresh_date)

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res = await compliance_template_staleness(template_root=str(root))
    assert res["ok"]
    assert res["scanned"] == 1
    assert res["stale"] == 0
    assert res["emitted"] == []

    actions = await fa.list_pending()
    assert actions == []


@pytest.mark.asyncio
async def test_duplicate_emit_is_guarded(tmp_path, monkeypatch):
    """Second run with a pending action for the same template must skip."""
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    stale_date = (datetime.date.today() - datetime.timedelta(days=400)).isoformat()
    _write_template(str(root), "default", "en", "tos", stale_date)

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res1 = await compliance_template_staleness(template_root=str(root))
    assert len(res1["emitted"]) == 1

    res2 = await compliance_template_staleness(template_root=str(root))
    assert res2["emitted"] == []
    assert res2["skipped_duplicate"] == 1

    actions = await fa.list_pending()
    assert len(actions) == 1  # still just the one


@pytest.mark.asyncio
async def test_after_resolve_a_new_action_can_be_emitted(tmp_path, monkeypatch):
    """Once the founder marks the prior action done, a fresh scan re-emits."""
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    stale_date = (datetime.date.today() - datetime.timedelta(days=400)).isoformat()
    _write_template(str(root), "default", "en", "dpa", stale_date)

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res1 = await compliance_template_staleness(template_root=str(root))
    assert len(res1["emitted"]) == 1
    action_id = res1["emitted"][0]
    # Resolve it.
    await fa.resolve(action_id, response_payload={"ack": True})

    res2 = await compliance_template_staleness(template_root=str(root))
    assert len(res2["emitted"]) == 1  # template still stale; new action emitted


@pytest.mark.asyncio
async def test_walks_subdirectories(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    stale = (datetime.date.today() - datetime.timedelta(days=200)).isoformat()
    fresh = datetime.date.today().isoformat()
    _write_template(str(root), "gdpr", "en", "privacy_policy", stale)
    _write_template(str(root), "default", "en", "tos", stale)
    _write_template(str(root), "ccpa", "en", "privacy_policy", fresh)

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res = await compliance_template_staleness(template_root=str(root))
    assert res["scanned"] == 3
    assert res["stale"] == 2
    assert len(res["emitted"]) == 2

    titles = {a.title for a in await fa.list_pending()}
    assert any("gdpr" in t for t in titles)
    assert any("default" in t for t in titles)


@pytest.mark.asyncio
async def test_missing_last_reviewed_counts_as_stale(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    root = tmp_path / "templates"
    d = root / "default" / "en"
    d.mkdir(parents=True)
    (d / "tos.md.j2").write_text("# tos\n", encoding="utf-8")
    # Meta without last_reviewed.
    (d / "tos.meta.json").write_text(
        json.dumps({"version": "1", "doc_type": "tos"}), encoding="utf-8",
    )

    from mr_roboto.executors.compliance_template_staleness import (
        compliance_template_staleness,
    )
    res = await compliance_template_staleness(template_root=str(root))
    assert res["stale"] == 1
    assert len(res["emitted"]) == 1


def test_cron_seed_registers_weekly_cadence():
    from general_beckman.cron_seed import INTERNAL_CADENCES
    matches = [
        c for c in INTERNAL_CADENCES
        if c.get("title") == "compliance_template_staleness"
    ]
    assert len(matches) == 1
    assert matches[0].get("interval_seconds") == 604800  # 7 days
    assert matches[0].get("payload", {}).get("_executor") == (
        "compliance_template_staleness"
    )
