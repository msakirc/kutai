"""Z8 T4C — incident playbook loader + matcher + generator executor."""
from __future__ import annotations

import json

import pytest


# ─────────────────────── playbook loader ─────────────────────────


def test_six_starter_playbooks_load():
    """All 6 v1 starter playbooks must load without error."""
    from src.ops.playbooks import list_playbooks

    pbs = list_playbooks(recipes_dir="recipes")
    ids = {pb.id for pb in pbs}
    expected = {
        "incident_playbook_db_disk_full_v1",
        "incident_playbook_payment_provider_down_v1",
        "incident_playbook_auth_provider_down_v1",
        "incident_playbook_cert_expiring_v1",
        "incident_playbook_error_spike_v1",
        "incident_playbook_uptime_drop_v1",
    }
    assert expected.issubset(ids), (
        f"missing playbooks: {expected - ids}; got: {sorted(ids)}"
    )


def test_playbook_has_action_sequence():
    """Each starter playbook must define at least one action verb."""
    from src.ops.playbooks import list_playbooks

    for pb in list_playbooks(recipes_dir="recipes"):
        if not pb.id.startswith("incident_playbook_"):
            continue
        assert pb.action_sequence, f"{pb.id} has empty action_sequence"
        for step in pb.action_sequence:
            assert "verb" in step, f"{pb.id} step missing verb: {step}"


# ─────────────────────── match_playbook ──────────────────────────


def test_match_playbook_db_disk_full():
    from src.ops.playbooks import match_playbook

    alert = {"integration": "betterstack", "event": "db_disk_alert"}
    runtime_state = {"db": {"disk_used_pct": 87}}
    pb = match_playbook(alert, runtime_state)
    assert pb is not None
    assert pb.id == "incident_playbook_db_disk_full_v1"


def test_match_playbook_runtime_state_blocks_under_threshold():
    """db.disk_used_pct=50 (under 85) must NOT match db_disk_full."""
    from src.ops.playbooks import match_playbook

    alert = {"integration": "betterstack", "event": "db_disk_alert"}
    runtime_state = {"db": {"disk_used_pct": 50}}
    pb = match_playbook(alert, runtime_state)
    assert pb is None, (
        "runtime_state condition (disk_used_pct > 85) failed to gate match"
    )


def test_match_playbook_error_spike():
    from src.ops.playbooks import match_playbook

    pb = match_playbook(
        {"integration": "sentry", "event": "issue_alert"},
        runtime_state={},
    )
    assert pb is not None
    assert pb.id == "incident_playbook_error_spike_v1"


def test_match_playbook_no_match_returns_none():
    from src.ops.playbooks import match_playbook

    pb = match_playbook(
        {"integration": "unknown", "event": "unknown"}, runtime_state={}
    )
    assert pb is None


# ─────────────────── stack matching (phase 13 path) ──────────────


def test_match_playbooks_for_stack_postgres():
    from src.ops.playbooks import match_playbooks_for_stack

    pbs = match_playbooks_for_stack(["fastapi", "postgres"])
    ids = {pb.id for pb in pbs}
    assert "incident_playbook_db_disk_full_v1" in ids


def test_match_playbooks_for_stack_string():
    """+"-joined string form must also work (matches mission.tech_stack format)."""
    from src.ops.playbooks import match_playbooks_for_stack

    pbs = match_playbooks_for_stack("fastapi+postgres+nextjs")
    ids = {pb.id for pb in pbs}
    assert "incident_playbook_db_disk_full_v1" in ids


def test_match_playbooks_for_stack_empty_when_no_overlap():
    from src.ops.playbooks import match_playbooks_for_stack

    pbs = match_playbooks_for_stack(["only_a_made_up_stack_xyz"])
    assert pbs == []


# ─────────────────── generate_playbooks executor ─────────────────


@pytest.mark.asyncio
async def test_generate_playbooks_returns_artifact():
    """Executor must emit the incident_playbooks artifact shape."""
    from mr_roboto.executors.generate_playbooks import run

    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "generate_playbooks",
            "tech_stack": ["fastapi", "postgres", "sentry"],
        },
    }
    res = await run(task)
    assert res["status"] == "ok"
    assert res["artifact_name"] == "incident_playbooks"
    art = res["artifact"]
    assert art["mission_id"] == 99
    assert "playbooks" in art
    ids = set(res["playbook_ids"])
    # Both postgres (db_disk_full) and sentry (error_spike) playbooks must match.
    assert "incident_playbook_db_disk_full_v1" in ids
    assert "incident_playbook_error_spike_v1" in ids


@pytest.mark.asyncio
async def test_generate_playbooks_empty_stack():
    """No tech_stack and no mission_id → empty artifact, not failure."""
    from mr_roboto.executors.generate_playbooks import run

    res = await run({"id": 1, "mission_id": None, "payload": {"action": "generate_playbooks"}})
    assert res["status"] == "ok"
    assert res["artifact"]["playbooks"] == []


@pytest.mark.asyncio
async def test_generate_playbooks_routes_through_mr_roboto():
    """End-to-end via mr_roboto.run() dispatch."""
    import mr_roboto

    task = {
        "id": 5,
        "mission_id": 1,
        "payload": {
            "action": "generate_playbooks",
            "tech_stack": "fastapi+postgres",
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["artifact_name"] == "incident_playbooks"


# ─────────────────── i2p_v3.json phase 13.3.playbooks step ──────────


def test_i2p_v3_has_playbooks_step():
    """The phase 13 generator step must be present and configured."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    steps = data.get("steps") or []
    matches = [s for s in steps if s.get("id") == "13.3.playbooks"]
    assert len(matches) == 1, "13.3.playbooks step missing or duplicated"
    step = matches[0]
    assert step["agent"] == "mechanical"
    assert step["payload"]["action"] == "generate_playbooks"
    assert "incident_playbooks" in step["output_artifacts"]
