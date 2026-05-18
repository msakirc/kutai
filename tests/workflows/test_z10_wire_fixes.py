"""z10-wire-fixes — F1 (push), F2 (mark_green), F3 (scenario_path), F7 (demo gate).

Verifies that audit-identified orphan primitives now have at least one
i2p_v3.json wire site, so production missions actually exercise them.
"""
from __future__ import annotations

import json
import os

import pytest


def _load_v3():
    path = os.path.join("src", "workflows", "i2p", "i2p_v3.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _by_id(steps, sid):
    for s in steps:
        if s.get("id") == sid:
            return s
    return None


# ── F1: payload.push=True on milestone commits ──────────────────────────


def test_f1_push_true_on_milestone_commits():
    """At least one i2p commit step must set ``payload.push=True``.

    Without this, Z10 T1A's atomic commit↔push contract is unreachable in
    production — the verb exists but no workflow step triggers it.
    """
    wf = _load_v3()
    pushed = [
        s for s in wf["steps"]
        if (s.get("payload") or {}).get("push") is True
    ]
    assert pushed, "no i2p step sets payload.push=True (F1 orphan)"
    # Each push step must use the git_commit mechanical action.
    for s in pushed:
        pay = s.get("payload") or {}
        assert pay.get("action") == "git_commit", (
            f"{s['id']}: push=True without action=git_commit — "
            "push is only honored by the git_commit verb"
        )


def test_f1_existing_commit_steps_carry_push():
    """The three pre-existing git_commit steps gained push=True."""
    wf = _load_v3()
    for sid in ("7.3.git_commit", "7.5.git_commit", "8.spike.git_commit"):
        s = _by_id(wf["steps"], sid)
        assert s is not None, f"{sid} missing"
        assert (s.get("payload") or {}).get("push") is True, (
            f"{sid}: F1 requires push=True"
        )


# ── F2: payload.mark_green=True on milestone gate steps (≤6 cap) ────────


def test_f2_mark_green_wired_with_cap():
    """At least one step sets mark_green=True; never more than the cap."""
    wf = _load_v3()
    greens = [
        s for s in wf["steps"]
        if (s.get("payload") or {}).get("mark_green") is True
    ]
    assert greens, "no i2p step sets payload.mark_green=True (F2 orphan)"
    assert len(greens) <= 6, (
        f"F2 cap: mark_green payload count must be <=6, got {len(greens)} "
        f"({[s['id'] for s in greens]})"
    )
    # mark_green only fires from the git_commit verb.
    for s in greens:
        pay = s.get("payload") or {}
        assert pay.get("action") == "git_commit", (
            f"{s['id']}: mark_green without action=git_commit — "
            "mark_green is invoked from the git_commit post-hook only"
        )


def test_f2_new_green_milestone_steps_inserted():
    """The 4.16.git_commit_green and 13.14.git_commit_green inserts exist."""
    wf = _load_v3()
    for sid, parent in (
        ("4.16.git_commit_green", "4.16"),
        ("13.14.git_commit_green", "13.14"),
    ):
        s = _by_id(wf["steps"], sid)
        assert s is not None, f"{sid} missing"
        pay = s.get("payload") or {}
        assert pay.get("push") is True
        assert pay.get("mark_green") is True
        assert parent in (s.get("depends_on") or []), (
            f"{sid}: must depend on its milestone parent {parent}"
        )


# ── F3: record_demo step payload no longer hardcodes scenario_path ──────


def test_f3_record_demo_payload_drops_scenario_hardcode():
    wf = _load_v3()
    s = _by_id(wf["steps"], "15.10b_record_demo")
    assert s is not None
    pay = s.get("payload") or {}
    assert "scenario_path" not in pay, (
        "F3: 15.10b_record_demo must not hardcode scenario_path — record_demo "
        "resolves from missions.demo_scenario_path / newest e2e spec at runtime"
    )


# ── F7: demo gate blocks 15.14 roadmap_update ───────────────────────────


def test_f7_verify_demo_blocks_roadmap_update():
    """15.10c_verify_demo → 15.14 dependency chain must hold."""
    wf = _load_v3()
    record = _by_id(wf["steps"], "15.10b_record_demo")
    verify = _by_id(wf["steps"], "15.10c_verify_demo")
    roadmap = _by_id(wf["steps"], "15.14")
    bundle = _by_id(wf["steps"], "15.14b_deliverable_bundle")
    assert record is not None and verify is not None
    assert roadmap is not None and bundle is not None
    assert "15.10b_record_demo" in (verify.get("depends_on") or [])
    assert "15.10c_verify_demo" in (roadmap.get("depends_on") or []), (
        "F7: 15.14 roadmap_update must depend on 15.10c_verify_demo so the "
        "demo gate blocks the mission-final roadmap update"
    )
    assert "15.10c_verify_demo" in (bundle.get("depends_on") or [])
    assert "15.14" in (bundle.get("depends_on") or [])


def test_f7_workflow_json_parses_after_inserts():
    """Defensive: ensure all our edits left valid JSON the loader can parse."""
    wf = _load_v3()
    assert isinstance(wf.get("steps"), list)
    ids = [s.get("id") for s in wf["steps"]]
    # No duplicate ids introduced.
    assert len(ids) == len(set(ids)), (
        f"duplicate step ids after wire-fix inserts: "
        f"{[i for i in ids if ids.count(i) > 1][:5]}"
    )
