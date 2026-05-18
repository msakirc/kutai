"""Z1 Tier 4 (T4B / B2) — propagate_asset_change contract tests.

Founder points at an asset (HTML element / screen_plan section / token
field) and describes a desired change. The action walks the
produces/consumes graph in ``i2p_v3.json`` and surfaces a list of
dependent artifacts with suggested patches. The output is a
``propagation_proposal.md`` consumed by the founder via Telegram.

The action does NOT regenerate anything itself — that's T4A's
``regen_artifact``. T4B only finds what would need to be regenerated.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mr_roboto.propagate_asset_change import propagate_asset_change


# Toy workflow steps — emulates the produces/consumes graph shape used
# by i2p_v3.json. Keeping it inline keeps the test independent of the
# real workflow JSON, which churns frequently.
_STEPS = [
    {
        "id": "5.0",
        "name": "design_tokens",
        "produces": ["mission_{mission_id}/.style/design_tokens.json"],
        "output_artifacts": ["design_tokens"],
    },
    {
        "id": "5.20a",
        "name": "per_screen_plans_chunk_a",
        "input_artifacts": ["design_tokens"],
        "output_artifacts": ["per_screen_plans_chunk_a"],
        "produces": ["mission_{mission_id}/.plans/screen_5_3.md"],
    },
    {
        "id": "5.30a",
        "name": "html_prototypes_chunk_a",
        "input_artifacts": ["per_screen_plans_chunk_a", "design_tokens"],
        "output_artifacts": ["html_prototypes_chunk_a"],
        "produces": ["mission_{mission_id}/.web/screen_5_3.html"],
    },
]


def test_change_to_token_propagates_to_screen_plan_and_html():
    res = propagate_asset_change(
        asset_path="mission_1/.style/design_tokens.json",
        change_description="bump primary color from blue to teal",
        steps=_STEPS,
        mission_id="1",
    )
    assert res["ok"] is True
    deps = {d["step_id"] for d in res["dependents"]}
    # design_tokens consumed by screen-plan and html steps
    assert "5.20a" in deps
    assert "5.30a" in deps
    # the originating step is also surfaced (it produces the asset)
    assert res["origin_step_id"] == "5.0"
    # at least one suggested patch per dependent
    for d in res["dependents"]:
        assert d.get("suggested_patch")


def test_change_to_html_does_not_propagate_to_upstream():
    res = propagate_asset_change(
        asset_path="mission_1/.web/screen_5_3.html",
        change_description="hero card too busy, simplify",
        steps=_STEPS,
        mission_id="1",
    )
    assert res["ok"] is True
    # html is a leaf — nothing downstream consumes it
    assert res["dependents"] == []
    assert res["origin_step_id"] == "5.30a"
    # but the proposer surfaces upstream candidates as "consider patching"
    assert any(s["step_id"] == "5.20a" for s in res["upstream_candidates"])


def test_unknown_asset_returns_error():
    res = propagate_asset_change(
        asset_path="mission_1/.nowhere/foo.txt",
        change_description="tweak",
        steps=_STEPS,
        mission_id="1",
    )
    assert res["ok"] is False
    assert "no producing step" in res["error"].lower()


def test_emits_proposal_markdown(tmp_path: Path):
    out_path = tmp_path / "propagation_proposal.md"
    res = propagate_asset_change(
        asset_path="mission_1/.style/design_tokens.json",
        change_description="bump primary color from blue to teal",
        steps=_STEPS,
        mission_id="1",
        out_path=str(out_path),
    )
    assert res["ok"] is True
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "propagation proposal" in body.lower()
    assert "5.20a" in body
    assert "5.30a" in body
    assert "teal" in body.lower()


def test_run_dispatch_routes(tmp_path: Path):
    from mr_roboto import run as mr_run
    out = tmp_path / "prop.md"
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "propagate_asset_change",
            "asset_path": "mission_1/.style/design_tokens.json",
            "change_description": "switch to teal",
            "steps": _STEPS,
            "mission_id": "1",
            "out_path": str(out),
        },
    }
    res = asyncio.run(mr_run(task))
    assert res.status == "completed", res.error
    assert out.exists()
