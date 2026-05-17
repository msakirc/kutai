"""Z5 T5 — mobile app-store distribution.

Covers:
  * i2p_v3.json parses after the T5 edits.
  * the `mobile_app_submission` conditional group resolves onto real steps.
  * the enhanced step 14.8 + the mechanical submit-chain siblings have a
    consistent schema.
  * the `mobile_release_rejection` recipe loads, matches `expo`, and
    round-trips through `instantiate_recipe`.
  * `z6_admission` emits a `vendor_enroll` founder-action when the submit
    step runs without `apple_appstore` / `google_play` credentials.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
I2P_PATH = REPO / "src" / "workflows" / "i2p" / "i2p_v3.json"
RECIPES_DIR = str(REPO / "recipes")


def _load_i2p() -> dict:
    return json.loads(I2P_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# i2p_v3.json structural checks
# ---------------------------------------------------------------------------

def test_i2p_v3_parses():
    d = _load_i2p()
    assert d["steps"], "i2p_v3.json has no steps"


def test_mobile_app_submission_group_resolves():
    """The conditional group's if_true must point at real step ids."""
    d = _load_i2p()
    step_ids = {s["id"] for s in d["steps"]}
    groups = d["metadata"]["conditional_groups"]
    grp = next(g for g in groups if g["group_id"] == "mobile_app_submission")

    assert grp["if_true"], "mobile_app_submission.if_true is still empty"
    for sid in grp["if_true"]:
        assert sid in step_ids, f"if_true references missing step {sid}"
    # The chain must lead with the enhanced 14.8 anchor.
    assert "14.8" in grp["if_true"]


def test_step_14_8_is_metadata_anchor():
    """14.8 generates store metadata + privacy labels (LLM step, full revert)."""
    d = _load_i2p()
    s = next(x for x in d["steps"] if x["id"] == "14.8")
    assert s["name"] == "app_store_submission"
    assert s["agent"] == "executor"
    # Demoted to `full` — it only drafts; the irreversible upload moved out.
    assert s["reversibility"] == "full"
    assert s["real_tool_kind"] == "apple_appstore|google_play"
    assert s.get("needs_real_tools") is True
    # produces the two store artifacts.
    produces = s.get("produces") or []
    assert any("store_metadata.json" in p for p in produces)
    assert any("privacy_nutrition_labels.json" in p for p in produces)
    # artifact schema names the new required fields.
    req = s["artifact_schema"]["app_store_submission"]["required_fields"]
    for field in ("stores", "materials", "bundle_id", "submission_channel"):
        assert field in req


def test_submit_chain_siblings_present_and_mechanical():
    """The three mechanical siblings exist with valid payload + depends_on."""
    d = _load_i2p()
    steps = {s["id"]: s for s in d["steps"]}

    for sid in ("14.8.screenshots", "14.8.submit", "14.8.review_status"):
        assert sid in steps, f"missing sibling step {sid}"
        s = steps[sid]
        assert s["agent"] == "mechanical"
        assert s["executor"] == "mechanical"
        assert isinstance(s["payload"], dict) and s["payload"].get("action")

    # screenshots reuses capture_screenshots in device mode.
    sc = steps["14.8.screenshots"]
    assert sc["payload"]["action"] == "capture_screenshots"
    assert sc["payload"]["capture_mode"] == "device"
    assert sc["depends_on"] == ["14.8"]

    # submit ships the binary via fastlane — irreversible, needs creds.
    sub = steps["14.8.submit"]
    assert sub["payload"]["action"] == "fastlane"
    assert sub["payload"]["lane"] in ("pilot", "supply")
    assert sub["reversibility"] == "irreversible"
    assert sub["real_tool_kind"] == "apple_appstore|google_play"
    assert sub["depends_on"] == ["14.8.screenshots"]

    # review_status polls via vendor_call.
    rev = steps["14.8.review_status"]
    assert rev["payload"]["action"] == "vendor_call"
    assert rev["depends_on"] == ["14.8.submit"]


def test_no_orphan_depends_on_in_submit_chain():
    """Every depends_on in the chain resolves to a real step."""
    d = _load_i2p()
    step_ids = {s["id"] for s in d["steps"]}
    for sid in ("14.8", "14.8.screenshots", "14.8.submit", "14.8.review_status"):
        s = next(x for x in d["steps"] if x["id"] == sid)
        for dep in s.get("depends_on", []):
            assert dep in step_ids, f"{sid} depends on missing {dep}"


# ---------------------------------------------------------------------------
# mobile_release_rejection recipe
# ---------------------------------------------------------------------------

def test_recipe_loads():
    from src.infra.recipes import load_recipe

    r = load_recipe(
        str(REPO / "recipes" / "mobile_release_rejection" / "v1" / "recipe.yaml")
    )
    assert r.name == "mobile_release_rejection"
    assert r.version == "v1"
    assert r.lessons_domain == "mobile_release_rejection"
    assert "expo" in (r.requires.get("tech_stack") or [])
    assert r.post_hooks  # has post-hooks


def test_recipe_in_list_recipes():
    from src.infra.recipes import list_recipes

    recs = list_recipes(RECIPES_DIR)
    names = {r.name for r in recs}
    assert "mobile_release_rejection" in names


def test_match_recipe_expo_returns_rejection_recipe():
    from src.infra.recipes import list_recipes, match_recipe

    recs = list_recipes(RECIPES_DIR)
    matches = match_recipe("expo", recs)
    matched_names = {r.name: score for r, score in matches}
    assert "mobile_release_rejection" in matched_names
    # `expo` is one of its declared tech_stacks → exact match.
    assert matched_names["mobile_release_rejection"] == 1.0


def test_instantiate_recipe_round_trips():
    from src.infra.recipes import list_recipes, match_recipe, instantiate_recipe

    recs = list_recipes(RECIPES_DIR)
    recipe = next(
        r for r, _ in match_recipe("expo", recs)
        if r.name == "mobile_release_rejection"
    )
    with tempfile.TemporaryDirectory() as td:
        res = instantiate_recipe(recipe, td, {"APP_NAME": "RoundTripApp"})
        assert res["ok"] is True
        # both templates land.
        written = set(res["files_written"])
        assert "rejection_response.md.template" in written
        assert "demo_account.json.template" in written
        # token substitution actually happened.
        resp = Path(td, "rejection_response.md.template").read_text(encoding="utf-8")
        assert "RoundTripApp" in resp
        assert "<<APP_NAME>>" not in resp


# ---------------------------------------------------------------------------
# Enrollment path — z6_admission fires vendor_enroll on missing credentials
# ---------------------------------------------------------------------------

async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z5_t5.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


@pytest.mark.asyncio
async def test_submit_step_missing_credentials_emits_vendor_enroll(
    tmp_path, monkeypatch
):
    """When 14.8.submit runs with apple_appstore/google_play creds absent,
    z6_admission must emit a vendor_enroll founder-action."""
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    # Adapter resolves (registered) but no credentials in the vault.
    async def _resolve_with_cred(kinds):
        return "apple_appstore" if "apple_appstore" in kinds else None
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter_with_cred",
        _resolve_with_cred,
    )

    async def _no_cred(_svc):
        return None
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _no_cred,
    )

    mid = await db_mod.add_mission("mobile mission", "")
    # Mirrors the task the expander builds from step 14.8.submit:
    # real_tool_kind copied onto context, needs_real_tools column set.
    task = {
        "id": 1,
        "mission_id": mid,
        "needs_real_tools": 1,
        "reversibility": "irreversible",
        "context": json.dumps({
            "workflow_step_id": "14.8.submit",
            "real_tool_kind": "apple_appstore|google_play",
        }),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    assert "no credential" in res.reason

    actions = await fa.list_by_mission(mid)
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == "vendor_enroll"
    assert a.blocking_step_id == "14.8.submit"
    # The rich enrollment card walks the founder through Apple enrollment.
    assert "Apple Developer Program" in a.title


@pytest.mark.asyncio
async def test_submit_step_admits_when_credentials_present(
    tmp_path, monkeypatch
):
    """With credentials stored and a prior cost_ack, the submit step admits."""
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    from general_beckman.z6_admission import check_z6_admission

    async def _resolve_with_cred(kinds):
        return "google_play" if "google_play" in kinds else None
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter_with_cred",
        _resolve_with_cred,
    )

    async def _has_cred(_svc):
        return {"service_account_json": {"x": 1}}
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _has_cred,
    )

    # 14.8.submit has no cost_estimate_usd, so the irreversible cost-ack
    # gate is skipped (cost 0). Admission should pass straight through.
    mid = await db_mod.add_mission("mobile mission", "")
    task = {
        "id": 2,
        "mission_id": mid,
        "needs_real_tools": 1,
        "reversibility": "irreversible",
        "context": json.dumps({
            "workflow_step_id": "14.8.submit",
            "real_tool_kind": "apple_appstore|google_play",
        }),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is True
    actions = await fa.list_by_mission(mid)
    assert actions == []
