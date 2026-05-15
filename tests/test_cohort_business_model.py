"""Z9 T5A + T5B — cohort awareness + B2B/B2C business-model branching.

Covers:
  T5A
   * mission.context['target_segment'] defaults to "any" (no migration).
   * validate_target_segment emits a WARNING (never blocks) for a Phase 8+
     mission with no explicit target_segment, and back-fills the default.
   * an explicit target_segment is passed through untouched (no warn).
   * the analytics_instrumentation recipe gains a `segment_predicate` field
     and its track_event shim templates carry the cohort-gate machinery.
  T5B
   * score_backlog scoring differs for b2b vs b2c on identical signals
     (B2B tilts churn/pricing/bug; B2C tilts feature_request).
   * track_event shim templates attach account_id when business_model=b2b
     and only user_id (distinct_id) for b2c.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.infra.db as _db_mod
from src.infra.db import init_db, add_mission, get_mission, insert_growth_event

WORKTREE_ROOT = Path(__file__).parent.parent
RECIPE_V1 = WORKTREE_ROOT / "recipes" / "analytics_instrumentation" / "v1"


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation.

    Forces SANDBOX_MODE=local so add_mission() does not block on a docker
    daemon call when provisioning a per-mission container.
    """
    try:
        from src.tools import shell as _shell_mod
        monkeypatch.setattr(_shell_mod, "SANDBOX_MODE", "local")
    except Exception:
        pass
    db_file = tmp_path / "cohort.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    await init_db()


# ───────────────────────── T5A — target_segment ──────────────────────────

@pytest.mark.asyncio
async def test_target_segment_default_is_any(tmp_path, monkeypatch):
    """A mission with no target_segment resolves to the 'any' default."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission("m", "d", context={})

    from mr_roboto.executors.validate_target_segment import (
        validate_target_segment,
        DEFAULT_SEGMENT,
    )
    res = await validate_target_segment(mid)
    assert res["ok"] is True
    assert res["target_segment"] == DEFAULT_SEGMENT == "any"
    assert res["explicit"] is False


@pytest.mark.asyncio
async def test_validate_target_segment_warns_and_backfills(
    tmp_path, monkeypatch
):
    """No target_segment → WARN (warn-only, ok stays True) + back-fill 'any'."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission("m", "d", context={"foo": "bar"})

    from mr_roboto.executors.validate_target_segment import (
        validate_target_segment,
    )
    res = await validate_target_segment(mid)
    # warn-only nag — never a block.
    assert res["ok"] is True
    assert res["warned"] is True
    assert res["explicit"] is False

    # The 'any' default is persisted into mission.context (idempotent).
    mission = await get_mission(mid)
    ctx = json.loads(mission["context"])
    assert ctx["target_segment"] == "any"
    assert ctx["foo"] == "bar"  # other keys untouched


@pytest.mark.asyncio
async def test_explicit_target_segment_passes_without_warn(
    tmp_path, monkeypatch
):
    """An explicit recognised segment is honoured and never warns."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(
        "m", "d", context={"target_segment": "week2_churners"}
    )

    from mr_roboto.executors.validate_target_segment import (
        validate_target_segment,
    )
    res = await validate_target_segment(mid)
    assert res["ok"] is True
    assert res["explicit"] is True
    assert res["warned"] is False
    assert res["unknown_value"] is False
    assert res["target_segment"] == "week2_churners"


@pytest.mark.asyncio
async def test_unknown_target_segment_is_warned_but_passes(
    tmp_path, monkeypatch
):
    """A free-form non-canonical segment proceeds (free-form is allowed)."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission(
        "m", "d", context={"target_segment": "enterprise_trials"}
    )

    from mr_roboto.executors.validate_target_segment import (
        validate_target_segment,
    )
    res = await validate_target_segment(mid)
    assert res["ok"] is True
    assert res["explicit"] is True
    assert res["unknown_value"] is True
    assert res["target_segment"] == "enterprise_trials"


@pytest.mark.asyncio
async def test_validate_target_segment_run_requires_mission_id():
    """The run(task) dispatcher entry rejects a missing mission_id."""
    from mr_roboto.executors.validate_target_segment import run
    res = await run({"payload": {}})
    assert res["ok"] is False
    assert "mission_id" in res["error"]


# ───────────────────── T5A — segment_predicate recipe ────────────────────

def test_recipe_declares_segment_predicate_field():
    """recipe.yaml gains an additive optional segment_predicate field."""
    data = yaml.safe_load((RECIPE_V1 / "recipe.yaml").read_text())
    assert "segment_predicate" in data
    # Default unset → no cohort gate.
    assert data["segment_predicate"] is None


def test_track_event_shims_carry_segment_predicate_machinery():
    """Both shim templates honour segment_predicate (cohort gate)."""
    client = (RECIPE_V1 / "client.template.ts").read_text()
    server = (RECIPE_V1 / "server.template.py").read_text()

    # Client shim: predicate constant + gate fn + emission guard.
    assert "SEGMENT_PREDICATE" in client
    assert "segmentMatches" in client
    assert "RECIPE_PARAM:SEGMENT_PREDICATE" in client
    # The capture call is gated by the predicate.
    assert "if (!segmentMatches())" in client

    # Server shim: predicate constant + gate fn + emission guard.
    assert "SEGMENT_PREDICATE" in server
    assert "_segment_matches" in server
    assert "RECIPE_PARAM:SEGMENT_PREDICATE" in server
    assert "if not _segment_matches():" in server


# ──────────────────── T5B — score_backlog b2b vs b2c ─────────────────────

def test_revenue_impact_tables_differ_by_business_model():
    """B2B tilts churn/pricing/bug up; B2C tilts feature_request higher."""
    from mr_roboto.executors.score_backlog import _REVENUE_IMPACT_BY_MODEL

    b2b = _REVENUE_IMPACT_BY_MODEL["b2b"]
    b2c = _REVENUE_IMPACT_BY_MODEL["b2c"]

    # B2B: a churned account = many lost seats → bug/pricing weigh heavier.
    assert b2b["bug"] > b2c["bug"]
    assert b2b["pricing_feedback"] > b2c["pricing_feedback"]
    # B2C: diffuse, growth-driven → feature requests weigh relatively more.
    assert b2c["feature_request"] > b2b["feature_request"]


def test_normalize_business_model_defaults_to_b2c():
    """Unknown / missing business_model coerces to the b2c default."""
    from mr_roboto.executors.score_backlog import _normalize_business_model
    assert _normalize_business_model(None) == "b2c"
    assert _normalize_business_model("") == "b2c"
    assert _normalize_business_model("nonsense") == "b2c"
    assert _normalize_business_model("B2B") == "b2b"
    assert _normalize_business_model(" hybrid ") == "hybrid"


@pytest.mark.asyncio
async def test_score_backlog_scores_differ_b2b_vs_b2c(tmp_path, monkeypatch):
    """Identical signals score differently under b2b vs b2c business models.

    Feed one feature_request cluster + one bug cluster, then score the same
    DB once as b2b and once as b2c (payload override). B2B must rank the
    bug cluster relatively higher (heavier bug revenue_impact); B2C must
    rank the feature_request cluster relatively higher.
    """
    await _fresh_db(tmp_path, monkeypatch)

    # Two missions sharing the same signal shape but different models.
    mid_b2b = await add_mission("b2b", "d", context={"business_model": "b2b"})
    mid_b2c = await add_mission("b2c", "d", context={"business_model": "b2c"})

    async def _seed(mid):
        for i in range(3):
            await insert_growth_event(mid, "classified_signal", {
                "label": "feature_request", "domain": "general",
                "external_id": f"fr-{mid}-{i}",
                "occurred_at": "2026-05-14 09:00:00",
                "content_excerpt": "please add dark mode",
            })
        for i in range(3):
            await insert_growth_event(mid, "classified_signal", {
                "label": "bug", "domain": "general",
                "external_id": f"bug-{mid}-{i}",
                "occurred_at": "2026-05-14 09:00:00",
                "content_excerpt": "crash on save",
            })

    await _seed(mid_b2b)
    await _seed(mid_b2c)

    from mr_roboto.executors.score_backlog import run as score_run

    res_b2b = await score_run({"mission_id": mid_b2b, "payload": {}})
    res_b2c = await score_run({"mission_id": mid_b2c, "payload": {}})
    assert res_b2b["ok"] and res_b2c["ok"]

    def _score(scored, label):
        for s in scored:
            if s["label"] == label:
                return s["score"]
        raise AssertionError(f"no {label} cluster")

    bug_b2b = _score(res_b2b["scored"], "bug")
    bug_b2c = _score(res_b2c["scored"], "bug")
    fr_b2b = _score(res_b2b["scored"], "feature_request")
    fr_b2c = _score(res_b2c["scored"], "feature_request")

    # Same frequency + age → score reflects only the model-aware weighting.
    # B2B weighs bugs heavier than B2C does.
    assert bug_b2b > bug_b2c
    # B2B weighs feature_request lighter than B2C does.
    assert fr_b2c > fr_b2b

    # The model is recorded in the inspectable formula breakdown.
    for s in res_b2b["scored"]:
        assert s["formula"]["business_model"] == "b2b"
    for s in res_b2c["scored"]:
        assert s["formula"]["business_model"] == "b2c"


@pytest.mark.asyncio
async def test_score_backlog_payload_business_model_override(
    tmp_path, monkeypatch
):
    """payload.business_model overrides the mission context value."""
    await _fresh_db(tmp_path, monkeypatch)
    mid = await add_mission("m", "d", context={"business_model": "b2c"})
    for i in range(2):
        await insert_growth_event(mid, "classified_signal", {
            "label": "churn_signal", "domain": "general",
            "external_id": f"ch-{i}", "occurred_at": "2026-05-14 09:00:00",
            "content_excerpt": "cancelling subscription",
        })

    from mr_roboto.executors.score_backlog import run as score_run
    res = await score_run(
        {"mission_id": mid, "payload": {"business_model": "b2b"}}
    )
    assert res["ok"]
    assert res["scored"][0]["formula"]["business_model"] == "b2b"


# ───────────────────── T5B — b2b event tagging ───────────────────────────

def test_b2b_event_tagging_attaches_account_id_and_user_id():
    """B2B track_event attaches account_id; B2C attaches user_id only."""
    client = (RECIPE_V1 / "client.template.ts").read_text()
    server = (RECIPE_V1 / "server.template.py").read_text()

    # Client: account_id is part of AnalyticsContext and attached for b2b.
    assert "account_id" in client
    assert 'businessModel === "b2b"' in client
    assert "enriched.account_id" in client

    # Server: set_analytics_context accepts account_id; track_event attaches
    # it alongside distinct_id (the user id) only when business_model=b2b.
    assert "account_id" in server
    assert 'business_model == "b2b"' in server
    assert 'enriched["account_id"]' in server
    # B2C path: distinct_id is always the per-user id (posthog.capture arg).
    assert "distinct_id=distinct_id" in server
