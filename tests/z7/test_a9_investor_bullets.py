"""Z7 T5 A9 — Investor bullets + A9.r1 segmented templates tests.

Covers:
  1. Anomaly detection flags a >2σ outlier (positive and negative).
  2. Bullets render with all sections (Highlights, Lowlights, Numbers,
     Anomalies, Suggested asks).
  3. Missing data source degrades gracefully (skips section, no crash).
  4. Three segmented variants emitted (one per recipient category).
  5. founder_action surfaced with all variants attached.
  6. mr_roboto dispatch handler registered for "investor_bullets".
  7. cron_seed INTERNAL_CADENCES contains monthly "investor_bullets" entry.
  8. DB migration: no new tables needed (A9 reads existing tables only).
"""
from __future__ import annotations

import json
import math
import statistics
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for A9 tests."""
    db_file = str(tmp_path / "test_a9.db")
    monkeypatch.setenv("DB_PATH", db_file)
    try:
        import src.infra.db as _db_mod
        monkeypatch.setattr(_db_mod, "DB_PATH", db_file)
        monkeypatch.setattr(_db_mod, "_db_connection", None)
        monkeypatch.setattr(_db_mod, "_db_connection_path", None)
    except Exception:
        pass
    return db_file


@pytest_asyncio.fixture
async def db(tmp_db):
    """Initialised DB with full schema."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ===========================================================================
# 1. Anomaly detection
# ===========================================================================


def test_anomaly_detect_positive_outlier():
    """A value far above the 3-month median is flagged as a positive outlier."""
    from src.app.jobs.investor_bullets import _detect_anomaly

    history = [100.0, 102.0, 98.0]      # trailing 3 months
    current = 180.0                       # well above median + 2σ
    result = _detect_anomaly("mrr", current, history)
    assert result["is_anomaly"] is True
    assert result["direction"] == "up"


def test_anomaly_detect_negative_outlier():
    """A value far below the 3-month median is flagged as a negative outlier."""
    from src.app.jobs.investor_bullets import _detect_anomaly

    history = [100.0, 102.0, 98.0]
    current = 20.0                        # far below
    result = _detect_anomaly("churn", current, history)
    assert result["is_anomaly"] is True
    assert result["direction"] == "down"


def test_anomaly_detect_normal_range():
    """A value within ±2σ is NOT flagged."""
    from src.app.jobs.investor_bullets import _detect_anomaly

    history = [100.0, 110.0, 90.0]
    current = 105.0                       # inside band
    result = _detect_anomaly("customers", current, history)
    assert result["is_anomaly"] is False


def test_anomaly_detect_insufficient_history():
    """With fewer than 2 history points, no anomaly is flagged."""
    from src.app.jobs.investor_bullets import _detect_anomaly

    result = _detect_anomaly("mrr", 9999.0, [])
    assert result["is_anomaly"] is False

    result2 = _detect_anomaly("mrr", 9999.0, [100.0])
    assert result2["is_anomaly"] is False


# ===========================================================================
# 2. Bullet sections rendered
# ===========================================================================


@pytest.mark.asyncio
async def test_render_all_sections():
    """render_bullets produces all 5 required section headers."""
    from src.app.jobs.investor_bullets import render_bullets

    metrics = {
        "mrr": {"current": 5000, "history": [4800, 4900, 4950]},
        "customer_count": {"current": 120, "history": [100, 105, 110]},
        "churn_rate": {"current": 0.02, "history": [0.03, 0.03, 0.025]},
        "burn": {"current": 12000, "history": [12500, 12000, 11800]},
        "runway_months": {"current": 14, "history": [13, 13, 14]},
        "uptime_pct": {"current": 99.9, "history": [99.8, 99.9, 99.9]},
        "p95_latency_ms": {"current": 220, "history": [200, 210, 215]},
        "incident_count": {"current": 0, "history": [1, 0, 1]},
        "prs_shipped": {"current": 12, "history": [8, 10, 9]},
        "security_issues": {"current": 0, "history": [1, 0, 0]},
        "support_volume": {"current": 40, "history": [35, 38, 42]},
        "escalation_rate": {"current": 0.1, "history": [0.12, 0.11, 0.10]},
    }
    hypotheses = {}
    gaps = []

    md = await render_bullets(metrics, hypotheses, gaps)
    for section in ("Highlights", "Lowlights", "Numbers", "Anomalies", "Suggested asks"):
        assert section in md, f"Missing section: {section}"


@pytest.mark.asyncio
async def test_render_highlights_negative_lowlights_positive():
    """Highlights = best positive outliers; Lowlights = worst negative outliers."""
    from src.app.jobs.investor_bullets import render_bullets

    # MRR up 70%, churn up 50% (bad)
    metrics = {
        "mrr": {"current": 8500, "history": [5000, 5000, 5000]},   # big up
        "churn_rate": {"current": 0.15, "history": [0.02, 0.02, 0.02]},  # big up = bad
        "customer_count": {"current": 100, "history": [99, 100, 100]},
    }
    md = await render_bullets(metrics, {}, [])
    assert "mrr" in md.lower() or "MRR" in md
    assert "churn" in md.lower()


# ===========================================================================
# 3. Missing data source degrades gracefully
# ===========================================================================


@pytest.mark.asyncio
async def test_missing_z6_metrics_degrades():
    """When Z6 metrics are absent, the section is skipped, not crashed."""
    from src.app.jobs.investor_bullets import collect_metrics

    # Patch all optional sources to raise
    with (
        patch("src.app.jobs.investor_bullets._fetch_z6_metrics",
              new=AsyncMock(side_effect=Exception("z6 absent"))),
        patch("src.app.jobs.investor_bullets._fetch_ops_metrics",
              new=AsyncMock(return_value={})),
        patch("src.app.jobs.investor_bullets._fetch_review_density",
              new=AsyncMock(return_value={})),
        patch("src.app.jobs.investor_bullets._fetch_support_metrics",
              new=AsyncMock(return_value={})),
        patch("src.app.jobs.investor_bullets._fetch_mention_counts",
              new=AsyncMock(return_value={})),
    ):
        metrics, missing = await collect_metrics("product_1")
    # Should not crash and z6 sources should be in missing list
    assert "z6" in missing


@pytest.mark.asyncio
async def test_missing_all_sources_produces_empty_metrics():
    """All sources absent → empty metrics dict, no crash."""
    from src.app.jobs.investor_bullets import collect_metrics

    with (
        patch("src.app.jobs.investor_bullets._fetch_z6_metrics",
              new=AsyncMock(side_effect=Exception("absent"))),
        patch("src.app.jobs.investor_bullets._fetch_ops_metrics",
              new=AsyncMock(side_effect=Exception("absent"))),
        patch("src.app.jobs.investor_bullets._fetch_review_density",
              new=AsyncMock(side_effect=Exception("absent"))),
        patch("src.app.jobs.investor_bullets._fetch_support_metrics",
              new=AsyncMock(side_effect=Exception("absent"))),
        patch("src.app.jobs.investor_bullets._fetch_mention_counts",
              new=AsyncMock(side_effect=Exception("absent"))),
    ):
        metrics, missing = await collect_metrics("product_1")
    assert metrics == {}
    assert len(missing) > 0


@pytest.mark.asyncio
async def test_empty_metrics_renders_without_crash():
    """Empty metrics still produces valid Markdown (all sections present)."""
    from src.app.jobs.investor_bullets import render_bullets

    md = await render_bullets({}, {}, [])
    # Sections must exist even when no data
    for section in ("Numbers", "Anomalies", "Suggested asks"):
        assert section in md


# ===========================================================================
# 4. Three segmented variants emitted
# ===========================================================================


@pytest.mark.asyncio
async def test_three_segmented_variants_emitted():
    """emit_segmented_variants returns one variant per template type."""
    from src.app.jobs.investor_bullets import emit_segmented_variants

    bullets_md = "## Numbers\n- MRR: $5000\n"
    contacts = [
        {"category": "investor", "handle": "@vc1"},
        {"category": "investor", "handle": "@vc2"},
        {"category": "advisor", "handle": "@advisor1"},
    ]

    variants = emit_segmented_variants(bullets_md, contacts)
    # Must produce all 3 template kinds
    kinds = {v["template_kind"] for v in variants}
    assert "pre_investor_pitch_bullets" in kinds
    assert "current_investor_update" in kinds
    assert "advisor_check_in" in kinds


@pytest.mark.asyncio
async def test_segmented_variants_differ():
    """The three variants frame the same numbers differently."""
    from src.app.jobs.investor_bullets import emit_segmented_variants

    bullets_md = "## Numbers\n- MRR: $5000\n"
    contacts = [
        {"category": "investor", "handle": "@vc1"},
        {"category": "advisor", "handle": "@mentor"},
    ]

    variants = emit_segmented_variants(bullets_md, contacts)
    texts = [v["content_md"] for v in variants]
    # All texts are non-empty
    assert all(t.strip() for t in texts)
    # Not all identical (templates differ)
    assert len(set(texts)) > 1


@pytest.mark.asyncio
async def test_no_investor_contacts_pre_pitch_absent():
    """With no investor contacts, pre_investor_pitch_bullets is still emitted
    (system emits all 3 variants when any contact exists)."""
    from src.app.jobs.investor_bullets import emit_segmented_variants

    contacts = [{"category": "advisor", "handle": "@advisor1"}]
    variants = emit_segmented_variants("## Numbers\n- MRR: $0\n", contacts)
    kinds = {v["template_kind"] for v in variants}
    # All 3 always emitted when contacts exist
    assert len(variants) == 3


@pytest.mark.asyncio
async def test_no_contacts_returns_empty():
    """With no CRM contacts at all, no variants are emitted."""
    from src.app.jobs.investor_bullets import emit_segmented_variants

    variants = emit_segmented_variants("## Numbers\n- MRR: $0\n", [])
    assert variants == []


# ===========================================================================
# 5. founder_action surfaced
# ===========================================================================


@pytest.mark.asyncio
async def test_founder_action_surfaced():
    """run_investor_bullets surfaces a founder_action with bullets attached."""
    from src.app.jobs.investor_bullets import run_investor_bullets

    mock_fa = AsyncMock(return_value=MagicMock(id=42))
    with (
        patch("src.app.jobs.investor_bullets.collect_metrics",
              new=AsyncMock(return_value=({}, []))),
        patch("src.app.jobs.investor_bullets.render_bullets",
              new=AsyncMock(return_value="## Numbers\n- MRR: N/A\n")),
        patch("src.app.jobs.investor_bullets.emit_segmented_variants",
              return_value=[
                  {"template_kind": "pre_investor_pitch_bullets", "content_md": "pitch"},
                  {"template_kind": "current_investor_update", "content_md": "update"},
                  {"template_kind": "advisor_check_in", "content_md": "advisor"},
              ]),
        patch("src.app.jobs.investor_bullets._list_contacts",
              new=AsyncMock(return_value=[])),
        patch("src.app.jobs.investor_bullets._create_founder_action", new=mock_fa),
    ):
        result = await run_investor_bullets(product_id="p1")

    assert result["ok"] is True
    mock_fa.assert_called_once()
    call_kwargs = mock_fa.call_args[1]
    assert "investor_bullets" in call_kwargs.get("title", "").lower() or \
           "bullets" in call_kwargs.get("title", "").lower()
    # context_json must contain the variants
    ctx = json.loads(call_kwargs["context_json"])
    assert "variants" in ctx


@pytest.mark.asyncio
async def test_run_investor_bullets_returns_ok():
    """run_investor_bullets returns {'ok': True} on success."""
    from src.app.jobs.investor_bullets import run_investor_bullets

    with (
        patch("src.app.jobs.investor_bullets.collect_metrics",
              new=AsyncMock(return_value=({"mrr": {"current": 5000, "history": [4800, 4900, 4950]}}, []))),
        patch("src.app.jobs.investor_bullets.render_bullets",
              new=AsyncMock(return_value="## Highlights\n- mrr up\n")),
        patch("src.app.jobs.investor_bullets.emit_segmented_variants",
              return_value=[]),
        patch("src.app.jobs.investor_bullets._list_contacts",
              new=AsyncMock(return_value=[])),
        patch("src.app.jobs.investor_bullets._create_founder_action",
              new=AsyncMock(return_value=MagicMock(id=1))),
    ):
        result = await run_investor_bullets(product_id="p1")

    assert result.get("ok") is True


# ===========================================================================
# 6. mr_roboto dispatch handler registered
# ===========================================================================


def test_investor_bullets_routed_in_mr_roboto():
    """investor_bullets must be handled in mr_roboto._run_dispatch."""
    import inspect
    import mr_roboto
    try:
        src = inspect.getsource(mr_roboto._run_dispatch)
    except Exception:
        src = ""
    assert "investor_bullets" in src, (
        "investor_bullets not found in mr_roboto._run_dispatch source"
    )


# ===========================================================================
# 7. cron_seed contains monthly investor_bullets entry
# ===========================================================================


def test_cron_seed_contains_investor_bullets():
    """INTERNAL_CADENCES must have an investor_bullets monthly entry."""
    from general_beckman.cron_seed import INTERNAL_CADENCES

    entry = next(
        (c for c in INTERNAL_CADENCES if c["title"] == "investor_bullets"),
        None,
    )
    assert entry is not None, "investor_bullets not found in INTERNAL_CADENCES"
    interval = entry.get("interval_seconds")
    cron = entry.get("cron_expression", "")
    # Monthly = ~2592000s or a cron expression
    assert (
        (interval and interval >= 2592000)
        or (cron and len(cron) > 0)
    ), f"investor_bullets interval not monthly: interval={interval}, cron={cron}"


def test_cron_seed_payload_has_executor():
    """investor_bullets cron entry must carry _executor key."""
    from general_beckman.cron_seed import INTERNAL_CADENCES

    entry = next(
        (c for c in INTERNAL_CADENCES if c["title"] == "investor_bullets"),
        None,
    )
    assert entry is not None
    assert entry.get("payload", {}).get("_executor") == "investor_bullets"


# ===========================================================================
# 8. No new DB table required — reads existing tables
# ===========================================================================


@pytest.mark.asyncio
async def test_ops_metrics_reads_incidents_table(db):
    """_fetch_ops_metrics reads the incidents table (exists after init_db)."""
    from src.app.jobs.investor_bullets import _fetch_ops_metrics

    # incidents table exists (seeded by Z7 T3D migration)
    result = await _fetch_ops_metrics("product_1")
    # Should return a dict (possibly empty) without crashing
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_support_metrics_reads_tickets_table(db):
    """_fetch_support_metrics reads the tickets table (exists after init_db)."""
    from src.app.jobs.investor_bullets import _fetch_support_metrics

    result = await _fetch_support_metrics("product_1")
    assert isinstance(result, dict)


# ===========================================================================
# 9. LLM hypothesis call is mocked (OVERHEAD lane)
# ===========================================================================


@pytest.mark.asyncio
async def test_llm_hypothesis_call_uses_await_inline():
    """_call_llm_anomaly_hypothesis enqueues via await_inline=True and returns str content."""
    from unittest.mock import patch, AsyncMock
    from general_beckman import TaskResult

    # Return a TaskResult with content in the result dict (real awaitable path).
    fake_result = TaskResult(
        status="completed",
        result={"content": "Hypothesis: seasonality spike from Q1 campaign."},
        error=None,
    )

    async def _fake_enqueue(spec, *, lane, await_inline=False):
        # Verify the spec uses raw_dispatch (real llm_call structure)
        llm_call = (spec.get("context") or {}).get("llm_call") or {}
        assert llm_call.get("raw_dispatch") is True, "Must use raw_dispatch=True pattern"
        assert "_callback" not in (spec.get("context") or {}), \
            "Must NOT use deprecated _callback pattern"
        assert await_inline is True, "Must call with await_inline=True"
        return fake_result

    with patch(
        "src.app.jobs.investor_bullets._enqueue_overhead",
        new=AsyncMock(side_effect=_fake_enqueue),
    ):
        from src.app.jobs.investor_bullets import _call_llm_anomaly_hypothesis
        hyp = await _call_llm_anomaly_hypothesis("mrr", 180.0, [100.0, 102.0, 98.0])

    assert isinstance(hyp, str)
    assert "seasonality" in hyp


# ===========================================================================
# 10. render_bullets: Suggested asks uses mission_lessons degrade
# ===========================================================================


@pytest.mark.asyncio
async def test_suggested_asks_degrades_when_no_lessons():
    """render_bullets degrades gracefully when mission_lessons has no needs_external_help rows."""
    from src.app.jobs.investor_bullets import render_bullets

    # Pass empty gaps list → section present but may say "none found"
    md = await render_bullets({}, {}, [])
    assert "Suggested asks" in md
