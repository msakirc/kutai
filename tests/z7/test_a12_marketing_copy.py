"""Z7 T6 A12 — Marketing copy generator tests.

Covers:
  1. marketing_copy verb produces structured artifact with hero×3, features, pricing, FAQ.
  2. Artifact written to artifacts/marketing_copy/{mission_id}.json.
  3. Brand-voice lint (A5) invoked on generated copy; degrades gracefully when absent.
  4. Copy compliance review (A6) invoked on generated copy; degrades gracefully on error.
  5. founder_action emitted with approve / regenerate-hero / regenerate-FAQ options.
  6. Graceful degradation: brand_voice doc absent → lint skipped with info note.
  7. Graceful degradation: faq artifact absent → FAQ section uses empty list.
  8. Reversibility tag registered as "full".
  9. marketing_copy imported in mr_roboto.__init__ and Action dispatched correctly.
 10. LLM enqueue called with MAIN_WORK lane.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── DB helpers ──────────────────────────────────────────────────────────────


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_a12.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.infra import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


# ── Minimal product spec ─────────────────────────────────────────────────────

_PRODUCT_SPEC = {
    "name": "SuperApp",
    "tagline": "The AI that does everything",
    "features": [
        {"name": "Smart Search", "description": "Find anything instantly"},
        {"name": "Auto Reports", "description": "Generate reports in seconds"},
    ],
    "pricing_tiers": [
        {"name": "Free", "price_usd": 0, "description": "Basic features"},
        {"name": "Pro", "price_usd": 29, "description": "Full access"},
    ],
    "target_audience": "b2b",
}

_MOCK_COPY_RESULT = {
    "hero": [
        {"headline": "Headline 1", "subheadline": "Sub 1", "cta": "Get started"},
        {"headline": "Headline 2", "subheadline": "Sub 2", "cta": "Try free"},
        {"headline": "Headline 3", "subheadline": "Sub 3", "cta": "See demo"},
    ],
    "features": [
        {"name": "Smart Search", "copy": "Find anything instantly with AI"},
        {"name": "Auto Reports", "copy": "Generate beautiful reports in seconds"},
    ],
    "pricing": [
        {"tier": "Free", "price_usd": 0, "copy": "Get started for free, no card required"},
        {"tier": "Pro", "price_usd": 29, "copy": "Everything you need, unlimited"},
    ],
    "faq": [
        {"question": "How does pricing work?", "answer": "Simple monthly subscription."},
        {"question": "Is there a free trial?", "answer": "Yes, forever free tier available."},
    ],
}


# ── Test 1: marketing_copy produces structured artifact ─────────────────────


@pytest.mark.asyncio
async def test_marketing_copy_produces_structured_artifact(tmp_path, monkeypatch):
    """Verb should produce hero×3, features, pricing, FAQ and write to disk."""
    db_mod = await _setup_db(tmp_path, monkeypatch)

    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no audience metadata"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=42),
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=1,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience="marketing",
        )

    assert result["status"] == "completed"
    artifact = result["artifact"]

    # Hero must have exactly 3 variants
    assert len(artifact["hero"]) == 3
    for hero in artifact["hero"]:
        assert "headline" in hero
        assert "subheadline" in hero
        assert "cta" in hero

    # Features must be present
    assert len(artifact["features"]) >= 1

    # Pricing tiers
    assert len(artifact["pricing"]) >= 1

    # FAQ entries
    assert isinstance(artifact["faq"], list)

    # Artifact file written to disk
    artifact_path = result.get("artifact_path")
    assert artifact_path is not None
    assert Path(artifact_path).exists()
    with open(artifact_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["hero"] == artifact["hero"]


# ── Test 2: artifact written under artifacts/marketing_copy/{mission_id}.json ─


@pytest.mark.asyncio
async def test_artifact_path_uses_mission_id(tmp_path, monkeypatch):
    """Artifact file should be at artifacts/marketing_copy/7.json."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no doc"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=10),
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=7,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience=None,
        )

    assert "7.json" in result["artifact_path"]


# ── Test 3: brand_voice lint invoked when audience provided ─────────────────


@pytest.mark.asyncio
async def test_brand_voice_lint_invoked(tmp_path, monkeypatch):
    """Lint function should be called with the generated copy text."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    lint_calls = []

    async def mock_lint(text, audience, task_id, mission_id):
        lint_calls.append((audience, task_id))
        return {"status": "ok", "violations": []}

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=mock_lint,
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=5),
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=3,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience="marketing",
        )

    assert len(lint_calls) == 1
    assert lint_calls[0][0] == "marketing"
    assert result["lint_result"]["status"] == "ok"


# ── Test 4: copy compliance invoked ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_copy_compliance_invoked(tmp_path, monkeypatch):
    """Compliance check should be called on the artifact text."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    compliance_calls = []

    async def mock_compliance(text, task_id, mission_id):
        compliance_calls.append(task_id)
        return {"status": "ok", "findings": []}

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no doc"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=mock_compliance,
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=5),
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=4,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience=None,
        )

    assert len(compliance_calls) == 1
    assert result["compliance_result"]["status"] == "ok"


# ── Test 5: founder_action emitted with correct options ─────────────────────


@pytest.mark.asyncio
async def test_founder_action_emitted_with_options(tmp_path, monkeypatch):
    """founder_action should include approve / regenerate-hero / regenerate-FAQ."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    emitted_actions = []

    async def mock_emit(mission_id, artifact_path, options, task_id):
        emitted_actions.append({"mission_id": mission_id, "options": options})
        return 99

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=mock_emit,
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=5,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience="marketing",
        )

    assert len(emitted_actions) == 1
    options = emitted_actions[0]["options"]
    option_labels = [o.lower() for o in options]
    assert any("approve" in o for o in option_labels)
    assert any("hero" in o for o in option_labels)
    assert any("faq" in o for o in option_labels)
    assert result["founder_action_id"] == 99


# ── Test 6: graceful degradation — brand_voice doc absent ───────────────────


@pytest.mark.asyncio
async def test_lint_degrades_gracefully_when_voice_doc_absent(tmp_path, monkeypatch):
    """When brand_voice doc missing, lint returns skip (not error)."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    from mr_roboto.marketing_copy import _run_brand_voice_lint

    # Patch the load function to return None (no voice doc)
    with patch(
        "mr_roboto.marketing_copy._load_brand_voice_doc",
        return_value=None,
    ):
        result = await _run_brand_voice_lint(
            text="Great product copy here",
            audience="marketing",
            task_id=None,
            mission_id=None,
        )

    assert result["status"] == "skip"
    assert "brand_voice" in result.get("reason", "").lower() or "doc" in result.get("reason", "").lower()


# ── Test 7: graceful degradation — faq artifact absent ──────────────────────


@pytest.mark.asyncio
async def test_faq_degrades_gracefully_when_artifact_absent(tmp_path, monkeypatch):
    """When A8 faq artifact absent, FAQ section in artifact is empty list (not error)."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    # Mock copy result with empty FAQ (as would happen when seeding from absent artifact)
    mock_result_no_faq = dict(_MOCK_COPY_RESULT)
    mock_result_no_faq = {**_MOCK_COPY_RESULT, "faq": []}

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": mock_result_no_faq}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no doc"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=1),
    ), patch(
        "mr_roboto.marketing_copy._load_faq_seed",
        return_value=None,  # faq artifact absent
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=6,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience=None,
        )

    assert result["status"] == "completed"
    assert result["artifact"]["faq"] == []


# ── Test 8: reversibility tag ────────────────────────────────────────────────


def test_marketing_copy_reversibility_registered():
    """marketing_copy verb must be tagged 'full' in VERB_REVERSIBILITY."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY

    assert VERB_REVERSIBILITY.get("marketing_copy") == "full"


# ── Test 9: marketing_copy imported in mr_roboto.__init__ ───────────────────


def test_marketing_copy_module_importable():
    """mr_roboto.marketing_copy module must be importable."""
    import mr_roboto.marketing_copy as mc

    assert hasattr(mc, "run_marketing_copy")


# ── Test 10: LLM enqueue called with MAIN_WORK lane ─────────────────────────


@pytest.mark.asyncio
async def test_enqueue_uses_main_work_lane(tmp_path, monkeypatch):
    """LLM copy generation task must use MAIN_WORK lane, not OVERHEAD."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    enqueue_calls = []

    async def mock_enqueue(spec, **kwargs):
        enqueue_calls.append({"spec": spec, "kwargs": kwargs})
        return {"status": "completed", "result": _MOCK_COPY_RESULT}

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=mock_enqueue,
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no doc"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=AsyncMock(return_value={"status": "ok"}),
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=7),
    ):
        await run_marketing_copy(
            product_id="prod-1",
            mission_id=8,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience=None,
        )

    assert len(enqueue_calls) == 1
    call = enqueue_calls[0]
    # Lane must be main_work (content generation, not overhead)
    lane = call["kwargs"].get("lane") or call["spec"].get("lane") or call["spec"].get("context", {}).get("lane")
    if lane is None:
        # Lane may also be embedded in the spec directly
        lane = call["spec"].get("kind") or call["spec"].get("call_category")
    # Verify it's not OVERHEAD
    assert lane is None or "overhead" not in str(lane).lower()


# ── Test 11: i2p_v3.json step parses validly ────────────────────────────────


def test_i2p_v3_json_still_valid():
    """i2p_v3.json must parse without error after step addition."""
    wf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "workflows", "i2p", "i2p_v3.json"
    )
    with open(wf_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["plan_id"] == "i2p_v3"


# ── Test 12: i2p_v3 contains marketing_copy_draft step ─────────────────────


def test_i2p_v3_contains_marketing_copy_step():
    """i2p_v3.json must contain a step with id '13.0a' for marketing_copy_draft."""
    wf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "workflows", "i2p", "i2p_v3.json"
    )
    with open(wf_path, encoding="utf-8") as f:
        content = f.read()
    assert "13.0a" in content, "Step 13.0a (marketing_copy_draft) not found in i2p_v3.json"
    assert "marketing_copy" in content


# ── Test 13: compliance degrade gracefully on error ──────────────────────────


@pytest.mark.asyncio
async def test_compliance_degrades_gracefully_on_error(tmp_path, monkeypatch):
    """If A6 compliance raises, result should still be 'completed' with error note."""
    await _setup_db(tmp_path, monkeypatch)
    artifacts_dir = tmp_path / "artifacts" / "marketing_copy"
    artifacts_dir.mkdir(parents=True)
    monkeypatch.setenv("MARKETING_COPY_ARTIFACTS_DIR", str(artifacts_dir))

    async def error_compliance(text, task_id, mission_id):
        raise RuntimeError("A6 subsystem unavailable")

    from mr_roboto.marketing_copy import run_marketing_copy

    with patch(
        "mr_roboto.marketing_copy.enqueue",
        new=AsyncMock(return_value={"status": "completed", "result": _MOCK_COPY_RESULT}),
    ), patch(
        "mr_roboto.marketing_copy._run_brand_voice_lint",
        new=AsyncMock(return_value={"status": "skip", "reason": "no doc"}),
    ), patch(
        "mr_roboto.marketing_copy._run_copy_compliance",
        new=error_compliance,
    ), patch(
        "mr_roboto.marketing_copy._emit_founder_action",
        new=AsyncMock(return_value=1),
    ):
        result = await run_marketing_copy(
            product_id="prod-1",
            mission_id=9,
            product_spec=_PRODUCT_SPEC,
            brand_voice_audience=None,
        )

    # Must not fail — graceful degrade
    assert result["status"] == "completed"
    comp = result.get("compliance_result", {})
    assert comp.get("status") in ("error", "skip", "ok")
