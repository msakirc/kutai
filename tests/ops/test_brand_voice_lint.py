"""Tests for Z7 A5 — brand_voice_lint posthook + brand_voice loader.

Coverage:
  - brand_voice loader: parse front-matter, load_brand_voice by slug
  - lint checks: prohibited term flagged, sentence-length drift, FK level,
    pronoun ratio, tone required/forbidden signals (mechanical)
  - missing voice doc: graceful skip
  - missing audience: graceful skip
  - LLM tone pass: mocked
  - founder_action annotation: mocked
  - full handle() integration path
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── Unit tests for brand_voice.py ────────────────────────────────────────


class TestBrandVoiceLoader:
    """Tests for src/ops/brand_voice.py."""

    def _loader(self):
        from src.ops.brand_voice import (
            _parse_brand_voice,
            load_brand_voice,
            load_brand_voice_from_path,
        )
        return _parse_brand_voice, load_brand_voice, load_brand_voice_from_path

    def _sample_md(self) -> str:
        return textwrap.dedent("""\
            ---
            slug: marketing
            display_name: Marketing Voice
            version: "1.0"
            prohibited_terms:
              - "cheap"
              - "guaranteed returns"
              - /(?i)unlimited\\s+free/
            target_avg_sentence_length_words: 18
            flesch_kincaid_reading_level_max: 10
            we_ratio_max: 0.3
            tone_required_signals:
              - confident
              - helpful
            tone_forbidden_signals:
              - fear
              - urgency_artificial
            ---

            ## Voice principles

            Speak as a peer.
        """)

    def test_parse_slug(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert voice.slug == "marketing"

    def test_parse_display_name(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert voice.display_name == "Marketing Voice"

    def test_parse_prohibited_terms(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert "cheap" in voice.prohibited_terms
        assert "guaranteed returns" in voice.prohibited_terms
        assert any("unlimited" in t for t in voice.prohibited_terms)

    def test_parse_sentence_length_target(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert voice.target_avg_sentence_length_words == 18

    def test_parse_fk_max(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert voice.flesch_kincaid_reading_level_max == 10.0

    def test_parse_we_ratio_max(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert voice.we_ratio_max == pytest.approx(0.3, abs=0.01)

    def test_parse_tone_required(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert "confident" in voice.tone_required_signals
        assert "helpful" in voice.tone_required_signals

    def test_parse_tone_forbidden(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert "fear" in voice.tone_forbidden_signals
        assert "urgency_artificial" in voice.tone_forbidden_signals

    def test_parse_body_captured(self):
        parse, _, _ = self._loader()
        voice = parse(self._sample_md())
        assert "Voice principles" in voice.raw_body_md

    def test_load_from_path_missing_returns_default(self, tmp_path):
        _, _, load_from_path = self._loader()
        voice = load_from_path(str(tmp_path / "nonexistent.md"))
        assert voice.slug == ""
        assert voice.prohibited_terms == []

    def test_load_brand_voice_returns_none_for_missing(self, tmp_path):
        _, load, _ = self._loader()
        voice = load("nonexistent_audience_xyz", voices_dir=str(tmp_path))
        assert voice is None

    def test_load_brand_voice_finds_file(self, tmp_path):
        _, load, _ = self._loader()
        md_file = tmp_path / "support.md"
        md_file.write_text(
            "---\nslug: support\ndisplay_name: Support Voice\n---\n",
            encoding="utf-8",
        )
        voice = load("support", voices_dir=str(tmp_path))
        assert voice is not None
        assert voice.slug == "support"

    def test_load_brand_voice_example_fallback(self, tmp_path):
        _, load, _ = self._loader()
        md_file = tmp_path / "investor.example.md"
        md_file.write_text(
            "---\nslug: investor\ndisplay_name: Investor Voice\n---\n",
            encoding="utf-8",
        )
        voice = load("investor", voices_dir=str(tmp_path))
        assert voice is not None
        assert voice.slug == "investor"


# ─── Lint check unit tests ─────────────────────────────────────────────────


class TestLintChecks:
    """Unit tests for individual lint-check functions."""

    def test_prohibited_term_exact_flagged(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_prohibited_terms,
        )
        violations = _check_prohibited_terms("This is cheap stuff.", ["cheap"])
        assert len(violations) == 1
        assert violations[0]["severity"] == "blocker"
        assert violations[0]["check"] == "prohibited_term"

    def test_prohibited_term_not_present_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_prohibited_terms,
        )
        violations = _check_prohibited_terms("This is quality stuff.", ["cheap"])
        assert violations == []

    def test_prohibited_term_regex_flagged(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_prohibited_terms,
        )
        violations = _check_prohibited_terms(
            "Unlimited Free offer!", ["/(?i)unlimited\\s+free/"]
        )
        assert len(violations) == 1
        assert violations[0]["severity"] == "blocker"

    def test_prohibited_term_regex_not_matched_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_prohibited_terms,
        )
        violations = _check_prohibited_terms(
            "Unlimited value offer!", ["/(?i)unlimited\\s+free/"]
        )
        assert violations == []

    def test_sentence_length_within_band_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_sentence_length,
        )
        # avg 18 words, target 18 → OK
        stats = {"avg_sentence_len": 18.0}
        violations = _check_sentence_length(stats, target=18)
        assert violations == []

    def test_sentence_length_over_band_warns(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_sentence_length,
        )
        # 18 * 1.25 = 22.5 → 30 words is > 22.5 → warning
        stats = {"avg_sentence_len": 30.0}
        violations = _check_sentence_length(stats, target=18)
        assert len(violations) == 1
        assert violations[0]["severity"] == "warning"
        assert violations[0]["check"] == "sentence_length"

    def test_sentence_length_under_band_warns(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_sentence_length,
        )
        # 18 * 0.75 = 13.5 → 5 words is < 13.5 → warning
        stats = {"avg_sentence_len": 5.0}
        violations = _check_sentence_length(stats, target=18)
        assert len(violations) == 1
        assert violations[0]["severity"] == "warning"

    def test_fk_within_ceiling_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_flesch_kincaid,
        )
        stats = {"fk_grade": 9.5}
        violations = _check_flesch_kincaid(stats, fk_max=10.0)
        assert violations == []

    def test_fk_exceeds_ceiling_warns(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_flesch_kincaid,
        )
        stats = {"fk_grade": 12.5}
        violations = _check_flesch_kincaid(stats, fk_max=10.0)
        assert len(violations) == 1
        assert violations[0]["severity"] == "warning"
        assert violations[0]["check"] == "flesch_kincaid"

    def test_pronoun_ratio_within_ceiling_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_pronoun_ratio,
        )
        # 2 we, 8 you → ratio = 0.2, ceiling 0.35 (0.3 + 0.05) → OK
        stats = {"we_count": 2, "you_count": 8}
        violations = _check_pronoun_ratio(stats, we_ratio_max=0.3)
        assert violations == []

    def test_pronoun_ratio_exceeds_ceiling_warns(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_pronoun_ratio,
        )
        # 8 we, 2 you → ratio = 0.8, ceiling 0.35 → warning
        stats = {"we_count": 8, "you_count": 2}
        violations = _check_pronoun_ratio(stats, we_ratio_max=0.3)
        assert len(violations) == 1
        assert violations[0]["severity"] == "warning"
        assert violations[0]["check"] == "pronoun_ratio"

    def test_pronoun_ratio_none_skipped(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_pronoun_ratio,
        )
        stats = {"we_count": 50, "you_count": 1}
        violations = _check_pronoun_ratio(stats, we_ratio_max=None)
        assert violations == []

    def test_pronoun_ratio_insufficient_data_skipped(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_pronoun_ratio,
        )
        # Only 3 total pronouns → skip (< 5 threshold)
        stats = {"we_count": 2, "you_count": 1}
        violations = _check_pronoun_ratio(stats, we_ratio_max=0.3)
        assert violations == []

    def test_tone_required_found_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_tone_signals_mechanical,
        )
        text = "We are confident and helpful in our approach."
        violations = _check_tone_signals_mechanical(
            text, required_signals=["confident"], forbidden_signals=[]
        )
        assert all(v["check"] != "tone_required" for v in violations)

    def test_tone_required_missing_in_window_flags(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_tone_signals_mechanical,
        )
        # Short text with no required signal
        text = "The product is available. You can try it. We ship fast."
        violations = _check_tone_signals_mechanical(
            text, required_signals=["confident"], forbidden_signals=[]
        )
        tone_reqs = [v for v in violations if v["check"] == "tone_required"]
        assert len(tone_reqs) >= 1
        assert tone_reqs[0]["severity"] == "blocker"

    def test_tone_forbidden_present_flags_blocker(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_tone_signals_mechanical,
        )
        text = "Act now before fear takes over. This is urgent_artificial."
        violations = _check_tone_signals_mechanical(
            text, required_signals=[], forbidden_signals=["fear"]
        )
        forbidden_hits = [v for v in violations if v["check"] == "tone_forbidden"]
        assert len(forbidden_hits) >= 1
        assert forbidden_hits[0]["severity"] == "blocker"

    def test_tone_forbidden_absent_passes(self):
        from general_beckman.posthook_handlers.brand_voice_lint import (
            _check_tone_signals_mechanical,
        )
        text = "This is a calm, helpful message."
        violations = _check_tone_signals_mechanical(
            text, required_signals=[], forbidden_signals=["fear"]
        )
        assert all(v["check"] != "tone_forbidden" for v in violations)


# ─── handle() integration tests ───────────────────────────────────────────


class TestHandleIntegration:
    """Integration tests for handle() with mocked LLM pass and founder_action."""

    def _make_task(self, audience: str = "marketing", **extra_ctx) -> dict:
        import json

        ctx = {"brand_voice_audience": audience, **extra_ctx}
        return {
            "id": 99,
            "mission_id": 42,
            "context": json.dumps(ctx),
        }

    def _make_result(self, text: str) -> dict:
        return {"result": text}

    def _good_text(self) -> str:
        # 5 sentences, clean language, 'confident' present, no 'fear'
        return (
            "You can build a confident product that your users trust. "
            "We provide clear guidance every step of the way. "
            "The process is helpful and straightforward for you. "
            "Your team will find our approach transparent. "
            "You will see results within the first week."
        )

    @pytest.fixture()
    def marketing_voice_dir(self, tmp_path):
        """Create a temporary brand_voices dir with marketing.md."""
        md = textwrap.dedent("""\
            ---
            slug: marketing
            display_name: Marketing Voice
            version: "1.0"
            prohibited_terms:
              - "cheap"
            target_avg_sentence_length_words: 18
            flesch_kincaid_reading_level_max: 12
            we_ratio_max: 0.5
            tone_required_signals:
              - confident
            tone_forbidden_signals:
              - fear
            ---
        """)
        voices_dir = tmp_path / "brand_voices"
        voices_dir.mkdir()
        (voices_dir / "marketing.md").write_text(md, encoding="utf-8")
        return str(voices_dir)

    @pytest.mark.asyncio
    async def test_missing_audience_returns_skip(self):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = {"id": 1, "mission_id": 10, "context": "{}"}
        result = await brand_voice_lint.handle(task, {"result": "some text"})
        assert result["status"] == "skip"
        assert "audience" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_missing_voice_doc_returns_skip(self, tmp_path):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="nonexistent_voice_zzz")
        with patch(
            "src.ops.brand_voice.load_brand_voice",
            return_value=None,
        ):
            result = await brand_voice_lint.handle(task, self._make_result("text"))
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_empty_artifact_returns_skip(self, marketing_voice_dir):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await brand_voice_lint.handle(task, {"result": ""})
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_prohibited_term_triggers_failed(self, marketing_voice_dir):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        bad_text = "This cheap product is the best on the market. " * 10

        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                brand_voice_lint,
                "_annotate_founder_action",
                new=AsyncMock(return_value=101),
            ),
        ):
            result = await brand_voice_lint.handle(task, self._make_result(bad_text))

        assert result["status"] == "failed"
        assert any(
            v["check"] == "prohibited_term" for v in result.get("violations", [])
        )
        assert result.get("founder_action_id") == 101

    @pytest.mark.asyncio
    async def test_clean_text_returns_ok(self, marketing_voice_dir):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        text = self._good_text()

        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await brand_voice_lint.handle(task, self._make_result(text))

        # Should pass (no 'cheap', 'confident' present, 'fear' absent)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_sentence_length_drift_flagged_as_warning_not_blocker(
        self, marketing_voice_dir
    ):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        # Very short sentences — avg will be ~3 words, far below target 18
        short_text = "Go. Run. Stop. Buy. Now. " * 20 + " confident"

        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                brand_voice_lint,
                "_annotate_founder_action",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await brand_voice_lint.handle(task, self._make_result(short_text))

        violations = result.get("violations", [])
        sl_violations = [v for v in violations if v["check"] == "sentence_length"]
        assert len(sl_violations) >= 1
        assert all(v["severity"] == "warning" for v in sl_violations)

    @pytest.mark.asyncio
    async def test_forbidden_tone_triggers_failed(self, marketing_voice_dir):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        # Contains 'fear' which is forbidden
        fear_text = "You should be afraid of missing out on this. Fear is real. " * 10

        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                brand_voice_lint,
                "_annotate_founder_action",
                new=AsyncMock(return_value=200),
            ),
        ):
            result = await brand_voice_lint.handle(task, self._make_result(fear_text))

        assert result["status"] == "failed"
        forbidden_hits = [
            v for v in result.get("violations", []) if v["check"] == "tone_forbidden"
        ]
        assert len(forbidden_hits) >= 1
        assert forbidden_hits[0]["severity"] == "blocker"

    @pytest.mark.asyncio
    async def test_llm_tone_pass_mocked_info_violation(self, marketing_voice_dir):
        from general_beckman.posthook_handlers import brand_voice_lint

        task = self._make_task(audience="marketing")
        text = self._good_text()

        mock_tone = [
            {
                "severity": "info",
                "check": "tone_score",
                "detail": "Tone score 4/10",
                "excerpt": "some text",
            }
        ]

        with (
            patch(
                "src.ops.brand_voice._default_voices_dir",
                return_value=marketing_voice_dir,
            ),
            patch.object(
                brand_voice_lint,
                "_run_llm_tone_pass",
                new=AsyncMock(return_value=mock_tone),
            ),
        ):
            result = await brand_voice_lint.handle(task, self._make_result(text))

        # Info-only violations → status="ok" (no blockers)
        assert result["status"] == "ok"
        info_violations = [
            v for v in result.get("violations", []) if v.get("check") == "tone_score"
        ]
        assert len(info_violations) == 1

    @pytest.mark.asyncio
    async def test_stub_handle_returns_ok_with_dummy_args(self):
        """Original stub contract: handle({id:1, mission_id:42}, {}) -> {status: ok}."""
        from general_beckman.posthook_handlers import brand_voice_lint

        # With no audience and no LLM import needed, should return skip/ok gracefully
        result = await brand_voice_lint.handle({"id": 1, "mission_id": 42}, {})
        # Empty audience → skip
        assert result.get("status") in ("ok", "skip")


# ─── husam.run migration tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tone_pass_calls_husam_and_parses_score():
    from general_beckman.posthook_handlers import brand_voice_lint as bvl

    async def fake_run(spec):
        return {"content": '{"score": 4, "flagged_sections": ["too salesy"]}'}

    with patch("husam.run", fake_run):
        findings = await bvl._run_llm_tone_pass(
            text="Buy now! Limited offer!",
            voice_display_name="Calm Professional",
            voice_body_md="Measured, factual, never pushy.",
            source_task_id=123,
        )
    assert all(f.get("check") != "tone_pass_skipped" for f in findings)


@pytest.mark.asyncio
async def test_tone_pass_skipped_when_husam_raises():
    from general_beckman.posthook_handlers import brand_voice_lint as bvl

    async def fake_run(spec):
        raise RuntimeError("no model available")

    with patch("husam.run", fake_run):
        findings = await bvl._run_llm_tone_pass(
            text="x", voice_display_name="v", voice_body_md="b", source_task_id=1,
        )
    assert findings and findings[0]["check"] == "tone_pass_skipped"


def test_brand_voice_lint_no_await_inline_in_module():
    import pathlib
    _root = pathlib.Path(__file__).resolve().parents[2]
    src = (_root / "packages" / "general_beckman" / "src" / "general_beckman"
           / "posthook_handlers" / "brand_voice_lint.py").read_text(encoding="utf-8")
    offenders = [ln for ln in src.splitlines()
                 if "await_inline=True" in ln and not ln.lstrip().startswith("#")]
    assert not offenders, f"brand_voice_lint still uses await_inline: {offenders}"


# ─── posthooks registry tests ─────────────────────────────────────────────


class TestPosthooksRegistry:
    """Verify Z7 kinds are registered in POST_HOOK_REGISTRY."""

    def test_z7_kinds_registered(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY, POST_HOOK_KINDS

        z7_kinds = {
            "briefing_compose",
            "brand_voice_lint",
            "copy_compliance_review",
            "audit_completeness_check",
        }
        for kind in z7_kinds:
            assert kind in POST_HOOK_REGISTRY, f"{kind} missing from POST_HOOK_REGISTRY"
            assert kind in POST_HOOK_KINDS, f"{kind} missing from POST_HOOK_KINDS"

    def test_brand_voice_lint_is_blocker(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY

        spec = POST_HOOK_REGISTRY["brand_voice_lint"]
        assert spec.default_severity == "blocker"

    def test_brand_voice_lint_is_moderate_cost(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY

        spec = POST_HOOK_REGISTRY["brand_voice_lint"]
        assert spec.cost_band == "moderate"

    def test_z7_stubs_importable(self):
        from general_beckman.posthook_handlers import (
            briefing_compose,
            brand_voice_lint,
            copy_compliance_review,
            audit_completeness_check,
        )
        for mod in (briefing_compose, brand_voice_lint, copy_compliance_review, audit_completeness_check):
            assert callable(getattr(mod, "handle", None)), (
                f"{mod.__name__} has no callable 'handle'"
            )
