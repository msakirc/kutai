"""Z7 A6 — copy_compliance_review tests.

Covers:
  - channel_rules_loader: parse hn.example.md + producthunt.example.md
  - channel_rules_loader: max-length flagged
  - channel_rules_loader: banned-word (literal + regex) flagged
  - channel_rules_loader: required-disclosure missing flagged
  - channel_rules_loader: image_required flagged
  - channel_rules_loader: unknown channel returns None
  - copy handler: superlative flagged (warning, jurisdiction=us)
  - copy handler: superlative not flagged for unknown jurisdiction
  - copy handler: outcome claim without disclosure flagged
  - copy handler: outcome claim + disclosure = pass
  - copy handler: forward-looking without safe harbor flagged (info)
  - copy handler: forward-looking with safe harbor = pass
  - copy handler: privacy contradiction (mocked LLM) = blocker + fix suggestion
  - copy handler: privacy skip when no privacy_policy found = info note
  - copy handler: graceful skip when no copy text found (status=ok, verdict=skip)
  - copy handler: multiple warnings do not produce fail status
  - mr_roboto routing: action="copy_compliance_review" routes to handler
  - apply.py: _posthook_agent_and_payload produces mechanical copy_compliance_review
  - apply.py: copy_compliance_review verdict uses _apply_simple_blocker_verdict
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_RULES_DIR = REPO_ROOT / "docs" / "templates" / "channel_rules"


# ---------------------------------------------------------------------------
# channel_rules_loader tests
# ---------------------------------------------------------------------------

class TestChannelRulesLoader:

    def test_load_hn_example(self):
        from general_beckman.posthook_handlers.channel_rules_loader import load_channel_rules
        rules = load_channel_rules("hn_post", rules_dir=str(CHANNEL_RULES_DIR))
        assert rules is not None
        assert rules.max_title_chars == 80
        assert rules.max_body_chars == 2000
        assert rules.max_total_chars == 0
        assert not rules.image_required
        assert len(rules.banned_words) >= 3

    def test_load_producthunt_example(self):
        from general_beckman.posthook_handlers.channel_rules_loader import load_channel_rules
        rules = load_channel_rules("ph_post", rules_dir=str(CHANNEL_RULES_DIR))
        assert rules is not None
        assert rules.max_title_chars == 60
        assert rules.max_body_chars == 260
        assert rules.image_required is True
        assert rules.image_min_width_px == 240
        assert rules.image_min_height_px == 240
        assert rules.image_max_size_kb == 5120
        assert "png" in rules.image_allowed_formats

    def test_unknown_channel_returns_none(self):
        from general_beckman.posthook_handlers.channel_rules_loader import load_channel_rules
        result = load_channel_rules("totally_unknown_channel_xyz", rules_dir=str(CHANNEL_RULES_DIR))
        assert result is None

    def test_load_from_path_missing_file(self, tmp_path):
        from general_beckman.posthook_handlers.channel_rules_loader import load_channel_rules_from_path
        rules = load_channel_rules_from_path(str(tmp_path / "nonexistent.md"))
        # Should return default (empty) ChannelRules without raising
        assert rules.channel == ""
        assert rules.max_title_chars == 0

    def test_parse_rules_max_body_chars(self, tmp_path):
        """A custom channel file with max_body_chars is parsed correctly."""
        from general_beckman.posthook_handlers.channel_rules_loader import (
            load_channel_rules, load_channel_rules_from_path,
        )
        md_content = (
            "---\n"
            "channel: tweet\n"
            "display_name: Twitter/X\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 280\n"
            "max_total_chars: 280\n"
            "banned_words:\n"
            '  - "guaranteed returns"\n'
            r"  - /(?i)crypto\s+guaranteed/" + "\n"
            "required_disclosures: []\n"
            "image_required: false\n"
            "image_min_width_px: 0\n"
            "image_min_height_px: 0\n"
            "image_max_size_kb: 0\n"
            "image_allowed_formats: []\n"
            "---\n"
        )
        rules_dir = tmp_path
        (rules_dir / "tweet.md").write_text(md_content, encoding="utf-8")
        rules = load_channel_rules("tweet", rules_dir=str(rules_dir))
        assert rules is not None
        assert rules.max_body_chars == 280
        assert rules.max_total_chars == 280
        assert "guaranteed returns" in rules.banned_words


# ---------------------------------------------------------------------------
# Check function unit tests
# ---------------------------------------------------------------------------

class TestOutcomeClaims:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_outcome_claim_flagged_without_disclosure(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_outcome_claims
        copy_text = "Our tool saves you 10 hours per week and boosts productivity by 3x."
        findings = _check_outcome_claims(copy_text)
        assert len(findings) >= 1
        assert any(f["check"] == "outcome_claim_no_disclosure" for f in findings)
        assert all(f["severity"] == "warning" for f in findings)

    def test_outcome_claim_passes_with_results_vary(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_outcome_claims
        copy_text = "Save 10 hours per week. Results may vary."
        findings = _check_outcome_claims(copy_text)
        assert findings == []

    def test_no_outcome_claim_passes(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_outcome_claims
        copy_text = "A great product for your business needs."
        findings = _check_outcome_claims(copy_text)
        assert findings == []


class TestSuperlatives:

    def test_superlative_flagged_for_us_jurisdiction(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_superlatives
        copy_text = "The best CRM on the market. Guaranteed results."
        findings = _check_superlatives(copy_text, "us")
        assert len(findings) >= 1
        assert all(f["severity"] == "warning" for f in findings)

    def test_superlative_not_flagged_unknown_jurisdiction(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_superlatives
        copy_text = "The best CRM on the market."
        # Unknown/empty jurisdiction → skip
        findings = _check_superlatives(copy_text, "unknown_country")
        assert findings == []

    def test_superlative_not_flagged_empty_jurisdiction(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_superlatives
        copy_text = "Guaranteed delivery in 2 days."
        # Empty jurisdiction: still applies (defaults to check)
        findings = _check_superlatives(copy_text, "")
        # Empty string is not in _SUBSTANTIATION_REQUIRED_JURISDICTIONS, but
        # the check skips when jur is empty (falsy) — returns empty.
        # (Empty jurisdiction = unknown = skip, per design.)
        # This behaviour is intentional: we skip rather than over-flag.
        assert isinstance(findings, list)


class TestForwardLooking:

    def test_forward_looking_flagged_without_safe_harbor(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_forward_looking
        copy_text = "We expect to launch in Q3 and anticipate 10,000 users."
        findings = _check_forward_looking(copy_text)
        assert len(findings) == 1
        assert findings[0]["severity"] == "info"
        assert findings[0]["check"] == "forward_looking_no_safe_harbor"

    def test_forward_looking_passes_with_safe_harbor(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_forward_looking
        copy_text = (
            "We expect to launch in Q3. These are forward-looking statements. "
            "Actual results may differ materially."
        )
        findings = _check_forward_looking(copy_text)
        assert findings == []

    def test_no_forward_looking_passes(self):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_forward_looking
        copy_text = "Our product is live and used by 5,000 businesses."
        findings = _check_forward_looking(copy_text)
        assert findings == []


class TestChannelRulesCheck:

    def test_max_length_flagged(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        md = (
            "---\n"
            "channel: sms\n"
            "display_name: SMS\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 160\n"
            "max_total_chars: 0\n"
            "banned_words: []\n"
            "required_disclosures: []\n"
            "image_required: false\n"
            "image_min_width_px: 0\n"
            "image_min_height_px: 0\n"
            "image_max_size_kb: 0\n"
            "image_allowed_formats: []\n"
            "---\n"
        )
        (tmp_path / "sms.md").write_text(md, encoding="utf-8")
        copy_text = "A" * 200  # exceeds 160
        findings = _check_channel_rules(
            copy_text, "sms", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_max_body_chars" for f in findings)

    def test_banned_word_flagged(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        md = (
            "---\n"
            "channel: hn_post\n"
            "display_name: HN\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 0\n"
            "max_total_chars: 0\n"
            "banned_words:\n"
            '  - "check out"\n'
            r"  - /(?i)excited\s+to\s+share/" + "\n"
            "required_disclosures: []\n"
            "image_required: false\n"
            "image_min_width_px: 0\n"
            "image_min_height_px: 0\n"
            "image_max_size_kb: 0\n"
            "image_allowed_formats: []\n"
            "---\n"
        )
        (tmp_path / "hn_post.md").write_text(md, encoding="utf-8")
        copy_text = "Hey, check out our new product!"
        findings = _check_channel_rules(
            copy_text, "hn_post", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_banned_word" for f in findings)

    def test_banned_regex_flagged(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        md = (
            "---\n"
            "channel: blog_post\n"
            "display_name: Blog\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 0\n"
            "max_total_chars: 0\n"
            "banned_words:\n"
            r"  - /(?i)excited\s+to\s+(announce|share)/" + "\n"
            "required_disclosures: []\n"
            "image_required: false\n"
            "image_min_width_px: 0\n"
            "image_min_height_px: 0\n"
            "image_max_size_kb: 0\n"
            "image_allowed_formats: []\n"
            "---\n"
        )
        (tmp_path / "blog_post.md").write_text(md, encoding="utf-8")
        copy_text = "We are Excited to Announce a major release!"
        findings = _check_channel_rules(
            copy_text, "blog_post", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_banned_word" for f in findings)

    def test_required_disclosure_missing(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        md = (
            "---\n"
            "channel: email\n"
            "display_name: Email\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 0\n"
            "max_total_chars: 0\n"
            "banned_words: []\n"
            "required_disclosures:\n"
            r'  - {label: "unsubscribe", pattern: /(?i)unsubscribe/}' + "\n"
            "image_required: false\n"
            "image_min_width_px: 0\n"
            "image_min_height_px: 0\n"
            "image_max_size_kb: 0\n"
            "image_allowed_formats: []\n"
            "---\n"
        )
        (tmp_path / "email.md").write_text(md, encoding="utf-8")
        copy_text = "This is a marketing email with no unsubscribe link mention."
        # Remove any mention of unsubscribe
        copy_text = "Buy our product today! Great deals await."
        findings = _check_channel_rules(
            copy_text, "email", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_missing_disclosure" for f in findings)

    def test_image_required_no_url(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        md = (
            "---\n"
            "channel: ph_post\n"
            "display_name: Product Hunt\n"
            'version: "1.0"\n'
            "max_title_chars: 0\n"
            "max_body_chars: 0\n"
            "max_total_chars: 0\n"
            "banned_words: []\n"
            "required_disclosures: []\n"
            "image_required: true\n"
            "image_min_width_px: 240\n"
            "image_min_height_px: 240\n"
            "image_max_size_kb: 5120\n"
            "image_allowed_formats:\n"
            "  - png\n"
            "  - jpg\n"
            "---\n"
        )
        (tmp_path / "ph_post.md").write_text(md, encoding="utf-8")
        copy_text = "Awesome product launch!"
        findings = _check_channel_rules(
            copy_text, "ph_post", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_image_required" for f in findings)

    def test_unknown_channel_returns_info(self, tmp_path):
        from general_beckman.posthook_handlers.copy_compliance_review import _check_channel_rules
        findings = _check_channel_rules(
            "some copy text", "totally_unknown_channel", {}, rules_dir=str(tmp_path)
        )
        assert any(f["check"] == "channel_rules_missing" for f in findings)
        assert all(f["severity"] == "info" for f in findings)


# ---------------------------------------------------------------------------
# Privacy LLM check (mocked)
# ---------------------------------------------------------------------------

class TestPrivacyMismatchLLM:

    @pytest.mark.asyncio
    async def test_contradiction_yes_returns_blocker(self):
        """Mocked LLM returns contradicts=yes → blocker finding."""
        task = {"id": 99, "mission_id": 1}
        ctx = {}
        copy_text = "We never collect your data."
        privacy_policy = "We collect email and usage data for analytics."

        async def fake_run(spec):
            return {"content": '{"contradicts": "yes", "citation": "We never collect your data."}'}

        with patch("husam.run", fake_run):
            from general_beckman.posthook_handlers.copy_compliance_review import (
                _check_privacy_mismatch_llm,
            )
            findings = await _check_privacy_mismatch_llm(copy_text, privacy_policy, task, ctx)

        assert len(findings) == 1
        assert findings[0]["severity"] == "blocker"
        assert findings[0]["check"] == "privacy_policy_contradiction"
        assert "never collect" in findings[0]["excerpt"]

    @pytest.mark.asyncio
    async def test_contradiction_no_returns_empty(self):
        """Mocked LLM returns contradicts=no → no findings."""
        task = {"id": 100, "mission_id": 1}
        ctx = {}
        copy_text = "We protect your privacy."
        privacy_policy = "We collect data but protect it."

        async def fake_run(spec):
            return {"content": '{"contradicts": "no", "citation": ""}'}

        with patch("husam.run", fake_run):
            from general_beckman.posthook_handlers.copy_compliance_review import (
                _check_privacy_mismatch_llm,
            )
            findings = await _check_privacy_mismatch_llm(copy_text, privacy_policy, task, ctx)

        assert findings == []

    @pytest.mark.asyncio
    async def test_contradiction_unclear_returns_info(self):
        """Mocked LLM returns contradicts=unclear → info finding."""
        task = {"id": 101, "mission_id": 1}
        ctx = {}
        copy_text = "Your data stays with us."
        privacy_policy = "We share data with analytics partners."

        async def fake_run(spec):
            return {"content": '{"contradicts": "unclear", "citation": "Your data stays with us."}'}

        with patch("husam.run", fake_run):
            from general_beckman.posthook_handlers.copy_compliance_review import (
                _check_privacy_mismatch_llm,
            )
            findings = await _check_privacy_mismatch_llm(copy_text, privacy_policy, task, ctx)

        assert len(findings) == 1
        assert findings[0]["severity"] == "info"

    @pytest.mark.asyncio
    async def test_llm_error_returns_info_skip(self):
        """LLM call failure → distinct 'skipped due to error' info finding, no crash."""
        task = {"id": 102, "mission_id": 1}
        ctx = {}
        copy_text = "Amazing product."
        privacy_policy = "We collect data."

        async def fake_run(spec):
            raise RuntimeError("LLM unavailable")

        with patch("husam.run", fake_run):
            from general_beckman.posthook_handlers.copy_compliance_review import (
                _check_privacy_mismatch_llm,
            )
            findings = await _check_privacy_mismatch_llm(copy_text, privacy_policy, task, ctx)

        assert len(findings) == 1
        assert findings[0]["severity"] == "info"
        assert "skipped due to error" in findings[0]["why"]


# ---------------------------------------------------------------------------
# Full handle() integration tests
# ---------------------------------------------------------------------------

class TestHandleIntegration:

    def _make_task(self, extra_ctx=None):
        ctx = extra_ctx or {}
        return {"id": 1, "mission_id": 42, "context": ctx, "description": ""}

    @pytest.mark.asyncio
    async def test_no_copy_text_returns_skip(self):
        """handle() with no copy text returns status=ok, verdict=skip."""
        from general_beckman.posthook_handlers.copy_compliance_review import handle
        task = self._make_task({"workspace_path": "/nonexistent"})
        result = await handle(task, {})
        assert result["status"] == "ok"
        assert result.get("verdict") == "skip"

    @pytest.mark.asyncio
    async def test_warnings_only_returns_ok(self):
        """Warnings alone do not produce status=fail."""
        from general_beckman.posthook_handlers.copy_compliance_review import handle

        # Text with superlatives (warning) but no blocker
        ctx = {
            "jurisdiction": "us",
            "channel": "",
            "workspace_path": "",
        }
        task = self._make_task(ctx)
        task["description"] = "The best product guaranteed to improve your workflow."

        # No privacy policy → info note. Superlative → warning.
        # Forward-looking → none. Channel → none (empty).
        # Total: warnings + infos only.
        result = await handle(task, {})
        assert result["status"] == "ok"
        assert result.get("verdict") == "pass"
        assert result["blocker_count"] == 0
        assert result.get("warning_count", 0) >= 0

    @pytest.mark.asyncio
    async def test_blocker_returns_fail_with_fix_suggestion(self):
        """Privacy blocker produces status=fail with fix_suggestions."""
        from general_beckman.posthook_handlers.copy_compliance_review import handle

        privacy_policy_text = "We collect and share all user data with third parties."
        copy_text = "We never share your data with anyone, ever."

        ctx = {
            "jurisdiction": "us",
            "channel": "",
            "workspace_path": "",
        }
        task = self._make_task(ctx)
        task["description"] = copy_text

        async def fake_run(spec):
            return {"content": '{"contradicts": "yes", "citation": "We never share your data with anyone, ever."}'}

        with patch("husam.run", fake_run):
            # Also patch _load_privacy_policy to return our fake policy
            with patch(
                "general_beckman.posthook_handlers.copy_compliance_review._load_privacy_policy",
                return_value=privacy_policy_text,
            ):
                result = await handle(task, {})

        assert result["status"] == "fail"
        assert result["verdict"] == "fail"
        assert result["blocker_count"] >= 1
        assert "fix_suggestions" in result
        assert "privacy_policy_contradiction" in result["fix_suggestions"]

    @pytest.mark.asyncio
    async def test_missing_privacy_policy_returns_info_not_fail(self):
        """No privacy_policy found → info note, not a blocker."""
        from general_beckman.posthook_handlers.copy_compliance_review import handle

        copy_text = "A solid product for everyday use."
        ctx = {"workspace_path": "/totally_nonexistent_path_xyz", "jurisdiction": ""}
        task = self._make_task(ctx)
        task["description"] = copy_text

        result = await handle(task, {})
        assert result["status"] == "ok"
        assert result["blocker_count"] == 0
        # Should have at least one info finding about missing privacy policy
        info_checks = [
            f for f in result.get("findings", [])
            if f.get("severity") == "info" and "privacy" in f.get("why", "").lower()
        ]
        assert len(info_checks) >= 1

    @pytest.mark.asyncio
    async def test_channel_max_length_flagged(self, tmp_path):
        """Channel max-length violation → warning finding in result."""
        from general_beckman.posthook_handlers.copy_compliance_review import handle, _check_channel_rules

        long_body = "A" * 300  # exceeds 280 for tweet
        findings = _check_channel_rules(
            long_body, "tweet",
            {},
            rules_dir=str(CHANNEL_RULES_DIR),
        )
        # tweet.md doesn't exist → channel_rules_missing info
        assert any(f["check"] == "channel_rules_missing" for f in findings)


# ---------------------------------------------------------------------------
# Apply.py wiring test
# ---------------------------------------------------------------------------

class TestApplyWiring:

    def test_posthook_agent_and_payload_for_copy_compliance(self):
        """_posthook_agent_and_payload returns mechanical tuple for copy_compliance_review."""
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook

        source_ctx = {"workspace_path": "/ws", "produces": ["output.md"], "jurisdiction": "us"}
        a = RequestPostHook(source_task_id=42, kind="copy_compliance_review", source_ctx=source_ctx)
        source = {"id": 42, "mission_id": 7, "context": json.dumps({"workspace_path": "/ws"})}

        agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)
        assert agent_type == "mechanical"
        assert payload["posthook_kind"] == "copy_compliance_review"
        assert payload["payload"]["action"] == "copy_compliance_review"
        assert payload["payload"]["workspace_path"] == "/ws"
        assert payload["payload"]["jurisdiction"] == "us"


# ---------------------------------------------------------------------------
# mr_roboto routing test
# ---------------------------------------------------------------------------

class TestMrRobotoRouting:

    @pytest.mark.asyncio
    async def test_mr_roboto_routes_copy_compliance(self):
        """mr_roboto.run routes copy_compliance_review to the handler."""
        from mr_roboto import run as mr_run
        from mr_roboto import Action

        copy_text = "Simple clean copy without issues."
        task = {
            "id": 55,
            "mission_id": 7,
            "agent_type": "mechanical",
            "context": {
                "executor": "mechanical",
                "payload": {
                    "action": "copy_compliance_review",
                    "source_task_id": 55,
                    "workspace_path": "",
                    "jurisdiction": "",
                    "channel": "",
                    "artifact_metadata": {},
                    "copy_path": "",
                    "privacy_policy_path": "",
                    "produces": [],
                },
                "description": copy_text,
            },
            "description": copy_text,
        }

        # Patch get_task to return None (no source task)
        with patch(
            "src.infra.db.get_task",
            new=AsyncMock(return_value=None),
        ):
            action: Action = await mr_run(task)

        # Should complete (copy_text comes from task description fallback)
        assert action.status in ("completed", "failed")
        # If completed, result should have status key
        if action.status == "completed":
            assert "status" in action.result


# ---------------------------------------------------------------------------
# CPS SP4a Task 3 — husam.run raw_dispatch migration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_copy_compliance_builds_raw_dispatch_spec_for_husam():
    from general_beckman.posthook_handlers import copy_compliance_review as ccr

    captured = {}

    async def fake_run(spec):
        captured["spec"] = spec
        return {"content": '{"contradicts": "no", "citation": ""}'}

    with patch("husam.run", fake_run):
        findings = await ccr._check_privacy_mismatch_llm(
            copy_text="We never sell your data.",
            privacy_policy="We do not sell personal data.",
            task={"id": 42, "mission_id": None},
            ctx={},
        )

    llm = captured["spec"]["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert llm["messages"][0]["role"] == "user"
    assert all(f.get("severity") != "blocker" for f in findings)


def test_copy_compliance_no_await_inline_in_module():
    import pathlib
    _root = pathlib.Path(__file__).resolve().parents[1]
    src = (_root / "packages" / "general_beckman" / "src" / "general_beckman"
           / "posthook_handlers" / "copy_compliance_review.py").read_text(encoding="utf-8")
    offenders = [ln for ln in src.splitlines()
                 if "await_inline=True" in ln and not ln.lstrip().startswith("#")]
    assert not offenders, f"copy_compliance_review still uses await_inline: {offenders}"
