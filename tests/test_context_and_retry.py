"""Tests for Batch 2: context window protection + API retry/backoff."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.workflows.engine.artifacts import (
    estimate_tokens,
    format_artifacts_for_prompt,
    MAX_CONTEXT_INJECTION_TOKENS,
    _truncate,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_string(self):
        assert estimate_tokens("hello") == 1

    def test_typical_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100  # 400 / 4

    def test_long_text(self):
        text = "x" * 48000
        assert estimate_tokens(text) == 12000


class TestContextWindowProtection:
    def test_small_artifacts_unchanged(self):
        """Small artifacts should pass through without truncation."""
        artifacts = {
            "spec": "Short specification text",
            "notes": "Brief notes",
        }
        result = format_artifacts_for_prompt(artifacts)
        assert "Short specification text" in result
        assert "Brief notes" in result

    def test_oversized_artifacts_trimmed(self):
        """Artifacts exceeding the token budget should be trimmed."""
        # Create artifacts that collectively exceed MAX_CONTEXT_INJECTION_TOKENS
        huge = "x" * (MAX_CONTEXT_INJECTION_TOKENS * 5)
        artifacts = {
            "primary_doc": huge,
            "secondary": "Should survive",
            "tertiary": huge,
        }
        result = format_artifacts_for_prompt(artifacts, max_total=999999)
        # Result should be capped
        result_tokens = estimate_tokens(result)
        assert result_tokens <= MAX_CONTEXT_INJECTION_TOKENS + 100  # small tolerance

    def test_tiered_strategy_budgets(self):
        """Context strategy should apply different budgets per tier."""
        artifacts = {
            "design_doc": "A" * 10000,
            "reference": "B" * 5000,
            "extra": "C" * 2000,
        }
        strategy = {
            "primary": ["design_doc"],
            "reference": ["reference"],
            "full_only_if_needed": ["extra"],
        }
        result = format_artifacts_for_prompt(artifacts, context_strategy=strategy)
        # Primary should get 8000 chars, reference 3000, extra 1500
        assert "design_doc" in result
        assert "reference" in result


class TestTruncate:
    def test_short_content_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_content_truncated(self):
        result = _truncate("a" * 200, 50)
        assert len(result) == 50
        assert result.endswith("...")


class TestPhaseSummaryBudget:
    def test_summaries_respect_budget(self):
        """Phase summaries should not exceed 1/4 of injection budget."""
        from src.workflows.engine.artifacts import ArtifactStore, get_phase_summaries

        store = ArtifactStore(use_db=False)

        # Manually populate cache with large summaries
        goal_id = 1
        key = store._goal_key(goal_id)
        store._cache[key] = {}
        # Each summary is large
        for n in range(15):
            store._cache[key][f"phase_{n}_summary"] = "x" * 20000

        summaries = _run(get_phase_summaries(store, goal_id, "phase_15"))
        total_chars = sum(len(v) for v in summaries.values())
        total_tokens = estimate_tokens("".join(summaries.values()))
        # Should be within the summary budget (1/4 of injection budget)
        assert total_tokens <= MAX_CONTEXT_INJECTION_TOKENS // 4 + 10


class TestAPIRetryBackoff:
    def test_retry_on_429(self):
        """HTTP 429 should trigger retry with backoff."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "test_api",
            "base_url": "https://api.test.com",
            "auth_type": "bearer",
            "actions": {
                "get_data": {
                    "method": "GET",
                    "path": "/data",
                    "required_params": [],
                    "max_retries": 2,
                }
            },
        }
        integration = HttpIntegration(config)

        call_count = 0

        async def mock_http(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"status_code": 429, "body": "rate limited", "headers": {}}
            return {"status_code": 200, "body": '{"ok": true}', "headers": {}}

        with patch("src.integrations.http_integration._get_http_func", return_value=mock_http):
            with patch("src.integrations.http_integration.asyncio.sleep", AsyncMock()):
                mock_cred = AsyncMock(return_value={"token": "test"})
                with patch("src.security.credential_store.get_credential", mock_cred):
                    result = _run(integration.execute("get_data", {}))

        assert result["status"] == "ok"
        assert call_count == 3

    def test_no_retry_on_400(self):
        """HTTP 400 (client error) should NOT trigger retry."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "test_api",
            "base_url": "https://api.test.com",
            "auth_type": "bearer",
            "actions": {
                "get_data": {
                    "method": "GET",
                    "path": "/data",
                    "required_params": [],
                    "max_retries": 2,
                }
            },
        }
        integration = HttpIntegration(config)

        call_count = 0

        async def mock_http(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"status_code": 400, "body": "bad request", "headers": {}}

        with patch("src.integrations.http_integration._get_http_func", return_value=mock_http):
            mock_cred = AsyncMock(return_value={"token": "test"})
            with patch("src.security.credential_store.get_credential", mock_cred):
                result = _run(integration.execute("get_data", {}))

        assert result["status"] == "error"
        assert call_count == 1  # no retries

    def test_retry_on_connection_error(self):
        """Connection errors should trigger retry."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "test_api",
            "base_url": "https://api.test.com",
            "auth_type": "none",
            "actions": {
                "get_data": {
                    "method": "GET",
                    "path": "/data",
                    "required_params": [],
                    "max_retries": 1,
                }
            },
        }
        integration = HttpIntegration(config)

        call_count = 0

        async def mock_http(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return {"status_code": 200, "body": '{"ok": true}', "headers": {}}

        with patch("src.integrations.http_integration._get_http_func", return_value=mock_http):
            with patch("src.integrations.http_integration.asyncio.sleep", AsyncMock()):
                mock_cred = AsyncMock(return_value={"token": "test"})
                with patch("src.security.credential_store.get_credential", mock_cred):
                    result = _run(integration.execute("get_data", {}))

        assert result["status"] == "ok"
        assert call_count == 2

    def test_all_retries_exhausted(self):
        """Should return error after all retries are exhausted."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "test_api",
            "base_url": "https://api.test.com",
            "auth_type": "none",
            "actions": {
                "get_data": {
                    "method": "GET",
                    "path": "/data",
                    "required_params": [],
                    "max_retries": 2,
                }
            },
        }
        integration = HttpIntegration(config)

        async def mock_http(*args, **kwargs):
            return {"status_code": 503, "body": "unavailable", "headers": {}}

        with patch("src.integrations.http_integration._get_http_func", return_value=mock_http):
            with patch("src.integrations.http_integration.asyncio.sleep", AsyncMock()):
                mock_cred = AsyncMock(return_value={"token": "test"})
                with patch("src.security.credential_store.get_credential", mock_cred):
                    result = _run(integration.execute("get_data", {}))

        assert result["status"] == "error"
