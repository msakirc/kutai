"""Tests for MetricsParser — Prometheus /metrics parsing."""
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from dallama.metrics import MetricsParser, MetricsSnapshot

SAMPLE_METRICS_UNDERSCORE = """
# HELP llamacpp_tokens_predicted_total Total predicted tokens
# TYPE llamacpp_tokens_predicted_total counter
llamacpp_tokens_predicted_total 1234
llamacpp_prompt_tokens_total 5678
llamacpp_tokens_predicted_seconds_total 98.5
llamacpp_prompt_seconds_total 12.3
llamacpp_tokens_predicted_seconds 12.5
llamacpp_prompt_tokens_seconds 461.8
llamacpp_requests_processing 1
llamacpp_requests_pending 0
llamacpp_kv_cache_usage_ratio 0.42
""".strip()

SAMPLE_METRICS_COLON = SAMPLE_METRICS_UNDERSCORE.replace("llamacpp_", "llamacpp:")

@pytest.fixture
def parser():
    return MetricsParser()

@pytest.mark.asyncio
async def test_parse_underscore_format(parser):
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_METRICS_UNDERSCORE
    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        snap = await parser.fetch("http://127.0.0.1:8080")
    assert snap.generation_tokens_per_second == 12.5
    assert snap.prompt_tokens_per_second == 461.8
    assert snap.kv_cache_usage_percent == 42.0
    assert snap.requests_processing == 1
    assert snap.requests_pending == 0
    assert snap.generation_tokens_total == 1234
    assert snap.prompt_tokens_total == 5678

@pytest.mark.asyncio
async def test_parse_colon_format(parser):
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_METRICS_COLON
    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        snap = await parser.fetch("http://127.0.0.1:8080")
    assert snap.generation_tokens_per_second == 12.5

@pytest.mark.asyncio
async def test_fetch_failure_returns_empty(parser):
    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.side_effect = httpx.ConnectError("refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        snap = await parser.fetch("http://127.0.0.1:8080")
    assert snap.generation_tokens_per_second == 0.0
    assert snap.generation_tokens_total == 0

@pytest.mark.asyncio
async def test_fetch_non_200_returns_empty(parser):
    mock_resp = AsyncMock()
    mock_resp.status_code = 503
    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance
        snap = await parser.fetch("http://127.0.0.1:8080")
    assert snap.generation_tokens_per_second == 0.0
