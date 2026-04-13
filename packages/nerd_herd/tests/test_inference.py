# tests/test_inference.py
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nerd_herd.inference import InferenceCollector

SAMPLE_METRICS_TEXT = """\
# HELP llamacpp_tokens_predicted_total Total predicted tokens
# TYPE llamacpp_tokens_predicted_total counter
llamacpp:tokens_predicted_total 1500
# HELP llamacpp_prompt_tokens_total Total prompt tokens
# TYPE llamacpp_prompt_tokens_total counter
llamacpp:prompt_tokens_total 3000
# HELP llamacpp_tokens_predicted_seconds Predicted tokens per second
# TYPE llamacpp_tokens_predicted_seconds counter
llamacpp:tokens_predicted_seconds 25.5
# HELP llamacpp_prompt_tokens_seconds Prompt tokens per second
# TYPE llamacpp_prompt_tokens_seconds counter
llamacpp:prompt_tokens_seconds 150.3
# HELP llamacpp_requests_processing Current processing requests
# TYPE llamacpp_requests_processing gauge
llamacpp:requests_processing 2
# HELP llamacpp_requests_pending Current pending requests
# TYPE llamacpp_requests_pending gauge
llamacpp:requests_pending 1
# HELP llamacpp_kv_cache_usage_ratio KV cache usage
# TYPE llamacpp_kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.45
"""


@pytest.fixture
def collector():
    return InferenceCollector(
        llama_server_url="http://127.0.0.1:8080",
        poll_interval=5,
    )


def test_collect_when_server_down(collector):
    result = collector.collect()
    assert result["inference_tokens_per_sec"] == 0.0
    assert result["kv_cache_ratio"] == 0.0


def test_parse_metrics(collector):
    collector._parse_and_record(SAMPLE_METRICS_TEXT)
    result = collector.collect()
    assert result["requests_processing"] == 2
    assert result["requests_pending"] == 1
    assert result["kv_cache_ratio"] == 0.45


def test_prometheus_metrics_returns_list(collector):
    metrics = collector.prometheus_metrics()
    assert isinstance(metrics, list)


def test_rate_computation(collector):
    """After two parse calls with different timestamps, rate should be > 0."""
    collector._parse_and_record(SAMPLE_METRICS_TEXT, ts=1000.0)
    # Simulate counter increase
    text2 = SAMPLE_METRICS_TEXT.replace("1500", "1600").replace("3000", "3200")
    collector._parse_and_record(text2, ts=1010.0)
    result = collector.collect()
    # gen rate: (1600-1500) / (1010-1000) = 10.0
    assert result["inference_tokens_per_sec"] == pytest.approx(10.0)
    # prompt rate: (3200-3000) / (1010-1000) = 20.0
    assert result["inference_prompt_tokens_per_sec"] == pytest.approx(20.0)
