"""Prometheus /metrics parser for llama-server."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Point-in-time performance snapshot from llama-server /metrics."""

    generation_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    kv_cache_usage_percent: float = 0.0
    requests_processing: int = 0
    requests_pending: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0


class MetricsParser:
    """Fetches and parses llama-server's Prometheus-format /metrics endpoint."""

    async def fetch(self, api_base: str) -> MetricsSnapshot:
        """Fetch metrics from *api_base*/metrics.

        Returns a zero-valued MetricsSnapshot on any failure (connection error,
        non-200 status, parse error) — never raises.
        """
        snap = MetricsSnapshot()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{api_base}/metrics", timeout=3.0)
                if resp.status_code != 200:
                    return snap
                self._parse_into(resp.text, snap)
        except Exception as exc:
            logger.debug("Failed to fetch llama-server metrics: %s", exc)
        return snap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_into(text: str, snap: MetricsSnapshot) -> None:
        """Parse Prometheus plain-text format into *snap* in-place."""
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # Strip label block: metric_name{label="v"} → metric_name
            raw_metric = parts[0].split("{")[0]

            try:
                val = float(parts[-1])
            except ValueError:
                continue

            # Normalize: llama.cpp uses both `llamacpp:foo` and `llamacpp_foo`
            m = raw_metric.replace(":", "_")

            if m == "llamacpp_tokens_predicted_total":
                snap.generation_tokens_total = int(val)
            elif m == "llamacpp_prompt_tokens_total":
                snap.prompt_tokens_total = int(val)
            elif m == "llamacpp_tokens_predicted_seconds":
                snap.generation_tokens_per_second = round(val, 1)
            elif m == "llamacpp_prompt_tokens_seconds":
                snap.prompt_tokens_per_second = round(val, 1)
            elif m == "llamacpp_requests_processing":
                snap.requests_processing = int(val)
            elif m == "llamacpp_requests_pending":
                snap.requests_pending = int(val)
            elif m == "llamacpp_kv_cache_usage_ratio":
                snap.kv_cache_usage_percent = round(val * 100, 1)
