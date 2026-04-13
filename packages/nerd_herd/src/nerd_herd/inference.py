"""Inference collector — fetches llama-server /metrics, pre-computes rates."""
from __future__ import annotations

import asyncio
import time

from prometheus_client import Gauge

from yazbunu import get_logger

from nerd_herd.ring_buffer import RingBuffer

logger = get_logger("nerd_herd.inference")

_g_gen_tps = Gauge("nerd_herd_inference_tokens_per_sec", "Generation tokens/sec (1m avg)")
_g_prompt_tps = Gauge("nerd_herd_inference_prompt_tokens_per_sec", "Prompt tokens/sec (1m avg)")
_g_kv = Gauge("nerd_herd_inference_kv_cache_ratio", "KV cache usage ratio")
_g_proc = Gauge("nerd_herd_inference_requests_processing", "Requests currently processing")
_g_pend = Gauge("nerd_herd_inference_requests_pending", "Requests pending in queue")

_ALL_GAUGES = [_g_gen_tps, _g_prompt_tps, _g_kv, _g_proc, _g_pend]


class InferenceCollector:
    """Fetches llama-server Prometheus metrics and pre-computes rates."""

    name = "inference"

    def __init__(
        self,
        llama_server_url: str = "http://127.0.0.1:8080",
        poll_interval: int = 5,
        ring_capacity: int = 60,
    ) -> None:
        self._url = llama_server_url.rstrip("/")
        self._poll_interval = poll_interval
        self._poll_task: asyncio.Task | None = None

        self._gen_tokens_buf = RingBuffer(capacity=ring_capacity)
        self._prompt_tokens_buf = RingBuffer(capacity=ring_capacity)

        self._kv_cache: float = 0.0
        self._requests_processing: int = 0
        self._requests_pending: int = 0

    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        logger.info("Inference collector started", url=self._url, interval=self._poll_interval)
        while True:
            try:
                await self._fetch_and_record()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Inference poll error", error=str(e))
            await asyncio.sleep(self._poll_interval)

    async def _fetch_and_record(self) -> None:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status != 200:
                        return
                    text = await resp.text()
                    self._parse_and_record(text)
        except Exception as e:
            logger.debug("llama-server metrics fetch failed", error=str(e))

    def _parse_and_record(self, text: str, ts: float | None = None) -> None:
        if ts is None:
            ts = time.time()

        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            raw = parts[0].split("{")[0].replace(":", "_")
            try:
                val = float(parts[-1])
            except ValueError:
                continue

            if raw == "llamacpp_tokens_predicted_total":
                self._gen_tokens_buf.append(ts, val)
            elif raw == "llamacpp_prompt_tokens_total":
                self._prompt_tokens_buf.append(ts, val)
            elif raw == "llamacpp_requests_processing":
                self._requests_processing = int(val)
            elif raw == "llamacpp_requests_pending":
                self._requests_pending = int(val)
            elif raw == "llamacpp_kv_cache_usage_ratio":
                self._kv_cache = round(val, 4)

    def collect(self) -> dict[str, float | int]:
        return {
            "inference_tokens_per_sec": round(self._gen_tokens_buf.rate(60), 1),
            "inference_prompt_tokens_per_sec": round(self._prompt_tokens_buf.rate(60), 1),
            "kv_cache_ratio": self._kv_cache,
            "requests_processing": self._requests_processing,
            "requests_pending": self._requests_pending,
        }

    def prometheus_metrics(self) -> list:
        data = self.collect()
        _g_gen_tps.set(data["inference_tokens_per_sec"])
        _g_prompt_tps.set(data["inference_prompt_tokens_per_sec"])
        _g_kv.set(data["kv_cache_ratio"])
        _g_proc.set(data["requests_processing"])
        _g_pend.set(data["requests_pending"])
        return list(_ALL_GAUGES)
