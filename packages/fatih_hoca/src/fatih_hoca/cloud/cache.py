"""Disk cache for provider /models responses.

Layout: ``<cache_dir>/<provider>.json``

Schema:
    {
      "fetched_at_unix": <float>,
      "fetched_at_iso": <str>,
      "status": <ProviderStatus>,
      "models": [<DiscoveredModel-as-dict>, ...]
    }

TTL semantics:
    - fresh:    age <= CACHE_TTL_SECONDS (7d)
    - stale:    CACHE_TTL_SECONDS < age <= EVICT_TTL_SECONDS (14d)
    - evicted:  age > EVICT_TTL_SECONDS  -> load() returns None
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.infra.logging_config import get_logger

from .types import DiscoveredModel, ProviderStatus

logger = get_logger("fatih_hoca.cloud.cache")

CACHE_TTL_SECONDS = 7 * 24 * 3600
EVICT_TTL_SECONDS = 14 * 24 * 3600


@dataclass
class CachedSnapshot:
    provider: str
    fetched_at_unix: float
    fetched_at_iso: str
    status: ProviderStatus
    models: list[DiscoveredModel]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at_unix

    @property
    def is_fresh(self) -> bool:
        return self.age_seconds <= CACHE_TTL_SECONDS

    @property
    def is_evicted(self) -> bool:
        return self.age_seconds > EVICT_TTL_SECONDS


def _path(cache_dir: Path, provider: str) -> Path:
    return Path(cache_dir) / f"{provider}.json"


def save(cache_dir: Path, provider: str, models: list[DiscoveredModel], status: ProviderStatus) -> None:
    p = _path(cache_dir, provider)
    p.parent.mkdir(parents=True, exist_ok=True)
    now_unix = time.time()
    payload = {
        "fetched_at_unix": now_unix,
        "fetched_at_iso": datetime.fromtimestamp(now_unix, tz=timezone.utc).isoformat(),
        "status": status,
        "models": [dataclasses.asdict(m) for m in models],
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


def load(cache_dir: Path, provider: str) -> CachedSnapshot | None:
    p = _path(cache_dir, provider)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("cache read failed for %s: %s", provider, e)
        return None
    snap = CachedSnapshot(
        provider=provider,
        fetched_at_unix=float(raw.get("fetched_at_unix", 0.0)),
        fetched_at_iso=str(raw.get("fetched_at_iso", "")),
        status=raw.get("status", "ok"),
        models=[DiscoveredModel(**m) for m in raw.get("models", [])],
    )
    if snap.is_evicted:
        logger.info("cache evicted for %s (age=%.0fs)", provider, snap.age_seconds)
        return None
    return snap
