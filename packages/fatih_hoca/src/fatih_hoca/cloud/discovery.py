"""Boot-time and periodic cloud provider discovery orchestrator.

Calls each provider adapter concurrently, writes results to disk cache,
diffs against previous snapshot (logs adds/removes), invokes per-provider
alert callback on auth_fail / non-recoverable errors.

The orchestrator does NOT touch the registry, KDV, or Telegram. The boot
caller consumes the returned results map and wires those subsystems.
``alert_fn`` is injected so this module stays free of telegram/salako
dependencies.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Mapping

from src.infra.logging_config import get_logger

from . import cache as cache_mod
from .types import ProviderResult

logger = get_logger("fatih_hoca.cloud.discovery")

AlertFn = Callable[[str, str, "str | None"], None]


class CloudDiscovery:
    def __init__(
        self,
        adapters: Mapping[str, object],  # name -> adapter with .fetch_models()
        cache_dir: Path,
        alert_fn: AlertFn,
    ) -> None:
        self._adapters = dict(adapters)
        self._cache_dir = Path(cache_dir)
        self._alert_fn = alert_fn

    async def refresh_all(self, api_keys: dict[str, str]) -> dict[str, ProviderResult]:
        """Probe every adapter we have a key for. Returns {provider: result}.

        Provider missing from api_keys is silently skipped (key truly absent).
        """
        targets = [(name, adapter, api_keys[name])
                   for name, adapter in self._adapters.items() if api_keys.get(name)]
        if not targets:
            return {}
        coros = [self._probe_one(name, adapter, key) for name, adapter, key in targets]
        results_list = await asyncio.gather(*coros, return_exceptions=False)
        return {r.provider: r for r in results_list}

    async def _probe_one(self, name: str, adapter, api_key: str) -> ProviderResult:
        try:
            live = await adapter.fetch_models(api_key)
        except Exception as e:  # noqa: BLE001 — adapters must not raise, but defend
            logger.error("adapter %s raised: %s", name, e)
            live = ProviderResult(provider=name, status="server_error",
                                  auth_ok=False, error=f"adapter exception: {e}")

        prior = cache_mod.load(self._cache_dir, name)

        # Successful fetch: persist and diff.
        if live.status == "ok":
            self._log_diff(name,
                           prior_models=[m.litellm_name for m in (prior.models if prior else [])],
                           new_models=[m.litellm_name for m in live.models])
            cache_mod.save(self._cache_dir, name, live.models, status="ok")
            return live

        # Rate-limited probe: still ok auth-wise, do not overwrite cache, do not alert.
        if live.status == "rate_limited":
            logger.warning("provider %s rate-limited at /models probe; keeping cache", name)
            if prior is not None:
                live.models = prior.models
                live.served_from_cache = True
            return live

        # Failure path. Try cache.
        if prior is not None and prior.is_fresh:
            logger.warning("provider %s probe failed (%s); serving fresh cache (age=%.0fs)",
                           name, live.status, prior.age_seconds)
            return ProviderResult(
                provider=name, status=live.status, auth_ok=True,
                models=prior.models, error=live.error, served_from_cache=True,
                fetched_at=prior.fetched_at_iso,
            )
        if prior is not None and not prior.is_fresh and not prior.is_evicted:
            logger.warning("provider %s probe failed (%s); serving STALE cache (age=%.0fs)",
                           name, live.status, prior.age_seconds)
            self._alert_fn(name, live.status, live.error)
            return ProviderResult(
                provider=name, status=live.status, auth_ok=False,
                models=prior.models, error=live.error, served_from_cache=True,
                fetched_at=prior.fetched_at_iso,
            )

        # No fresh, no stale. Provider goes dark.
        logger.error("provider %s unreachable and no cache: %s %s",
                     name, live.status, live.error)
        self._alert_fn(name, live.status, live.error)
        return live

    def _log_diff(self, provider: str, prior_models: list[str], new_models: list[str]) -> None:
        prior_set = set(prior_models)
        new_set = set(new_models)
        added = sorted(new_set - prior_set)
        removed = sorted(prior_set - new_set)
        if added or removed:
            logger.info("provider %s diff: added=%s removed=%s", provider, added, removed)
        else:
            logger.debug("provider %s: no model-list changes (%d models)", provider, len(new_set))
