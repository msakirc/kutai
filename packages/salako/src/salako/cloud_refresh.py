"""Cloud discovery refresh executor.

Periodic salako handler — fired every 6h by the ``cloud_refresh`` internal
cadence. Re-runs fatih_hoca's cloud discovery probe, re-registers any new
models, and refreshes the benchmark match + review artifact.

Boundary: this module legitimately bridges fatih_hoca with the broader
system. fatih_hoca itself does NOT import salako or KDV — those wirings
live here at the bridge layer. KDV mark_provider_enabled hookup is in a
separate task (T18) and adds a small extra step inside _refresh_cloud_subsystem.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("salako.cloud_refresh")


# Map provider name -> env var holding its API key. Single source of truth.
# Keep in sync with src/app/run.py::_env_key_map and src/app/config.py.
_ENV_KEY_MAP = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _read_api_keys() -> dict[str, str]:
    """Re-read env vars at refresh time so rotated keys are picked up."""
    out: dict[str, str] = {}
    for provider, var in _ENV_KEY_MAP.items():
        v = os.getenv(var, "")
        if v:
            out[provider] = v
    return out


async def _refresh_cloud_subsystem() -> dict[str, Any]:
    """Drive a fresh discovery cycle through fatih_hoca, re-register
    discovered models, and re-run bench match.

    Returns a small JSON-serialisable summary of what happened.
    """
    import fatih_hoca
    from fatih_hoca.benchmark_cloud_match import apply_cloud_benchmarks, write_review_artifact

    api_keys = _read_api_keys()
    if not api_keys:
        logger.info("cloud_refresh: no API keys present; nothing to do")
        return {"providers_probed": 0, "providers_ok": 0, "models_registered": 0}

    results = await fatih_hoca._run_cloud_discovery(api_keys=api_keys)
    fatih_hoca.discovery_results = results

    # Re-register discovered models (in-place updates if already present).
    registered = 0
    for provider, result in results.items():
        if not result.auth_ok:
            continue
        for dm in result.models:
            try:
                fatih_hoca.registry.register_cloud_from_discovered(
                    fatih_hoca._registry, provider, dm,
                )
                registered += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("register_cloud_from_discovered failed for %s/%s: %s",
                               provider, getattr(dm, "litellm_name", "?"), e)

    # Update selector's available_providers in place so picks reflect new state.
    if fatih_hoca._selector is not None:
        fatih_hoca._selector._available_providers = {
            p for p, r in results.items() if r.auth_ok
        }

    # Cross-package bridge: tell KDV which providers are currently enabled.
    # Idempotent — first call sets the timestamp, repeats are no-ops.
    try:
        from src.core.router import get_kdv
        kdv = get_kdv()
        for provider, result in results.items():
            if result.auth_ok:
                kdv.mark_provider_enabled(provider)
    except Exception as e:  # noqa: BLE001
        logger.warning("KDV mark_provider_enabled wireup failed: %s", e)

    # Build aa_lookup from current registry, then re-apply bench match.
    from fatih_hoca.cloud.family import normalize as _normalize
    aa_lookup: dict[str, dict[str, float]] = {}
    for m in fatih_hoca._registry.all_models():
        if not m.benchmark_scores:
            continue
        family = m.family or _normalize(m.provider, m.litellm_name or m.name)
        if not family:
            continue
        aa_lookup.setdefault(family, dict(m.benchmark_scores))

    cache_root = Path(".benchmark_cache")
    approved_path = cache_root / "cloud_match_approved.json"
    review_path = cache_root / "cloud_match_review.json"
    try:
        models_list = list(fatih_hoca._registry.all_models())
        apply_cloud_benchmarks(models_list, aa_lookup, approved_path=approved_path)
        write_review_artifact(models_list, aa_lookup, output_path=review_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("bench match refresh failed: %s", e)

    summary = {
        "providers_probed": len(results),
        "providers_ok": sum(1 for r in results.values() if r.auth_ok),
        "models_registered": registered,
    }
    logger.info(
        "cloud_refresh: probed=%d ok=%d registered=%d",
        summary["providers_probed"], summary["providers_ok"], summary["models_registered"],
    )
    return summary


async def run(task: dict) -> dict[str, Any]:
    """Salako mechanical executor entry point.

    Returns the refresh summary. Exceptions propagate to the salako
    dispatcher which converts them into ``Action(status='failed')``.
    """
    return await _refresh_cloud_subsystem()
