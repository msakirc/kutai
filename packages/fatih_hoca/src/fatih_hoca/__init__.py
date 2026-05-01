"""Fatih Hoca — model manager: scoring, selection, swap budget."""
from __future__ import annotations

from fatih_hoca.types import Pick, Failure, SwapBudget
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements, AGENT_REQUIREMENTS, CAPABILITY_TO_TASK
from fatih_hoca.capabilities import Cap, ALL_CAPABILITIES, TASK_PROFILES
from fatih_hoca.ranking import ScoredModel
from fatih_hoca.selector import Selector

__all__ = [
    "init", "select", "all_models",
    "Pick", "Failure", "ModelInfo", "ModelRequirements", "ScoredModel",
    "AGENT_REQUIREMENTS", "CAPABILITY_TO_TASK",
    "Cap", "ALL_CAPABILITIES", "TASK_PROFILES",
    "discovery_results",
]

_selector: Selector | None = None
_registry: ModelRegistry | None = None

# Populated by init() — boot caller reads to wire KDV etc.
discovery_results: dict = {}

_ADAPTERS = None  # lazily built on first init()


def _build_adapters() -> dict[str, object]:
    """Lazy import so test files that monkeypatch _run_cloud_discovery don't pay HTTP-adapter import cost."""
    from .cloud.providers.groq import GroqAdapter
    from .cloud.providers.openai import OpenAIAdapter
    from .cloud.providers.anthropic import AnthropicAdapter
    from .cloud.providers.gemini import GeminiAdapter
    from .cloud.providers.cerebras import CerebrasAdapter
    from .cloud.providers.sambanova import SambanovaAdapter
    from .cloud.providers.openrouter import OpenRouterAdapter
    return {
        "groq": GroqAdapter(),
        "openai": OpenAIAdapter(),
        "anthropic": AnthropicAdapter(),
        "gemini": GeminiAdapter(),
        "cerebras": CerebrasAdapter(),
        "sambanova": SambanovaAdapter(),
        "openrouter": OpenRouterAdapter(),
    }


async def _run_cloud_discovery(api_keys: dict[str, str]) -> dict:
    """Module-level seam: tests monkeypatch this. Real impl below builds the
    discovery + throttle + adapters and returns the results map."""
    return await _run_cloud_discovery_impl(
        api_keys=api_keys,
        cache_dir=_default_cache_dir,
        alert_state_path=_default_alert_state_path,
        user_alert_fn=_default_alert_fn,
    )


# These three are set at the top of init() so _run_cloud_discovery (the seam)
# can read them. Pattern keeps the seam signature small (only api_keys) so
# tests can monkeypatch with a one-arg async function.
_default_cache_dir: str = ".benchmark_cache/cloud_models"
_default_alert_state_path: str = ".benchmark_cache/cloud_alert_throttle.json"
_default_alert_fn = None  # type: ignore[assignment]


async def _run_cloud_discovery_impl(
    api_keys: dict[str, str],
    cache_dir: str,
    alert_state_path: str,
    user_alert_fn,
) -> dict:
    from pathlib import Path as _Path
    from .cloud.discovery import CloudDiscovery
    from .cloud.alert_throttle import AlertThrottle

    global _ADAPTERS
    if _ADAPTERS is None:
        _ADAPTERS = _build_adapters()

    throttle = AlertThrottle(_Path(alert_state_path))

    def _alert(provider: str, status: str, error):
        if not throttle.should_alert(provider, current_state=status):
            return
        if user_alert_fn is None:
            return
        try:
            user_alert_fn(provider, status, error)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("user_alert_fn raised: %s", e)

    discovery = CloudDiscovery(
        adapters=_ADAPTERS,
        cache_dir=_Path(cache_dir),
        alert_fn=_alert,
    )
    return await discovery.refresh_all(api_keys=api_keys)


def _run_async(coro):
    """Run an async coroutine from sync context.

    When an event loop is already running (e.g. pytest-asyncio, or called
    from an async context via a sync shim), we spin up a NEW loop in a
    dedicated thread so we don't deadlock the running loop.
    """
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside a running loop — run in a fresh thread with its own loop.
        result_holder = [None]
        exception_holder = [None]

        def _thread_target():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result_holder[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                exception_holder[0] = e
            finally:
                new_loop.close()

        t = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = t.submit(_thread_target)
        future.result()  # blocks current (non-loop) thread until done
        t.shutdown(wait=False)
        if exception_holder[0] is not None:
            raise exception_holder[0]
        return result_holder[0]
    else:
        # No running loop — use asyncio.run() which creates a new loop.
        return asyncio.run(coro)


def init(
    models_dir: str | None = None,
    catalog_path: str | None = None,
    nerd_herd: object = None,
    available_providers: set[str] | None = None,
    api_keys: dict[str, str] | None = None,
    cloud_cache_dir: str = ".benchmark_cache/cloud_models",
    cloud_alert_state_path: str = ".benchmark_cache/cloud_alert_throttle.json",
    alert_fn=None,
) -> list[str]:
    """Initialize the Fatih Hoca model registry, selector, and run cloud discovery.

    Parameters
    ----------
    models_dir : str, optional
        Directory to scan for GGUF files (local models).
    catalog_path : str, optional
        Path to a YAML model catalog (cloud + YAML-declared local models).
    nerd_herd : object, optional
        Nerd Herd instance providing system snapshots. If None, a no-op
        stub is used (snapshot() returns an empty SystemSnapshot).
    available_providers : set[str], optional
        Set of cloud provider names that have API keys configured.
        Cloud models whose provider is not in this set are filtered out.
        If None, all cloud models are eligible (no API key check).
    api_keys : dict[str, str], optional
        Mapping of {provider_name: api_key}. When provided, run discovery
        probe per provider; auth_ok=True providers join the selector's
        available set.
    cloud_cache_dir : str
        Directory for cloud model discovery cache files.
    cloud_alert_state_path : str
        Path for the alert throttle state JSON file.
    alert_fn : callable, optional
        callable(provider: str, status: str, error: str | None). Called when
        discovery confirms a provider failure (after throttle check). Fatih
        Hoca does NOT import telegram or salako — boot caller bridges here.

    Returns
    -------
    list[str]
        Names of all models registered.
    """
    global _selector, _registry, discovery_results
    global _default_cache_dir, _default_alert_state_path, _default_alert_fn

    if nerd_herd is None:
        from nerd_herd.types import SystemSnapshot

        class _NoopNerdHerd:
            def snapshot(self) -> SystemSnapshot:
                return SystemSnapshot()

        nerd_herd = _NoopNerdHerd()

    _registry = ModelRegistry()
    model_names: list[str] = []

    if catalog_path:
        models = _registry.load_yaml(catalog_path)
        model_names.extend(m.name for m in models)

    if models_dir:
        models = _registry.load_gguf_dir(models_dir)
        model_names.extend(m.name for m in models)

    # Load persisted speed measurements + demoted flags into model objects.
    # Without this, all models default to 10 tok/s → timeouts are too short.
    _registry._load_speed_cache()

    # ── Cloud discovery ──────────────────────────────────────────────────────
    discovery_results = {}
    if api_keys:
        # Set module-level defaults so _run_cloud_discovery seam can find them.
        _default_cache_dir = cloud_cache_dir
        _default_alert_state_path = cloud_alert_state_path
        _default_alert_fn = alert_fn
        try:
            discovery_results = _run_async(_run_cloud_discovery(api_keys))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("cloud discovery failed at init: %s", e)
            discovery_results = {}

        # Fire alert_fn for providers with auth_ok=False (if caller provided one).
        if alert_fn is not None:
            for provider, result in discovery_results.items():
                if not result.auth_ok:
                    try:
                        alert_fn(provider, result.status, result.error)
                    except Exception:
                        pass

        # Register discovered cloud models BEFORE benchmark enrichment so
        # they participate in the AA match step.
        from .registry import register_cloud_from_discovered
        for provider, result in discovery_results.items():
            if not result.auth_ok:
                continue
            for dm in result.models:
                try:
                    register_cloud_from_discovered(_registry, provider, dm)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "register_cloud_from_discovered failed for %s/%s: %s",
                        provider, getattr(dm, "litellm_name", "?"), e,
                    )

        # ── Cross-validate yaml-loaded cloud models against discovery ──
        # Yaml entries (models.yaml) load BEFORE discovery and stay in the
        # registry even if the provider has retired the id (Gemini retires
        # *-preview-MM-DD slugs, models.yaml/gemini-flash-thinking ↔
        # gemini/gemini-2.5-flash-preview-05-20 was the production
        # culprit). Without cross-validation, selector ranks the dead id
        # via cached benchmark scores, every call 404s.
        #
        # For each provider whose discovery succeeded (auth_ok), build the
        # set of live litellm_names. Any registered cloud model on that
        # provider whose litellm_name is missing from the live set gets
        # mark_dead'd here — keeps the entry in the registry so other
        # subsystems that lookup-by-name don't break, but the eligibility
        # filter excludes it.
        #
        # Skipped when discovery failed (auth_fail / network / rate_limit)
        # — empty live set on a transient outage would mark every model
        # dead. Better to keep candidates and let the runtime 404 hook
        # mark_dead on actual call failure.
        import logging as _logging
        _log = _logging.getLogger(__name__)
        for provider, result in discovery_results.items():
            if not result.auth_ok or result.status != "ok":
                continue
            live_litellm = {dm.litellm_name for dm in result.models}
            stale = []
            for m in _registry.all_models():
                if m.is_local or m.provider != provider:
                    continue
                if m.litellm_name not in live_litellm:
                    stale.append(m.litellm_name)
            for ln in stale:
                _registry.mark_dead(ln)
            if stale:
                _log.warning(
                    "discovery cross-check: %d stale yaml entries on %s "
                    "(retired/typo) — marked dead: %s",
                    len(stale), provider, ", ".join(stale[:5]),
                )
            # Symmetric path: revive any persisted-dead id that has come
            # back in this provider's live set. Without this, a model
            # marked dead in a prior session stays dead forever even if
            # the provider restored it (OR re-routes a model to a new
            # upstream, Gemini publishes a previously-retired slug, etc.)
            revived = []
            for ln in list(live_litellm):
                if _registry.is_dead(ln):
                    _registry.revive(ln)
                    revived.append(ln)
            if revived:
                _log.info(
                    "discovery cross-check: %d previously-dead ids "
                    "revived on %s: %s",
                    len(revived), provider, ", ".join(revived[:5]),
                )

    # ── Benchmark enrichment: populate ModelInfo.benchmark_scores from cached AA data ──
    import logging
    logger = logging.getLogger(__name__)
    try:
        from src.models.benchmark.benchmark_fetcher import enrich_registry_with_benchmarks

        enrich_registry_with_benchmarks(_registry)
        all_models_list = _registry.all_models()
        matched = sum(1 for m in all_models_list if m.benchmark_scores)
        total = len(all_models_list)
        unmatched = [m.name for m in all_models_list if not m.benchmark_scores]

        logger.info(
            "benchmark coverage: %d/%d matched (unmatched=%d)",
            matched, total, len(unmatched),
        )
        if unmatched:
            logger.warning(
                "benchmark coverage: %d unmatched models — %s",
                len(unmatched), ", ".join(unmatched[:10]),
            )
    except Exception as e:
        logger.warning("benchmark enrichment failed at init: %s", e)

    # ── Blend profile + benchmark into final capabilities vector ──
    try:
        from src.models.auto_tuner import blend_capability_scores

        for m in _registry.all_models():
            if not m.benchmark_scores:
                continue
            blended = blend_capability_scores(
                profile_scores=dict(m.capabilities),
                benchmark_scores=dict(m.benchmark_scores),
                grading_scores={},
                grading_call_count=0,
            )
            m.capabilities = blended
    except Exception as e:
        logger.warning("capability blending failed at init: %s", e)

    # ── Cloud bench match (family-aware, gated by operator approval) ──
    try:
        from .benchmark_cloud_match import apply_cloud_benchmarks, write_review_artifact

        # Build {family: {capability: score}} lookup from per-model benchmark_scores
        # populated by the local enricher above. Cross-provider clones share via family.
        from .cloud.family import normalize as _family_normalize
        aa_lookup: dict[str, dict[str, float]] = {}
        for m in _registry.all_models():
            if not m.benchmark_scores:
                continue
            family = m.family or _family_normalize(m.provider, m.litellm_name or m.name)
            if not family:
                continue
            aa_lookup.setdefault(family, dict(m.benchmark_scores))

        # Resolve approval + review artifact paths inside the cloud cache dir.
        from pathlib import Path as _Path
        cache_root = _Path(cloud_cache_dir) if cloud_cache_dir else _Path(".benchmark_cache")
        approved_path = cache_root / "cloud_match_approved.json"
        review_path = cache_root / "cloud_match_review.json"

        models_list = list(_registry.all_models())
        apply_cloud_benchmarks(models_list, aa_lookup, approved_path=approved_path)
        write_review_artifact(models_list, aa_lookup, output_path=review_path)
    except Exception as e:
        logger.warning("cloud benchmark match failed at init: %s", e)

    # ── Compute final available_providers ────────────────────────────────────
    # Caller-provided set is the universe ("I have keys for these"). When
    # discovery ran, intersect with the auth_ok subset. When discovery did
    # not run (no api_keys passed), trust the caller-provided set as-is.
    if discovery_results:
        auth_ok_providers = {p for p, r in discovery_results.items() if r.auth_ok}
        if available_providers is not None:
            final_providers = available_providers & auth_ok_providers
        else:
            final_providers = auth_ok_providers
    else:
        final_providers = available_providers

    _selector = Selector(
        registry=_registry,
        nerd_herd=nerd_herd,
        available_providers=final_providers,
    )
    return model_names


def select(**kwargs) -> Pick | None:
    """
    Select the best model for a task.

    Keyword arguments are forwarded to Selector.select(). Returns a Pick
    with the chosen model and estimated min_time_seconds, or None if no
    eligible model was found or if init() has not been called.

    See Selector.select() for the full list of keyword arguments.
    """
    if _selector is None:
        return None
    return _selector.select(**kwargs)


def all_models() -> list[ModelInfo]:
    """
    Return all models currently in the registry.

    Returns an empty list if init() has not been called.
    """
    if _registry is None:
        return []
    return _registry.all_models()
