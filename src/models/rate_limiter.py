# src/models/rate_limiter.py
"""Shim — delegates to kuleden_donen_var package.

All real logic lives in packages/kuleden_donen_var/.
This file preserves import paths during migration.
"""
from kuleden_donen_var.rate_limiter import RateLimitState, RateLimitManager  # noqa: F401
from kuleden_donen_var.header_parser import RateLimitSnapshot  # noqa: F401

# ─── Initial Provider Limits (fallback before header discovery) ──────────────
_INITIAL_PROVIDER_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": 30, "tpm": 131072},
    "gemini": {"rpm": 15, "tpm": 1000000},
    "cerebras": {"rpm": 30, "tpm": 131072},
    "sambanova": {"rpm": 20, "tpm": 100000},
    "openai": {"rpm": 500, "tpm": 2000000},
    "anthropic": {"rpm": 50, "tpm": 80000},
}

PROVIDER_AGGREGATE_LIMITS = _INITIAL_PROVIDER_LIMITS

# ─── Singleton ───────────────────────────────────────────────
_manager: RateLimitManager | None = None


def get_rate_limit_manager() -> RateLimitManager:
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
        _init_from_registry()
    return _manager


def _init_from_registry() -> None:
    """Auto-register all cloud models from the model registry."""
    try:
        from src.models.model_registry import get_registry
        registry = get_registry()
        manager = get_rate_limit_manager()

        for model in registry.cloud_models():
            agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
            manager.register_model(
                litellm_name=model.litellm_name,
                provider=model.provider,
                rpm=model.rate_limit_rpm,
                tpm=model.rate_limit_tpm,
                provider_aggregate_rpm=agg.get("rpm"),
                provider_aggregate_tpm=agg.get("tpm"),
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Rate limit init failed: {e}")
