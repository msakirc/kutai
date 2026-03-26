"""Shopping configuration module. Load from YAML if available, with sensible defaults."""

from __future__ import annotations
import os
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "shopping_config.yaml"

# Defaults
_DEFAULTS = {
    "rate_limits": {
        "akakce": {"delay_seconds": 10, "daily_budget": 200},
        "trendyol": {"delay_seconds": 5, "daily_budget": 100},
        "hepsiburada": {"delay_seconds": 15, "daily_budget": 50},
        "amazon_tr": {"delay_seconds": 3, "daily_budget": 500},
        "eksisozluk": {"delay_seconds": 5, "daily_budget": 50},
        "technopat": {"delay_seconds": 5, "daily_budget": 100},
        "donanimhaber": {"delay_seconds": 5, "daily_budget": 100},
        "sikayetvar": {"delay_seconds": 5, "daily_budget": 100},
        "sahibinden": {"delay_seconds": 15, "daily_budget": 30},
        "google_cse": {"delay_seconds": 1, "daily_budget": 100},
        "getir": {"delay_seconds": 5, "daily_budget": 100},
        "migros": {"delay_seconds": 5, "daily_budget": 100},
        "koctas": {"delay_seconds": 5, "daily_budget": 50},
        "ikea": {"delay_seconds": 5, "daily_budget": 50},
    },
    "cache_ttl": {
        "specs": 2592000,      # 30 days
        "prices": 86400,       # 24 hours
        "reviews": 604800,     # 7 days
        "search_results": 43200,  # 12 hours
    },
    "execution": {
        "prefer_remote": False,
        "always_remote": False,
        "local_search_daily_limit": 50,
        "max_searches_per_session": 20,
    },
    "llm": {
        "model": "local",  # placeholder for llama.cpp model
        "temperature": 0.3,
        "max_tokens": 2048,
    },
    "features": {
        "seasonal_analysis": True,
        "substitution_suggestions": True,
        "used_market_check": False,
        "energy_cost_calculation": True,
    },
    "user_defaults": {
        "language": "tr",
        "currency": "TRY",
        "preferred_stores": [],
    },
}


def load_config() -> dict:
    config = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            yaml_config = yaml.safe_load(f) or {}
        _deep_merge(config, yaml_config)
    return config


def _deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# Singleton
_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_rate_limit(domain: str) -> dict:
    return get_config()["rate_limits"].get(domain, {"delay_seconds": 10, "daily_budget": 50})
