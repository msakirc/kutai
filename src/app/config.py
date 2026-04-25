# config.py
"""
Central configuration — API keys, constants, environment.
Model pool logic has moved to model_registry.py.
"""

import os

# Load .env at module import. Any process that imports anything from
# src.app.config (which is essentially everything in this repo) gets the
# .env file resolved BEFORE DB_PATH or any other os.getenv is read. This
# is the safety net for standalone scripts (one-off CLIs, reseed scripts,
# tests run outside the wrapper). Previously a script invoked via
# `python -c "from src.infra.db import get_db"` saw os.getenv("DB_PATH")
# return None and silently forked to the data/kutai.db default — while
# the wrapper-launched KutAI loaded .env and used the prod path. Result:
# the script wrote to one DB, prod read from another, edits ghosted.
# (Caught 2026-04-25 during a prompt_versions reseed that hit the wrong
# DB and left prod DB untouched.) load_dotenv is a no-op when env vars
# are already set, so the wrapper's existing load is unaffected.
try:
    from dotenv import load_dotenv as _load_dotenv
    _ENV_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    _load_dotenv(_ENV_PATH, override=False)
except ImportError:  # python-dotenv not installed in this venv
    pass

from src.infra.logging_config import get_logger
from src.models.model_registry import get_registry

logger = get_logger("app.config")

# ─── Core Settings ───────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")

# ─── Core Settings ───────────────────────────────────────────────────────────

# DB_PATH must be absolute so that any process — tests, CLIs, subprocesses that
# don't load .env — all resolve to the same database. A relative default would
# silently fork the DB into the caller's cwd. The load_dotenv() call at the
# top of this module is what makes the .env override actually fire here for
# standalone scripts; without it the default below kicked in instead.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.getenv("DB_PATH") or os.path.join(_PROJECT_ROOT, "data", "kutai.db")
if not os.path.isabs(DB_PATH):
    raise RuntimeError(
        f"DB_PATH must be absolute to prevent forked databases; got {DB_PATH!r}"
    )

# Canonical workspace root — project-root/workspace/.  All modules that
# read or write mission output MUST derive from this (scaffolder,
# orchestrator, tools/shell, tools/git_ops, tools/workspace).  Gitignored
# so mission output never pollutes the repo.
WORKSPACE_ROOT = os.environ.get(
    "WORKSPACE_DIR",
    os.path.join(_PROJECT_ROOT, "workspace"),
)
DOCKER_CONTAINER_NAME = "orchestrator-sandbox"
MODEL_DIR = os.getenv("MODEL_DIR", "")

# Global upper bound for agent iterations.  Individual agents override with
# lower values; see each agent class for per-agent rationale comments.
MAX_AGENT_ITERATIONS = 8
MAX_TOOL_OUTPUT_LENGTH = 4000
MAX_CONTEXT_CHAIN_LENGTH = 12000

MAX_CONCURRENT_MISSIONS = int(os.getenv("MAX_CONCURRENT_GOALS", "3"))
PROJECTS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "projects.json"
)

COST_BUDGET_DAILY: float = float(os.getenv("COST_BUDGET_DAILY", "1.0"))

# ─── API Key Detection ───────────────────────────────────────────────────────

AVAILABLE_KEYS: dict[str, bool] = {
    "groq":      bool(os.getenv("GROQ_API_KEY", "")),
    "openai":    bool(os.getenv("OPENAI_API_KEY", "")),
    "anthropic": bool(os.getenv("ANTHROPIC_API_KEY", "")),
    "gemini":    bool(os.getenv("GEMINI_API_KEY", "")),
    "cerebras":  bool(os.getenv("CEREBRAS_API_KEY", "")),
    "sambanova": bool(os.getenv("SAMBANOVA_API_KEY", "")),
}

# ─── Geocoding API Keys ─────────────────────────────────────────────────────
HERE_API_KEY = os.getenv("HERE_API_KEY", "")
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY", "")

# ─── Benchmark API Keys ────────────────────────────────────────────────────
ARTIFICIAL_ANALYSIS_API_KEY = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "")

# ─── Task Priority Levels ────────────────────────────────────────────────────

TASK_PRIORITY = {
    "critical": 10,
    "high": 8,
    "normal": 5,
    "low": 3,
    "background": 1,
}

# ─── Backward Compatibility ──────────────────────────────────────────────────
# Other modules that imported MODEL_POOL from config.py can use this.
# It's a read-only view derived from the registry on first access.

_model_pool_cache: dict | None = None


def get_model_pool() -> dict:
    """
    Backward-compatible MODEL_POOL derived from the registry.
    Maps old format: {name: {litellm_name, capabilities, quality, ...}}
    """
    global _model_pool_cache
    if _model_pool_cache is not None:
        return _model_pool_cache

    try:
        registry = get_registry()

        pool = {}
        for name, m in registry.models.items():
            # Map 15-dimension capabilities back to a simple capability list
            # (top capabilities above threshold 5.0)
            top_caps = [
                cap for cap, score in sorted(
                    m.capabilities.items(), key=lambda x: x[1], reverse=True
                )
                if score >= 5.0
            ][:6]

            pool[name] = {
                "litellm_name": m.litellm_name,
                "capabilities": top_caps,
                "quality": round(m.best_score()),
                "speed": (
                    "very_fast" if m.tokens_per_second > 100
                    else "fast" if m.tokens_per_second > 30 or m.total_params_b < 10
                    else "medium"
                ),
                "rate_limit": m.rate_limit_rpm,
                "provider": m.provider,
                "max_tokens": m.max_tokens,
                "context_length": m.context_length,
                "supports_function_calling": m.supports_function_calling,
                "supports_response_format": m.supports_json_mode,
            }

        _model_pool_cache = pool
        return pool

    except Exception as e:
        logger.error(f"Failed to build MODEL_POOL from registry: {e}")
        return {}


# Lazy property — code doing `from config import MODEL_POOL` gets this
class _LazyModelPool:
    """Lazy proxy that builds MODEL_POOL on first attribute access."""
    def __init__(self):
        self._pool = None

    def _ensure(self):
        if self._pool is None:
            self._pool = get_model_pool()

    def __getitem__(self, key):
        self._ensure()
        return self._pool[key]

    def __contains__(self, key):
        self._ensure()
        return key in self._pool

    def __iter__(self):
        self._ensure()
        return iter(self._pool)

    def __len__(self):
        self._ensure()
        return len(self._pool)

    def items(self):
        self._ensure()
        return self._pool.items()

    def keys(self):
        self._ensure()
        return self._pool.keys()

    def values(self):
        self._ensure()
        return self._pool.values()

    def get(self, key, default=None):
        self._ensure()
        return self._pool.get(key, default)


MODEL_POOL = _LazyModelPool()


# ─── Startup Display ────────────────────────────────────────────────────────

def print_config() -> None:
    registry = get_registry()

    key_status = {p: ("ok" if ok else "missing") for p, ok in AVAILABLE_KEYS.items()}
    logger.info(
        "Config summary",
        api_keys=key_status,
        model_dir=MODEL_DIR or "(not set)",
        workspace=WORKSPACE_ROOT,
        docker=DOCKER_CONTAINER_NAME,
        max_iterations=MAX_AGENT_ITERATIONS,
        daily_budget=COST_BUDGET_DAILY,
    )
    # print_summary was on the old ModelRegistry; fatih_hoca logs during build
    total = len(registry.models) if hasattr(registry, 'models') else '?'
    logger.info("Config summary", model_count=total)
