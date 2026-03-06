# config.py
"""
Central configuration — API keys, model pool, tier mapping, constants.
"""
import os
import subprocess
import logging


logger = logging.getLogger(__name__)

# ─── Core Settings ───────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")

DB_PATH = "orchestrator.db"
WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "workspace")
)
DOCKER_CONTAINER_NAME = "orchestrator-sandbox"

MAX_AGENT_ITERATIONS = 8
MAX_TOOL_OUTPUT_LENGTH = 4000
MAX_CONTEXT_CHAIN_LENGTH = 12000   # chars of prior-step output to inject

# ─── Cost Budget (Phase 4) ──────────────────────────────────────────────────
COST_BUDGET_DAILY: float = float(os.getenv("COST_BUDGET_DAILY", "1.0"))

# ─── Thinking/Reasoning Models (Phase 4) ────────────────────────────────────
# Substrings to detect thinking-capable models. These models should not
# have temperature set and need increased timeouts.
THINKING_MODELS: list[str] = [
    "o1", "o3", "o4",        # OpenAI reasoning
    "qwq",                    # Alibaba QwQ
    "deepseek-r1",            # DeepSeek R1
    "gemini-2.5-flash",       # Gemini thinking
]

# ─── API Key Detection ───────────────────────────────────────────────────────

AVAILABLE_KEYS: dict[str, bool] = {
    "groq":      bool(os.getenv("GROQ_API_KEY", "")),
    "openai":    bool(os.getenv("OPENAI_API_KEY", "")),
    "anthropic": bool(os.getenv("ANTHROPIC_API_KEY", "")),
    "gemini":    bool(os.getenv("GEMINI_API_KEY", "")),
    "cerebras":  bool(os.getenv("CEREBRAS_API_KEY", "")),
    "sambanova": bool(os.getenv("SAMBANOVA_API_KEY", "")),
}

# ─── Ollama Detection ────────────────────────────────────────────────────────

def _detect_ollama_models() -> list[str]:
    """
    Detect locally-available Ollama models.
    Tries the HTTP API first (works in Docker / remote setups),
    then falls back to the CLI.
    """
    # Try HTTP API
    try:
        import httpx                       # usually available via litellm
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    # Fallback: CLI
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            models: list[str] = []
            for line in result.stdout.strip().split("\n")[1:]:   # skip header
                if line.strip():
                    models.append(line.split()[0])
            return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


OLLAMA_MODELS: list[str] = _detect_ollama_models()
OLLAMA_AVAILABLE: bool = len(OLLAMA_MODELS) > 0

# ─── Model Pool ──────────────────────────────────────────────────────────────
# Rich model definitions with capabilities and quality scores.
# The router uses this for smart model selection; MODEL_TIERS is derived
# from it for backward compatibility and simple tier-based routing.

MODEL_POOL: dict[str, dict] = {}

# ── Local: Ollama ──
if OLLAMA_AVAILABLE:
    _ollama_defs: dict[str, dict] = {
        "ollama-qwen3-8b": {
            "litellm_name": "ollama/qwen3:8b-q4_K_M",
            "match_fragment": "qwen3",
            "capabilities": ["planning", "reasoning", "general"],
            "quality": 7, "speed": "medium", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 4096,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
        "ollama-qwen25-coder-7b": {
            "litellm_name": "ollama/qwen2.5-coder:7b-instruct-q4_K_M",
            "match_fragment": "qwen2.5-coder:7b",
            "capabilities": ["coding", "debugging", "general"],
            "quality": 7, "speed": "medium", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 4096,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
        "ollama-qwen25-coder-3b": {
            "litellm_name": "ollama/qwen2.5-coder:3b-instruct-q5_K_M",
            "match_fragment": "qwen2.5-coder:3b",
            "capabilities": ["coding", "quick_code"],
            "quality": 5, "speed": "fast", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 2048,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
        "ollama-qwen25-7b": {
            "litellm_name": "ollama/qwen2.5:7b-instruct-q4_K_M",
            "match_fragment": "qwen2.5:7b",
            "capabilities": ["general", "writing", "analysis"],
            "quality": 6, "speed": "medium", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 4096,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
        "ollama-llama32-3b": {
            "litellm_name": "ollama/llama3.2:3b-instruct-q5_K_M",
            "match_fragment": "llama3.2",
            "capabilities": ["routing", "classification", "simple"],
            "quality": 4, "speed": "fast", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 1024,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
        "ollama-phi4-mini": {
            "litellm_name": "ollama/phi4-mini:3.8b-q4_K_M",
            "match_fragment": "phi4-mini",
            "capabilities": ["reasoning", "simple", "general"],
            "quality": 5, "speed": "fast", "rate_limit": 999,
            "provider": "ollama", "max_tokens": 2048,
            "supports_function_calling": False,
            "supports_response_format": True,
        },
    }
    for _key, _cfg in _ollama_defs.items():
        _fragment = _cfg["match_fragment"]
        if any(_fragment in m for m in OLLAMA_MODELS):
            MODEL_POOL[_key] = {
                k: v for k, v in _cfg.items() if k != "match_fragment"
            }

# ── Cloud: Groq (free tier ~30 rpm) ──
if AVAILABLE_KEYS["groq"]:
    MODEL_POOL["groq-llama-8b"] = {
        "litellm_name": "groq/llama-3.1-8b-instant",
        "capabilities": ["routing", "classification", "simple", "general"],
        "quality": 5, "speed": "very_fast", "rate_limit": 30,
        "provider": "groq", "max_tokens": 1024,
        "supports_function_calling": True,
    }
    MODEL_POOL["groq-llama-70b"] = {
        "litellm_name": "groq/llama-3.3-70b-versatile",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 8, "speed": "fast", "rate_limit": 30,
        "provider": "groq", "max_tokens": 4096,
        "supports_function_calling": True,
    }

# ── Cloud: Google Gemini (free tier ~15 rpm) ──
if AVAILABLE_KEYS["gemini"]:
    MODEL_POOL["gemini-flash"] = {
        "litellm_name": "gemini/gemini-2.0-flash",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 8, "speed": "fast", "rate_limit": 15,
        "provider": "gemini", "max_tokens": 8192,
        "supports_function_calling": True,
    }
    MODEL_POOL["gemini-flash-preview"] = {
        "litellm_name": "gemini/gemini-2.5-flash-preview-05-20",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 9, "speed": "fast", "rate_limit": 15,
        "provider": "gemini", "max_tokens": 8192,
        "supports_function_calling": True,
    }

# ── Cloud: Cerebras (free tier ~30 rpm) ──
if AVAILABLE_KEYS["cerebras"]:
    MODEL_POOL["cerebras-llama-70b"] = {
        "litellm_name": "cerebras/llama3.3-70b",
        "capabilities": ["general", "coding", "reasoning", "writing"],
        "quality": 8, "speed": "very_fast", "rate_limit": 30,
        "provider": "cerebras", "max_tokens": 4096,
        "supports_function_calling": True,
    }

# ── Cloud: SambaNova (free tier ~20 rpm) ──
if AVAILABLE_KEYS["sambanova"]:
    MODEL_POOL["sambanova-qwen3-32b"] = {
        "litellm_name": "sambanova/Qwen3-32B",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 9, "speed": "fast", "rate_limit": 20,
        "provider": "sambanova", "max_tokens": 4096,
        "supports_function_calling": True,
    }

# ── Cloud: OpenAI (paid) ──
if AVAILABLE_KEYS["openai"]:
    MODEL_POOL["gpt-4o-mini"] = {
        "litellm_name": "gpt-4o-mini",
        "capabilities": [
            "general", "coding", "planning", "writing", "analysis",
        ],
        "quality": 8, "speed": "fast", "rate_limit": 500,
        "provider": "openai", "max_tokens": 4096,
        "supports_function_calling": True,
    }
    MODEL_POOL["gpt-4o"] = {
        "litellm_name": "gpt-4o",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 9, "speed": "medium", "rate_limit": 500,
        "provider": "openai", "max_tokens": 8192,
        "supports_function_calling": True,
    }

# ── Cloud: Anthropic (paid) ──
if AVAILABLE_KEYS["anthropic"]:
    MODEL_POOL["claude-sonnet"] = {
        "litellm_name": "claude-sonnet-4-20250514",
        "capabilities": [
            "general", "coding", "planning",
            "reasoning", "writing", "analysis",
        ],
        "quality": 10, "speed": "medium", "rate_limit": 50,
        "provider": "anthropic", "max_tokens": 8192,
        "supports_function_calling": True,
    }

'''
# ─── Tier Helpers ────────────────────────────────────────────────────────────

# def get_models_for_tier(tier: str) -> list[str]:
#     """
#     Return MODEL_POOL keys suitable for *tier*, ordered best-first.
#
#     Overlap between tiers is intentional — it provides more fallback options.
#     """
#     if tier == "routing":
#         candidates = [
#             k for k, v in MODEL_POOL.items()
#             if "routing" in v["capabilities"]
#             or "classification" in v["capabilities"]
#         ]
#         if not candidates:
#             candidates = [k for k, v in MODEL_POOL.items() if v["quality"] <= 5]
#     elif tier == "cheap":
#         candidates = [k for k, v in MODEL_POOL.items() if v["quality"] <= 6]
#     elif tier == "code":
#         candidates = [k for k, v in MODEL_POOL.items()
#                       if "coding" in v["capabilities"]]
#     elif tier == "medium":
#         candidates = [k for k, v in MODEL_POOL.items() if v["quality"] >= 6]
#     elif tier == "expensive":
#         candidates = [k for k, v in MODEL_POOL.items() if v["quality"] >= 8]
#     else:
#         candidates = list(MODEL_POOL.keys())
#
#     # Sort: routing/cheap prefer local (unlimited); everything else by quality
#     if tier in ("routing", "cheap"):
#         candidates.sort(key=lambda k: (
#             0 if MODEL_POOL[k]["provider"] == "ollama" else 1,
#             -MODEL_POOL[k]["quality"],
#         ))
#     elif tier == "code":
#         # Prefer models whose *primary* capability is coding
#         candidates.sort(key=lambda k: (
#             0 if MODEL_POOL[k]["capabilities"][0] == "coding" else 1,
#             -MODEL_POOL[k]["quality"],
#         ))
#     else:
#         _free_cloud = {"groq", "cerebras", "sambanova", "gemini"}
#         candidates.sort(key=lambda k: (
#             0 if MODEL_POOL[k]["provider"] == "ollama" else
#             1 if MODEL_POOL[k]["provider"] in _free_cloud else 2,
#             -MODEL_POOL[k]["quality"],
#         ))
#
#     return candidates


# ─── MODEL_TIERS (derived from MODEL_POOL) ──────────────────────────────────
# Backward-compatible dict consumed by the router and other modules.

def _build_tier_config(
    tier: str,
    temperature: float,
    description: str,
) -> dict | None:
    """Build a single MODEL_TIERS entry from the pool."""
    candidates = [m['litellm_name'] for m in select_model(tier)]
    if not candidates:
        return None
    primary = candidates[0]
    return {
        "model":       MODEL_POOL[primary]["litellm_name"],
        "fallbacks":   [MODEL_POOL[k]["litellm_name"] for k in candidates[1:]],
        "max_tokens":  MODEL_POOL[primary]["max_tokens"],
        "temperature": temperature,
        "description": description,
    }


MODEL_TIERS: dict[str, dict] = {}

_tier_definitions = [
    ("routing",   0.0, "Task classification only"),
    ("cheap",     0.3, "Simple Q&A, formatting, lookups"),
    ("code",      0.1, "Code generation and debugging"),
    ("medium",    0.3, "Summaries, planning, moderate reasoning"),
    ("expensive", 0.3, "Complex analysis, full code gen, architecture"),
]
for _tname, _ttemp, _tdesc in _tier_definitions:
    _tcfg = _build_tier_config(_tname, _ttemp, _tdesc)
    if _tcfg:
        MODEL_TIERS[_tname] = _tcfg

# ─── Classifier Model ───────────────────────────────────────────────────────

_routing_pool = [m['litellm_name'] for m in select_model("routing")]
if _routing_pool:
    CLASSIFIER_MODEL: str = MODEL_POOL[_routing_pool[0]]["litellm_name"]
elif MODEL_POOL:
    _cheapest = min(MODEL_POOL, key=lambda k: MODEL_POOL[k]["quality"])
    CLASSIFIER_MODEL = MODEL_POOL[_cheapest]["litellm_name"]
else:
    CLASSIFIER_MODEL = "groq/llama-3.1-8b-instant"     # last-resort default

'''
# ─── Fallback & Agent Mapping ────────────────────────────────────────────────

FALLBACK_ORDER: list[str] = ["expensive", "medium", "code", "cheap", "routing"]

AGENT_TIER_MAP: dict[str, str] = {
    "planner":    "medium",
    "coder":      "code",
    "executor":   "cheap",
    "researcher": "medium",
    "writer":     "medium",
    "reviewer":   "medium",
}

# ─── Task Priority Levels ────────────────────────────────────────────────────

TASK_PRIORITY = {
    "critical": 10,     # User actively waiting (Telegram conversation)
    "high": 8,          # Goal planning, urgent
    "normal": 5,        # Standard background subtasks
    "low": 3,           # Maintenance, optional
    "background": 1,    # Scheduled, nice-to-have
}

# ─── Startup Display ────────────────────────────────────────────────────────

def print_config() -> None:
    # Lazy import to avoid circular dependency
    from router import MODEL_TIERS, CLASSIFIER_MODEL

    print("=" * 60)
    print("  🔑 API Keys:")
    for provider, available in AVAILABLE_KEYS.items():
        print(f"     {'✅' if available else '❌'} {provider}")

    print(f"\n  🦙 Ollama: ", end="")
    if OLLAMA_AVAILABLE:
        print(f"✅ {len(OLLAMA_MODELS)} model(s)")
        for m in OLLAMA_MODELS:
            print(f"     • {m}")
    else:
        print("❌ offline")

    print(f"\n  📦 Model Pool ({len(MODEL_POOL)} models):")
    for key, cfg in MODEL_POOL.items():
        caps = ", ".join(cfg["capabilities"][:3])
        print(
            f"     {key:30s} q={cfg['quality']:>2} | "
            f"{cfg['provider']:10s} | {caps}"
        )

    print(f"\n  📊 Active Tiers:")
    for tier_name, tier_cfg in MODEL_TIERS.items():
        n_fb = len(tier_cfg.get("fallbacks", []))
        fb_str = f" (+{n_fb} fallback{'s' if n_fb != 1 else ''})" if n_fb else ""
        print(f"     {tier_name:12s}: {tier_cfg['model']}{fb_str}")

    print(f"\n\n  🧭 Classifier: {CLASSIFIER_MODEL}")
    print(f"  📁 Workspace:  {WORKSPACE_ROOT}")
    print(f"  🐳 Docker:     {DOCKER_CONTAINER_NAME}")
    print(f"  🔄 Max iters:  {MAX_AGENT_ITERATIONS}")
    print("=" * 60)
