# config.py
import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")

# Only define tiers for models you ACTUALLY have keys for
# Order matters: fallback goes DOWN this list, not up

AVAILABLE_KEYS = {
    "groq": bool(os.getenv("GROQ_API_KEY", "")),
    "openai": bool(os.getenv("OPENAI_API_KEY", "")),
    "anthropic": bool(os.getenv("ANTHROPIC_API_KEY", "")),
}

# Build tiers dynamically based on what keys you have
MODEL_TIERS = {}

# Tier: cheap — always available if you have Groq
if AVAILABLE_KEYS["groq"]:
    MODEL_TIERS["cheap"] = {
        "model": "groq/llama-3.1-8b-instant",
        "max_tokens": 1024,
        "description": "Simple lookups, formatting, classification, Q&A"
    }

# Tier: medium
if AVAILABLE_KEYS["groq"]:
    MODEL_TIERS["medium"] = {
        "model": "groq/llama-3.3-70b-versatile",  # FREE on Groq, much smarter
        "max_tokens": 4096,
        "description": "Summaries, drafting, moderate reasoning"
    }
elif AVAILABLE_KEYS["openai"]:
    MODEL_TIERS["medium"] = {
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
        "description": "Summaries, drafting, moderate reasoning"
    }

# Tier: expensive
if AVAILABLE_KEYS["openai"]:
    MODEL_TIERS["expensive"] = {
        "model": "gpt-4o",
        "max_tokens": 8192,
        "description": "Complex analysis, coding, critical decisions"
    }
elif AVAILABLE_KEYS["anthropic"]:
    MODEL_TIERS["expensive"] = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "description": "Complex analysis, coding, critical decisions"
    }
elif AVAILABLE_KEYS["groq"]:
    # No paid keys at all? Use Groq's best as fallback
    MODEL_TIERS["expensive"] = {
        "model": "groq/llama-3.3-70b-versatile",
        "max_tokens": 8192,
        "description": "Best available free model"
    }

# The classifier always uses the cheapest model
CLASSIFIER_MODEL = "groq/llama-3.1-8b-instant"

# Fallback order: DOWNWARD (cheaper), not upward
FALLBACK_ORDER = ["expensive", "medium", "cheap"]

DB_PATH = "orchestrator.db"

def print_config():
    """Print active configuration on startup."""
    print("=" * 50)
    print(" Available API Keys:")
    for provider, available in AVAILABLE_KEYS.items():
        status = "✅" if available else "❌"
        print(f"   {status} {provider}")
    print(f"\n Active Model Tiers:")
    for tier, config in MODEL_TIERS.items():
        print(f"   {tier}: {config['model']}")
    print("=" * 50)
