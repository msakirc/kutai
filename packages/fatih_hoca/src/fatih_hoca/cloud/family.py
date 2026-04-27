"""Cross-provider model family normalization.

Maps provider-specific litellm_name (e.g. ``groq/llama-3.3-70b-versatile``)
to a canonical family key (``llama-3.3-70b``). Same family across providers
shares benchmark cache entry.

Rules ordered most-specific first; first regex hit wins. Unmatched names
fall back to lower-cased provider-stripped form and are flagged via
``is_known_family()`` so new releases surface for manual rule addition.
"""
from __future__ import annotations

import re
from src.infra.logging_config import get_logger

logger = get_logger("fatih_hoca.cloud.family")

# (regex pattern matched against stripped/lowered name, family key)
_FAMILY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^llama-?3\.3-70b"), "llama-3.3-70b"),
    (re.compile(r"^llama3\.3-70b"), "llama-3.3-70b"),
    (re.compile(r"^llama-?3\.1-70b"), "llama-3.1-70b"),
    (re.compile(r"^llama-?3\.1-8b"), "llama-3.1-8b"),
    (re.compile(r"^llama3\.1-8b"), "llama-3.1-8b"),
    (re.compile(r"^llama-?3-70b"), "llama-3-70b"),
    (re.compile(r"^llama-?3-8b"), "llama-3-8b"),
    (re.compile(r"^qwen-?3-?32b"), "qwen3-32b"),
    (re.compile(r"^qwen3-32b"), "qwen3-32b"),
    (re.compile(r"^qwen-?2\.5-72b"), "qwen2.5-72b"),
    (re.compile(r"^qwen-?2\.5-coder-32b"), "qwen2.5-coder-32b"),
    (re.compile(r"^claude-3-?5-sonnet"), "claude-3.5-sonnet"),
    (re.compile(r"^claude-sonnet-4"), "claude-sonnet-4"),
    (re.compile(r"^claude-opus-4"), "claude-opus-4"),
    (re.compile(r"^claude-haiku-4"), "claude-haiku-4"),
    (re.compile(r"^gpt-4o-mini"), "gpt-4o-mini"),
    (re.compile(r"^gpt-4o"), "gpt-4o"),
    (re.compile(r"^o1-preview"), "o1-preview"),
    (re.compile(r"^o1-mini"), "o1-mini"),
    (re.compile(r"^o1"), "o1"),
    (re.compile(r"^gemini-2\.5-flash"), "gemini-2.5-flash"),
    (re.compile(r"^gemini-2\.0-flash"), "gemini-2.0-flash"),
    (re.compile(r"^gemini-1\.5-pro"), "gemini-1.5-pro"),
    (re.compile(r"^gemini-1\.5-flash"), "gemini-1.5-flash"),
    (re.compile(r"^mixtral-8x7b"), "mixtral-8x7b"),
]

_KNOWN_FAMILIES: set[str] = {key for _, key in _FAMILY_RULES}


def _strip_provider_prefix(litellm_name: str) -> str:
    """Strip leading ``provider/`` segment(s) and ``meta-`` model-org prefix.

    Openrouter uses two segments (``openrouter/meta-llama/...``) — strip
    everything except the last segment. Some hosts (SambaNova) include a
    ``Meta-`` org prefix in the model id (e.g. ``Meta-Llama-3.3-70B-Instruct``);
    strip that too so the llama-3.3-70b rule matches uniformly.
    """
    parts = litellm_name.split("/")
    last = parts[-1] if len(parts) >= 2 else litellm_name
    # Drop ``meta-`` org prefix that some hosts include in the model id.
    lower = last.lower()
    if lower.startswith("meta-"):
        last = last[5:]
    return last


def normalize(provider: str, litellm_name: str) -> str:
    """Return canonical family key for a (provider, litellm_name) pair."""
    stripped = _strip_provider_prefix(litellm_name).lower()
    for pattern, family in _FAMILY_RULES:
        if pattern.match(stripped):
            return family
    logger.info("family_unknown provider=%s litellm_name=%s", provider, litellm_name)
    return stripped


def is_known_family(family: str) -> bool:
    """True iff family was produced by a regex rule (not fallback)."""
    return family in _KNOWN_FAMILIES
