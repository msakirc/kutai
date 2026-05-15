"""
src/util/lang.py — Z7 T2B Multilingual base utilities.

Public API
----------
detect_language(text, default="en") -> str
    Detect the ISO 639-1 language code for *text*.  Never raises.  Returns
    *default* when the text is empty, too short, or undetectable.

normalize_lang(code) -> str
    Coerce locale variants (e.g. "tr-TR", "en_US", "TR") → canonical two-
    letter ISO 639-1 code ("tr", "en").

lang_collection_name(base, lang) -> str
    Compose a per-language Chroma collection name following the convention
    established by Z10 T3C mission namespacing:

        lang_collection_name("support_docs", "tr") → "support_docs_tr"

    The separator is ``_`` (single underscore) — same style as the COLLECTIONS
    list in vector_store.py.  The ``lang`` argument is normalised before
    composing, so "tr-TR" and "tr" both produce the same result.

    A8 (T4 FAQ regen) MUST use this helper so every per-language Chroma index
    follows a consistent naming scheme.  Do NOT create collections directly
    here — this is only a naming helper.

lang_artifact_path(base_name, lang, default_lang="en") -> str
    Return the canonical file path for a per-language artifact.

    Convention (mirrors how i2p step IDs encode language):
      • lang == default_lang  →  ``{base_name}.md``   (no suffix)
      • otherwise             →  ``{base_name}_{lang}.md``

    Examples:
      lang_artifact_path("faq", "tr")         → "faq_tr.md"
      lang_artifact_path("faq", "en")         → "faq.md"
      lang_artifact_path("faq", "en-US")      → "faq.md"
      lang_artifact_path("policy", "de")      → "policy_de.md"

    A8 MUST use this helper when writing per-language FAQ/policy markdown so
    files land at predictable paths that the briefing surface and RAG ingestion
    pipelines can discover without extra metadata.

LANG_DISPLAY
    dict[str, str] mapping ISO 639-1 codes → display names for the languages
    KutAI cares about.  Extend as needed.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("util.lang")

# ─── Seed for deterministic langdetect output ─────────────────────────────────
# langdetect uses a non-deterministic Naive Bayes detector by default.
# DetectorFactory.seed fixes it so repeated calls on the same text always
# return the same result.  This is safe to call at import time.
try:
    from langdetect import DetectorFactory as _DetectorFactory
    _DetectorFactory.seed = 0
except Exception:
    pass  # langdetect unavailable at import? handled lazily in detect_language.


# ─── Display names for languages KutAI cares about ───────────────────────────

LANG_DISPLAY: dict[str, str] = {
    "tr": "Türkçe",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "ar": "العربية",
    "zh": "中文",
    "ru": "Русский",
    "ja": "日本語",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_lang(code: str) -> str:
    """Coerce a locale code to a canonical two-letter ISO 639-1 string.

    Examples::

        normalize_lang("tr-TR")  → "tr"
        normalize_lang("en_US")  → "en"
        normalize_lang("TR")     → "tr"
        normalize_lang("en")     → "en"
    """
    if not code:
        return code
    # Split on hyphen or underscore, take the first segment, lowercase.
    base = code.replace("_", "-").split("-")[0].lower()
    return base


def detect_language(text: str, default: str = "en") -> str:
    """Detect the ISO 639-1 language code for *text*.

    Uses langdetect under the hood with a fixed seed (set at module import)
    so results are deterministic across calls.

    Parameters
    ----------
    text:
        The text to analyse.  If empty, whitespace-only, or too short
        (< 10 chars after stripping), *default* is returned without calling
        langdetect — the detector is unreliable on micro-strings.
    default:
        The code to return when detection is impossible.  Defaults to "en".

    Returns
    -------
    str
        An ISO 639-1 code, e.g. "en", "tr".  Never raises.
    """
    stripped = (text or "").strip()
    if len(stripped) < 10:
        return default

    try:
        from langdetect import detect as _detect
        code = _detect(stripped)
        # langdetect returns codes like "zh-cn"; normalise to just the base.
        return normalize_lang(code)
    except Exception as exc:
        logger.debug("detect_language: langdetect raised %s — returning default %r", exc, default)
        return default


def lang_collection_name(base: str, lang: str) -> str:
    """Return the per-language Chroma collection name for *base* + *lang*.

    Naming convention::

        lang_collection_name("support_docs", "tr")    → "support_docs_tr"
        lang_collection_name("support_docs", "tr-TR") → "support_docs_tr"
        lang_collection_name("faq", "en")             → "faq_en"

    The separator is a single underscore, consistent with the existing
    COLLECTIONS list in ``src/memory/vector_store.py``.

    Note: this helper only *names* collections — it never creates them.
    Use ``vector_store.embed_and_store`` / ``query`` with the returned name
    as the collection argument (after ensuring the collection exists).
    """
    normalised = normalize_lang(lang)
    return f"{base}_{normalised}"


def lang_artifact_path(
    base_name: str,
    lang: str,
    default_lang: str = "en",
) -> str:
    """Return the canonical file path for a per-language markdown artifact.

    Convention
    ----------
    * ``lang == default_lang`` (after normalisation) → ``{base_name}.md``
    * otherwise → ``{base_name}_{lang}.md``

    Examples::

        lang_artifact_path("faq", "tr")         → "faq_tr.md"
        lang_artifact_path("faq", "en")         → "faq.md"
        lang_artifact_path("faq", "en-US")      → "faq.md"
        lang_artifact_path("policy", "de")      → "policy_de.md"

    Parameters
    ----------
    base_name:
        Base filename without extension (e.g. "faq", "policy").
    lang:
        ISO 639-1 code or locale string (normalised before comparison).
    default_lang:
        The language whose artifacts get the bare ``{base}.md`` path.
        Defaults to "en".
    """
    normalised = normalize_lang(lang)
    if normalised == normalize_lang(default_lang):
        return f"{base_name}.md"
    return f"{base_name}_{normalised}.md"
