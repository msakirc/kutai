"""Tests for src/util/lang.py — Z7 T2B multilingual base."""
from __future__ import annotations

import pytest


# ─── detect_language ──────────────────────────────────────────────────────────

def test_detect_english():
    from src.util.lang import detect_language
    result = detect_language(
        "The quick brown fox jumps over the lazy dog. This is a normal English sentence."
    )
    assert result == "en"


def test_detect_turkish():
    from src.util.lang import detect_language
    result = detect_language(
        "Merhaba dünya! Bu bir Türkçe cümledir. Bugün hava çok güzel."
    )
    assert result == "tr"


def test_detect_empty_returns_default():
    from src.util.lang import detect_language
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


def test_detect_custom_default():
    from src.util.lang import detect_language
    assert detect_language("", default="tr") == "tr"


def test_detect_too_short_returns_default():
    from src.util.lang import detect_language
    # Very short/single-char input — may be undetectable; must not raise
    result = detect_language("a")
    assert isinstance(result, str)
    assert len(result) == 2


def test_detect_garbage_returns_default():
    from src.util.lang import detect_language
    # Garbage input must not raise, returns default
    result = detect_language("!@#$%^&*()", default="en")
    assert isinstance(result, str)


def test_detect_deterministic():
    """Same input must return same output across repeated calls."""
    from src.util.lang import detect_language
    text = "Bu test metninin her seferinde aynı sonucu vermesi gerekir."
    results = {detect_language(text) for _ in range(5)}
    assert len(results) == 1, f"Non-deterministic output: {results}"


def test_detect_deterministic_english():
    from src.util.lang import detect_language
    text = "This is a test sentence that must always return the same language."
    results = {detect_language(text) for _ in range(5)}
    assert len(results) == 1


# ─── normalize_lang ───────────────────────────────────────────────────────────

def test_normalize_lang_passthrough():
    from src.util.lang import normalize_lang
    assert normalize_lang("tr") == "tr"
    assert normalize_lang("en") == "en"


def test_normalize_lang_region_variants():
    from src.util.lang import normalize_lang
    assert normalize_lang("tr-TR") == "tr"
    assert normalize_lang("en-US") == "en"
    assert normalize_lang("en-GB") == "en"
    assert normalize_lang("zh-CN") == "zh"


def test_normalize_lang_underscore_variants():
    from src.util.lang import normalize_lang
    assert normalize_lang("tr_TR") == "tr"
    assert normalize_lang("en_US") == "en"


def test_normalize_lang_uppercase():
    from src.util.lang import normalize_lang
    assert normalize_lang("TR") == "tr"
    assert normalize_lang("EN") == "en"


# ─── lang_collection_name ─────────────────────────────────────────────────────

def test_lang_collection_name_basic():
    from src.util.lang import lang_collection_name
    assert lang_collection_name("support_docs", "tr") == "support_docs_tr"
    assert lang_collection_name("support_docs", "en") == "support_docs_en"


def test_lang_collection_name_other_langs():
    from src.util.lang import lang_collection_name
    assert lang_collection_name("faq", "de") == "faq_de"
    assert lang_collection_name("knowledge", "fr") == "knowledge_fr"


def test_lang_collection_name_normalizes():
    """Collection name helper normalizes lang codes before composing."""
    from src.util.lang import lang_collection_name
    assert lang_collection_name("support_docs", "tr-TR") == "support_docs_tr"
    assert lang_collection_name("support_docs", "en-US") == "support_docs_en"


# ─── lang_artifact_path ───────────────────────────────────────────────────────

def test_lang_artifact_path_turkish():
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "tr") == "faq_tr.md"


def test_lang_artifact_path_english_default():
    """For English (the default lang), returns base.md not base_en.md."""
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "en") == "faq.md"


def test_lang_artifact_path_custom_default():
    from src.util.lang import lang_artifact_path
    # Turkish could be the default in a given context → faq.md
    assert lang_artifact_path("faq", "tr", default_lang="tr") == "faq.md"


def test_lang_artifact_path_other_langs():
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "de") == "faq_de.md"
    assert lang_artifact_path("policy", "fr") == "policy_fr.md"


def test_lang_artifact_path_normalizes():
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "tr-TR") == "faq_tr.md"
    assert lang_artifact_path("faq", "en-US") == "faq.md"


# ─── LANG_DISPLAY ─────────────────────────────────────────────────────────────

def test_lang_display_contains_required_langs():
    from src.util.lang import LANG_DISPLAY
    assert "en" in LANG_DISPLAY
    assert "tr" in LANG_DISPLAY


def test_lang_display_values_are_strings():
    from src.util.lang import LANG_DISPLAY
    for code, name in LANG_DISPLAY.items():
        assert isinstance(code, str), f"Key {code!r} is not a string"
        assert isinstance(name, str), f"Value {name!r} for {code!r} is not a string"
        assert len(code) >= 2, f"Code {code!r} too short"
