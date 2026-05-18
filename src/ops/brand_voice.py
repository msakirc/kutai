"""Brand voice loader — Z7 A5.

Parses ``docs/templates/brand_voices/<audience>.md`` YAML front-matter into a
structured :class:`BrandVoice` object. Reusable by the ``brand_voice_lint``
posthook handler and future agents.

Usage::

    voice = load_brand_voice("marketing")          # looks up brand_voices/marketing.md
    voice = load_brand_voice_from_path("/abs/path/to/marketing.md")
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("ops.brand_voice")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class BrandVoice:
    """Structured representation of one brand-voice .md file.

    Attributes
    ----------
    slug:                    Canonical audience id (matches filename stem).
    display_name:            Human-readable name.
    version:                 Front-matter version string.
    prohibited_terms:        List of exact strings or ``/regex/`` patterns.
    target_avg_sentence_length_words:
                             Target average sentence length in words.
                             Drift > 25% triggers a warning.
    flesch_kincaid_reading_level_max:
                             Flesch-Kincaid grade-level ceiling (inclusive).
    we_ratio_max:            ``we / (we + you)`` ceiling; ``None`` = unchecked.
    tone_required_signals:   At least one must appear per 200-word window.
    tone_forbidden_signals:  None may appear anywhere.
    raw_body_md:             Optional guidance text after the YAML block.
    """
    slug: str = ""
    display_name: str = ""
    version: str = "1.0"
    prohibited_terms: list[str] = field(default_factory=list)
    target_avg_sentence_length_words: int = 18
    flesch_kincaid_reading_level_max: float = 10.0
    we_ratio_max: Optional[float] = 0.3
    tone_required_signals: list[str] = field(default_factory=list)
    tone_forbidden_signals: list[str] = field(default_factory=list)
    raw_body_md: str = ""


# ---------------------------------------------------------------------------
# Default search root
# ---------------------------------------------------------------------------


def _default_voices_dir() -> str:
    """Return the canonical docs/templates/brand_voices/ path."""
    # packages/.../posthook_handlers → NOT this file's location.
    # This file lives at src/ops/brand_voice.py — walk up 3 levels to repo root.
    here = Path(__file__).resolve()
    # src/ops/brand_voice.py → up 3 levels → repo root
    repo_root = here.parents[2]
    return str(repo_root / "docs" / "templates" / "brand_voices")


# ---------------------------------------------------------------------------
# Lightweight YAML front-matter parser (no PyYAML dependency)
# ---------------------------------------------------------------------------


def _extract_frontmatter(text: str) -> tuple[str, str]:
    """Split Markdown text into (yaml_block, body_md)."""
    stripped = text.strip()
    if not stripped.startswith("---"):
        return "", stripped

    rest = stripped[3:]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return rest.strip(), ""

    yaml_str = rest[:end_idx].strip()
    body_md = rest[end_idx + 4:].strip()
    return yaml_str, body_md


def _strip_inline_comment(s: str) -> str:
    if not s:
        return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s
    if s.startswith("/"):
        slash_idx = s.rfind("/", 1)
        if slash_idx > 0:
            return s[: slash_idx + 1]
    comment_idx = s.find("  #")
    if comment_idx != -1:
        return s[:comment_idx].strip()
    return s


def _unquote(s: str) -> str:
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
    return s


def _coerce(s: str):
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s in ("null", "~"):
        return None
    unq = _unquote(s)
    if unq != s:
        return unq
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_yaml_lite(yaml_str: str) -> dict:
    """Parse the small YAML subset used in brand-voice files."""
    result: dict = {}
    lines = yaml_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        rest = _strip_inline_comment(rest)

        if not rest or rest == "[]":
            items: list = []
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    i += 1
                    continue
                if not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break
                stripped_next = next_line.strip()
                if stripped_next.startswith("#"):
                    i += 1
                    continue
                if stripped_next.startswith("- "):
                    item_val = stripped_next[2:].strip()
                    items.append(_unquote(item_val))
                    i += 1
                else:
                    break
            result[key] = items
            continue

        result[key] = _coerce(rest)
    return result


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def _parse_brand_voice(text: str) -> BrandVoice:
    """Parse raw brand-voice .md text into a BrandVoice dataclass."""
    yaml_str, body_md = _extract_frontmatter(text)
    data = _parse_yaml_lite(yaml_str) if yaml_str else {}

    def _float(key: str, default: float) -> float:
        val = data.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int) -> int:
        val = data.get(key, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    we_ratio_raw = data.get("we_ratio_max", 0.3)
    we_ratio: Optional[float]
    if we_ratio_raw is None:
        we_ratio = None
    else:
        try:
            we_ratio = float(we_ratio_raw)
        except (TypeError, ValueError):
            we_ratio = 0.3

    return BrandVoice(
        slug=str(data.get("slug") or ""),
        display_name=str(data.get("display_name") or ""),
        version=str(data.get("version") or "1.0"),
        prohibited_terms=[str(t) for t in (data.get("prohibited_terms") or []) if t],
        target_avg_sentence_length_words=_int("target_avg_sentence_length_words", 18),
        flesch_kincaid_reading_level_max=_float("flesch_kincaid_reading_level_max", 10.0),
        we_ratio_max=we_ratio,
        tone_required_signals=[str(s) for s in (data.get("tone_required_signals") or []) if s],
        tone_forbidden_signals=[str(s) for s in (data.get("tone_forbidden_signals") or []) if s],
        raw_body_md=body_md,
    )


def load_brand_voice_from_path(path: str) -> BrandVoice:
    """Load brand voice from an absolute file path.

    Returns a default empty :class:`BrandVoice` if the file is missing
    or unreadable — caller should treat absent voice as a skip signal.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return _parse_brand_voice(text)
    except (OSError, FileNotFoundError) as exc:
        logger.debug("brand_voice: file not found: %s (%s)", path, exc)
        return BrandVoice()
    except Exception as exc:
        logger.warning("brand_voice: parse error: %s (%s)", path, exc)
        return BrandVoice()


def load_brand_voice(audience: str, voices_dir: str | None = None) -> "BrandVoice | None":
    """Load a brand-voice doc by audience slug.

    Searches ``<voices_dir>/<audience>.md`` (and ``.example.md`` variants for
    built-in examples). Returns ``None`` when no voice file is found — callers
    should degrade gracefully (skip with an info-level note, never crash).

    Parameters
    ----------
    audience:
        Audience slug (e.g. ``"marketing"``, ``"support"``, ``"investor"``).
    voices_dir:
        Override the default search directory. Defaults to
        ``docs/templates/brand_voices/``.
    """
    if not audience:
        return None

    _voices_dir = voices_dir or _default_voices_dir()
    slug = audience.strip().lower().replace("-", "_").replace(" ", "_")

    candidates = [
        os.path.join(_voices_dir, f"{slug}.md"),
        os.path.join(_voices_dir, f"{slug}.example.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            voice = load_brand_voice_from_path(path)
            # Accept even lightly populated docs
            if voice.slug or voice.prohibited_terms or voice.tone_required_signals:
                return voice
            if not voice.slug:
                voice.slug = slug
            return voice
    return None
