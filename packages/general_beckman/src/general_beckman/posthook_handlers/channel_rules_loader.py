"""Z7 A6 — channel-rules loader.

Parses ``docs/templates/channel_rules/<channel>.md`` YAML front-matter into a
structured :class:`ChannelRules` object. The loader is used exclusively by the
``copy_compliance_review`` posthook handler.

Usage::

    rules = load_channel_rules("hn_post")          # looks up channel_rules/hn_post.md
    rules = load_channel_rules_from_path("/abs/path/to/ph_post.md")
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from yazbunu import get_logger

logger = get_logger("beckman.posthooks.channel_rules_loader")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ChannelRules:
    """Structured representation of one channel-rules .md file.

    Attributes
    ----------
    channel:        Canonical channel id (matches the ``channel`` front-matter key).
    display_name:   Human-readable name (e.g. "Hacker News Show HN").
    version:        Front-matter version string.
    max_title_chars:  Hard title character limit. 0 = no limit.
    max_body_chars:   Hard body character limit. 0 = no limit.
    max_total_chars:  Combined limit. 0 = no limit.
    banned_words:   List of banned strings or ``/regex/`` patterns.
    required_disclosures: List of ``{label, pattern}`` dicts.
    image_required: Whether an image URL is required.
    image_min_width_px, image_min_height_px, image_max_size_kb: 0 = no check.
    image_allowed_formats: Empty list = any format.
    raw_body_md:    Optional guidance text after the YAML block (for LLM prompts).
    """
    channel: str = ""
    display_name: str = ""
    version: str = "1.0"
    max_title_chars: int = 0
    max_body_chars: int = 0
    max_total_chars: int = 0
    banned_words: list[str] = field(default_factory=list)
    required_disclosures: list[dict] = field(default_factory=list)
    image_required: bool = False
    image_min_width_px: int = 0
    image_min_height_px: int = 0
    image_max_size_kb: int = 0
    image_allowed_formats: list[str] = field(default_factory=list)
    raw_body_md: str = ""


# ---------------------------------------------------------------------------
# Default search root
# ---------------------------------------------------------------------------

def _default_rules_dir() -> str:
    """Return the canonical docs/templates/channel_rules/ path."""
    # Walk up from this file to the repo root, then down to docs/templates/channel_rules/
    here = Path(__file__).resolve()
    # packages/general_beckman/src/general_beckman/posthook_handlers/channel_rules_loader.py
    # → up 6 levels → repo root
    repo_root = here.parents[5]
    return str(repo_root / "docs" / "templates" / "channel_rules")


# ---------------------------------------------------------------------------
# YAML front-matter parser (no PyYAML dependency — lightweight subset)
# ---------------------------------------------------------------------------

def _extract_frontmatter(text: str) -> tuple[str, str]:
    """Split Markdown text into (yaml_block, body) strings.

    Expects optional ``---`` delimiters. Returns (yaml_str, body_md).
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return "", stripped

    # Find the closing ---
    rest = stripped[3:]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return rest.strip(), ""

    yaml_str = rest[:end_idx].strip()
    body_md = rest[end_idx + 4:].strip()  # skip \n---
    return yaml_str, body_md


def _parse_yaml_lite(yaml_str: str) -> dict:
    """Parse a small YAML subset used in channel_rules files.

    Handles:
    - scalar strings / numbers / booleans
    - list items starting with ``- ``
    - nested objects (2-space indent) under list items: ``{label: ..., pattern: ...}``
    - ``/regex/`` patterns preserved as strings
    - Inline comments starting with ``#`` stripped from scalar lines
    - Quoted scalar strings (single or double quotes)
    """
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

        # Strip inline comment (but not inside /regex/ or quotes)
        rest = _strip_inline_comment(rest)

        # Empty value → check if next lines are list items
        if not rest or rest == "[]":
            items: list = []
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    i += 1
                    continue
                # Check indentation — list item must be indented
                if not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break
                stripped_next = next_line.strip()
                if stripped_next.startswith("#"):
                    i += 1
                    continue
                if stripped_next.startswith("- "):
                    item_val = stripped_next[2:].strip()
                    # Sub-dict: ``{label: foo, pattern: bar}``
                    if item_val.startswith("{") and item_val.endswith("}"):
                        items.append(_parse_inline_dict(item_val))
                    else:
                        items.append(_unquote(item_val))
                    i += 1
                else:
                    break
            result[key] = items if (not rest or rest == "[]") else rest
            continue

        # Scalar value
        result[key] = _coerce(rest)
    return result


def _strip_inline_comment(s: str) -> str:
    """Remove trailing inline comment (``# ...``) from a scalar value.

    Skips when inside a /regex/ or quoted string.
    """
    if not s:
        return s
    # If surrounded by quotes, leave as-is
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s
    # If /regex/: don't strip
    if s.startswith("/"):
        slash_idx = s.rfind("/", 1)
        if slash_idx > 0:
            return s[:slash_idx + 1]
    # Strip # comment
    comment_idx = s.find("  #")
    if comment_idx != -1:
        return s[:comment_idx].strip()
    if s.endswith("#"):
        parts = s.rsplit("#", 1)
        return parts[0].strip()
    return s


def _unquote(s: str) -> str:
    """Remove surrounding single or double quotes."""
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
    return s


def _coerce(s: str):
    """Coerce a scalar YAML string to Python bool/int/float/str."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s == "null" or s == "~":
        return None
    # Strip surrounding quotes
    unq = _unquote(s)
    if unq != s:
        return unq
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_inline_dict(s: str) -> dict:
    """Parse a simple ``{key: value, key2: value2}`` inline YAML dict."""
    inner = s.strip("{} \t")
    result: dict = {}
    for part in inner.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        result[k.strip()] = _unquote(v.strip())
    return result


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def _parse_rules(text: str) -> ChannelRules:
    """Parse raw channel-rules .md text into a ChannelRules dataclass."""
    yaml_str, body_md = _extract_frontmatter(text)
    data = _parse_yaml_lite(yaml_str) if yaml_str else {}

    def _int(key: str, default: int = 0) -> int:
        val = data.get(key, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def _bool_val(key: str, default: bool = False) -> bool:
        val = data.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")
        return bool(val)

    disclosures_raw = data.get("required_disclosures") or []
    disclosures: list[dict] = []
    for d in disclosures_raw:
        if isinstance(d, dict):
            disclosures.append(d)
        elif isinstance(d, str) and d:
            disclosures.append({"label": d, "pattern": d})

    banned = [str(w) for w in (data.get("banned_words") or []) if w]
    formats = [str(f) for f in (data.get("image_allowed_formats") or []) if f]

    return ChannelRules(
        channel=str(data.get("channel") or ""),
        display_name=str(data.get("display_name") or ""),
        version=str(data.get("version") or "1.0"),
        max_title_chars=_int("max_title_chars"),
        max_body_chars=_int("max_body_chars"),
        max_total_chars=_int("max_total_chars"),
        banned_words=banned,
        required_disclosures=disclosures,
        image_required=_bool_val("image_required"),
        image_min_width_px=_int("image_min_width_px"),
        image_min_height_px=_int("image_min_height_px"),
        image_max_size_kb=_int("image_max_size_kb"),
        image_allowed_formats=formats,
        raw_body_md=body_md,
    )


def load_channel_rules_from_path(path: str) -> ChannelRules:
    """Load channel rules from an absolute file path.

    Returns a default empty :class:`ChannelRules` if the file is missing
    or unreadable (caller should handle gracefully).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return _parse_rules(text)
    except (OSError, FileNotFoundError) as exc:
        logger.debug("channel_rules: file not found: %s (%s)", path, exc)
        return ChannelRules()
    except Exception as exc:
        logger.warning("channel_rules: parse error: %s (%s)", path, exc)
        return ChannelRules()


def load_channel_rules(channel: str, rules_dir: str | None = None) -> ChannelRules | None:
    """Load channel rules by channel name.

    Searches ``<rules_dir>/<channel>.md`` (and the ``.example.md`` variants
    for built-in examples). Returns ``None`` if no rule file is found.

    Parameters
    ----------
    channel:
        Channel id string (e.g. ``"hn_post"``, ``"ph_post"``).
    rules_dir:
        Override the default search directory. Defaults to
        ``docs/templates/channel_rules/``.
    """
    if not channel:
        return None

    _rules_dir = rules_dir or _default_rules_dir()

    # Normalise: lowercase, replace spaces/hyphens with underscores
    channel_key = channel.strip().lower().replace("-", "_").replace(" ", "_")

    # Primary candidates: exact name match + .example.md suffix
    candidates = [
        os.path.join(_rules_dir, f"{channel_key}.md"),
        os.path.join(_rules_dir, f"{channel_key}.example.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            rules = load_channel_rules_from_path(path)
            if rules.channel or rules.max_title_chars or rules.banned_words:
                return rules
            if not rules.channel:
                rules.channel = channel_key
                return rules

    # Fallback: scan all .md / .example.md files in the directory and match
    # by the ``channel:`` key declared in the front-matter.  This handles
    # aliased example files (e.g. ``producthunt.example.md`` declares
    # ``channel: ph_post``).
    if not os.path.isdir(_rules_dir):
        return None
    try:
        filenames = os.listdir(_rules_dir)
    except OSError:
        return None
    for fname in sorted(filenames):
        if not (fname.endswith(".md") or fname.endswith(".example.md")):
            continue
        fpath = os.path.join(_rules_dir, fname)
        rules = load_channel_rules_from_path(fpath)
        if rules.channel and rules.channel.lower() == channel_key:
            return rules
    return None
