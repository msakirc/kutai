"""HTML prototype shape verifier — Z1 Tier 3 (C9+A11).

Mechanical post-hook that reads a paraflow-shape per-screen HTML file and
asserts the structural contract emitted by step ``5.2
generate_html_prototypes``:

    1. ``<!DOCTYPE html>`` declaration
    2. Mobile viewport sized 390×844 — either via ``style="width:390px"``
       /``height:844px`` or via Tailwind ``w-[390px]`` / ``min-h-[844px]``
    3. Tailwind script tag present (jsdelivr / unpkg / cdn.tailwindcss.com)
    4. Every ``<img>`` has a non-empty ``alt`` attribute
    5. No ``<img>`` ``src`` is empty / ``#`` / ``about:blank``
    6. Optional: every color value present in supplied ``design_tokens``
       (skipped when ``design_tokens`` not provided)

Pure check (regex-based; no HTML parsing dependency). Caller (post-hook
wiring) provides the HTML via ``payload['html_text']`` or a
``html_paths`` list.

Returns
-------
dict
    ``ok`` (bool), ``per_file`` (list when paths given), problem fields.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# Shared HTML primitives — one definition for the `<img>` tag matcher + attr
# parser across the swap chain and both verify posthooks.
from mr_roboto._html_common import IMG_RE as _IMG_RE, parse_attrs as _parse_attrs

_DOCTYPE_RE = re.compile(r"<!DOCTYPE\s+html\s*>", re.IGNORECASE)

# Width 390 + height 844 — check both Tailwind arbitrary-value classes
# and inline-style equivalents.
_W_PATTERNS = (
    re.compile(r"\bw-\[\s*390px\s*\]"),
    re.compile(r"width\s*:\s*390px"),
)
_H_PATTERNS = (
    re.compile(r"\b(?:min-)?h-\[\s*844px\s*\]"),
    re.compile(r"\b(?:min-)?height\s*:\s*844px"),
)

_TAILWIND_PATTERNS = (
    re.compile(r"cdn\.tailwindcss\.com", re.IGNORECASE),
    re.compile(r"cdn\.jsdelivr\.net/npm/tailwind", re.IGNORECASE),
    re.compile(r"unpkg\.com/tailwind", re.IGNORECASE),
    re.compile(r"static\.paraflowcontent\.com/.*tailwind", re.IGNORECASE),
)

# Color tokens — `#RRGGBB`, `#RGB`, `rgb(...)`, `rgba(...)`. Tailwind
# named colors (`text-red-500`) are out of scope — those are handled by
# Tailwind's own engine and aren't "raw" values from a design_tokens.json.
_HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?\b")
_RGB_COLOR_RE = re.compile(r"rgba?\([^)]+\)", re.IGNORECASE)


def _flatten_color_values(tokens: Any) -> set[str]:
    """Walk a design_tokens.json structure and collect every leaf string
    that looks like a color value (hex, rgb, rgba). Comparison is
    case-insensitive and whitespace-stripped."""
    out: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)
        elif isinstance(node, str):
            stripped = node.strip()
            if _HEX_COLOR_RE.fullmatch(stripped):
                out.add(stripped.lower())
            elif _RGB_COLOR_RE.fullmatch(stripped):
                # Normalize whitespace inside rgb(...).
                out.add(re.sub(r"\s+", "", stripped.lower()))

    _walk(tokens)
    return out


def _normalize_color(value: str) -> str:
    s = value.strip().lower()
    if s.startswith("rgb"):
        return re.sub(r"\s+", "", s)
    return s


def _check_dimensions(html: str) -> tuple[bool, bool]:
    has_w = any(p.search(html) for p in _W_PATTERNS)
    has_h = any(p.search(html) for p in _H_PATTERNS)
    return has_w, has_h


def _has_tailwind(html: str) -> bool:
    return any(p.search(html) for p in _TAILWIND_PATTERNS)


def _img_problems(html: str) -> tuple[int, list[dict[str, str]]]:
    """Return ``(img_count, problems)``."""
    problems: list[dict[str, str]] = []
    imgs = list(_IMG_RE.finditer(html))
    for m in imgs:
        attrs = _parse_attrs(m.group(1) or "")
        alt = attrs.get("alt")
        src = attrs.get("src", "").strip()
        if alt is None or not alt.strip():
            problems.append({
                "tag": m.group(0)[:80],
                "issue": "missing or empty alt",
            })
        if src in {"", "#", "about:blank"}:
            problems.append({
                "tag": m.group(0)[:80],
                "issue": f"invalid src {src!r}",
            })
    return len(imgs), problems


def _hardcoded_color_offenders(
    html: str, allowed_colors: set[str] | None
) -> list[str]:
    if allowed_colors is None:
        return []
    offenders: list[str] = []
    seen: set[str] = set()
    for m in _HEX_COLOR_RE.finditer(html):
        norm = _normalize_color(m.group(0))
        if norm in seen:
            continue
        seen.add(norm)
        if norm not in allowed_colors:
            offenders.append(m.group(0))
    for m in _RGB_COLOR_RE.finditer(html):
        norm = _normalize_color(m.group(0))
        if norm in seen:
            continue
        seen.add(norm)
        if norm not in allowed_colors:
            offenders.append(m.group(0))
    return offenders[:20]


def _verify_one(
    html: str,
    *,
    allowed_colors: set[str] | None,
) -> dict[str, Any]:
    problems: list[str] = []

    if not _DOCTYPE_RE.search(html):
        problems.append("missing <!DOCTYPE html>")

    has_w, has_h = _check_dimensions(html)
    if not has_w:
        problems.append("missing 390px width (Tailwind w-[390px] or width:390px)")
    if not has_h:
        problems.append("missing 844px height (Tailwind h-[844px]/min-h-[844px] or height:844px)")

    if not _has_tailwind(html):
        problems.append("missing Tailwind script tag")

    img_count, img_problems = _img_problems(html)
    if img_problems:
        problems.append(f"{len(img_problems)} <img> tag(s) malformed")

    color_offenders = _hardcoded_color_offenders(html, allowed_colors)
    if color_offenders:
        problems.append(
            f"{len(color_offenders)} hardcoded color value(s) outside design_tokens"
        )

    ok = not problems

    return {
        "ok": ok,
        "doctype_present": bool(_DOCTYPE_RE.search(html)),
        "has_390_width": has_w,
        "has_844_height": has_h,
        "has_tailwind": _has_tailwind(html),
        "img_count": img_count,
        "img_problems": img_problems,
        "color_offenders": color_offenders,
        "problems": problems,
    }


def verify_html_prototype_shape(
    *,
    html_text: str | None = None,
    html_paths: list[str] | None = None,
    design_tokens: Any = None,
) -> dict[str, Any]:
    """Validate one or more paraflow-shape per-screen HTML files."""
    allowed_colors = (
        _flatten_color_values(design_tokens)
        if design_tokens is not None else None
    )

    if html_text is not None:
        return _verify_one(html_text, allowed_colors=allowed_colors)

    if not html_paths:
        return {
            "ok": False,
            "error": "no html_text or html_paths provided",
            "per_file": [],
        }

    per_file: list[dict[str, Any]] = []
    all_ok = True
    for p in html_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                html = fh.read()
        except OSError as e:
            per_file.append({"path": p, "ok": False, "error": str(e)})
            all_ok = False
            continue
        res = _verify_one(html, allowed_colors=allowed_colors)
        res["path"] = p
        per_file.append(res)
        if not res.get("ok"):
            all_ok = False

    return {"ok": all_ok, "per_file": per_file}
