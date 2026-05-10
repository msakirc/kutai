"""HTML ``data-oid`` annotator — Z1 Tier 4 (T4B / C17 + A20).

Onlook async-adapted: each semantic block in a generated HTML prototype
gets a stable ``data-oid="<artifact_slug>:<section>"`` attribute. The
oid is the anchor the spec-patch proposer uses to reverse-look up which
spec node a founder edit touched without re-parsing or guessing.

Wiring
------
Mechanical post-processor that runs AFTER ``verify_html_prototype_shape``
and BEFORE ``verify_screen_consistency`` on every chunk-pair step. Reads
the freshly-emitted HTML, walks the DOM via BeautifulSoup, assigns
oids per semantic block, and re-saves in place.

Schema
------
``data-oid`` value: ``"<artifact_slug>:<section>"``

- ``artifact_slug`` is the caller-supplied stable identifier (e.g.
  ``"screen_5_3"`` or the plan ID). Caller derives this from the file
  basename or workflow context.
- ``section`` is one of: ``header / main / nav / footer / aside``,
  or ``section`` for a generic ``<section>`` tag, or ``section_2``,
  ``section_3``... for repeats. ``<div>`` blocks tagged with a class
  that matches a known role (``hero``, ``cta``, ``card``,
  ``form``) are also annotated.

Existing ``data-oid`` attributes are preserved; this is idempotent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag


# Canonical semantic-block tag names that ALWAYS get annotated.
_SEMANTIC_TAGS: tuple[str, ...] = (
    "header",
    "main",
    "nav",
    "footer",
    "aside",
    "section",
    "article",
)

# When a <div> carries one of these class hints, treat it as a
# semantic block too. Founder will care about these — they are the
# "card / cta / hero" units they point at.
_ROLE_CLASSES: tuple[str, ...] = (
    "hero",
    "cta",
    "card",
    "form",
    "modal",
    "dialog",
    "list",
    "tab-bar",
    "tabbar",
    "navbar",
)


def _section_label(tag: Tag) -> str | None:
    """Return the section role for a tag, or None if it isn't semantic.

    Matches:
    - canonical semantic tags by name
    - ``<div>`` whose class list overlaps :data:`_ROLE_CLASSES`
    """
    name = (tag.name or "").lower()
    if name in _SEMANTIC_TAGS:
        return name

    if name == "div":
        cls = tag.get("class") or []
        if isinstance(cls, str):
            cls = cls.split()
        for c in cls:
            cl = (c or "").lower()
            for role in _ROLE_CLASSES:
                if role in cl:
                    return role.replace("-", "_")
    return None


def _annotate_soup(soup: BeautifulSoup, artifact_slug: str) -> int:
    """Annotate every semantic block in the soup. Returns the count of
    NEW oids assigned (existing ones are preserved)."""
    counts: dict[str, int] = {}
    annotated = 0
    for tag in soup.find_all(True):  # noqa: B007  — visit every tag
        if not isinstance(tag, Tag):
            continue
        if tag.has_attr("data-oid"):
            continue
        label = _section_label(tag)
        if not label:
            continue
        n = counts.get(label, 0) + 1
        counts[label] = n
        suffix = label if n == 1 else f"{label}_{n}"
        tag["data-oid"] = f"{artifact_slug}:{suffix}"
        annotated += 1
    return annotated


def annotate_html_oids(
    *,
    html_text: str | None = None,
    html_paths: Iterable[str] | None = None,
    artifact_slug: str | None = None,
) -> dict[str, Any]:
    """Walk the DOM and assign ``data-oid`` to semantic blocks.

    Parameters
    ----------
    html_text:
        Raw HTML string. If supplied, the annotated HTML is returned
        in ``annotated_html``.
    html_paths:
        Iterable of paths to HTML files. Each file is rewritten in
        place and a per-file count is reported in ``per_file``.
    artifact_slug:
        Stable identifier baked into every oid. For path mode, when
        omitted the file stem is used.

    Returns
    -------
    dict
        ``ok`` (bool), ``annotated_count``, ``annotated_html`` (str
        when ``html_text`` was given), ``per_file`` (list when
        ``html_paths`` was given).
    """
    if html_text is None and not html_paths:
        return {"ok": False, "error": "must supply html_text or html_paths"}

    if html_text is not None:
        slug = artifact_slug or "artifact"
        soup = BeautifulSoup(html_text, "html.parser")
        n = _annotate_soup(soup, slug)
        return {
            "ok": True,
            "annotated_count": n,
            "annotated_html": str(soup),
        }

    per_file: list[dict[str, Any]] = []
    total = 0
    for raw in (html_paths or []):
        p = Path(raw)
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            per_file.append(
                {"path": str(p), "ok": False, "error": str(e), "annotated_count": 0}
            )
            continue
        slug = artifact_slug or p.stem
        soup = BeautifulSoup(text, "html.parser")
        n = _annotate_soup(soup, slug)
        if n > 0:
            try:
                p.write_text(str(soup), encoding="utf-8")
            except Exception as e:
                per_file.append(
                    {
                        "path": str(p),
                        "ok": False,
                        "error": f"write failed: {e}",
                        "annotated_count": 0,
                    }
                )
                continue
        per_file.append(
            {"path": str(p), "ok": True, "annotated_count": n}
        )
        total += n

    return {
        "ok": all(f["ok"] for f in per_file) if per_file else True,
        "annotated_count": total,
        "per_file": per_file,
    }
