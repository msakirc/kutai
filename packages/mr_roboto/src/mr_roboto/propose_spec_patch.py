"""HTML diff → spec-patch proposer — Z1 Tier 4 (T4B / C17 + A20).

Founder edits an annotated HTML offline and re-uploads. The diff walks
both DOMs by ``data-oid`` (Onlook async-adapted), pairs nodes by stable
anchor, and surfaces:

- copy edits (text content changes)
- style edits (inline style / class attribute changes)
- structure edits (child count or tag-name changes)

Each surfaced change is mapped to a suggested upstream artifact:

- color shift → style guide / design_tokens.json
- copy edit → screen plan or a copy doc
- structure / layout → screen plan

Output: ``spec_patch_proposal.md`` for founder review. Acceptance is a
separate clarify-shape interaction; this action only proposes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag


_COLOR_RE = re.compile(r"(#[0-9a-fA-F]{3,8}\b|rgb\([^)]*\)|rgba\([^)]*\))")


def _index_by_oid(soup: BeautifulSoup) -> dict[str, Tag]:
    """Return {data_oid_value: tag} for every annotated tag.

    Duplicates (same oid on multiple tags) keep the first — annotator
    uses suffixes to dedupe so this should never collide in practice;
    when it does, surfacing the first is better than dropping all.
    """
    out: dict[str, Tag] = {}
    for t in soup.find_all(attrs={"data-oid": True}):
        if not isinstance(t, Tag):
            continue
        oid = t.get("data-oid")
        if isinstance(oid, str) and oid not in out:
            out[oid] = t
    return out


def _normalized_text(t: Tag) -> str:
    """Return a normalized text payload for the tag — collapses
    whitespace so the diff doesn't trip on indentation changes."""
    return re.sub(r"\s+", " ", t.get_text(" ", strip=True)).strip()


def _style_attrs(t: Tag) -> dict[str, str]:
    """Style-relevant attributes pulled flat for diffing."""
    out: dict[str, str] = {}
    for k in ("style", "class", "color"):
        v = t.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            v = " ".join(v)
        out[k] = str(v)
    return out


def _kinds_for_change(
    orig: Tag,
    edited: Tag,
) -> tuple[list[str], dict[str, Any]]:
    """Compare ``orig`` and ``edited`` and return the kinds of changes
    detected plus a small metadata payload."""
    kinds: list[str] = []
    detail: dict[str, Any] = {}

    o_text = _normalized_text(orig)
    e_text = _normalized_text(edited)
    if o_text != e_text:
        # Heuristic: short text → "copy" edit, long → "text"
        kinds.append("copy" if max(len(o_text), len(e_text)) <= 200 else "text")
        detail["text_before"] = o_text
        detail["text_after"] = e_text

    o_style = _style_attrs(orig)
    e_style = _style_attrs(edited)
    if o_style != e_style:
        kinds.append("style")
        detail["style_before"] = o_style
        detail["style_after"] = e_style
        # Detect color shifts specifically — these route to design_tokens.
        before_colors = sorted(set(_COLOR_RE.findall(o_style.get("style", ""))))
        after_colors = sorted(set(_COLOR_RE.findall(e_style.get("style", ""))))
        if before_colors != after_colors:
            kinds.append("color")
            detail["color_before"] = before_colors
            detail["color_after"] = after_colors

    o_struct = [c.name for c in orig.find_all(True, recursive=False)]
    e_struct = [c.name for c in edited.find_all(True, recursive=False)]
    if o_struct != e_struct:
        kinds.append("structure")
        detail["children_before"] = o_struct
        detail["children_after"] = e_struct

    return kinds, detail


def _suggested_target(kinds: list[str]) -> str:
    """Map change kinds → recommended upstream artifact.

    Cheap heuristic; founder reviewer makes the final call.
    """
    if "color" in kinds:
        return "design_tokens (style guide color slot)"
    if "style" in kinds:
        return "style guide / design_tokens"
    if "structure" in kinds:
        return "screen plan (layout section)"
    if "copy" in kinds or "text" in kinds:
        return "screen plan (copy slot)"
    return "screen plan"


def _render_proposal_markdown(
    html_path: str,
    edited_html_path: str,
    changes: list[dict],
    missing_oids: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Spec Patch Proposal")
    lines.append("")
    lines.append(f"- **Original**: `{html_path}`")
    lines.append(f"- **Edited**: `{edited_html_path}`")
    lines.append(f"- **Changes detected**: {len(changes)}")
    lines.append("")
    if not changes:
        lines.append("_No `data-oid`-anchored changes detected._")
        return "\n".join(lines) + "\n"

    for c in changes:
        lines.append(f"## `{c['data_oid']}` ({', '.join(c['kinds'])})")
        lines.append("")
        lines.append(f"- **Suggested target**: {c['suggested_target']}")
        d = c.get("detail") or {}
        if "text_before" in d:
            lines.append(f"- Copy: `{d.get('text_before','')}` → `{d.get('text_after','')}`")
        if "color_before" in d:
            lines.append(
                f"- Color: {d.get('color_before')} → {d.get('color_after')}"
            )
        elif "style_before" in d:
            lines.append(
                f"- Style: `{d.get('style_before')}` → `{d.get('style_after')}`"
            )
        if "children_before" in d:
            lines.append(
                f"- Structure: {d.get('children_before')} → {d.get('children_after')}"
            )
        lines.append("")
    if missing_oids:
        lines.append("## Notes")
        lines.append("")
        lines.append(
            f"- {len(missing_oids)} edited tag(s) lacked `data-oid` and "
            "could not be reverse-mapped. Run `annotate_html_oids` on "
            "the originals before editing offline."
        )
    return "\n".join(lines) + "\n"


def propose_spec_patch_from_html_diff(
    *,
    html_path: str,
    edited_html_path: str,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Diff two HTML files and propose spec-level patches per change.

    Parameters
    ----------
    html_path:
        Original (annotated) HTML.
    edited_html_path:
        Founder-edited HTML.
    out_path:
        Optional ``spec_patch_proposal.md`` output path.

    Returns
    -------
    dict
        ``ok``, ``changes`` (list of {data_oid, kinds, detail,
        suggested_target}), ``missing_oids`` (list), ``proposal_md``,
        ``proposal_path`` (when written), ``error`` (when ``ok=False``).
    """
    try:
        orig_text = Path(html_path).read_text(encoding="utf-8")
        edit_text = Path(edited_html_path).read_text(encoding="utf-8")
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}

    orig_soup = BeautifulSoup(orig_text, "html.parser")
    edit_soup = BeautifulSoup(edit_text, "html.parser")

    orig_oids = _index_by_oid(orig_soup)
    edit_oids = _index_by_oid(edit_soup)

    changes: list[dict[str, Any]] = []

    for oid, orig_tag in orig_oids.items():
        edit_tag = edit_oids.get(oid)
        if edit_tag is None:
            # The oid was removed in the edit — caller may have
            # restructured. Surface as a "removed" change kind.
            changes.append({
                "data_oid": oid,
                "kinds": ["removed"],
                "detail": {"removed": True},
                "suggested_target": "screen plan (section removed)",
            })
            continue
        kinds, detail = _kinds_for_change(orig_tag, edit_tag)
        if not kinds:
            continue
        changes.append({
            "data_oid": oid,
            "kinds": kinds,
            "detail": detail,
            "suggested_target": _suggested_target(kinds),
        })

    # Edited tags that have NO oid in the original — caller stripped or
    # added new sections. Surface count for the founder; we can't pair
    # them to a spec node.
    missing_oids: list[str] = []
    edited_unannotated_count = 0
    for t in edit_soup.find_all(True):
        if not isinstance(t, Tag):
            continue
        if not t.has_attr("data-oid"):
            # Only count tags that LOOK like semantic blocks
            if (t.name or "").lower() in (
                "header", "main", "nav", "footer",
                "aside", "section", "article",
            ):
                edited_unannotated_count += 1
    if edited_unannotated_count:
        missing_oids.append(
            f"{edited_unannotated_count} semantic tag(s) lacked data-oid"
        )

    proposal_md = _render_proposal_markdown(
        html_path=html_path,
        edited_html_path=edited_html_path,
        changes=changes,
        missing_oids=missing_oids,
    )

    proposal_path: str | None = None
    if out_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            Path(out_path).write_text(proposal_md, encoding="utf-8")
            proposal_path = out_path
        except Exception as e:
            return {"ok": False, "error": f"write proposal failed: {e}"}

    return {
        "ok": True,
        "changes": changes,
        "missing_oids": missing_oids,
        "proposal_md": proposal_md,
        "proposal_path": proposal_path,
    }
