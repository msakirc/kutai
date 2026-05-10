"""Z1 Tier 5A (P6) — verify referenced compliance templates exist on disk.

Post-hook on step ``1.11a compliance_overlay``. Reads the overlay artifact
(or the explicit ``template_ids`` payload) and asserts each ``template_id``
resolves to a file under the repo's ``compliance_templates/`` root.

Fail-fast: if ANY referenced template is missing, return failed with the
list of missing IDs. Reviewers/founder must add the template before the
mission can proceed past the overlay step.
"""
from __future__ import annotations

import json
import os
from typing import Any


# Repo-root template directory. Mirrors the v3 plan: jurisdiction × lang ×
# doc_type. Resolution is best-effort: we accept ``<id>.md.j2`` anywhere
# under the root (recursive walk) so callers can pass the bare doc-type
# template id without locking jurisdictions.
TEMPLATE_ROOT = "compliance_templates"


def _candidate_filenames(template_id: str) -> list[str]:
    """Return the filenames we'd accept for a given template id."""
    return [
        f"{template_id}.md.j2",
        f"{template_id}.md",
        f"{template_id}.j2",
    ]


def _walk_templates(root: str) -> set[str]:
    """Return a set of every filename present under ``root`` (recursive)."""
    found: set[str] = set()
    if not os.path.isdir(root):
        return found
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            found.add(fn)
    return found


def compliance_template_present(
    template_ids: list[str] | None = None,
    overlay_path: str | None = None,
    overlay_obj: dict[str, Any] | None = None,
    template_root: str | None = None,
) -> dict[str, Any]:
    """Verify every referenced template exists under ``compliance_templates/``.

    Inputs (any one is sufficient):

    - ``template_ids``: explicit list (highest precedence)
    - ``overlay_obj``: the parsed compliance_overlay dict
    - ``overlay_path``: path to ``compliance_overlay.json`` on disk

    Returns ``{"ok": bool, "missing": [...], "checked": [...], "root": str}``.
    """
    root = template_root or TEMPLATE_ROOT

    ids: list[str] = []
    if template_ids:
        ids = list(template_ids)
    else:
        if overlay_obj is None and overlay_path:
            try:
                with open(overlay_path, "r", encoding="utf-8") as fh:
                    overlay_obj = json.load(fh)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"could not read overlay: {e}",
                    "missing": [],
                    "checked": [],
                    "root": root,
                }
        if overlay_obj:
            for doc in overlay_obj.get("required_documents") or []:
                tid = doc.get("template_id")
                if tid and tid not in ids:
                    ids.append(tid)

    if not ids:
        # Nothing to check — that's OK (e.g. fingerprint with no jurisdictions).
        return {"ok": True, "missing": [], "checked": [], "root": root}

    available = _walk_templates(root)
    missing: list[str] = []
    for tid in ids:
        if not any(cand in available for cand in _candidate_filenames(tid)):
            missing.append(tid)

    return {
        "ok": not missing,
        "missing": missing,
        "checked": ids,
        "root": root,
    }
