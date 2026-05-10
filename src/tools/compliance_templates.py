"""Z1 Tier 5A (P6) — LLM-callable compliance template renderer.

Renders a Jinja2 template from ``compliance_templates/`` against a
fingerprint dict and returns the rendered Markdown. Hand-curated templates
only — see ``compliance_templates/README.md`` for the contract.

Lookup order:
1. ``compliance_templates/<jurisdiction>/<lang>/<doc_type>.md.j2``
2. ``compliance_templates/default/<lang>/<doc_type>.md.j2``
3. ``compliance_templates/default/en/<doc_type>.md.j2``

A sibling ``.meta.json`` (``{"version": "...", "last_reviewed": "YYYY-MM-DD"}``)
is read when present; templates older than ``STALE_DAYS`` are flagged but
still rendered (warning, not block — block is reviewer's call).
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("tools.compliance_templates")

TEMPLATE_ROOT = "compliance_templates"
STALE_DAYS = 180


def _resolve_template(
    doc_type: str,
    jurisdiction: str,
    lang: str,
    root: str,
) -> str | None:
    candidates = [
        os.path.join(root, jurisdiction, lang, f"{doc_type}.md.j2"),
        os.path.join(root, "default", lang, f"{doc_type}.md.j2"),
        os.path.join(root, "default", "en", f"{doc_type}.md.j2"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def compliance_template_render(
    fingerprint: dict[str, Any],
    doc_type: str,
    lang: str = "en",
    template_root: str | None = None,
) -> dict[str, Any]:
    """Render a compliance template.

    Returns ``{"ok": bool, "rendered": str, "template_id": str,
    "template_version": str, "stale": bool, "template_path": str}``
    on success, or ``{"ok": False, "error": str}`` when no template
    matches.
    """
    try:
        from jinja2 import Template
    except ImportError:  # pragma: no cover — jinja2 is a soft dep
        return {"ok": False, "error": "jinja2 not installed"}

    root = template_root or TEMPLATE_ROOT
    juris_list = fingerprint.get("jurisdictions") or []
    juris = juris_list[0] if juris_list else "default"

    tpl_path = _resolve_template(doc_type, juris, lang, root)
    if not tpl_path:
        return {
            "ok": False,
            "error": f"no template for {doc_type} in {juris}/{lang} (root={root})",
        }

    meta_path = tpl_path.replace(".md.j2", ".meta.json")
    version = "0"
    stale = False
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            version = str(meta.get("version", "0"))
            lr = meta.get("last_reviewed")
            if lr:
                try:
                    last_reviewed = datetime.date.fromisoformat(str(lr))
                    age = (datetime.date.today() - last_reviewed).days
                    if age > STALE_DAYS:
                        stale = True
                except ValueError:
                    pass
        except Exception as e:
            logger.warning("compliance_template_render: meta read fail %s: %s", meta_path, e)

    try:
        with open(tpl_path, "r", encoding="utf-8") as fh:
            tpl_src = fh.read()
        rendered = Template(tpl_src).render(
            **fingerprint,
            generated_at=datetime.datetime.utcnow().isoformat(),
        )
    except Exception as e:
        return {"ok": False, "error": f"render failed: {e}"}

    return {
        "ok": True,
        "rendered": rendered,
        "template_id": doc_type,
        "template_version": version,
        "stale": stale,
        "template_path": tpl_path,
    }
