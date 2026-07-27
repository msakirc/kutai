"""ADR register verifier — Tier 2 of Z1 (P3).

Validates the per-mission ``register.md`` against the on-disk ADR JSON
decision files. The i2p pipeline writes one ``<slug>_decision.json`` per
architecture domain (steps 4.1/4.2/4.2a/4.4/4.6/4.8/4.9/4.10) and step
4.14 writes ``register.md`` indexing them.

Matching is by **domain slug** (the deterministic filename), NOT by ADR
id: the register's ADR ids and each JSON doc's internal ``adr_id`` are
both LLM-invented and never reconcile, so any id-equality check is
impossible-by-design (this is why the previous filename==ADR-id model
never once passed on a real mission). Instead the gate asserts:

  1. **Coverage** — every required architecture domain has a
     ``<slug>_decision.json`` on disk (no topic silently missing).
  2. **Register is real** — present, non-empty, and indexes at least as
     many ADR entries as there are required domains (not a stub/lying
     index that claims completeness without listing the decisions).

Pure I/O wrapper; no LLM. Paths are resolved by the caller against the
workspace root (see ``_resolve_path_list`` in ``mr_roboto.__init__``);
``workspace_path`` is accepted as an enumeration fallback.

Returns
-------
dict
    ``ok`` (bool), ``required`` (domain slugs asked for), ``on_disk``
    (domain slugs found as ``*_decision.json``), ``missing_domains``
    (required but absent), ``orphan_domains`` (on disk but not required,
    informational — never fails), ``register_adr_count`` (ADR ids parsed
    from the register).
"""
from __future__ import annotations

import os
import re
from typing import Any

_ADR_ID_RE = re.compile(r"\bADR-\d{4}-\d{2}-\d{2}-\d{2,4}(?:-[A-Za-z0-9_-]+)?\b")

# Canonical architecture-decision domains produced by i2p steps
# 4.1/4.2/4.2a/4.4/4.6/4.8/4.9/4.10 — each writes ``.adr/<slug>_decision.json``.
# Single source of truth for the register-coverage gate; a mission whose
# ADR set differs can override via the ``required_domains`` argument
# (wired from the check payload).
REQUIRED_ADR_DOMAINS: list[str] = [
    "architecture_pattern",
    "tech_stack",
    "component_library",
    "database_schema",
    "auth_design",
    "third_party_selections",
    "infrastructure_designs",
    "communication_designs",
]

_DECISION_SUFFIX = "_decision"


def _gather_register(
    register_text: str | None, register_path: str | None
) -> tuple[str, str | None]:
    """Return ``(text, parent_dir)``; parent_dir is None when only text given."""
    if register_text is not None:
        return register_text, None
    if not register_path:
        return "", None
    try:
        with open(register_path, encoding="utf-8") as fh:
            text = fh.read()
        parent = os.path.dirname(register_path) or "."
        return text, parent
    except OSError:
        return "", os.path.dirname(register_path) if register_path else None


def _domain_slug(filename: str) -> str:
    """``auth_design_decision.json`` -> ``auth_design`` (strip suffix + ext)."""
    stem = filename[:-5] if filename.endswith(".json") else filename
    if stem.endswith(_DECISION_SUFFIX):
        stem = stem[: -len(_DECISION_SUFFIX)]
    return stem


def _enumerate_on_disk(search_dirs: list[str]) -> list[str]:
    """Collect domain slugs from ``*.json`` ADR docs across candidate dirs."""
    seen: set[str] = set()
    for d in search_dirs:
        if d and os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".json"):
                    seen.add(_domain_slug(name))
    return sorted(seen)


def verify_adr_register(
    *,
    register_text: str | None = None,
    register_path: str | None = None,
    adr_dir: str | None = None,
    workspace_path: str | None = None,
    required_domains: list[str] | None = None,
    allow_empty_register: bool = False,
) -> dict[str, Any]:
    """Validate the ADR register against the on-disk ADR decision docs.

    See module docstring for the contract and output schema.
    """
    text, inferred_dir = _gather_register(register_text, register_path)

    required = (
        list(required_domains) if required_domains else list(REQUIRED_ADR_DOMAINS)
    )

    if not text.strip() and not allow_empty_register:
        return {
            "ok": False,
            "error": "empty or missing register",
            "required": required,
            "on_disk": [],
            "missing_domains": required,
            "orphan_domains": [],
            "register_adr_count": 0,
        }

    search_dirs: list[str] = []
    for d in (adr_dir, inferred_dir):
        if d and d not in search_dirs:
            search_dirs.append(d)
    if workspace_path:
        wsd = os.path.join(workspace_path, ".adr")
        if wsd not in search_dirs:
            search_dirs.append(wsd)

    on_disk = _enumerate_on_disk(search_dirs)
    on_disk_set = set(on_disk)

    missing_domains = [d for d in required if d not in on_disk_set]
    orphan_domains = [d for d in on_disk if d not in required]

    referenced = {m.group(0) for m in _ADR_ID_RE.finditer(text)}
    register_adr_count = len(referenced)

    if not referenced and allow_empty_register:
        # Escape hatch: register landed before any ADR was indexed.
        return {
            "ok": True,
            "required": required,
            "on_disk": on_disk,
            "missing_domains": [],
            "orphan_domains": orphan_domains,
            "register_adr_count": 0,
        }

    register_complete = register_adr_count >= len(required)
    ok = (not missing_domains) and register_complete

    return {
        "ok": ok,
        "required": required,
        "on_disk": on_disk,
        "missing_domains": missing_domains,
        "orphan_domains": orphan_domains,
        "register_adr_count": register_adr_count,
    }
