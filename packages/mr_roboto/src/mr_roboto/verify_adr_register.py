"""ADR register verifier — Tier 2 of Z1 (P3).

Reads ``register.md`` (the per-mission ADR index) and asserts:
  1. Every ADR id referenced in the register has a corresponding JSON file
     on disk under the same ``.adr/`` directory.
  2. There are no orphan ADR JSON files in ``.adr/`` that the register
     does not reference.
  3. The register format is parseable (one row per ADR, contains the
     ``ADR-YYYY-MM-DD-NNN`` id).

Pure I/O wrapper; no LLM. Caller passes ``register_path`` and the parent
``.adr/`` dir is inferred from it (or passed explicitly).

Returns
-------
dict
    ``ok`` (bool), ``referenced`` (list of ids in register), ``on_disk``
    (list of ADR-id stems found as JSON files), ``missing_files`` (in
    register but not on disk), ``orphan_files`` (on disk but not in
    register).
"""
from __future__ import annotations

import os
import re
from typing import Any

_ADR_ID_RE = re.compile(r"\bADR-\d{4}-\d{2}-\d{2}-\d{2,4}(?:-[A-Za-z0-9_-]+)?\b")


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


def verify_adr_register(
    *,
    register_text: str | None = None,
    register_path: str | None = None,
    adr_dir: str | None = None,
    allow_empty_register: bool = False,
) -> dict[str, Any]:
    """Validate the ADR register against the on-disk ADR JSON files.

    See module docstring for output schema.
    """
    text, inferred_dir = _gather_register(register_text, register_path)
    if not text.strip() and not allow_empty_register:
        return {
            "ok": False,
            "error": "empty or missing register",
            "referenced": [],
            "on_disk": [],
            "missing_files": [],
            "orphan_files": [],
        }

    referenced = sorted({m.group(0) for m in _ADR_ID_RE.finditer(text)})

    target_dir = adr_dir or inferred_dir
    on_disk: list[str] = []
    if target_dir and os.path.isdir(target_dir):
        for name in os.listdir(target_dir):
            if not name.endswith(".json"):
                continue
            stem = name[:-5]
            if _ADR_ID_RE.fullmatch(stem):
                on_disk.append(stem)
    on_disk.sort()

    missing_files = [aid for aid in referenced if aid not in on_disk]
    orphan_files = [aid for aid in on_disk if aid not in referenced]

    if not referenced and allow_empty_register:
        # Empty-register escape hatch — used when the workflow lands the
        # register before any ADR has been written (legacy phase-4 missions).
        return {
            "ok": True,
            "referenced": [],
            "on_disk": on_disk,
            "missing_files": [],
            "orphan_files": orphan_files,
        }

    ok = not missing_files and not orphan_files and bool(referenced)
    return {
        "ok": ok,
        "referenced": referenced,
        "on_disk": on_disk,
        "missing_files": missing_files,
        "orphan_files": orphan_files,
    }
