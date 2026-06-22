"""Whole-word presence check: does an artifact contain the canonical product name?

Pure / deterministic — no I/O. The mr_roboto dispatch branch supplies the
`product_name` (read from the artifact store) and the artifact texts (read from
disk); this module only decides present/absent. When `product_name` is empty or
None the check is a defensive SKIP (ok=True) — we never hard-block on our own
missing precondition; the reviewer backstop (1.13 check 10) covers that case.
"""
from __future__ import annotations

import re
from typing import Any


def _whole_word_present(text: str, name: str) -> bool:
    if not text or not name:
        return False
    return re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE) is not None


def verify_contains_product_name(
    *, product_name: str | None, artifact_texts: list[str]
) -> dict[str, Any]:
    name = (product_name or "").strip()
    if not name:
        return {
            "ok": True, "skipped": "no product_name pinned",
            "product_name": None, "checked": len(artifact_texts), "found": False,
        }
    found = any(_whole_word_present(t or "", name) for t in artifact_texts)
    return {
        "ok": bool(found), "product_name": name,
        "checked": len(artifact_texts), "found": found,
    }
