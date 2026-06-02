"""Mechanical verb — prior-art candidate fetch (i2p step 1.0b).

Reads the 1.0a query artifact, runs the deterministic multi-source fetch
in ``src.research.prior_art.fetch_candidates``, and writes the candidates
artifact for the 1.0c synthesis LLM. No LLM call, no fabrication: every
candidate URL was really fetched + HEAD-resolved.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.prior_art_fetch")


def _resolve(path: str | None) -> str | None:
    if not path or os.path.isabs(path) or os.path.isfile(path):
        return path
    try:
        from src.tools.workspace import WORKSPACE_DIR
        return os.path.join(WORKSPACE_DIR, path)
    except Exception:
        return path


async def prior_art_fetch(
    queries_path: str,
    candidates_path: str,
    *,
    db_path: str | None = None,
) -> dict[str, Any]:
    qpath = _resolve(queries_path)
    if not qpath or not os.path.isfile(qpath):
        return {"ok": False, "error": f"queries artifact missing: {queries_path}"}
    try:
        with open(qpath, encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as e:
        return {"ok": False, "error": f"failed to read queries: {e}"}

    queries = spec.get("queries") or []
    keywords = spec.get("domain_keywords") or []
    tier = spec.get("ambition_tier") or "private_beta"

    from src.research.prior_art import fetch_candidates
    out = await fetch_candidates(
        queries=queries, domain_keywords=keywords,
        ambition_tier=tier, db_path=db_path)

    cpath = _resolve(candidates_path) or candidates_path
    os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
    with open(cpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, default=str)

    return {"ok": True,
            "candidate_count": len(out.get("candidates") or []),
            "candidates_path": candidates_path}
