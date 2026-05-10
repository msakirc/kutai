"""Z1 Tier 6A (A7) — idea fingerprint + cross-mission dedup at intake.

Post-hook on step ``0.1 product_charter``. After the idea-brief artifact
lands, vector-search the embedding against past mission ``idea_brief``
artifacts (ChromaDB collection ``mission_ideas``). If similarity to any
prior mission exceeds the threshold (default 0.85, override via
``KUTAI_IDEA_DEDUP_THRESHOLD``) the action emits
``mission_<id>/similar_missions.md`` and returns ``status="needs_review"``
so the founder can decide via Telegram (Continue / Branch from #N / Abort).

The collection ``mission_ideas`` is intentionally separate from the
existing seven (``episodic``, ``semantic``, ``codebase`` …) — its rows
are mission-scoped fingerprints, not retrieval-time RAG sources, and we
never want them mixing into agent context.

Indexing is decoupled from search: callers invoke
:func:`index_idea_fingerprint` after the founder accepts (Continue or
"new mission") so the in-flight idea doesn't match itself.
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.find_similar_missions")

# Separate collection so mission idea fingerprints do not pollute the
# RAG pipeline. Same chroma_data dir as the seven core collections.
MISSION_IDEAS_COLLECTION = "mission_ideas"

DEFAULT_SIMILARITY_THRESHOLD = 0.85


def _threshold() -> float:
    """Resolve similarity threshold (env-overridable)."""
    raw = os.getenv("KUTAI_IDEA_DEDUP_THRESHOLD")
    if not raw:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "find_similar_missions: invalid KUTAI_IDEA_DEDUP_THRESHOLD=%r; "
            "using default %.2f",
            raw, DEFAULT_SIMILARITY_THRESHOLD,
        )
        return DEFAULT_SIMILARITY_THRESHOLD


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _load_idea_text(workspace_path: str) -> str | None:
    """Read the freshest idea-brief / charter text from the workspace.

    Tries (in order): ``.charter/product_charter.md``, ``idea_brief.md``,
    ``.charter/idea_brief.md``. Returns the raw text or None.
    """
    candidates = [
        os.path.join(workspace_path, ".charter", "product_charter.md"),
        os.path.join(workspace_path, "idea_brief.md"),
        os.path.join(workspace_path, ".charter", "idea_brief.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except Exception as e:
                logger.warning(
                    "find_similar_missions: failed to read %s: %s", path, e,
                )
    return None


async def _ensure_collection() -> Any:
    """Lazily create + return the `mission_ideas` chroma collection."""
    from src.memory import vector_store

    if not vector_store.is_ready():
        await vector_store.init_store()

    # Side-channel — vector_store keeps its own COLLECTIONS list; we
    # reuse the underlying client directly so mission_ideas does not
    # need to be in the core RAG list.
    client = getattr(vector_store, "_client", None)
    if client is None:
        return None

    cols = getattr(vector_store, "_collections", {}) or {}
    if MISSION_IDEAS_COLLECTION in cols:
        return cols[MISSION_IDEAS_COLLECTION]

    import asyncio
    try:
        # Reuse the same RefuseEmbedFunction guard that vector_store
        # builds at init — pull it off any existing collection.
        sample_ef = None
        for c in cols.values():
            sample_ef = getattr(c, "_embedding_function", None)
            if sample_ef is not None:
                break

        from src.memory.embeddings import (
            EMBEDDING_MODEL,
            get_expected_dimension,
        )
        kwargs = dict(
            name=MISSION_IDEAS_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": get_expected_dimension(),
            },
        )
        if sample_ef is not None:
            kwargs["embedding_function"] = sample_ef
        col = await asyncio.to_thread(
            client.get_or_create_collection, **kwargs
        )
        cols[MISSION_IDEAS_COLLECTION] = col
        return col
    except Exception as e:
        logger.warning(
            "find_similar_missions: ensure_collection failed: %s", e,
        )
        return None


async def _embed(text: str, is_query: bool) -> list[float] | None:
    from src.memory.embeddings import get_embedding
    return await get_embedding(text, is_query=is_query)


async def find_similar_missions(
    mission_id: int,
    idea_summary: str | None = None,
    *,
    workspace_path: str | None = None,
    top_k: int = 3,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Search prior mission idea fingerprints for similarity matches.

    Returns a dict shaped::

        {
            "ok": bool,                 # False when threshold breached
            "matches": [                # always sorted high → low
                {"mission_id": int, "similarity": float,
                 "title": str, "final_status_note": str},
                ...
            ],
            "threshold": float,
            "report_path": str | None,  # similar_missions.md when matches
            "checked": bool,            # True if a query actually ran
            "reason": str | None,       # diagnostic on no-op paths
        }

    "ok" is False when at least one match exceeds the threshold (the
    founder must decide). Empty collection / no idea text → ok=True.
    """
    thr = threshold if threshold is not None else _threshold()
    ws = _resolve_workspace(mission_id, workspace_path)

    text = (idea_summary or "").strip()
    if not text:
        loaded = _load_idea_text(ws)
        text = (loaded or "").strip()

    if not text:
        return {
            "ok": True,
            "matches": [],
            "threshold": thr,
            "report_path": None,
            "checked": False,
            "reason": "no idea text available",
        }

    col = await _ensure_collection()
    if col is None:
        return {
            "ok": True,
            "matches": [],
            "threshold": thr,
            "report_path": None,
            "checked": False,
            "reason": "mission_ideas collection unavailable",
        }

    import asyncio
    try:
        count = await asyncio.to_thread(col.count)
    except Exception as e:
        logger.warning("find_similar_missions: count failed: %s", e)
        count = 0
    if count == 0:
        return {
            "ok": True,
            "matches": [],
            "threshold": thr,
            "report_path": None,
            "checked": True,
            "reason": "empty collection",
        }

    embedding = await _embed(text, is_query=True)
    if embedding is None:
        return {
            "ok": True,
            "matches": [],
            "threshold": thr,
            "report_path": None,
            "checked": False,
            "reason": "embedding unavailable",
        }

    try:
        results = await asyncio.to_thread(
            lambda: col.query(
                query_embeddings=[embedding],
                n_results=min(top_k + 1, count),  # +1 in case self-row
            )
        )
    except Exception as e:
        logger.warning("find_similar_missions: query failed: %s", e)
        return {
            "ok": True,
            "matches": [],
            "threshold": thr,
            "report_path": None,
            "checked": False,
            "reason": f"query failed: {e}",
        }

    matches: list[dict[str, Any]] = []
    if results and results.get("ids") and results["ids"][0]:
        ids = results["ids"][0]
        metas = results.get("metadatas", [[]])[0] or [{}] * len(ids)
        distances = results.get("distances", [[]])[0] or [0.0] * len(ids)
        for i, doc_id in enumerate(ids):
            meta = metas[i] or {}
            other_mid = meta.get("mission_id")
            # Skip self — never match a mission against its own embedding.
            try:
                if int(other_mid) == int(mission_id):
                    continue
            except (TypeError, ValueError):
                pass
            # cosine distance → similarity (chromadb cosine is 1 - sim).
            sim = max(0.0, min(1.0, 1.0 - float(distances[i])))
            matches.append({
                "mission_id": (
                    int(other_mid) if other_mid is not None else None
                ),
                "doc_id": doc_id,
                "similarity": sim,
                "title": str(meta.get("title", "") or ""),
                "final_status_note": str(meta.get("final_status_note", "") or ""),
            })

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    matches = matches[:top_k]

    breaches = [m for m in matches if m["similarity"] >= thr]
    report_path: str | None = None
    if breaches:
        report_path = _write_report(ws, breaches, thr)

    return {
        "ok": not breaches,
        "matches": matches,
        "threshold": thr,
        "report_path": report_path,
        "checked": True,
        "reason": None,
    }


def _write_report(
    workspace_path: str, matches: list[dict[str, Any]], threshold: float,
) -> str:
    """Emit ``similar_missions.md`` for founder review."""
    os.makedirs(workspace_path, exist_ok=True)
    out = os.path.join(workspace_path, "similar_missions.md")
    lines: list[str] = [
        "# Similar prior missions",
        "",
        f"_threshold: {threshold:.2f}_",
        "",
        "The current idea is highly similar to one or more prior missions.",
        "The founder must choose: **Continue**, **Branch from a prior**, or **Abort**.",
        "",
    ]
    for m in matches:
        lines.append(
            f"## Prior mission #{m.get('mission_id')} "
            f"(similarity {m['similarity']:.3f})"
        )
        if m.get("title"):
            lines.append(f"- **Title:** {m['title']}")
        if m.get("final_status_note"):
            lines.append(f"- **Final status note:** {m['final_status_note']}")
        lines.append("")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except Exception as e:
        logger.warning(
            "find_similar_missions: failed to write report %s: %s", out, e,
        )
    return out


async def index_idea_fingerprint(
    mission_id: int,
    idea_summary: str | None = None,
    *,
    workspace_path: str | None = None,
    title: str = "",
    final_status_note: str = "",
) -> dict[str, Any]:
    """Embed + store the mission's idea fingerprint in ``mission_ideas``.

    Idempotent: doc_id = ``mission_<id>`` so re-indexing overwrites.
    Called only after the founder confirms a new mission (or "Continue"
    on the dedup decision) — never before, because the freshly indexed
    embedding would self-match on the next intake.
    """
    ws = _resolve_workspace(mission_id, workspace_path)
    text = (idea_summary or "").strip()
    if not text:
        text = (_load_idea_text(ws) or "").strip()
    if not text:
        return {"ok": False, "reason": "no idea text"}

    col = await _ensure_collection()
    if col is None:
        return {"ok": False, "reason": "collection unavailable"}

    embedding = await _embed(text, is_query=False)
    if embedding is None:
        return {"ok": False, "reason": "embedding failed"}

    import asyncio
    import time
    doc_id = f"mission_{int(mission_id)}"
    meta = {
        "mission_id": int(mission_id),
        "title": title or "",
        "final_status_note": final_status_note or "",
        "indexed_at": time.time(),
    }
    try:
        await asyncio.to_thread(
            col.upsert,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text[:8192]],  # cap so we don't blow the segment
            metadatas=[meta],
        )
    except Exception as e:
        logger.warning("index_idea_fingerprint: upsert failed: %s", e)
        return {"ok": False, "reason": f"upsert failed: {e}"}
    return {"ok": True, "doc_id": doc_id}
