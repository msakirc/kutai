"""Static arg-binding + bind cache for parametric artifacts.

Phase 2 is static-only — NO LLM-bind. Pipeline per matched parametric
artifact:
  1. try each inputs_schema.<field>.bind_from dotted path against the
     task context; first non-null wins; fall back to declared default.
  2. if every field filled → prebind-ready (complete=True).
  3. if any null → caller consults the embedding-keyed bind cache.
  4. cold cache-miss → caller renders plain prose inject (no LLM call).

The bind cache is keyed by an embedding of the relevant task-context
fields; a hit ≥ BIND_CACHE_HIT_THRESHOLD reuses the cached args.
"""
from __future__ import annotations

import json
import struct

from yazbunu import get_logger

logger = get_logger("intersect.binding")

BIND_CACHE_HIT_THRESHOLD: float = 0.92


def _resolve_path(ctx: dict, dotted: str):
    """Walk a dotted path through nested dicts; return None on any miss."""
    cur = ctx
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def static_bind(artifact, task_ctx: dict) -> tuple[dict, bool]:
    """Resolve all inputs_schema fields statically.

    Returns ``(bound_args, complete)``. ``complete`` is True when every
    schema field resolved (via a bind_from path or a declared default).
    Non-parametric artifacts return ``({}, True)``.
    """
    schema = getattr(artifact, "inputs_schema", None) or {}
    if not schema:
        return {}, True

    bound: dict = {}
    complete = True
    for field, rules in schema.items():
        if not isinstance(rules, dict):
            continue
        value = None
        for path in rules.get("bind_from", []) or []:
            value = _resolve_path(task_ctx, str(path))
            if value is not None:
                break
        if value is None and "default" in rules:
            value = rules["default"]
        bound[field] = value
        if value is None:
            complete = False
    return bound, complete


async def _embed_ctx(task_ctx: dict) -> bytes:
    """Embed a stable JSON of the task context for cache keying.

    Uses the project embedding model (multilingual-e5-base, 768d).
    Falls back to an empty blob on any failure — a missing embedding
    simply means every cache lookup misses (safe degrade).
    """
    try:
        from src.memory.embeddings import get_embedding
        text = json.dumps(task_ctx, sort_keys=True, ensure_ascii=False)[:2000]
        vec = await get_embedding(text, is_query=False)
        if vec is None:
            return b""
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("bind-cache embed failed: %s", exc)
        return b""


def _cosine(a: bytes, b: bytes) -> float:
    """Cosine similarity between two packed float32 blobs.

    Returns 1.0 when both blobs are identical (short-circuit — also
    handles the b'' == b'' case from failed embeddings, where the
    cache key is deterministically the same).
    """
    if a == b and len(a) > 0:
        return 1.0
    if not a or not b or len(a) != len(b):
        return 0.0
    n = len(a) // 4
    va = struct.unpack(f"{n}f", a)
    vb = struct.unpack(f"{n}f", b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def lookup_bind_cache(artifact, task_ctx: dict) -> dict | None:
    """Return cached bound args if a row matches ctx ≥ threshold, else None."""
    try:
        from dabidabi import get_db
        db = await get_db()
        target = await _embed_ctx(task_ctx)
        manifest_id = getattr(artifact, "artifact_id", None)
        cur = await db.execute(
            "SELECT id, ctx_embedding, bound_args_json "
            "FROM yalayut_bind_cache WHERE manifest_id = ?",
            (manifest_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
        best_id = None
        best_args = None
        best_sim = 0.0
        for row_id, emb, args_json in rows:
            sim = _cosine(target, emb or b"")
            if sim >= BIND_CACHE_HIT_THRESHOLD and sim > best_sim:
                best_sim, best_id, best_args = sim, row_id, args_json
        if best_id is None:
            return None
        await db.execute(
            "UPDATE yalayut_bind_cache "
            "SET hit_count = hit_count + 1, last_used_at = datetime('now') "
            "WHERE id = ?",
            (best_id,),
        )
        await db.commit()
        return json.loads(best_args) if best_args else None
    except Exception as exc:
        logger.debug("bind-cache lookup failed: %s", exc)
        return None


async def write_bind_cache(artifact, task_ctx: dict, bound_args: dict) -> None:
    """Persist a freshly-bound arg set for future cache hits."""
    try:
        from dabidabi import get_db
        db = await get_db()
        emb = await _embed_ctx(task_ctx)
        manifest_id = getattr(artifact, "artifact_id", None)
        await db.execute(
            "INSERT INTO yalayut_bind_cache "
            "(manifest_id, ctx_embedding, bound_args_json, hit_count, "
            " created_at, last_used_at) "
            "VALUES (?, ?, ?, 0, datetime('now'), datetime('now'))",
            (manifest_id, emb,
             json.dumps(bound_args, ensure_ascii=False)),
        )
        await db.commit()
    except Exception as exc:
        logger.debug("bind-cache write failed: %s", exc)
