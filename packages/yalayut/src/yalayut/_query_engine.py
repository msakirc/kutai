"""query(task_ctx) -> ranked list[Artifact].

Hot read path. Vector cosine similarity of the task text embedding against
every enabled artifact's stored embedding. yalayut owns the index; the
intersect (Phase 3) calls query() and then decides exposure.

Two entry points:
  query()    — production: embeds task text via src.memory.embeddings
  query_db() — testable core: takes a precomputed query_embedding, no I/O
               beyond the passed db. query() is a thin embed-then-query_db.
"""
from __future__ import annotations

import math

import aiosqlite

from yalayut.contracts import Artifact, IndexRow, TaskContext
from yalayut.index import blob_to_embedding

_DEFAULT_TOP_K = 12


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 for mismatched/empty vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_manifest_fields(manifest_path: str | None) -> tuple[list, dict]:
    """Load intent_keywords and inputs_schema from a manifest file.

    Returns (intent_keywords, inputs_schema). On any failure (missing file,
    parse error) returns ([], {}) — must not break the hot read path.
    """
    if not manifest_path:
        return [], {}
    try:
        from yalayut.manifest import parse_manifest_yaml
        with open(manifest_path, encoding="utf-8") as fh:
            m = parse_manifest_yaml(fh.read())
        return list(m.intent_keywords or []), dict(m.inputs_schema or {})
    except Exception:
        return [], {}


def _to_artifact(row: IndexRow, score: float) -> Artifact:
    intent_keywords, inputs_schema = _load_manifest_fields(row.manifest_path)
    return Artifact(
        artifact_id=row.id, name=row.name, name_original=row.name_original,
        artifact_type=row.artifact_type, kind=row.kind,
        vet_tier=row.vet_tier if row.vet_tier is not None else 3,
        score=score, exposure_class=row.exposure_class,
        applies_to=row.applies_to, mechanizable=row.mechanizable,
        body_excerpt=row.body_excerpt, payload={},
        source=row.source or "",
        owner=row.owner,
        env_status=row.env_status if row.env_status is not None else "ready",
        intent_keywords=intent_keywords,
        inputs_schema=inputs_schema,
    )


async def query_db(
    db: aiosqlite.Connection,
    task_ctx: TaskContext,
    query_embedding: list[float],
    top_k: int = _DEFAULT_TOP_K,
) -> list[Artifact]:
    """Rank every enabled artifact by cosine similarity to query_embedding."""
    cur = await db.execute(
        "SELECT * FROM yalayut_index WHERE enabled = 1"
    )
    rows = await cur.fetchall()
    scored: list[Artifact] = []
    for r in rows:
        emb = blob_to_embedding(r["embedding"])
        score = _cosine(query_embedding, emb)
        ir = IndexRow(
            id=r["id"], artifact_type=r["artifact_type"], kind=r["kind"],
            source=r["source"], owner=r["owner"], name=r["name"],
            name_original=r["name_original"], version=r["version"],
            manifest_path=r["manifest_path"], body_excerpt=r["body_excerpt"],
            vet_tier=r["vet_tier"], exposure_class=r["exposure_class"],
            applies_to=r["applies_to"], mechanizable=bool(r["mechanizable"]),
            model_hint=r["model_hint"], enabled=bool(r["enabled"]),
            env_status=r["env_status"] if r["env_status"] is not None else "ready",
        )
        scored.append(_to_artifact(ir, score))
    scored.sort(key=lambda a: a.score, reverse=True)
    return scored[:top_k]


async def query(task_ctx: dict, top_k: int = _DEFAULT_TOP_K) -> list[Artifact]:
    """Production entry: embed the task text, then rank the index.

    task_ctx is a raw KutAI task dict. Embedding uses KutAI's shared
    multilingual-e5-base utility (lazy import — keeps yalayut import-light and
    avoids a hard dep at module load).
    """
    from dabidabi import get_db
    from src.memory.embeddings import get_embedding

    ctx = TaskContext.from_task(task_ctx)
    text = ctx.query_text()
    if not text:
        return []
    emb = await get_embedding(text, is_query=True)
    if emb is None:
        return []
    db = await get_db()
    return await query_db(db, ctx, emb, top_k=top_k)
