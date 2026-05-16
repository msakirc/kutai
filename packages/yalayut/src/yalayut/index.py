"""yalayut_index storage + read.

Embeddings are stored as raw float32 BLOBs (multilingual-e5-base, 768d).
exposure_class is the tier-derived CEILING default — Phase 3's intersect makes
the real per-task decision; storing a default keeps the column non-null and
lets query() return something coherent before intersect exists.
"""
from __future__ import annotations

import array
import json

import aiosqlite

from yalayut.contracts import IndexRow, Manifest

# tier -> default exposure ceiling (spec Tier→exposure table). intersect
# refines per-task; this is the stored default.
_TIER_DEFAULT_EXPOSURE = {0: "inject", 1: "inject", 2: "quarantine",
                          3: "quarantine"}

_BODY_EXCERPT_LEN = 500


def embedding_to_blob(vec: list[float]) -> bytes:
    """Pack a float list into a compact float32 blob."""
    return array.array("f", vec).tobytes()


def blob_to_embedding(blob: bytes | None) -> list[float]:
    """Unpack a float32 blob back into a list. None/empty -> []."""
    if not blob:
        return []
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def _row_to_indexrow(r: aiosqlite.Row) -> IndexRow:
    return IndexRow(
        id=r["id"], artifact_type=r["artifact_type"], kind=r["kind"],
        source=r["source"], owner=r["owner"], name=r["name"],
        name_original=r["name_original"], version=r["version"],
        manifest_path=r["manifest_path"], body_excerpt=r["body_excerpt"],
        vet_tier=r["vet_tier"], exposure_class=r["exposure_class"],
        applies_to=r["applies_to"], mechanizable=bool(r["mechanizable"]),
        model_hint=r["model_hint"], enabled=bool(r["enabled"]),
    )


async def store(
    db: aiosqlite.Connection,
    manifest: Manifest,
    body: str,
    tier: int,
    audit: dict,
    embedding: list[float],
    manifest_path: str | None = None,
) -> int:
    """Insert (or update on UNIQUE conflict) one artifact. Returns row id.

    Enable policy (spec Lifecycle step 6): T0/T1 auto-enable; T2 quarantined-
    until-founder-promotes in v1 (enabled=0); T3 quarantine (enabled=0).
    """
    enabled = 1 if tier in (0, 1) else 0
    exposure = _TIER_DEFAULT_EXPOSURE.get(tier, "quarantine")
    excerpt = (body or "")[:_BODY_EXCERPT_LEN]
    await db.execute(
        """
        INSERT INTO yalayut_index
          (artifact_type, kind, source, owner, name, name_original, version,
           manifest_path, body_excerpt, embedding, vet_tier, exposure_class,
           applies_to, vet_state, vet_hash, source_max, check_max_json,
           signature, mechanizable, model_hint, env_status, enabled,
           created_at, vetted_at)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'vetted', ?, ?, ?, ?, ?, ?,
           'ready', ?, strftime('%Y-%m-%d %H:%M:%S','now'),
           strftime('%Y-%m-%d %H:%M:%S','now'))
        ON CONFLICT(source, name, version) DO UPDATE SET
           kind=excluded.kind, owner=excluded.owner,
           name_original=excluded.name_original,
           manifest_path=excluded.manifest_path,
           body_excerpt=excluded.body_excerpt, embedding=excluded.embedding,
           vet_tier=excluded.vet_tier, exposure_class=excluded.exposure_class,
           applies_to=excluded.applies_to, vet_hash=excluded.vet_hash,
           source_max=excluded.source_max,
           check_max_json=excluded.check_max_json,
           mechanizable=excluded.mechanizable, model_hint=excluded.model_hint,
           enabled=excluded.enabled,
           vetted_at=strftime('%Y-%m-%d %H:%M:%S','now')
        """,
        (
            manifest.artifact_type, manifest.kind, manifest.source,
            manifest.owner, manifest.name, manifest.name_original,
            manifest.version, manifest_path, excerpt,
            embedding_to_blob(embedding), tier, exposure,
            manifest.applies_to, str(abs(hash(body)) % (10 ** 12)),
            audit.get("source_max"), json.dumps(audit.get("check_maxes", {})),
            None, 1 if manifest.mechanizable else 0, manifest.model_hint,
            enabled,
        ),
    )
    await db.commit()
    cur = await db.execute(
        "SELECT id FROM yalayut_index WHERE source=? AND name=? AND version=?",
        (manifest.source, manifest.name, manifest.version),
    )
    row = await cur.fetchone()
    return row["id"]


async def get(db: aiosqlite.Connection, artifact_id: int) -> IndexRow | None:
    """Fetch one artifact by id."""
    cur = await db.execute(
        "SELECT * FROM yalayut_index WHERE id = ?", (artifact_id,)
    )
    r = await cur.fetchone()
    return _row_to_indexrow(r) if r else None


async def read_all_enabled(db: aiosqlite.Connection) -> list[IndexRow]:
    """Every enabled artifact, for query() to score."""
    cur = await db.execute(
        "SELECT * FROM yalayut_index WHERE enabled = 1"
    )
    return [_row_to_indexrow(r) for r in await cur.fetchall()]


async def read_embeddings(
    db: aiosqlite.Connection,
) -> list[tuple[int, list[float]]]:
    """(id, embedding) for every enabled artifact — query() hot path."""
    cur = await db.execute(
        "SELECT id, embedding FROM yalayut_index WHERE enabled = 1"
    )
    return [
        (r["id"], blob_to_embedding(r["embedding"]))
        for r in await cur.fetchall()
    ]
