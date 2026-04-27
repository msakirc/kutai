# memory/vector_store.py
"""
Phase 11.1 — Vector Store

ChromaDB-backed vector store with automatic embedding via the shared
embeddings module.  Runs in-process (no server needed).

Collections:
  - episodic:       Task results and execution history
  - semantic:       Facts, preferences, learned knowledge
  - codebase:       Code chunks and symbols
  - errors:         Failure patterns and recovery strategies
  - conversations:  User interaction history
  - shopping:       Products, reviews, shopping sessions
  - web_knowledge:  Cached web search results and extracted facts

Public API:
    await init_store()
    await embed_and_store(text, metadata, collection)
    await query(text, collection, top_k)
    await delete(ids, collection)
    await get_collection_count(collection)
"""
import asyncio
import logging
import os
import time
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# --- Injectable dependencies (default to src.memory.embeddings) -----------
_embed_fn = None
_dimension_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from src.memory.embeddings import get_embedding
        _embed_fn = get_embedding
    return _embed_fn


def _get_dimension_fn():
    global _dimension_fn
    if _dimension_fn is None:
        from src.memory.embeddings import get_expected_dimension
        _dimension_fn = get_expected_dimension
    return _dimension_fn


# ─── Constants ────────────────────────────────────────────────────────────────

COLLECTIONS = [
    "episodic",
    "semantic",
    "codebase",
    "errors",
    "conversations",
    "shopping",
    "web_knowledge",
]

_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "chroma",
)


# ─── ChromaDB Client ─────────────────────────────────────────────────────────

_client = None
_collections: dict = {}
_initialized = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

_CORRUPT_SEGMENT_MARKERS = (
    "nothing found on disk",
    "hnsw segment reader",
    "segment file",
    "error in compaction",
    "failed to apply logs",
    "metadata segment",
)


def _looks_like_segment_corrupt(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CORRUPT_SEGMENT_MARKERS)


def _recreate_collection(collection: str):
    """Drop + recreate a collection. Returns the new collection or None on failure."""
    if _client is None:
        return None
    try:
        _client.delete_collection(collection)
    except Exception:
        pass
    try:
        expected_dim = _get_dimension_fn()()
        from src.memory.embeddings import EMBEDDING_MODEL
        fresh = _client.get_or_create_collection(
            name=collection,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": expected_dim,
            },
        )
        _collections[collection] = fresh
        return fresh
    except Exception as e:
        logger.error(f"Failed to recreate '{collection}': {e}")
        return None


def _rescue_and_rebuild_sync(collection: str) -> tuple[bool, int, int]:
    """
    Tier 1 rescue: read all rows via col.get, drop+recreate, re-insert.

    Preserves data when corruption is in HNSW segment but sqlite rows readable.
    Returns (success, rescued_count, recreated_only_fallback).

    success=True, rescued_count>0  → full rescue
    success=True, rescued_count=0  → empty collection (nothing to rescue)
    success=False                  → fell back to nuke (Tier 3)
    """
    if _client is None or collection not in _collections:
        return (False, 0, 0)

    col = _collections[collection]
    rescued = None
    try:
        rescued = col.get(include=["documents", "embeddings", "metadatas"])
    except Exception as e:
        logger.warning(
            f"Tier 1 rescue read failed on '{collection}' ({e!s}) — "
            f"falling back to nuke"
        )

    fresh = _recreate_collection(collection)
    if fresh is None:
        return (False, 0, 0)

    if not rescued:
        return (True, 0, 1)

    ids = rescued.get("ids") or []
    docs = rescued.get("documents") or [None] * len(ids)
    metas = rescued.get("metadatas") or [{}] * len(ids)
    embs = rescued.get("embeddings") or [None] * len(ids)

    if not ids:
        return (True, 0, 0)

    try:
        kwargs = {"ids": ids, "documents": docs, "metadatas": metas}
        if any(e is not None for e in embs):
            kwargs["embeddings"] = embs
        fresh.upsert(**kwargs)
        logger.info(
            f"Tier 1 rescue rebuilt '{collection}' with {len(ids)} rows preserved"
        )
        return (True, len(ids), 0)
    except Exception as e:
        logger.error(
            f"Tier 1 rebuild upsert failed on '{collection}' ({e!s}) — "
            f"collection now empty (Tier 3 fallback)"
        )
        return (True, 0, 1)


async def _self_heal(collection: str) -> bool:
    """Run Tier 1 rescue (with Tier 3 fallback) off the event loop."""
    success, _rescued, _nuked = await asyncio.to_thread(
        _rescue_and_rebuild_sync, collection
    )
    return success


async def integrity_probe() -> dict[str, str]:
    """
    Pre-flight: cheap touch on every collection. Heals corrupt ones before
    serving traffic. Returns per-collection status: ok | healed | failed.
    """
    if not _initialized:
        return {}
    results: dict[str, str] = {}
    for name, col in list(_collections.items()):
        try:
            await asyncio.to_thread(col.count)
            results[name] = "ok"
            continue
        except Exception as e:
            if not _looks_like_segment_corrupt(e):
                results[name] = f"unknown_error: {e!s}"
                continue
        logger.warning(
            f"Pre-flight: '{name}' looks corrupt — running rescue+rebuild"
        )
        ok = await _self_heal(name)
        results[name] = "healed" if ok else "failed"
    return results


async def wal_checkpoint(db_path: str | None = None) -> bool:
    """Run sqlite WAL checkpoint(TRUNCATE) on chroma.sqlite3. Releases WAL bloat."""
    import sqlite3
    path = db_path or os.path.join(_DB_DIR, "chroma.sqlite3")
    if not os.path.exists(path):
        return False
    def _run() -> bool:
        try:
            conn = sqlite3.connect(path, timeout=10.0)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")
            return False
    return await asyncio.to_thread(_run)


async def snapshot_chroma(keep: int = 3) -> str | None:
    """
    Copy data/chroma → data/chroma.bak.<YYYYMMDD-HHMMSS>/. Prune to last `keep`.
    Returns destination path or None on failure.
    """
    import shutil
    src = _DB_DIR
    if not os.path.isdir(src):
        return None
    parent = os.path.dirname(src)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(parent, f"chroma.bak.{stamp}")

    def _copy() -> str | None:
        try:
            shutil.copytree(src, dst, dirs_exist_ok=False)
            backups = sorted(
                p for p in os.listdir(parent)
                if p.startswith("chroma.bak.")
            )
            for old in backups[:-keep]:
                shutil.rmtree(os.path.join(parent, old), ignore_errors=True)
            return dst
        except Exception as e:
            logger.warning(f"Chroma snapshot failed: {e}")
            return None
    return await asyncio.to_thread(_copy)


async def init_store(persist_dir: str | None = None, embed_fn=None, dimension_fn=None) -> bool:
    """
    Initialize the ChromaDB client and create collections.

    ChromaDB is a required dependency. Raises ImportError if not installed.
    Returns True on success, False on other errors.
    """
    global _client, _collections, _initialized, _embed_fn, _dimension_fn

    if _initialized:
        return True

    if embed_fn is not None:
        _embed_fn = embed_fn
    if dimension_fn is not None:
        _dimension_fn = dimension_fn

    db_dir = persist_dir or _DB_DIR

    import chromadb
    from chromadb.config import Settings

    os.makedirs(db_dir, exist_ok=True)

    try:
        _client = await asyncio.to_thread(
            lambda: chromadb.PersistentClient(
                path=db_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        )

        expected_dim = _get_dimension_fn()()
        from src.memory.embeddings import EMBEDDING_MODEL

        # Create or get collections with dimension tracking
        for name in COLLECTIONS:
            col = await asyncio.to_thread(
                _client.get_or_create_collection,
                name=name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dimension": expected_dim,
                },
            )

            # Verify dimension consistency
            col_meta = col.metadata or {}
            stored_dim = col_meta.get("embedding_dimension")
            if stored_dim and stored_dim != expected_dim:
                logger.warning(
                    f"Collection '{name}' has dimension {stored_dim} but "
                    f"current model expects {expected_dim}. "
                    f"Consider re-embedding this collection."
                )

            _collections[name] = col

        _initialized = True
        total = await asyncio.to_thread(
            lambda: sum(c.count() for c in _collections.values())
        )
        logger.info(
            f"Vector store initialized: {len(_collections)} collections, "
            f"{total} total documents (dir: {db_dir})"
        )
        return True

    except Exception as e:
        logger.error(f"Vector store init failed: {e}")
        return False


def is_ready() -> bool:
    """Check if vector store has been initialized."""
    return _initialized


# ─── Store ────────────────────────────────────────────────────────────────────

async def embed_and_store(
    text: str,
    metadata: dict,
    collection: str = "semantic",
    doc_id: str | None = None,
) -> Optional[str]:
    """
    Embed text and store in the specified collection.

    Args:
        text:       Text to embed and store.
        metadata:   Dict of metadata (all values must be str, int, float, or bool).
        collection: Collection name (one of COLLECTIONS).
        doc_id:     Optional document ID. Auto-generated if not provided.

    Returns:
        The document ID, or None if storage failed.
    """
    if not _initialized:
        if not await init_store():
            return None

    if collection not in _collections:
        logger.error(f"Unknown collection: {collection}")
        return None

    if not text or not text.strip():
        return None

    # Get embedding (as passage, not query)
    embedding = await _get_embed_fn()(text, is_query=False)

    # Clean metadata — ChromaDB only allows str/int/float/bool values
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        elif v is None:
            clean_meta[k] = ""
        else:
            clean_meta[k] = str(v)

    # Add standard metadata
    clean_meta["stored_at"] = time.time()
    clean_meta.setdefault("access_count", 0)
    clean_meta.setdefault("last_accessed", time.time())

    # Generate or use provided ID
    if not doc_id:
        import hashlib
        doc_id = hashlib.sha256(
            f"{collection}:{text[:200]}:{time.time()}".encode()
        ).hexdigest()[:24]

    col = _collections[collection]
    add_kwargs = {
        "ids": [doc_id],
        "documents": [text],
        "metadatas": [clean_meta],
    }
    if embedding is not None:
        add_kwargs["embeddings"] = [embedding]

    try:
        await asyncio.to_thread(col.upsert, **add_kwargs)
        logger.debug(
            f"Stored in '{collection}': {doc_id} "
            f"({len(text)} chars, embedding={'yes' if embedding else 'no'})"
        )
        return doc_id

    except Exception as e:
        if _looks_like_segment_corrupt(e):
            logger.warning(
                f"Collection '{collection}' segments corrupt on store "
                f"({e!s}) — running Tier 1 rescue, retrying once"
            )
            healed = await _self_heal(collection)
            if healed:
                try:
                    await asyncio.to_thread(
                        _collections[collection].upsert, **add_kwargs
                    )
                    return doc_id
                except Exception as retry_exc:
                    logger.error(
                        f"Store retry failed on '{collection}' after heal: {retry_exc}"
                    )
        logger.error(f"Failed to store in '{collection}': {e}")
        return None


# ─── Query ────────────────────────────────────────────────────────────────────

async def query(
    text: str,
    collection: str = "semantic",
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Query a collection by text similarity.

    Args:
        text:       Query text (will be embedded).
        collection: Collection to search.
        top_k:      Number of results to return.
        where:      Optional metadata filter (ChromaDB where clause).

    Returns:
        List of dicts with keys: id, text, metadata, distance.
        Sorted by relevance (closest first).
    """
    if not _initialized:
        if not await init_store():
            return []

    if collection not in _collections:
        logger.error(f"Unknown collection: {collection}")
        return []

    if not text or not text.strip():
        return []

    col = _collections[collection]

    if await asyncio.to_thread(col.count) == 0:
        return []

    # Get embedding for query (as query, not passage)
    embedding = await _get_embed_fn()(text, is_query=True)

    if embedding is None:
        logger.warning(f"Embedding unavailable — skipping vector query on '{collection}'")
        return []

    try:
        query_kwargs = {
            "n_results": min(top_k, await asyncio.to_thread(col.count)),
            "query_embeddings": [embedding],
        }

        if where:
            query_kwargs["where"] = where

        results = await asyncio.to_thread(lambda: col.query(**query_kwargs))

        # Parse results into a clean list
        docs: list[dict] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            texts = results["documents"][0] if results.get("documents") else [None] * len(ids)
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for i, doc_id in enumerate(ids):
                docs.append({
                    "id": doc_id,
                    "text": texts[i] or "",
                    "metadata": metas[i] or {},
                    "distance": distances[i],
                })

                # Update access metadata
                try:
                    meta = metas[i] or {}
                    meta["access_count"] = meta.get("access_count", 0) + 1
                    meta["last_accessed"] = time.time()
                    await asyncio.to_thread(
                        col.update,
                        ids=[doc_id],
                        metadatas=[meta],
                    )
                except Exception:
                    pass  # non-critical

        return docs

    except Exception as e:
        # Self-heal corrupt segments. Without this, every RAG-enabled
        # agent dispatch logged "Nothing found on disk" repeatedly without
        # recovery, and corrupt episodic compaction blocked the event loop
        # past Yaşar Usta's 120s heartbeat (2026-04-27).
        if _looks_like_segment_corrupt(e):
            logger.warning(
                f"Collection '{collection}' segments corrupt on query "
                f"({e!s}) — running Tier 1 rescue"
            )
            healed = await _self_heal(collection)
            if healed:
                return []
        logger.error(f"Query failed on '{collection}': {e}")
        return []


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete(
    ids: list[str],
    collection: str = "semantic",
) -> int:
    """
    Delete documents by ID from a collection.

    Returns the number of documents deleted.
    """
    if not _initialized:
        return 0

    if collection not in _collections:
        return 0

    if not ids:
        return 0

    try:
        col = _collections[collection]
        await asyncio.to_thread(col.delete, ids=ids)
        logger.debug(f"Deleted {len(ids)} doc(s) from '{collection}'")
        return len(ids)
    except Exception as e:
        logger.error(f"Delete failed on '{collection}': {e}")
        return 0


# ─── Utilities ────────────────────────────────────────────────────────────────

async def get_collection_count(collection: str) -> int:
    """Return the number of documents in a collection."""
    if not _initialized:
        return 0
    if collection not in _collections:
        return 0
    return await asyncio.to_thread(_collections[collection].count)


async def get_all_counts() -> dict[str, int]:
    """Return document counts for all collections."""
    if not _initialized:
        return {}
    def _all():
        return {name: col.count() for name, col in _collections.items()}
    return await asyncio.to_thread(_all)


async def delete_by_metadata(
    collection: str,
    where: dict,
) -> int:
    """
    Delete documents matching a metadata filter.

    Returns count of deleted documents.
    """
    if not _initialized or collection not in _collections:
        return 0

    try:
        col = _collections[collection]
        # ChromaDB doesn't return count on delete, so query first
        results = await asyncio.to_thread(col.get, where=where)
        if results and results.get("ids"):
            ids = results["ids"]
            await asyncio.to_thread(col.delete, ids=ids)
            return len(ids)
        return 0
    except Exception as e:
        logger.error(f"delete_by_metadata failed on '{collection}': {e}")
        return 0


async def reset_store() -> None:
    """Reset all collections (for testing). Deletes all data."""
    global _initialized, _collections, _client
    if _client:
        try:
            _client.reset()
        except Exception:
            pass
    _initialized = False
    _collections = {}
    _client = None
