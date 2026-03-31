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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_data",
)


# ─── ChromaDB Client ─────────────────────────────────────────────────────────

_client = None
_collections: dict = {}
_initialized = False


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
        _client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        expected_dim = _get_dimension_fn()()

        # Create or get collections with dimension tracking
        for name in COLLECTIONS:
            col = _client.get_or_create_collection(
                name=name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": "intfloat/multilingual-e5-small",
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
        total = sum(c.count() for c in _collections.values())
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

    try:
        col = _collections[collection]
        add_kwargs = {
            "ids": [doc_id],
            "documents": [text],
            "metadatas": [clean_meta],
        }
        if embedding is not None:
            add_kwargs["embeddings"] = [embedding]

        col.upsert(**add_kwargs)

        logger.debug(
            f"Stored in '{collection}': {doc_id} "
            f"({len(text)} chars, embedding={'yes' if embedding else 'no'})"
        )
        return doc_id

    except Exception as e:
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

    if col.count() == 0:
        return []

    # Get embedding for query (as query, not passage)
    embedding = await _get_embed_fn()(text, is_query=True)

    try:
        query_kwargs = {
            "n_results": min(top_k, col.count()),
        }

        if embedding is not None:
            query_kwargs["query_embeddings"] = [embedding]
        else:
            # Fallback: let ChromaDB handle text-based query
            query_kwargs["query_texts"] = [text]

        if where:
            query_kwargs["where"] = where

        results = col.query(**query_kwargs)

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
                    col.update(
                        ids=[doc_id],
                        metadatas=[meta],
                    )
                except Exception:
                    pass  # non-critical

        return docs

    except Exception as e:
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
        col.delete(ids=ids)
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
    return _collections[collection].count()


async def get_all_counts() -> dict[str, int]:
    """Return document counts for all collections."""
    if not _initialized:
        return {}
    return {name: col.count() for name, col in _collections.items()}


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
        results = col.get(where=where)
        if results and results.get("ids"):
            ids = results["ids"]
            col.delete(ids=ids)
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
