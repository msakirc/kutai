# memory/embeddings.py
"""
Phase 11 — Shared Embedding Utility

Provides get_embedding() for use by vector_store, RAG pipeline, and
task classifier.

Priority order:
  1. In-memory LRU cache
  2. sentence-transformers on CPU (multilingual-e5-base, always available)

Also exposes a batch helper: get_embeddings(texts).
"""
import hashlib
from collections import OrderedDict
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_TOKENS = 512  # model token limit
EMBEDDING_BATCH_SIZE = 32


# ─── In-Memory LRU Cache ────────────────────────────────────────────────────

_CACHE_MAX_SIZE = 5000


class _LRUCache:
    """Proper LRU cache using OrderedDict."""

    def __init__(self, max_size: int = _CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[list[float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_embedding_cache = _LRUCache()


def _cache_key(text: str) -> str:
    """Short hash for embedding cache lookup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def clear_cache() -> None:
    """Clear the embedding cache (useful for tests)."""
    _embedding_cache.clear()


# ─── sentence-transformers (CPU) ────────────────────────────────────────────

_st_model = None
_st_load_attempted = False


def _preprocess_e5(text: str, is_query: bool = True) -> str:
    """Add e5 prefix. E5 models need 'query: ' or 'passage: ' prefix."""
    prefix = "query: " if is_query else "passage: "
    return prefix + text


def _get_st_embedding(
    text: str, is_query: bool = True
) -> Optional[list[float]]:
    """Get embedding from sentence-transformers (CPU, synchronous)."""
    global _st_model, _st_load_attempted
    if _st_load_attempted and _st_model is None:
        return None

    if _st_model is None:
        _st_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            # Pin to CPU — per CLAUDE.md embedding must not contend with
            # llama-server for GPU. Default device='cuda' caused access
            # violations (0xC0000005) when llama-server was holding VRAM
            # (2026-04-22 mission 45 crash).
            _st_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            logger.info(f"Loaded sentence-transformers {EMBEDDING_MODEL}")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")
            return None

    try:
        processed = _preprocess_e5(text, is_query=is_query)
        vec = _st_model.encode(processed, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.debug(f"sentence-transformers encode failed: {e}")
        return None


def _get_st_embeddings_batch(
    texts: list[str], is_query: bool = True
) -> list[Optional[list[float]]]:
    """Batch embedding via sentence-transformers."""
    global _st_model, _st_load_attempted
    if _st_load_attempted and _st_model is None:
        return [None] * len(texts)

    if _st_model is None:
        _st_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            # Pin to CPU — per CLAUDE.md embedding must not contend with
            # llama-server for GPU. Default device='cuda' caused access
            # violations (0xC0000005) when llama-server was holding VRAM
            # (2026-04-22 mission 45 crash).
            _st_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            logger.info(f"Loaded sentence-transformers {EMBEDDING_MODEL}")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")
            return [None] * len(texts)

    try:
        processed = [_preprocess_e5(t, is_query=is_query) for t in texts]
        vecs = _st_model.encode(
            processed,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.debug(f"sentence-transformers batch encode failed: {e}")
        return [None] * len(texts)


# ─── Dimension Validation ────────────────────────────────────────────────────

_expected_dimension: int | None = None


def set_expected_dimension(dim: int) -> None:
    """Lock expected embedding dimension. Mismatches will be rejected."""
    global _expected_dimension
    _expected_dimension = dim


def get_expected_dimension() -> int:
    """Return the expected embedding dimension."""
    return _expected_dimension or EMBEDDING_DIMENSION


def _validate_dimension(embedding: list[float]) -> bool:
    """Check embedding dimension matches expected. Log warning on mismatch."""
    global _expected_dimension
    dim = len(embedding)
    if _expected_dimension is None:
        _expected_dimension = dim
        return True
    if dim != _expected_dimension:
        logger.warning(
            f"Embedding dimension mismatch: got {dim}, expected "
            f"{_expected_dimension}. Rejecting to prevent corruption."
        )
        return False
    return True


# ─── Public API ──────────────────────────────────────────────────────────────

async def get_embedding(
    text: str, is_query: bool = True
) -> Optional[list[float]]:
    """
    Get embedding vector for text.

    Tries in order:
      1. In-memory LRU cache
      2. sentence-transformers on CPU (multilingual-e5-base)
      3. Returns None if not available

    Args:
        text:     Text to embed.
        is_query: True for queries, False for documents/passages.

    Returns:
        List of floats (embedding vector), or None.
    """
    if not text or not text.strip():
        return None

    text = text[:2048]  # ~512 tokens

    ck = _cache_key(text)
    cached = _embedding_cache.get(ck)
    if cached is not None:
        return cached

    emb = _get_st_embedding(text, is_query=is_query)

    if emb is not None:
        if not _validate_dimension(emb):
            return None
        _embedding_cache.put(ck, emb)

    return emb


async def get_embeddings(
    texts: list[str], is_query: bool = True
) -> list[Optional[list[float]]]:
    """
    Batch version of get_embedding.

    Uses sentence-transformers batch encoding for uncached texts.
    """
    if not texts:
        return []

    results: list[Optional[list[float]]] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    # Check cache first
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        text = text[:2048]
        ck = _cache_key(text)
        cached = _embedding_cache.get(ck)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if not uncached_texts:
        return results

    # Batch sentence-transformers
    batch_results = _get_st_embeddings_batch(uncached_texts, is_query=is_query)

    for idx, emb in zip(uncached_indices, batch_results):
        if emb is not None and _validate_dimension(emb):
            text = uncached_texts[uncached_indices.index(idx)]
            _embedding_cache.put(_cache_key(text), emb)
            results[idx] = emb

    return results


def embedding_available() -> bool:
    """Quick check: is sentence-transformers available?"""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
