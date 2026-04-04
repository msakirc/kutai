# memory/embeddings.py
"""
Phase 11 — Shared Embedding Utility

Provides get_embedding() for use by vector_store, RAG pipeline, and
task classifier.

Priority order:
  1. In-memory LRU cache
  2. sentence-transformers on CPU (EmbeddingGemma-300M, always available)
  3. Ollama embedding on GPU (when llama-server is idle)

Also exposes a batch helper: get_embeddings(texts).
"""
import hashlib
from collections import OrderedDict
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_TOKENS = 2048  # model token limit (4x improvement over e5-small)
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


# ─── sentence-transformers (CPU, primary) ────────────────────────────────────

_st_model = None
_st_load_attempted = False
_st_model_name: str | None = None


def _get_st_embedding(
    text: str, is_query: bool = True
) -> Optional[list[float]]:
    """Get embedding from sentence-transformers (CPU, synchronous)."""
    global _st_model, _st_load_attempted, _st_model_name
    if _st_load_attempted and _st_model is None:
        return None

    if _st_model is None:
        _st_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(EMBEDDING_MODEL)
            _st_model_name = EMBEDDING_MODEL
            logger.info(f"Loaded sentence-transformers {EMBEDDING_MODEL}")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")
            return None

    try:
        if is_query:
            vec = _st_model.encode_query(text, show_progress_bar=False)
        else:
            vec = _st_model.encode_document(text, show_progress_bar=False)
        return vec.tolist()
    except (AttributeError, TypeError):
        # Fallback for models without encode_query/encode_document
        vec = _st_model.encode(text, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.debug(f"sentence-transformers encode failed: {e}")
        return None


def _get_st_embeddings_batch(
    texts: list[str], is_query: bool = True
) -> list[Optional[list[float]]]:
    """Batch embedding via sentence-transformers."""
    global _st_model, _st_load_attempted, _st_model_name
    if _st_load_attempted and _st_model is None:
        return [None] * len(texts)

    if _st_model is None:
        _st_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(EMBEDDING_MODEL)
            _st_model_name = EMBEDDING_MODEL
            logger.info(f"Loaded sentence-transformers {EMBEDDING_MODEL}")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")
            return [None] * len(texts)

    try:
        if is_query:
            vecs = _st_model.encode_query(
                texts,
                show_progress_bar=False,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
        else:
            vecs = _st_model.encode_document(
                texts,
                show_progress_bar=False,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
        return [v.tolist() for v in vecs]
    except (AttributeError, TypeError):
        vecs = _st_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.debug(f"sentence-transformers batch encode failed: {e}")
        return [None] * len(texts)


# ─── Ollama Embedding (GPU, secondary) ──────────────────────────────────────

_OLLAMA_MODELS = ["nomic-embed-text"]  # 768d, matches EmbeddingGemma-300M
_ollama_working_model: str | None = None


async def _get_ollama_embedding(text: str) -> Optional[list[float]]:
    """Get embedding from Ollama (tries multiple models)."""
    global _ollama_working_model
    try:
        import httpx
    except ImportError:
        return None

    # If we already found a working model, try it first
    models = (
        [_ollama_working_model] if _ollama_working_model else _OLLAMA_MODELS
    )

    for model_name in models:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": model_name, "prompt": text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data.get("embedding")
                    if embedding:
                        _ollama_working_model = model_name
                        return embedding
        except Exception:
            continue

    # If cached model failed, retry all
    if _ollama_working_model:
        _ollama_working_model = None
        return await _get_ollama_embedding(text)

    return None


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
      2. sentence-transformers on CPU (EmbeddingGemma-300M)
      3. Ollama embedding models (GPU)
      4. Returns None if nothing available

    Args:
        text:     Text to embed.
        is_query: True for queries, False for documents/passages.

    Returns:
        List of floats (embedding vector), or None.
    """
    if not text or not text.strip():
        return None

    # Truncate to model limit (~8192 chars ≈ 2048 tokens)
    text = text[:6144]  # conservative limit for 2048 tokens (Turkish is ~3 chars/token)

    ck = _cache_key(text)
    cached = _embedding_cache.get(ck)
    if cached is not None:
        return cached

    # Try sentence-transformers first (CPU, always available, multilingual)
    emb = _get_st_embedding(text, is_query=is_query)

    # Fallback: Ollama (GPU)
    if emb is None:
        emb = await _get_ollama_embedding(text)

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

    Uses sentence-transformers batch encoding for uncached texts,
    falls back to serial Ollama for any remaining.
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
        text = text[:6144]
        ck = _cache_key(text)
        cached = _embedding_cache.get(ck)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if not uncached_texts:
        return results

    # Try batch sentence-transformers
    batch_results = _get_st_embeddings_batch(uncached_texts, is_query=is_query)

    for idx, emb in zip(uncached_indices, batch_results):
        if emb is not None and _validate_dimension(emb):
            text = uncached_texts[uncached_indices.index(idx)]
            _embedding_cache.put(_cache_key(text), emb)
            results[idx] = emb

    # Fallback: Ollama for any still-missing
    for i, idx in enumerate(uncached_indices):
        if results[idx] is None:
            emb = await _get_ollama_embedding(uncached_texts[i])
            if emb is not None and _validate_dimension(emb):
                _embedding_cache.put(_cache_key(uncached_texts[i]), emb)
                results[idx] = emb

    return results


def embedding_available() -> bool:
    """Quick check: is any embedding backend likely available?"""
    # Check sentence-transformers first (preferred)
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        pass

    # Check Ollama
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    return False
