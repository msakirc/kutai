# memory/embeddings.py
"""
Phase 11 — Shared Embedding Utility

Provides get_embedding() for use by vector_store, RAG pipeline, and
task classifier.  Tries Ollama first, then sentence-transformers as
a fallback.

Also exposes a batch helper: get_embeddings(texts).
"""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── In-Memory Cache ──────────────────────────────────────────────────────────

_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX_SIZE = 5000


def _cache_key(text: str) -> str:
    """Short hash for embedding cache lookup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def clear_cache() -> None:
    """Clear the embedding cache (useful for tests)."""
    _embedding_cache.clear()


# ─── Ollama Embedding ─────────────────────────────────────────────────────────

_OLLAMA_MODELS = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]
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


# ─── sentence-transformers Fallback ───────────────────────────────────────────

_st_model = None
_st_load_attempted = False


def _get_st_embedding(text: str) -> Optional[list[float]]:
    """Get embedding from sentence-transformers (CPU, synchronous)."""
    global _st_model, _st_load_attempted
    if _st_load_attempted and _st_model is None:
        return None

    if _st_model is None:
        _st_load_attempted = True
        try:
            import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers all-MiniLM-L6-v2")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")
            return None

    try:
        vec = _st_model.encode(text, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.debug(f"sentence-transformers encode failed: {e}")
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

async def get_embedding(text: str) -> Optional[list[float]]:
    """
    Get embedding vector for text.

    Tries in order:
      1. In-memory cache
      2. Ollama embedding models
      3. sentence-transformers (local CPU)
      4. Returns None if nothing available

    Returns:
        List of floats (embedding vector), or None.
    """
    if not text or not text.strip():
        return None

    # Truncate very long texts
    text = text[:2000]

    ck = _cache_key(text)
    if ck in _embedding_cache:
        return _embedding_cache[ck]

    # Try Ollama
    emb = await _get_ollama_embedding(text)

    # Fallback: sentence-transformers
    if emb is None:
        emb = _get_st_embedding(text)

    if emb is not None:
        # Evict oldest entries if cache is full
        if len(_embedding_cache) >= _CACHE_MAX_SIZE:
            # Remove ~10% of entries
            keys_to_remove = list(_embedding_cache.keys())[
                : _CACHE_MAX_SIZE // 10
            ]
            for k in keys_to_remove:
                del _embedding_cache[k]
        _embedding_cache[ck] = emb

    return emb


async def get_embeddings(texts: list[str]) -> list[Optional[list[float]]]:
    """Batch version of get_embedding — returns parallel list of embeddings."""
    return [await get_embedding(t) for t in texts]


def embedding_available() -> bool:
    """Quick check: is any embedding backend likely available?"""
    # Check Ollama
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    # Check sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        pass

    return False
