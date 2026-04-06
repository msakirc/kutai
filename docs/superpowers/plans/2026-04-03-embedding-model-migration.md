# Embedding Model Migration: multilingual-e5-small → EmbeddingGemma-300M

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `intfloat/multilingual-e5-small` (384d) with `google/embeddinggemma-300m` (768d, Matryoshka) for better multilingual embedding quality while keeping CPU inference.

**Architecture:** Update the embedding module to load EmbeddingGemma-300M via sentence-transformers, change dimension constants to 768, update ChromaDB collection metadata, and provide a one-shot migration script that re-embeds all existing ChromaDB data. The Ollama fallback stays unchanged (it already produces different dimensions and is validated separately).

**Tech Stack:** sentence-transformers, ChromaDB, Python 3.10

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/memory/embeddings.py` | Modify | Change model name, dims, max tokens, replace e5 prefix logic with GemmaEmbed encode API |
| `src/memory/vector_store.py` | Modify | Update hardcoded model name in collection metadata |
| `scripts/migrate_embeddings.py` | Create | One-shot migration: re-embed all ChromaDB collections |
| `tests/test_embeddings_standalone.py` | Modify | Update tests for new model constants |
| `tests/test_embedding_migration.py` | Create | Test migration script logic |
| `CLAUDE.md` | Modify | Update model name and dimension references |

---

### Task 1: Update Embedding Constants and Model Loading

**Files:**
- Modify: `src/memory/embeddings.py:24-29` (constants)
- Modify: `src/memory/embeddings.py:85-117` (preprocessing and encoding)

- [ ] **Step 1: Update constants**

In `src/memory/embeddings.py`, change the configuration block:

```python
# ─── Configuration ───────────────────────────────────────────────────────────

EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_TOKENS = 2048  # model token limit (4x improvement over e5-small)
EMBEDDING_BATCH_SIZE = 32
```

- [ ] **Step 2: Replace e5 preprocessing with GemmaEmbed encoding**

Replace the `_preprocess_e5` function and `_get_st_embedding` function:

```python
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
    except AttributeError:
        # Fallback for models without encode_query/encode_document
        vec = _st_model.encode(text, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.debug(f"sentence-transformers encode failed: {e}")
        return None
```

- [ ] **Step 3: Update batch encoding**

Replace `_get_st_embeddings_batch`:

```python
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
    except AttributeError:
        vecs = _st_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.debug(f"sentence-transformers batch encode failed: {e}")
        return [None] * len(texts)
```

- [ ] **Step 4: Remove the `_preprocess_e5` function**

Delete the `_preprocess_e5` function entirely (lines 85-88). It is no longer used.

- [ ] **Step 5: Update text truncation limit**

In `get_embedding()` (line 252) and `get_embeddings()` (line 296), update the truncation:

```python
# Old: text = text[:2048]  (≈ 512 tokens)
text = text[:8192]  # ≈ 2048 tokens
```

- [ ] **Step 6: Verify the module imports cleanly**

Run:
```bash
python -c "from src.memory.embeddings import EMBEDDING_MODEL, EMBEDDING_DIMENSION; print(f'{EMBEDDING_MODEL} @ {EMBEDDING_DIMENSION}d')"
```

Expected: `google/embeddinggemma-300m @ 768d`

- [ ] **Step 7: Commit**

```bash
git add src/memory/embeddings.py
git commit -m "feat(embeddings): migrate from multilingual-e5-small to EmbeddingGemma-300M"
```

---

### Task 2: Update Vector Store Collection Metadata

**Files:**
- Modify: `src/memory/vector_store.py:114-121`

- [ ] **Step 1: Replace hardcoded model name with import**

In `src/memory/vector_store.py`, change the collection creation to use the constant from embeddings instead of a hardcoded string:

At the top of `init_store()`, after the `expected_dim = _get_dimension_fn()()` line, add:

```python
from src.memory.embeddings import EMBEDDING_MODEL
```

Then update the collection creation metadata:

```python
col = _client.get_or_create_collection(
    name=name,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": expected_dim,
    },
)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
python -c "from src.memory.vector_store import COLLECTIONS; print(COLLECTIONS)"
```

Expected: List of 7 collection names.

- [ ] **Step 3: Commit**

```bash
git add src/memory/vector_store.py
git commit -m "refactor(vector_store): use EMBEDDING_MODEL constant instead of hardcoded model name"
```

---

### Task 3: Write the ChromaDB Migration Script

**Files:**
- Create: `scripts/migrate_embeddings.py`

This script re-embeds all documents in all ChromaDB collections using the new model. It must:
1. Load existing documents from each collection
2. Generate new embeddings with EmbeddingGemma-300M
3. Upsert back with updated embeddings and metadata
4. Handle errors gracefully (skip individual failures, report at end)

- [ ] **Step 1: Write the migration test**

Create `tests/test_embedding_migration.py`:

```python
"""Tests for the embedding migration script."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_migrate_collection_reembeds_documents():
    """Migration should re-embed all documents in a collection."""
    from scripts.migrate_embeddings import migrate_collection

    # Mock collection with 2 documents
    mock_col = MagicMock()
    mock_col.count.return_value = 2
    mock_col.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": ["hello world", "test document"],
        "metadatas": [{"type": "fact"}, {"type": "skill"}],
    }

    mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

    stats = await migrate_collection(mock_col, "semantic", mock_embed_fn, batch_size=10)

    assert stats["total"] == 2
    assert stats["success"] == 2
    assert stats["failed"] == 0
    assert mock_col.update.called


@pytest.mark.asyncio
async def test_migrate_collection_skips_empty_documents():
    """Migration should skip documents with empty text."""
    from scripts.migrate_embeddings import migrate_collection

    mock_col = MagicMock()
    mock_col.count.return_value = 3
    mock_col.get.return_value = {
        "ids": ["doc1", "doc2", "doc3"],
        "documents": ["hello", None, ""],
        "metadatas": [{"type": "fact"}, {"type": "fact"}, {"type": "fact"}],
    }

    mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

    stats = await migrate_collection(mock_col, "test", mock_embed_fn, batch_size=10)

    assert stats["total"] == 3
    assert stats["skipped"] == 2  # None and empty string
    assert stats["success"] == 1


@pytest.mark.asyncio
async def test_migrate_collection_handles_embed_failure():
    """Migration should continue when individual embeddings fail."""
    from scripts.migrate_embeddings import migrate_collection

    mock_col = MagicMock()
    mock_col.count.return_value = 2
    mock_col.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": ["hello", "world"],
        "metadatas": [{"type": "fact"}, {"type": "fact"}],
    }

    # First call succeeds, second returns None
    mock_embed_fn = AsyncMock(side_effect=[[0.1] * 768, None])

    stats = await migrate_collection(mock_col, "test", mock_embed_fn, batch_size=10)

    assert stats["success"] == 1
    assert stats["failed"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_embedding_migration.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Write the migration script**

Create `scripts/migrate_embeddings.py`:

```python
"""
One-shot migration: re-embed all ChromaDB collections with EmbeddingGemma-300M.

Usage:
    python -m scripts.migrate_embeddings [--dry-run] [--collection NAME]

This script:
  1. Loads all documents from each ChromaDB collection
  2. Re-embeds them using the new model (google/embeddinggemma-300m)
  3. Upserts the new embeddings back into ChromaDB
  4. Reports success/failure counts

Safe to re-run — uses upsert so existing docs are updated, not duplicated.
"""
import argparse
import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def migrate_collection(
    col, name: str, embed_fn, batch_size: int = 32
) -> dict:
    """
    Re-embed all documents in a single ChromaDB collection.

    Args:
        col:        ChromaDB collection object
        name:       Collection name (for logging)
        embed_fn:   Async callable: text -> list[float] or None
        batch_size: How many documents to process at once

    Returns:
        Dict with keys: total, success, failed, skipped
    """
    total = col.count()
    stats = {"total": total, "success": 0, "failed": 0, "skipped": 0}

    if total == 0:
        return stats

    # Fetch all documents (ChromaDB get() returns all if no filter)
    data = col.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]

        update_ids = []
        update_embeddings = []

        for doc_id, doc_text, meta in zip(batch_ids, batch_docs, batch_metas):
            if not doc_text or not doc_text.strip():
                stats["skipped"] += 1
                continue

            try:
                embedding = await embed_fn(doc_text)
                if embedding is None:
                    stats["failed"] += 1
                    continue
                update_ids.append(doc_id)
                update_embeddings.append(embedding)
                stats["success"] += 1
            except Exception as e:
                print(f"  Error embedding {doc_id}: {e}")
                stats["failed"] += 1

        if update_ids:
            col.update(ids=update_ids, embeddings=update_embeddings)

    return stats


async def run_migration(dry_run: bool = False, target_collection: str | None = None):
    """Run the full migration across all collections."""
    from src.memory.embeddings import (
        get_embedding,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        clear_cache,
    )
    from src.memory.vector_store import COLLECTIONS, init_store

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print()

    # Clear embedding cache so we use the new model fresh
    clear_cache()

    # Initialize vector store
    ok = await init_store()
    if not ok:
        print("ERROR: Could not initialize vector store")
        sys.exit(1)

    from src.memory.vector_store import _collections

    collections_to_migrate = (
        [target_collection] if target_collection else COLLECTIONS
    )

    total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for col_name in collections_to_migrate:
        col = _collections.get(col_name)
        if col is None:
            print(f"  {col_name}: NOT FOUND — skipping")
            continue

        count = col.count()
        print(f"  {col_name}: {count} documents")

        if dry_run:
            total_stats["total"] += count
            continue

        start = time.time()

        async def embed_as_document(text: str):
            return await get_embedding(text, is_query=False)

        stats = await migrate_collection(
            col, col_name, embed_as_document, batch_size=32
        )
        elapsed = time.time() - start

        print(
            f"    ✓ {stats['success']} ok, {stats['failed']} failed, "
            f"{stats['skipped']} skipped ({elapsed:.1f}s)"
        )

        for k in total_stats:
            total_stats[k] += stats[k]

    print()
    print("=" * 50)
    if dry_run:
        print(f"DRY RUN: {total_stats['total']} documents would be re-embedded")
    else:
        print(
            f"Done: {total_stats['success']} ok, {total_stats['failed']} failed, "
            f"{total_stats['skipped']} skipped out of {total_stats['total']} total"
        )


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB embeddings to new model")
    parser.add_argument("--dry-run", action="store_true", help="Count documents without re-embedding")
    parser.add_argument("--collection", type=str, help="Migrate a single collection (default: all)")
    args = parser.parse_args()

    asyncio.run(run_migration(dry_run=args.dry_run, target_collection=args.collection))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:
```bash
pytest tests/test_embedding_migration.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_embeddings.py tests/test_embedding_migration.py
git commit -m "feat(migration): add ChromaDB re-embedding script for model migration"
```

---

### Task 4: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md embedding references**

In `CLAUDE.md`, update the embedding model reference in Common Pitfalls:

```
Old: - Embedding model is `intfloat/multilingual-e5-small` (384 dims) — don't mix with other models
New: - Embedding model is `google/embeddinggemma-300m` (768 dims) — don't mix with other models
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update embedding model reference to EmbeddingGemma-300M"
```

---

### Task 5: Run Existing Tests

**Files:**
- Test: `tests/test_embeddings_standalone.py`
- Test: `tests/test_vector_store_standalone.py`

- [ ] **Step 1: Run all embedding and vector store tests**

Run:
```bash
pytest tests/test_embeddings_standalone.py tests/test_vector_store_standalone.py tests/test_rag_standalone.py -v
```

Expected: All PASS. These tests check structural properties (stdlib logging, function signatures) — not model-specific behavior — so they should pass without changes.

- [ ] **Step 2: Run the full test suite**

Run:
```bash
pytest tests/ -v --timeout=30
```

Expected: All existing tests pass. If any test hardcodes `384` or `multilingual-e5-small`, fix those inline.

- [ ] **Step 3: Commit any test fixes if needed**

```bash
git add tests/
git commit -m "test: fix tests for EmbeddingGemma-300M migration"
```

---

### Task 6: Run the Migration (Manual Step)

**This task is run manually after KutAI is stopped.**

- [ ] **Step 1: Stop KutAI**

Use `/stop` via Telegram to gracefully stop the orchestrator.

- [ ] **Step 2: Dry run to see document counts**

```bash
python -m scripts.migrate_embeddings --dry-run
```

Review output — confirm document counts per collection look reasonable.

- [ ] **Step 3: Run the actual migration**

```bash
python -m scripts.migrate_embeddings
```

This will take a while on CPU (depends on total document count). Monitor output for failures.

- [ ] **Step 4: Restart KutAI**

Use `/start` via Telegram or restart the wrapper.

- [ ] **Step 5: Verify embeddings work end-to-end**

Send a test message via Telegram and verify RAG context injection works (check logs for embedding dimensions and vector store queries).
