# memory/decay.py
"""
Phase 11.6 — Memory Forgetting & Decay

Implements memory lifecycle management:
  - Relevance scoring based on access_count × recency
  - Pruning of low-relevance memories
  - Per-collection caps to prevent unbounded growth
  - Protected categories (user_preference, error patterns) with high importance floors

Public API:
    stats = await run_decay_cycle()     # Run one pruning cycle
    score = compute_relevance(metadata)  # Score a single memory
"""
import time

from src.infra.logging_config import get_logger
from .vector_store import (
    COLLECTIONS,
    is_ready,
    get_collection_count,
    get_all_counts, _collections,
)

logger = get_logger("memory.decay")


# ─── Configuration ────────────────────────────────────────────────────────────

# Maximum documents per collection before pruning kicks in
COLLECTION_CAPS: dict[str, int] = {
    "episodic": 10_000,
    "semantic": 10_000,
    "codebase": 15_000,
    "errors": 5_000,
    "conversations": 5_000,
}

# Minimum relevance score to keep (0.0 - 1.0)
RELEVANCE_THRESHOLD = 0.05

# Types that are protected from decay (high importance floor)
PROTECTED_TYPES = {"user_preference", "error_recovery"}

# How aggressively to prune when over cap (fraction of excess to remove)
PRUNE_FRACTION = 0.2

# Half-life for recency decay (in days)
RECENCY_HALF_LIFE_DAYS = 30.0


# ─── Relevance Scoring ───────────────────────────────────────────────────────

def compute_relevance(metadata: dict) -> float:
    """
    Compute a relevance score for a memory entry.

    Score = access_frequency × recency_weight × importance_floor

    - access_frequency: log(1 + access_count) normalized
    - recency_weight:   exponential decay based on last_accessed
    - importance_floor: protected types get a boost

    Returns:
        Float between 0.0 and 1.0.
    """
    import math

    access_count = metadata.get("access_count", 0)
    last_accessed = metadata.get("last_accessed", 0)
    stored_at = metadata.get("stored_at", 0)
    doc_type = metadata.get("type", "")
    importance = metadata.get("importance", 5)

    # ── Access frequency component ──
    # log(1 + count) gives diminishing returns for very frequently accessed items
    access_score = math.log(1 + access_count) / math.log(1 + 100)  # normalized to 100 accesses
    access_score = min(access_score, 1.0)

    # ── Recency component ──
    ref_time = last_accessed or stored_at or time.time()
    age_days = (time.time() - ref_time) / 86400.0
    if age_days < 0:
        age_days = 0
    recency_score = math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)

    # ── Importance floor ──
    # Protected types have a guaranteed minimum score
    if doc_type in PROTECTED_TYPES:
        importance = max(importance, 8)

    importance_score = importance / 10.0

    # ── Composite score ──
    # Weight: 30% access frequency, 40% recency, 30% importance
    score = access_score * 0.3 + recency_score * 0.4 + importance_score * 0.3

    return round(score, 4)


# ─── Decay Cycle ──────────────────────────────────────────────────────────────

async def run_decay_cycle() -> dict:
    """
    Execute one memory decay/pruning cycle across all collections.

    Steps:
      1. For each collection, check if it's over its cap
      2. Score all documents in over-cap collections
      3. Delete documents below the relevance threshold
      4. If still over cap, delete lowest-scored until within limit

    Returns:
        Dict with stats: {collection_name: {"before": N, "deleted": N, "after": N}}
    """
    if not is_ready():
        return {}

    stats: dict[str, dict] = {}
    total_deleted = 0

    for name in COLLECTIONS:
        col = _collections.get(name)
        if not col:
            continue

        count = col.count()
        cap = COLLECTION_CAPS.get(name, 10_000)

        if count == 0:
            stats[name] = {"before": 0, "deleted": 0, "after": 0}
            continue

        deleted = 0

        # Only run full scoring if collection is approaching or over cap
        # or periodically to clean truly stale entries
        should_prune = count > cap * 0.8  # start pruning at 80% of cap

        if should_prune:
            # Get all documents from the collection
            try:
                all_docs = col.get(
                    include=["metadatas"],
                    limit=count,
                )
            except Exception as e:
                logger.error(f"Decay: failed to get docs from '{name}': {e}")
                stats[name] = {"before": count, "deleted": 0, "after": count}
                continue

            if not all_docs or not all_docs.get("ids"):
                stats[name] = {"before": count, "deleted": 0, "after": count}
                continue

            ids = all_docs["ids"]
            metas = all_docs.get("metadatas", [{}] * len(ids))

            # Score all documents
            scored = []
            for doc_id, meta in zip(ids, metas):
                meta = meta or {}
                score = compute_relevance(meta)
                scored.append((doc_id, score, meta))

            # Sort by score ascending (lowest first — candidates for deletion)
            scored.sort(key=lambda x: x[1])

            # Phase 1: Delete below absolute threshold
            ids_to_delete = []
            for doc_id, score, meta in scored:
                doc_type = meta.get("type", "")
                if doc_type in PROTECTED_TYPES:
                    continue  # never delete protected types
                if score < RELEVANCE_THRESHOLD:
                    ids_to_delete.append(doc_id)

            # Phase 2: If still over cap, remove lowest-scored until within cap
            remaining_after_threshold = count - len(ids_to_delete)
            if remaining_after_threshold > cap:
                excess = remaining_after_threshold - cap
                to_remove = int(excess * PRUNE_FRACTION) + 1
                candidates = [
                    (doc_id, score, meta)
                    for doc_id, score, meta in scored
                    if doc_id not in set(ids_to_delete)
                    and meta.get("type", "") not in PROTECTED_TYPES
                ]
                for doc_id, _score, _meta in candidates[:to_remove]:
                    ids_to_delete.append(doc_id)

            # Execute deletion in batches
            if ids_to_delete:
                batch_size = 500
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    try:
                        col.delete(ids=batch)
                        deleted += len(batch)
                    except Exception as e:
                        logger.error(
                            f"Decay: failed to delete batch from '{name}': {e}"
                        )

        after = col.count()
        stats[name] = {
            "before": count,
            "deleted": deleted,
            "after": after,
        }
        total_deleted += deleted

    if total_deleted > 0:
        logger.info(
            f"Memory decay cycle: deleted {total_deleted} documents. "
            f"Stats: {stats}"
        )
    else:
        logger.debug("Memory decay cycle: no documents pruned.")

    return stats


async def get_decay_stats() -> dict:
    """
    Get current memory usage statistics for monitoring.

    Returns dict per collection: count, cap, usage_pct.
    """
    counts = await get_all_counts()
    result = {}
    for name, count in counts.items():
        cap = COLLECTION_CAPS.get(name, 10_000)
        result[name] = {
            "count": count,
            "cap": cap,
            "usage_pct": round(count / cap * 100, 1) if cap > 0 else 0,
        }
    return result
