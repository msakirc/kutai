"""Write-time selective-encoding gate (smart-RAG Phase 1).

Applied at the ``embed_and_store`` choke point so every writer inherits the
policy with no per-caller change. Two Phase-1 duties:

  P1a  KILL the *implicit* per-task "accepted" ``user_feedback`` firehose
       (never read via any filtered path; real ratings live in the SQLite
       ``task_feedback`` table). Explicit corrections (modified/rejected) are
       PRESERVED — they carry real user-correction text not mirrored elsewhere.
  P1c  Quality-filter ONLY firehose/auto types (``task_result``): reject
       degenerate / template-echo / too-short text before embedding. Curated
       and user-authored types pass through untouched (handoff §4).

Novelty-merge (P1b) and TTL/supersession (P3) extend this seam in later phases.
Killswitch: ``KUTAI_ENCODE_POLICY=off`` disables all gating.
"""
from __future__ import annotations

import os

_MIN_LEN = 12  # below this there is nothing to retrieve on

# Auto/firehose types that get the quality filter. Curated + user-authored
# types are deliberately NOT here — they must never be dropped.
_QUALITY_FILTERED_TYPES = {"task_result"}


def _enabled() -> bool:
    return os.getenv("KUTAI_ENCODE_POLICY", "on").strip().lower() != "off"


def should_store(text: str, metadata: dict | None) -> tuple[bool, str]:
    """Decide whether ``text`` (with ``metadata``) is worth embedding.

    Returns ``(allowed, reason)``; ``reason`` is ``""`` when allowed, else a
    short machine-readable tag for logging.
    """
    if not _enabled():
        return True, ""
    metadata = metadata or {}
    mem_type = metadata.get("type")

    # P1a — kill only the implicit per-task "accepted" feedback firehose.
    if mem_type == "user_feedback" and metadata.get("feedback_type") == "accepted":
        return False, "killed_implicit_feedback"

    # P1c — quality filter scoped to firehose/auto types only.
    if mem_type in _QUALITY_FILTERED_TYPES:
        t = (text or "").strip()
        if len(t) < _MIN_LEN:
            return False, "too_short"
        # Reuse the skills-system pollution regex (imported lazily to avoid a
        # circular import; embed_and_store already imports this module lazily).
        from src.memory.skills import _DESC_POLLUTION_RE
        if _DESC_POLLUTION_RE.search(t):
            return False, "pollution"

    return True, ""
