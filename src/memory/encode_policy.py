"""Write-time selective-encoding gate (smart-RAG Phase 1).

Applied at the ``embed_and_store`` choke point so every writer inherits the
policy with no per-caller change. Two Phase-1 duties:

  P1a  KILL the *implicit* per-task "accepted" ``user_feedback`` firehose
       (never read via any filtered path; real ratings live in the SQLite
       ``task_feedback`` table). Explicit corrections (modified/rejected) are
       PRESERVED — they carry real user-correction text not mirrored elsewhere.
  P1c  Length-floor ONLY firehose/auto types (``task_result``): drop too-short
       text before embedding. Semantic degeneracy is handled upstream by
       dogru_mu_samet at the sole task_result writer (episodic.store_task_result),
       not re-checked here. Curated and user-authored types pass through
       untouched (handoff §4).

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

    # P1c — length floor scoped to firehose/auto types only. Semantic
    # degeneracy for task_result is already caught by dogru_mu_samet at the
    # sole writer (episodic.store_task_result, episodic.py:59); the
    # skill-description pollution regex was mis-fit for multi-line task prose
    # (measured on the live DB: it dropped good YAML/markdown design artifacts),
    # so it is deliberately NOT reused here.
    if mem_type in _QUALITY_FILTERED_TYPES:
        t = (text or "").strip()
        if len(t) < _MIN_LEN:
            return False, "too_short"

    return True, ""
