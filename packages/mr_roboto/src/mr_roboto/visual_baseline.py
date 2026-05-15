"""Z4 T5A — cross-mission baseline store.

Per-mission baselines (``mission_{id}/.visual/baseline/``) do not carry
across missions.  This module provides a repo-root ``.visual_baseline/``
store keyed by a hash of the current design tokens so a stable design system
can reuse baselines mission-to-mission.

Public API
----------
- ``token_hash(design_tokens)`` → stable 12-char hex string (or ``"notokens"``)
- ``cross_mission_baseline_dir(repo_root, thash)`` → path string
- ``resolve_baseline(captured_basename, *, mission_baseline_dir, cross_dir)``
  → first matching path or None
- ``promote_to_cross_mission(frame_path, cross_dir)`` → destination path
- ``tokens_changed(workspace_path, current_hash)`` → bool
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.visual_baseline")

# Sentinel used when design_tokens is absent/empty
_NO_TOKENS_SENTINEL = "notokens"

# Filename that records the current token hash inside a mission's .visual/ dir
_TOKEN_HASH_FILE = ".token_hash"


def token_hash(design_tokens: dict | None) -> str:
    """Return a stable 12-character hex hash of *design_tokens*.

    The hash is order-independent: dicts are serialised with sorted keys so
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` yield the same hash.

    Parameters
    ----------
    design_tokens:
        Arbitrary mapping; ``None`` or empty dict → ``"notokens"`` sentinel.

    Returns
    -------
    12-character lowercase hex string, or the ``"notokens"`` sentinel.
    """
    if not design_tokens:
        return _NO_TOKENS_SENTINEL
    try:
        canonical = json.dumps(design_tokens, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:12]
    except Exception as exc:
        logger.warning("visual_baseline: token_hash failed (%s), using sentinel", exc)
        return _NO_TOKENS_SENTINEL


def cross_mission_baseline_dir(repo_root: str, thash: str) -> str:
    """Return the cross-mission baseline directory for *thash*.

    Path: ``{repo_root}/.visual_baseline/{thash}/``
    """
    return os.path.join(repo_root, ".visual_baseline", thash)


def resolve_baseline(
    captured_basename: str,
    *,
    mission_baseline_dir: str | None,
    cross_dir: str | None,
) -> str | None:
    """Find a baseline file for *captured_basename*, checking per-mission first.

    Resolution order
    ----------------
    1. ``{mission_baseline_dir}/{captured_basename}`` — per-mission baseline.
    2. ``{cross_dir}/{captured_basename}`` — cross-mission baseline.
    3. ``None`` — no baseline found (AUDIT mode).

    Parameters
    ----------
    captured_basename:
        The filename of the captured screenshot (e.g. ``root_light_375.png``).
    mission_baseline_dir:
        Directory containing per-mission approved baselines.  May be ``None``
        when the caller has no mission baseline to check.
    cross_dir:
        Cross-mission baseline directory (see ``cross_mission_baseline_dir``).
        May be ``None`` when tokens are absent.

    Returns
    -------
    Absolute path to the first matching baseline, or ``None``.
    """
    if mission_baseline_dir:
        candidate = os.path.join(mission_baseline_dir, captured_basename)
        if os.path.isfile(candidate):
            logger.debug("visual_baseline: per-mission baseline hit: %s", candidate)
            return candidate

    if cross_dir:
        candidate = os.path.join(cross_dir, captured_basename)
        if os.path.isfile(candidate):
            logger.debug("visual_baseline: cross-mission baseline hit: %s", candidate)
            return candidate

    return None


def promote_to_cross_mission(frame_path: str, cross_dir: str) -> str:
    """Copy *frame_path* into *cross_dir* (idempotent).

    If a file with the same basename already exists in *cross_dir*, it is
    silently overwritten — idempotency is guaranteed by the copy itself.

    Parameters
    ----------
    frame_path:
        Absolute path to the frame to promote.
    cross_dir:
        Destination cross-mission baseline directory.

    Returns
    -------
    Absolute path of the copied file in *cross_dir*.
    """
    os.makedirs(cross_dir, exist_ok=True)
    dest = os.path.join(cross_dir, os.path.basename(frame_path))
    shutil.copy2(frame_path, dest)
    logger.debug("visual_baseline: promoted %s → %s", frame_path, dest)
    return dest


def tokens_changed(workspace_path: str, current_hash: str) -> bool:
    """Return ``True`` when the stored token hash differs from *current_hash*.

    The stored hash lives at ``{workspace_path}/.visual/{_TOKEN_HASH_FILE}``.
    When the file is absent (first run) this also returns ``True``.

    A WARNING is logged when the hash has changed so callers know that
    cross-mission baselines for the old hash may be stale.  No auto-deletion
    is performed — the decision is left to the operator.

    Parameters
    ----------
    workspace_path:
        Mission workspace root (e.g. ``/path/to/mission_42/``).
    current_hash:
        The hash computed from the current design tokens.

    Returns
    -------
    ``True`` when the hash has changed or no stored hash exists.
    """
    hash_file = os.path.join(workspace_path, ".visual", _TOKEN_HASH_FILE)
    try:
        with open(hash_file, "r", encoding="utf-8") as fh:
            stored = fh.read().strip()
    except OSError:
        # File absent → treat as changed (first run).
        logger.debug("visual_baseline: no stored token hash at %s", hash_file)
        return True

    if stored != current_hash:
        logger.warning(
            "visual_baseline: design tokens changed (was=%s now=%s); "
            "cross-mission baselines for the old hash may be stale — "
            "promote new approved frames to update them",
            stored,
            current_hash,
        )
        return True

    return False


def _write_token_hash(workspace_path: str, thash: str) -> None:
    """Persist *thash* to ``{workspace_path}/.visual/.token_hash``."""
    visual_dir = os.path.join(workspace_path, ".visual")
    os.makedirs(visual_dir, exist_ok=True)
    hash_file = os.path.join(visual_dir, _TOKEN_HASH_FILE)
    try:
        with open(hash_file, "w", encoding="utf-8") as fh:
            fh.write(thash)
        logger.debug("visual_baseline: wrote token hash %s → %s", thash, hash_file)
    except OSError as exc:
        logger.warning("visual_baseline: could not write token hash: %s", exc)
