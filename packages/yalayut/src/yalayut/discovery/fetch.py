"""Staging + promotion for fetched artifacts.

Disk layout (spec Data model):
  vendor/skills/.staging/<source-slug>/<name>/      — during fetch
  vendor/skills/<source-slug>/<name>/v<version>/    — after tier-classify enable
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Repo-root-relative vendor dir. Tests monkeypatch this.
_VENDOR_ROOT = Path("vendor") / "skills"


def _slug(source: str) -> str:
    """Filesystem-safe slug for a source id."""
    return (
        source.replace("github:", "").replace("/", "_")
        .replace("@", "_").replace(":", "_")
    )


def stage_dir(source: str, name: str) -> Path:
    """Return (creating) the staging dir for one artifact fetch."""
    d = _VENDOR_ROOT / ".staging" / _slug(source) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def promote(
    staging: Path, source: str, name: str, version: str
) -> Path:
    """Move a staged artifact to its versioned vendor home. Returns the
    final path. Overwrites an existing version dir (re-fetch of same v)."""
    final = _VENDOR_ROOT / _slug(source) / name / f"v{version}"
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(final))
    return final
