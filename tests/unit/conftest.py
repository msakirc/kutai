"""Conftest for unit tests: add package src dirs to sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve to the repo root (tests/unit -> tests -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

for _pkg_src in [
    _REPO_ROOT / "packages" / "fatih_hoca" / "src",
    _REPO_ROOT / "packages" / "nerd_herd" / "src",
    _REPO_ROOT / "packages" / "dallama" / "src",
]:
    _pkg_src_str = str(_pkg_src)
    if _pkg_src_str not in sys.path:
        sys.path.insert(0, _pkg_src_str)
