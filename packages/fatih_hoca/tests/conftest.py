"""Pytest configuration for fatih_hoca tests.

Prepends this worktree's packages/fatih_hoca/src to sys.path so that pytest
loads the worktree's fatih_hoca code, NOT the shared-venv editable install
(which points at the main repo via __editable__.fatih_hoca-0.1.0.pth).

Also adds the tests/ directory so that sim.state (and future tests/sim/*
modules) are importable without polluting the runtime package.
"""
import sys
import pathlib

_here = pathlib.Path(__file__).parent

# Prepend worktree src FIRST so it shadows the shared-venv editable install.
# The shared venv's .pth file points at the main repo's packages/fatih_hoca/src,
# which would otherwise win at import time over any worktree changes.
_worktree_src = _here.parent / "src"
if str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))

# Allow `from sim.state import ...` in tests/sim/
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
