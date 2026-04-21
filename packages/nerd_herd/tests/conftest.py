"""Pytest configuration for nerd_herd tests.

Prepends this worktree's packages/nerd_herd/src to sys.path so that pytest
loads the worktree's nerd_herd code, NOT the shared-venv editable install
(which points at the main repo via an .pth file).
"""
import sys
import pathlib

_here = pathlib.Path(__file__).parent
_worktree_src = _here.parent / "src"
if str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))
