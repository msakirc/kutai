"""Pytest configuration for kuleden_donen_var tests.

Prepends this worktree's packages/kuleden_donen_var/src to sys.path so
pytest loads the worktree source, NOT the shared-venv editable install
(which points at the main repo via __editable__.kuleden_donen_var-*.pth).

Without this, new modules added in the worktree (e.g. in_flight.py) are
not visible because the editable path to main wins at import time.
"""
import pathlib
import sys

_here = pathlib.Path(__file__).parent
_worktree_src = _here.parent / "src"

if str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))

# Evict any already-imported kuleden_donen_var module so fresh resolution
# picks up the worktree path on next import.
for _mod in list(sys.modules):
    if _mod == "kuleden_donen_var" or _mod.startswith("kuleden_donen_var."):
        del sys.modules[_mod]
