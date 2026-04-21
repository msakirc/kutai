"""Worktree-root conftest.

Prepends every packages/*/src directory to sys.path BEFORE test collection
so pytest resolves each package to the worktree source rather than the
shared-venv editable install (which points at the main repo).

Having a single root conftest avoids pluggy "Plugin already registered
under a different name" collisions caused by multiple `tests/conftest.py`
files under `packages/*/tests/`.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent

# Worktree root (so `src.*` imports resolve to worktree code).
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PACKAGE_SRCS = [
    _ROOT / "packages" / "fatih_hoca" / "src",
    _ROOT / "packages" / "nerd_herd" / "src",
    _ROOT / "packages" / "kuleden_donen_var" / "src",
    _ROOT / "packages" / "general_beckman" / "src",
    _ROOT / "packages" / "hallederiz_kadir" / "src",
    _ROOT / "packages" / "dallama" / "src",
    _ROOT / "packages" / "dogru_mu_samet" / "src",
    _ROOT / "packages" / "vecihi" / "src",
    _ROOT / "packages" / "yasar_usta" / "src",
    _ROOT / "packages" / "yazbunu" / "src",
]
for p in _PACKAGE_SRCS:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Evict any already-imported worktree-overridden packages so fresh lookup
# picks up the worktree path. Necessary because the shared venv's .pth
# file may have pre-imported them before this conftest runs.
for _mod in list(sys.modules):
    root = _mod.split(".", 1)[0]
    if root in {
        "fatih_hoca", "nerd_herd", "kuleden_donen_var", "general_beckman",
        "hallederiz_kadir", "dallama", "dogru_mu_samet", "vecihi",
        "yasar_usta", "yazbunu",
    }:
        del sys.modules[_mod]
