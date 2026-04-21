import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # worktree root

# Prepend worktree root so top-level src.* imports resolve to worktree
# (not main-tree via editable install)
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
