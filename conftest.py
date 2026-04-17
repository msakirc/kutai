"""Root conftest.py to set up paths for all tests including worktree tests."""
import sys
from pathlib import Path

# Add all package source paths
repo_root = Path(__file__).parent
packages_to_add = [
    "packages/fatih_hoca/src",
    "packages/nerd_herd/src",
    "packages/dallama/src",
    "packages/yasar_usta/src",
    ".worktrees/fatih-hoca-intelligence/packages/fatih_hoca/src",
    ".worktrees/fatih-hoca-intelligence/packages/hallederiz_kadir/src",
    ".worktrees/fatih-hoca-intelligence/packages/dogru_mu_samet/src",
    ".worktrees/fatih-hoca-intelligence/packages/kuleden_donen_var/src",
    ".worktrees/fatih-hoca-intelligence/packages/vecihi/src",
]

for pkg in packages_to_add:
    pkg_path = repo_root / pkg
    if pkg_path.exists():
        sys.path.insert(0, str(pkg_path))
