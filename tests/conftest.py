import os
import sys
from pathlib import Path

# Z1 Tier 5C (B4) — Critic gate is default-on in production but must be
# default-off in unit tests so that pre-existing router tests don't hit
# the real LLMDispatcher. Dedicated gate tests opt back in via
# `monkeypatch.delenv("KUTAI_CRITIC_GATE", raising=False)`.
#
# Lives in tests/conftest.py (not worktree-root conftest.py) so the
# setdefault only fires during pytest runs from under tests/. Production
# orchestrator paths must keep the var unset.
os.environ.setdefault("KUTAI_CRITIC_GATE", "off")

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

# Phase B (registry slice): the model-registry tables (model_stats, model_pick_log,
# providers, models, registry_events) are created by dabidabi.init_db() ONLY via the
# schema-registration side-effect of importing fatih_hoca (their inline DDL was removed
# from init_db). Production guarantees this through an explicit import in src/app/run.py;
# tests must do the same here, or any test that calls init_db() without importing
# fatih_hoca gets "no such table: model_pick_log".
import fatih_hoca  # noqa: E402,F401
