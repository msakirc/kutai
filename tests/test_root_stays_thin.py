"""Modularization guardrail (P7, 2026-06-07).

Locks in the P1–P6 de-accretion so the root doesn't silently re-fatten. The
live bot re-adds feature logic to the orchestrator over time (documented in
docs/2026-05-31-root-debt-map.md); these assertions are the ratchet.

Three invariants:
  1. Declared thin shims stay thin (a fat impl must not regrow inside a file
     whose whole job is a back-compat re-export).
  2. Layer purity: LLM *execution* (litellm) stays in packages — src/core may
     delegate to hallederiz_kadir (the dispatcher owns that seam) but must not
     import litellm directly or re-implement execution.
  3. No package __init__ stub masks a fat src/ twin (the workflow_engine
     half-extraction: packages/workflow_engine is a stub; the real engine is
     src/workflows/engine — prod must use the latter).

Ceilings carry headroom and a "raise only with a refactor PR" contract: bump
one only alongside a commit that genuinely restructures, never to wave a
regression through.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _loc(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


# ── 1. Thin shims stay thin ──────────────────────────────────────────────────
# Genuine re-export shims — these must never grow real logic. local_model_manager
# is deliberately EXCLUDED: it is a fat DaLLaMa-facing wrapper with real load/
# swap orchestration (see ceilings below), not a shim.
_SHIM_MAX = 80
_THIN_SHIMS = [
    "src/models/model_registry.py",
    "src/models/gpu_monitor.py",
    "src/models/rate_limiter.py",
    "src/models/introspection.py",
    "src/models/context_sizing.py",
    "src/core/router.py",
    "src/core/result_router.py",
    "src/core/task_context.py",
]
_FAT_MODELS_ALLOWLIST = {"local_model_manager.py"}


@pytest.mark.parametrize("rel", _THIN_SHIMS)
def test_declared_shim_stays_thin(rel):
    path = ROOT / rel
    assert path.exists(), f"declared shim missing: {rel}"
    loc = _loc(path)
    assert loc < _SHIM_MAX, (
        f"{rel} is {loc} LOC — a 'thin shim' must stay < {_SHIM_MAX}. "
        f"If it grew real logic, move that logic to its owning package."
    )


def test_no_new_fat_file_in_src_models():
    """Every src/models/*.py is either a known thin shim or the one allowlisted
    fat wrapper. A new fat file here means logic that belongs in a package."""
    offenders = []
    for path in sorted((ROOT / "src" / "models").glob("*.py")):
        if path.name in _FAT_MODELS_ALLOWLIST:
            continue
        if _loc(path) >= _SHIM_MAX:
            offenders.append(f"{path.name} ({_loc(path)} LOC)")
    assert not offenders, (
        "fat module(s) in src/models that are neither a thin shim nor the "
        f"allowlisted local_model_manager: {offenders}. Move logic into the "
        "owning package or add to _FAT_MODELS_ALLOWLIST with a ceiling."
    )


# ── 2. Layer purity — LLM execution stays in packages ────────────────────────
def _src_core_files():
    return sorted((ROOT / "src" / "core").rglob("*.py"))


def test_src_core_never_imports_litellm():
    """litellm calls live in HaLLederiz Kadir. No src/core module may import it
    directly — that would mean re-implementing execution outside the package."""
    offenders = []
    for path in _src_core_files():
        text = path.read_text(encoding="utf-8")
        if "import litellm" in text or "from litellm" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"src/core imports litellm directly (execution must stay in "
        f"hallederiz_kadir): {offenders}"
    )


def test_hallederiz_kadir_only_imported_by_dispatcher():
    """src/core delegates LLM execution to HaLLederiz Kadir ONLY through the
    dispatcher (the ask→load→call seam). Any other src/core importer means the
    boundary leaked."""
    offenders = []
    for path in _src_core_files():
        if path.name == "llm_dispatcher.py":
            continue  # the designated seam
        text = path.read_text(encoding="utf-8")
        if "import hallederiz_kadir" in text or "from hallederiz_kadir" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"hallederiz_kadir imported outside the dispatcher seam: {offenders}"
    )


# ── 3. No stub package masks a fat src/ twin ─────────────────────────────────
def test_src_uses_real_workflow_engine_not_stub():
    """The real engine is src.workflows.engine (~8.8k LOC); packages/
    workflow_engine is a thin re-export stub. Prod code under src/ must import
    the real engine, never the bare ``workflow_engine`` stub — otherwise the
    half-extraction silently becomes the live path."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("import workflow_engine") or s.startswith("from workflow_engine"):
                offenders.append(f"{path.relative_to(ROOT)}: {s}")
    assert not offenders, (
        "src/ imports the workflow_engine STUB instead of src.workflows.engine: "
        f"{offenders}"
    )


def test_workflow_engine_stub_stays_a_stub():
    """If the stub ever grows into a fat twin of the real engine, that's the
    masking debt materializing — fail so it's a deliberate decision."""
    init = ROOT / "packages/workflow_engine/src/workflow_engine/__init__.py"
    assert init.exists(), "workflow_engine stub __init__ missing"
    assert _loc(init) < 120, (
        f"workflow_engine stub __init__ is {_loc(init)} LOC — it is supposed "
        "to be a thin re-export of src.workflows.engine, not a second engine."
    )


# ── Ratchet ceilings (raise ONLY with a real refactor PR) ────────────────────
# Headroom above the post-P5/P6 sizes so a small honest change doesn't trip the
# wire, but the live bot's slow re-accretion does. Do NOT bump to pass a commit
# that merely re-adds logic — bump only when the same PR restructures.
_CEILINGS = {
    "src/core/orchestrator.py": 690,        # 619 after P5
    "src/core/llm_dispatcher.py": 690,      # 623; SP5 request()-shim delete pending
    "src/models/local_model_manager.py": 600,  # 540 after P6
}


@pytest.mark.parametrize("rel,ceiling", list(_CEILINGS.items()))
def test_fat_file_under_ceiling(rel, ceiling):
    path = ROOT / rel
    assert path.exists(), f"missing: {rel}"
    loc = _loc(path)
    assert loc <= ceiling, (
        f"{rel} is {loc} LOC, over its {ceiling} ceiling. Extract logic to its "
        f"owning package; raise the ceiling ONLY in a PR that restructures."
    )
