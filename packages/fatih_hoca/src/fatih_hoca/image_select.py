"""Purpose-built image-model scorer. Sibling to selector.py (text). Plan 1
shipped cloud-only with a stub eviction-cost. Plan 2 adds the local
clair_obscur entry and the real eviction-cost + VRAM-fit gate, reading the
LIVE in-process nerd_herd singleton snapshot (residency/VRAM). load_mode is
the one exception — it is read from the client-backed module-level
``nerd_herd.snapshot()`` because /mode lands on the sidecar, never on the
orchestrator-process singleton (see ``_client_load_mode``)."""
from __future__ import annotations

import os

from .image_providers import image_catalog
from .registry import ImageModelInfo
from .types import Pick, SelectionFailure


_EVICTION_HUGE = 100.0
_EVICTION_HIGH = 50.0
_EVICTION_LOW = 2.0
_WARM_BATCH_BONUS = 1.0


def _snapshot():
    """Read LIVE in-process nerd_herd state via the singleton.

    Must read ``nerd_herd._get_singleton().snapshot()`` — NOT module-level
    ``nerd_herd.snapshot()`` (which returns the NerdHerdClient's CACHED value,
    stale w.r.t. residency) and NOT ``refresh_snapshot()`` (async — calling it
    from this sync helper returns a coroutine, so every ``getattr`` reads
    False/0 and local would never be selected). The singleton is exactly
    where ``record_image_server_state()`` (clair_obscur) and ``record_swap()``
    (husam) write, and its ``snapshot()`` builds ``vram_available_mb`` from a
    live GPU read — so residency and freed VRAM are reflected synchronously.
    Tests monkeypatch this helper directly."""
    try:
        import nerd_herd
        return nerd_herd._get_singleton().snapshot()
    except Exception:
        from nerd_herd.types import SystemSnapshot
        return SystemSnapshot()


def _client_load_mode() -> str:
    """Read load_mode from the CLIENT-backed module-level
    ``nerd_herd.snapshot()`` (the NerdHerdClient cache, refreshed ~2s by
    run.py's _snapshot_refresh_loop) — NOT from ``_snapshot()``.

    Process split: in prod NerdHerd runs as a SIDECAR process. ``/mode
    minimal`` flows telegram_bot → load_manager.set_load_mode →
    NerdHerdClient → sidecar, so the orchestrator-process singleton's
    LoadManager stays "full" forever — a singleton read would make the
    Minimal veto dead in prod. Residency/VRAM stay on the singleton seam
    (``_snapshot``), which is where ``record_image_server_state()`` writes.

    Defaults to "full" on any error / missing field. Tests monkeypatch this
    helper (the real-seam test goes through the client round-trip instead)."""
    try:
        import nerd_herd
        return str(getattr(nerd_herd.snapshot(), "load_mode", "full") or "full")
    except Exception:
        return "full"


def _provider_available(m: ImageModelInfo, hf_available: bool | None) -> bool:
    if m.provider == "huggingface":
        return os.getenv("HF_TOKEN") is not None if hf_available is None else hf_available
    if m.provider == "pollinations":
        return True
    if m.provider == "clair_obscur":
        # Read env DIRECTLY (the module-level clair_obscur.available() reads a
        # cached singleton config, stale/untestable here) AND require the exe
        # to exist on disk. Design §10: an absent/misconfigured backend must be
        # filtered at SELECTION time — a path that doesn't exist must NOT pass
        # selection, or hoca picks local and husam's start() fails at dispatch
        # (wasted swap + retry). "Filter early, no crash."
        exe = os.getenv("CLAIR_OBSCUR_EXE", "")
        return bool(exe) and os.path.exists(exe)
    return False


def _eviction_cost(m: ImageModelInfo, snap=None) -> float:
    """Real eviction cost (Plan 2). Cloud providers always score 0.

    Reads the snapshot ONCE per ``select_image`` call (passed in via ``snap``);
    falls back to ``_snapshot()`` for direct/legacy callers."""
    if not getattr(m, "is_local", False):
        return 0.0
    s = snap if snap is not None else _snapshot()
    if getattr(s, "image_server_resident", False):
        return 0.0
    in_flight = len(getattr(s, "in_flight_calls", []) or [])
    if in_flight == 0:
        local = getattr(s, "local", None)
        in_flight = int(getattr(local, "requests_processing", 0) or 0)
    if in_flight > 0:
        return _EVICTION_HUGE
    llm_loaded = bool(getattr(getattr(s, "local", None), "model_name", None))
    qp = getattr(s, "queue_profile", None)
    llm_queue = int(getattr(qp, "total_ready_count", 0) or 0) if qp else 0
    if llm_loaded or llm_queue > 0:
        return _EVICTION_HIGH
    return _EVICTION_LOW


def _warm_batch_bonus(m: ImageModelInfo, snap) -> float:
    if not getattr(m, "is_local", False):
        return 0.0
    return _WARM_BATCH_BONUS if getattr(snap, "image_server_resident", False) else 0.0


def select_image(
    *,
    quality_tier: str = "fast",
    failures: list[str] | None = None,
    hf_available: bool | None = None,
    remaining_budget_usd: float | None = None,
) -> Pick | SelectionFailure:
    # IMPORTANT: failures is a list of STRING provider/model names.
    failed = set(failures or [])
    snap = _snapshot()
    load_mode = _client_load_mode()
    candidates: list[tuple[float, ImageModelInfo]] = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        # Minimal load-mode = cloud-only (mirrors selector.py's
        # load_mode_minimal eligibility veto, which never fires for images
        # because select() short-circuits to select_image() first). Under
        # Minimal a local image pick would shut down the loaded llama and
        # grab ~4.5GB VRAM.
        # TWO SEAMS by design: residency/VRAM come from the in-process
        # singleton (``snap`` — where record_image_server_state writes), but
        # load_mode comes from the client-backed nerd_herd.snapshot()
        # (sidecar truth — the singleton never sees /mode in prod).
        if m.is_local and load_mode == "minimal":
            continue
        # VRAM-fit eligibility (mirrors selector.py's needs_vision gate).
        # Local: refuse if free VRAM (after a hypothetical llama unload — we
        # add a conservative 4GB local-recoverable allowance) can't fit.
        if m.is_local and m.vram_mb > 0:
            free_mb = int(getattr(snap, "vram_available_mb", 0) or 0)
            llm_loaded_mb = 4000 if getattr(
                getattr(snap, "local", None), "model_name", None
            ) else 0
            if (free_mb + llm_loaded_mb) < m.vram_mb:
                continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        score = m.quality_rank - _eviction_cost(m, snap) + _warm_batch_bonus(m, snap)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_summary = "; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5])
    best_score, best = candidates[0]
    return Pick(model=best, min_time_seconds=0.0, score=best_score, top_summary=top_summary)
