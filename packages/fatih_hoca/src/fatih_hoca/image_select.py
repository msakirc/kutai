"""Purpose-built image-model scorer. Sibling to selector.py (text). Plan 1
shipped cloud-only with a stub eviction-cost. Plan 2 adds the local
clair_obscur entry and the real eviction-cost + VRAM-fit gate. All system
state is read through ONE merged snapshot view (``_effective_snapshot``)
that composes the sidecar-backed CLIENT snapshot with the few fields only
the in-process singleton sees — see the seam table on that helper."""
from __future__ import annotations

import os

from .image_providers import image_catalog
from .registry import ImageModelInfo
from .types import Pick, SelectionFailure


_EVICTION_HUGE = 100.0
_EVICTION_HIGH = 50.0
_EVICTION_LOW = 2.0
_WARM_BATCH_BONUS = 1.0

# S14 hard threshold (another process owns >=30% of VRAM → veto local).
# Sourced from nerd_herd so the image path can't drift from the selector's
# contention signal; fallback keeps fatih_hoca importable standalone.
try:
    from nerd_herd.signals.s14_contention import (
        EXTERNAL_GPU_VETO_FRACTION as _EXTERNAL_GPU_VETO,
    )
except Exception:  # pragma: no cover
    _EXTERNAL_GPU_VETO = 0.30


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


def _effective_snapshot():
    """ONE merged snapshot view — client base + in-process overlay.

    Process split: in prod NerdHerd runs as a SIDECAR. Pushes
    (push_local_state, push_in_flight, /mode) land client→sidecar, so the
    orchestrator-process singleton NEVER sees them; conversely residency and
    queue_profile are written in-process and the sidecar never sees those.
    Reading everything off one seam silently kills the other half (the
    +4000 llama-unload allowance and the _EVICTION_HUGE in-flight guard were
    dead in prod when this module read only the singleton).

    Seam table (which process is authoritative, and why):

      CLIENT base — module-level ``nerd_herd.snapshot()`` (NerdHerdClient
      cache, refreshed ~2s by run.py's _snapshot_refresh_loop):
        load_mode               /mode lands on the sidecar only
        local.model_name        dispatcher push_local_state → sidecar
        in_flight_calls         beckman push_in_flight → sidecar
        user_idle_s, foreground_fullscreen, ram_*, external_gpu_fraction
                                sidecar owns the desktop/GPU sensors
        vram_available_mb       sidecar polls the GPU itself (real value;
                                the singleton also reads GPU live — either
                                works, client kept as the base)

      SINGLETON overlay — ``_snapshot()`` (in-process writes only):
        image_server_resident   clair_obscur record_image_server_state()
        image_server_vram_mb    (same writer)
        queue_profile           beckman push_queue_profile (in-process)
        recent_swap_count       this process's own swap-budget window

    Degrade: no wired client / any client failure → singleton-only view
    (exactly the pre-merge behavior). Tests monkeypatch ``_snapshot`` for
    the singleton side and install a NerdHerdClient with a cached snapshot
    for the client side — or monkeypatch this helper directly.

    Future: a nerd_herd-owned merged view (sidecar echoes in-process fields
    back, or the client composes) is the right altitude; kept here for now
    to avoid widening the nerd_herd API in a fix batch."""
    import copy

    sing = _snapshot()
    try:
        import nerd_herd
        from nerd_herd.client import get_default
        if get_default() is None:
            return sing
        client_snap = nerd_herd.snapshot()
    except Exception:
        return sing
    if client_snap is None:
        return sing
    merged = copy.copy(client_snap)  # never mutate the client's cache
    merged.image_server_resident = bool(
        getattr(sing, "image_server_resident", False))
    merged.image_server_vram_mb = int(
        getattr(sing, "image_server_vram_mb", 0) or 0)
    merged.queue_profile = getattr(sing, "queue_profile", None)
    merged.recent_swap_count = int(
        getattr(sing, "recent_swap_count", 0) or 0)
    return merged


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
    falls back to ``_effective_snapshot()`` for direct/legacy callers."""
    if not getattr(m, "is_local", False):
        return 0.0
    s = snap if snap is not None else _effective_snapshot()
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
    snap = _effective_snapshot()
    load_mode = str(getattr(snap, "load_mode", "full") or "full")
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
        # grab ~4.5GB VRAM. load_mode rides the merged snapshot's CLIENT
        # base (sidecar truth — the in-process singleton never sees /mode
        # in prod; see _effective_snapshot's seam table).
        if m.is_local and load_mode == "minimal":
            continue
        # Desktop-veto parity (S13/S14): select(needs_image=True) bypasses
        # ALL selector eligibility, so mirror the hard-veto semantics here
        # under M4 (nerd_herd modifiers.M4_load_mode_weights): "full"
        # SILENCES desktop signals, heavy/shared honor them, minimal is the
        # blanket veto above. Eligibility-only — local is skipped, cloud
        # unaffected. This mirrors selector S13 (foreground-fullscreen
        # sentinel) and S14 (external-GPU contention) pending a shared
        # eligibility helper (future altitude fix).
        if m.is_local and load_mode != "full":
            if getattr(snap, "foreground_fullscreen", False):
                continue
            ext_gpu = float(getattr(snap, "external_gpu_fraction", 0.0) or 0.0)
            if ext_gpu >= _EXTERNAL_GPU_VETO:
                continue
        # VRAM-fit eligibility (mirrors selector.py's needs_vision gate).
        # Local: refuse if free VRAM (after a hypothetical llama unload — we
        # add a conservative 4GB local-recoverable allowance) can't fit.
        if m.is_local and m.vram_mb > 0:
            free_mb = int(getattr(snap, "vram_available_mb", 0) or 0)
            # Residency credit: when the image server is already RESIDENT,
            # its own footprint (~4.5GB) is exactly the VRAM the model
            # occupies — reusing it warm needs no NEW VRAM. Raw free VRAM
            # already has that footprint subtracted, so without the credit
            # image #2..N of a warm batch on the 8GB GPU always fails this
            # gate and skips local.
            if getattr(snap, "image_server_resident", False):
                free_mb += int(getattr(snap, "image_server_vram_mb", 0) or 0)
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
