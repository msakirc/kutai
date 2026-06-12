"""FIX 3.2 — ONE merged client+singleton snapshot view for the image lane.

Process split (prod): NerdHerd runs as a SIDECAR. The in-process singleton
sees ONLY what is written in-process — image-server residency (clair_obscur)
and queue_profile (beckman). Everything else (load_mode, presence/contention,
local.model_name via push_local_state, in_flight_calls via push_in_flight)
lives on the CLIENT snapshot. Before this fix the consumer made per-field
seam decisions and read llm_loaded / in_flight off the SINGLETON — where
they are permanently None/empty in prod, leaving the +4000 llama-unload
allowance dead and the _EVICTION_HUGE in-flight guard inverted.

These tests exercise the REAL merge through the nerd_herd client round-trip
(only the cached client snapshot is faked, as run.py's refresh loop would
populate it)."""
import pytest

from fatih_hoca.image_select import (
    _EVICTION_HUGE,
    _effective_snapshot,
    _eviction_cost,
    select_image,
)
from fatih_hoca.types import Pick, SelectionFailure


def _set_client(monkeypatch, snap):
    from nerd_herd import client as nh_client
    c = nh_client.NerdHerdClient()
    c._cached_snapshot = snap
    monkeypatch.setattr(nh_client, "_default", c)
    return c


def _clear_client(monkeypatch):
    from nerd_herd import client as nh_client
    monkeypatch.setattr(nh_client, "_default", None)


def _singleton_snap(*, image_resident=False, image_vram=0, queue_ready=0,
                    recent_swaps=0, vram_mb=6000):
    class _Local:
        model_name = None
        requests_processing = 0
    class _QP:
        total_ready_count = queue_ready
    class _S:
        local = _Local()
        queue_profile = _QP() if queue_ready else None
        in_flight_calls = []
        image_server_resident = image_resident
        image_server_vram_mb = image_vram
        recent_swap_count = recent_swaps
        vram_available_mb = vram_mb
        load_mode = "full"   # singleton NEVER sees /mode in prod
    return _S()


@pytest.fixture
def real_exe(tmp_path, monkeypatch):
    p = tmp_path / "clair_obscur_server.exe"
    p.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(p))
    return str(p)


def test_merged_view_composes_client_base_with_singleton_overlay(monkeypatch):
    """Client carries load_mode/presence/llm-loaded/in-flight; singleton
    carries residency/queue_profile/recent_swap_count. The merged view must
    take each field from its authoritative process."""
    from nerd_herd.types import (InFlightCall, LocalModelState,
                                 SystemSnapshot)
    _set_client(monkeypatch, SystemSnapshot(
        vram_available_mb=5000,
        load_mode="shared",
        foreground_fullscreen=True,
        external_gpu_fraction=0.4,
        local=LocalModelState(model_name="qwen2.5"),
        in_flight_calls=[InFlightCall(
            call_id="c1", task_id=1, category="main_work", model="qwen2.5",
            provider="local", is_local=True, started_at=0.0)],
    ))
    monkeypatch.setattr(
        "fatih_hoca.image_select._snapshot",
        lambda: _singleton_snap(image_resident=True, image_vram=4500,
                                queue_ready=3, recent_swaps=2))

    merged = _effective_snapshot()
    # client-authoritative (sidecar receives the pushes / owns the sensors)
    assert merged.load_mode == "shared"
    assert merged.foreground_fullscreen is True
    assert merged.external_gpu_fraction == 0.4
    assert merged.local.model_name == "qwen2.5"
    assert len(merged.in_flight_calls) == 1
    assert merged.vram_available_mb == 5000
    # singleton-authoritative (written in-process, sidecar never sees them)
    assert merged.image_server_resident is True
    assert merged.image_server_vram_mb == 4500
    assert merged.queue_profile.total_ready_count == 3
    assert merged.recent_swap_count == 2


def test_merge_does_not_mutate_client_cache(monkeypatch):
    """The overlay must happen on a copy — NOT on the client's cached
    snapshot object (other consumers read that cache)."""
    from nerd_herd.types import SystemSnapshot
    c = _set_client(monkeypatch, SystemSnapshot(load_mode="heavy"))
    monkeypatch.setattr(
        "fatih_hoca.image_select._snapshot",
        lambda: _singleton_snap(image_resident=True, image_vram=4500))
    merged = _effective_snapshot()
    assert merged.image_server_resident is True
    assert c._cached_snapshot.image_server_resident is False


def test_client_down_falls_back_to_singleton_only(monkeypatch):
    """No wired client (orchestrator boot, tests) → singleton-only view,
    i.e. exactly today's behavior."""
    _clear_client(monkeypatch)
    sing = _singleton_snap(image_resident=True, image_vram=4500)
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: sing)
    assert _effective_snapshot() is sing


def test_client_error_falls_back_to_singleton_only(monkeypatch):
    from nerd_herd import client as nh_client

    class _Boom(nh_client.NerdHerdClient):
        def snapshot(self):
            raise RuntimeError("client exploded")

    monkeypatch.setattr(nh_client, "_default", _Boom())
    sing = _singleton_snap()
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: sing)
    assert _effective_snapshot() is sing


def test_eviction_huge_fires_on_client_in_flight(monkeypatch):
    """The _EVICTION_HUGE in-flight guard was dead in prod: in_flight_calls
    live on the CLIENT snapshot (push_in_flight → sidecar) while the cost
    function read the singleton's permanently-empty list. Through the merged
    view a client-side in-flight call must yield HUGE for a cold local."""
    from fatih_hoca.image_providers import image_catalog
    from nerd_herd.types import InFlightCall, SystemSnapshot
    _set_client(monkeypatch, SystemSnapshot(
        vram_available_mb=6000,
        in_flight_calls=[InFlightCall(
            call_id="c1", task_id=1, category="main_work", model="qwen2.5",
            provider="local", is_local=True, started_at=0.0)],
    ))
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _singleton_snap())  # singleton: idle, empty
    local = next(m for m in image_catalog() if m.is_local)
    assert _eviction_cost(local, _effective_snapshot()) == _EVICTION_HUGE


def test_llm_loaded_allowance_revived_via_client(monkeypatch, real_exe):
    """The +4000 llama-unload allowance read local.model_name off the
    SINGLETON, where it is permanently None in prod (push_local_state goes
    to the sidecar) — dead code. With the merged view, a client-side loaded
    llama grants the allowance: free 2500 + 4000 >= 4500 → local eligible
    (all cloud failed → local is picked instead of SelectionFailure)."""
    from nerd_herd.types import LocalModelState, SystemSnapshot
    _set_client(monkeypatch, SystemSnapshot(
        vram_available_mb=2500,
        local=LocalModelState(model_name="qwen2.5"),
    ))
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _singleton_snap(vram_mb=2500))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality",
                        failures=["huggingface/flux-schnell",
                                  "pollinations/flux"],
                        hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.provider == "clair_obscur"


def test_load_mode_comes_off_merged_client_base(monkeypatch, real_exe):
    """select_image's Minimal veto must see the CLIENT load_mode through the
    merged view (singleton stays 'full' forever in prod)."""
    from nerd_herd.types import SystemSnapshot
    _set_client(monkeypatch, SystemSnapshot(
        vram_available_mb=6000, load_mode="minimal"))
    monkeypatch.setattr(
        "fatih_hoca.image_select._snapshot",
        lambda: _singleton_snap(image_resident=True, image_vram=4500))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[],
                        hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False
