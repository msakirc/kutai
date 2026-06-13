import pytest

from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick, SelectionFailure


def _snap(*, llm_in_flight=0, llm_loaded=False, llm_queue=0,
          image_resident=False, vram_mb=None, mode="full",
          fullscreen=False, ext_gpu=0.0):
    # Realistic 8GB-GPU story: non-resident idle GPU has ~6000MB free; with
    # the ~4.5GB image server RESIDENT, raw free VRAM is only ~2800MB. The
    # old default (6000 free WITH the server resident) was physically
    # impossible on the 8GB card and masked the missing residency credit
    # in the VRAM-fit gate (warm image #2..N always skipped local).
    if vram_mb is None:
        vram_mb = 2800 if image_resident else 6000
    class _Local:
        model_name = "qwen2.5" if llm_loaded else None
        requests_processing = llm_in_flight
    class _QP:
        total_ready_count = llm_queue
    class _S:
        local = _Local()
        queue_profile = _QP()
        in_flight_calls = []
        image_server_resident = image_resident
        image_server_vram_mb = 4500 if image_resident else 0
        vram_available_mb = vram_mb
        load_mode = mode
        foreground_fullscreen = fullscreen
        external_gpu_fraction = ext_gpu
    return _S()


@pytest.fixture
def real_exe(tmp_path, monkeypatch):
    """A real, existing file path for CLAIR_OBSCUR_EXE. Selection eligibility
    now requires the exe to exist on disk (design §10: filter absent backend
    at selection time), so tests that expect local ELIGIBLE must point at a
    file that actually exists."""
    p = tmp_path / "clair_obscur_server.exe"
    p.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(p))
    return str(p)


def test_huge_when_llm_in_flight(monkeypatch, real_exe):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_in_flight=1))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_high_when_llm_loaded(monkeypatch, real_exe):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_loaded=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_low_when_idle_first_call_still_cloud(monkeypatch, real_exe):
    """8.0 (HF) > 7.5 − 2.0 = 5.5 (local LOW) → HF wins cold start."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _snap())
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_resident_with_warm_bonus_picks_local(monkeypatch, real_exe):
    """Image-server warm → local score = 7.5 + 1.0 = 8.5 > HF 8.0."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_residency_credit_makes_warm_local_eligible(monkeypatch, real_exe):
    """Warm-batch image #2..N on the 8GB GPU: raw free VRAM (2800) is BELOW
    the model's 4500MB need because the resident server itself holds ~4.5GB.
    The VRAM-fit gate must credit the resident server's own footprint
    (reuse-by-warm needs no NEW VRAM) — then local is eligible and wins via
    the warm bonus (7.5 + 1.0 = 8.5 > HF 8.0)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, vram_mb=2800))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.provider == "clair_obscur"


def test_no_residency_credit_when_not_resident(monkeypatch, real_exe):
    """Cold server + genuinely low free VRAM: no credit applies, local stays
    gated. With every cloud provider failed this surfaces as a
    SelectionFailure — proving the gate (not scoring) filtered local."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=2800))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality",
                        failures=["huggingface/flux-schnell",
                                  "pollinations/flux"],
                        hf_available=True)
    assert isinstance(pick, SelectionFailure)
    assert pick.reason == "availability"


def test_vram_too_low_filters_local(monkeypatch, real_exe):
    # Exe EXISTS (real_exe) so provider-availability passes; local must still
    # be filtered on the VRAM-fit gate alone.
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=2000))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_minimal_load_mode_vetoes_local(monkeypatch, real_exe):
    """Minimal = cloud-only (mirrors selector's load_mode_minimal veto).

    Warm-resident local would normally win (8.5 > 8.0), but under Minimal a
    local image pick would shut down the loaded llama and grab ~4.5GB VRAM —
    so local must be skipped even with plenty of VRAM. load_mode rides the
    merged snapshot (no client wired here → singleton-only fallback; the
    client-path split is covered by
    test_minimal_veto_reads_client_seam_not_singleton)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="minimal"))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_minimal_load_mode_only_local_left_fails(monkeypatch, real_exe):
    """If every cloud provider has failed, Minimal yields SelectionFailure
    rather than falling back to a local image model."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="minimal"))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality",
                        failures=["huggingface/flux-schnell",
                                  "pollinations/flux"],
                        hf_available=True)
    assert isinstance(pick, SelectionFailure)
    assert pick.reason == "availability"


def test_full_load_mode_keeps_local_pickable(monkeypatch, real_exe):
    """Explicit load_mode='full' leaves local eligible (warm-resident
    local wins as before)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="full"))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_minimal_veto_reads_client_seam_not_singleton(monkeypatch, real_exe):
    """Process-split regression (sidecar): in prod ``/mode minimal`` lands on
    the SIDECAR NerdHerd via NerdHerdClient — the orchestrator-process
    singleton's LoadManager stays "full" forever. The veto must see the
    client-backed load_mode through the merged view (_effective_snapshot's
    CLIENT base), NOT the singleton seam.

    Here the singleton seam (``_snapshot``) explicitly says load_mode="full"
    with warm-resident local (which would win), while the client cache —
    reached through the REAL ``nerd_herd.snapshot()`` → ``get_default()`` →
    ``client.snapshot()`` round-trip — says "minimal". Only the outermost
    client fetch is faked (the cached snapshot, normally refreshed over HTTP);
    the veto comparison itself is NOT mocked. If the veto still read the
    singleton, local would be picked and this test would fail."""
    from nerd_herd import client as nh_client
    from nerd_herd.types import SystemSnapshot

    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="full"))
    monkeypatch.setenv("HF_TOKEN", "x")

    c = nh_client.NerdHerdClient()
    # Client base needs realistic VRAM too (it is the merged view's base):
    # the test must prove the MINIMAL VETO excludes local, not the VRAM gate.
    c._cached_snapshot = SystemSnapshot(load_mode="minimal",
                                        vram_available_mb=6000)
    monkeypatch.setattr(nh_client, "_default", c)

    # Sanity: the two surfaces genuinely disagree (the process split).
    import fatih_hoca.image_select as image_select_mod
    assert image_select_mod._snapshot().load_mode == "full"
    import nerd_herd
    assert nerd_herd.snapshot().load_mode == "minimal"

    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


# ── FIX 3.3: S13/S14 desktop-veto parity in the image path ─────────────
# select(needs_image=True) bypasses ALL selector eligibility, so the image
# path must mirror the hard-veto semantics under M4: load_mode "full"
# SILENCES desktop signals; heavy/shared honor them; minimal is already a
# blanket local veto. Eligibility-only — cloud is never affected. Warm-
# resident fixtures would otherwise WIN (8.5 > 8.0), so a cloud pick proves
# the veto (not scoring) excluded local.


def test_fullscreen_shared_vetoes_local(monkeypatch, real_exe):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="shared",
                                      fullscreen=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_fullscreen_full_mode_keeps_local(monkeypatch, real_exe):
    """M4 'full' silences desktop signals — fullscreen must NOT veto."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="full",
                                      fullscreen=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_external_gpu_heavy_vetoes_local(monkeypatch, real_exe):
    """S14 hard threshold: another process owns >=30% VRAM → veto local."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="heavy",
                                      ext_gpu=0.4))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_external_gpu_full_mode_keeps_local(monkeypatch, real_exe):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="full",
                                      ext_gpu=0.4))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_external_gpu_below_threshold_keeps_local(monkeypatch, real_exe):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True, mode="shared",
                                      ext_gpu=0.2))
    monkeypatch.setenv("HF_TOKEN", "x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_local_unavailable_filters_local(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=8000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("CLAIR_OBSCUR_EXE", raising=False)
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False
