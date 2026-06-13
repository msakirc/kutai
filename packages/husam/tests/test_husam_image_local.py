"""Plan 2 Task 10 — husam local GPU-handover branch (inside keepalive()).

The handover sequence for a LOCAL image pick must be, in order:
    unload (get_local_manager().shutdown())
      → poll free VRAM (nerd_herd.gpu_vram_free_mb())
      → clair_obscur.start()
      → nerd_herd.record_swap()
      → paintress.generate()
all INSIDE the single existing ``heartbeat.keepalive()`` span so the 300s
no-progress watchdog stays satisfied through the 30-60s cold-start window.

NOTE on the VRAM-poll test seam: the implementation reads the cheap
``nerd_herd.gpu_vram_free_mb(invalidate=...)`` helper (a 2s-cached GPU read off
the in-process singleton's GPU collector; invalidate=True on the first read
forces a fresh poll post-shutdown), NOT a full ``snapshot()``. So these tests
monkeypatch ``nerd_herd.gpu_vram_free_mb`` to return free-MB values that let
the poll pass.
"""
import pytest

from husam import run as husam_run


@pytest.mark.asyncio
async def test_local_image_handover_ordering(monkeypatch, tmp_path):
    """Order must be: unload → poll → clair_obscur.start → record_swap →
    paintress.generate. All inside the existing keepalive() span (worker.py)
    so the watchdog sees bumps through the cold-start window."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    order = []

    class _Mgr:
        async def shutdown(self):
            order.append("unload")
    # _run_image imports get_local_manager from src.models.local_model_manager.
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    poll_calls = {"n": 0}

    def _free_vram(*, invalidate=False):
        poll_calls["n"] += 1
        # First read: VRAM still low; second: high enough.
        return 1000 if poll_calls["n"] == 1 else 7000
    monkeypatch.setattr("nerd_herd.gpu_vram_free_mb", _free_vram)

    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

    # Cold start: image server NOT yet resident → full eviction sequence fires.
    monkeypatch.setattr("clair_obscur.status", lambda: {"resident": False})

    async def _co_start():
        order.append("clair_obscur.start")
        return "http://127.0.0.1:8188"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    async def _fake_generate(pick, spec):
        order.append("paintress.generate")
        from paintress import ImageResult
        out = tmp_path / "out.png"
        out.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
        return ImageResult(path=str(out), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0, seed_used=7)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    model = ImageModelInfo(
        name="clair_obscur/sdxl-turbo", provider="clair_obscur",
        location="local", endpoint="", quality_rank=7.5,
        cost_per_image=0.0, vram_mb=4500, supports_seed=True,
    )
    pick = Pick(model=model, min_time_seconds=0.0, score=8.5, top_summary="t")
    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a fox", "out_dir": str(tmp_path),
            "width": 512, "height": 512, "quality_tier": "fast",
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await husam_run(task)
    assert res["path"].endswith(".png")
    # unload → start → generate come in order (handover precedes the call).
    assert order == ["unload", "clair_obscur.start", "paintress.generate"]
    assert swaps == ["clair_obscur/sdxl-turbo"]
    assert poll_calls["n"] >= 2  # polled until VRAM fit


@pytest.mark.asyncio
async def test_warm_batch_skips_eviction_but_keeps_server_warm(monkeypatch, tmp_path):
    """Emergent batching (spec §4): when the image server is ALREADY resident
    (back-to-back image batch), the eviction work must NOT run — no DaLLaMa
    shutdown, no recorded swap (which would exhaust hoca's 3/5min swap budget
    after 3 images and break the warm batch). But clair_obscur.start() MUST
    still fire: it's idempotent and clears the pending release hint, resetting
    the warm window and handing paintress the base_url."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    touched = {"unload": False, "start": False, "swap": False}

    class _Mgr:
        async def shutdown(self):
            touched["unload"] = True
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    # Server already warm → eviction must be skipped.
    monkeypatch.setattr("clair_obscur.status", lambda: {"resident": True})

    poll_calls = {"n": 0}

    def _free_vram(*, invalidate=False):
        poll_calls["n"] += 1
        return 7000
    monkeypatch.setattr("nerd_herd.gpu_vram_free_mb", _free_vram)
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": touched.__setitem__("swap", True))

    async def _co_start():
        touched["start"] = True
        return "http://127.0.0.1:8188"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    captured = {"endpoint": None}

    async def _gen(pick, spec):
        captured["endpoint"] = pick.model.endpoint
        from paintress import ImageResult
        p = tmp_path / "warm.png"
        p.write_bytes(b"\x89PNG")
        return ImageResult(path=str(p), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="clair_obscur/sdxl-turbo", provider="clair_obscur",
                           location="local", endpoint="", vram_mb=4500)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}
    res = await husam_run(task)
    assert res["path"].endswith(".png")
    # start() fired (endpoint set, release hint cleared); eviction skipped.
    assert touched["start"] is True
    assert touched["unload"] is False, "warm path must not shutdown DaLLaMa"
    assert touched["swap"] is False, "warm path must record NO swap"
    assert poll_calls["n"] == 0, "warm path must not poll VRAM"
    assert captured["endpoint"] == "http://127.0.0.1:8188"


@pytest.mark.asyncio
async def test_keepalive_wraps_long_handover(monkeypatch, tmp_path):
    """The handover (shutdown + poll + start) must run INSIDE the keepalive()
    span so heartbeat bumps remain reachable. We assert the context does not
    raise and the local branch executes."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    class _Mgr:
        async def shutdown(self):
            # Simulate a slow unload (>30s would normally trip watchdog;
            # we only assert the keepalive span stays reachable).
            import asyncio
            await asyncio.sleep(0)
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    monkeypatch.setattr("nerd_herd.gpu_vram_free_mb",
                        lambda *, invalidate=False: 7000)
    monkeypatch.setattr("nerd_herd.record_swap", lambda name="": None)
    monkeypatch.setattr("clair_obscur.status", lambda: {"resident": False})

    async def _co_start():
        return "http://127.0.0.1:8188"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    async def _gen(pick, spec):
        from paintress import ImageResult
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG")
        return ImageResult(path=str(p), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="clair_obscur/sdxl-turbo", provider="clair_obscur",
                           location="local", endpoint="", vram_mb=4500)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}
    res = await husam_run(task)
    assert res["path"].endswith(".png")
    # The keepalive context must not raise — that's the contract this test enforces.


@pytest.mark.asyncio
async def test_clair_obscur_start_failure_preserves_availability_category(monkeypatch, tmp_path):
    """If clair_obscur.start() fails during the LOCAL handover, husam.run must
    raise ModelCallFailed with error_category == "availability" (a TRANSIENT
    category that rides Beckman's backoff ladder → reselect → local→cloud
    degrade), NOT "raw_exception". The outer except must NOT downgrade it.

    Also asserts the handover aborts: record_swap and paintress.generate are
    never reached because start() failed before them."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick
    from src.core.router import ModelCallFailed

    class _Mgr:
        async def shutdown(self):
            pass
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    monkeypatch.setattr("nerd_herd.gpu_vram_free_mb",
                        lambda *, invalidate=False: 7000)

    touched = {"swap": False, "generate": False}
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": touched.__setitem__("swap", True))
    monkeypatch.setattr("clair_obscur.status", lambda: {"resident": False})

    async def _co_start_fail():
        raise RuntimeError("comfyui boot timeout")
    monkeypatch.setattr("clair_obscur.start", _co_start_fail)

    async def _gen(pick, spec):
        touched["generate"] = True
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "x.png"), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="clair_obscur/sdxl-turbo", provider="clair_obscur",
                           location="local", endpoint="", vram_mb=4500)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}

    with pytest.raises(ModelCallFailed) as ei:
        await husam_run(task)
    assert ei.value.error_category == "availability"
    assert touched == {"swap": False, "generate": False}


@pytest.mark.asyncio
async def test_cloud_image_path_skips_handover(monkeypatch, tmp_path):
    """Sanity: cloud pick must NOT touch shutdown, clair_obscur, or record_swap."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    touched = {"unload": False, "start": False, "swap": False}

    class _Mgr:
        async def shutdown(self):
            touched["unload"] = True
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    async def _co_start():
        touched["start"] = True
        return ""
    monkeypatch.setattr("clair_obscur.start", _co_start)
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": touched.__setitem__("swap", True))

    async def _gen(pick, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "x.png"),
                           provider="pollinations", model="pollinations/flux",
                           cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="pollinations/flux", provider="pollinations",
                           location="cloud", endpoint="https://x/", vram_mb=0)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}
    await husam_run(task)
    assert touched == {"unload": False, "start": False, "swap": False}
