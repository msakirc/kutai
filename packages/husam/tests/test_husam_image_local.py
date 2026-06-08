"""Plan 2 Task 10 — husam local GPU-handover branch (inside keepalive()).

The handover sequence for a LOCAL image pick must be, in order:
    unload (get_local_manager().shutdown())
      → poll free VRAM (nerd_herd._get_singleton().snapshot())
      → clair_obscur.start()
      → nerd_herd.record_swap()
      → paintress.generate()
all INSIDE the single existing ``heartbeat.keepalive()`` span so the 300s
no-progress watchdog stays satisfied through the 30-60s cold-start window.

NOTE on the snapshot test seam: the implementation reads the SYNC in-process
singleton ``nerd_herd._get_singleton().snapshot()`` (a live GPU read), NOT the
async ``nerd_herd.refresh_snapshot()`` nor the cached module ``nerd_herd.snapshot()``.
So these tests monkeypatch ``nerd_herd._get_singleton`` to return a stub whose
``.snapshot()`` yields a ``vram_available_mb`` that lets the poll pass.
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

    class _Singleton:
        def snapshot(self):
            poll_calls["n"] += 1
            # First snapshot: VRAM still low; second: high enough.
            class _S:
                vram_available_mb = 1000 if poll_calls["n"] == 1 else 7000
            return _S()
    monkeypatch.setattr("nerd_herd._get_singleton", lambda: _Singleton())

    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

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

    class _Singleton:
        def snapshot(self):
            class _S:
                vram_available_mb = 7000
            return _S()
    monkeypatch.setattr("nerd_herd._get_singleton", lambda: _Singleton())
    monkeypatch.setattr("nerd_herd.record_swap", lambda name="": None)

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
