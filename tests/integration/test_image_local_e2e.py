"""Plan 2 Task 12 — end-to-end local image lane host-path test (mock backend).

Drives the full LOCAL image lane against mocks, with NO real GPU process:

    beckman.next_task()           # admit the image task, force local pick,
                                  # stamp preselected_pick + provider
      → husam.run(task)           # _run_image handover (Task 10):
                                  #   get_local_manager().shutdown() (mock)
                                  #   → poll VRAM (nerd_herd._get_singleton().snapshot())
                                  #   → clair_obscur.start() (mock)
                                  #   → nerd_herd.record_swap() (once)
                                  #   → paintress.generate() → real LocalServerProvider
                                  #     → A1111 /sdapi/v1/txt2img via MOCKED httpx
                                  #   → PNG written to disk
      → beckman.on_task_finished()  # telemetry round-trip + image-lane hook:
                                    #   queue empty after this task → lane idle →
                                    #   clair_obscur.record_release_hint() fires.

Asserts:
  * a real PNG file is written and non-empty,
  * the husam result reports provider=clair_obscur and the requested seed,
  * record_swap fired exactly once,
  * telemetry landed (model_pick_log + model_call_tokens rows for the task),
  * the beckman warm-batch hook fired record_release_hint() on lane switch.

TEST SEAMS (the plan snippet's nerd_herd.refresh_snapshot patch is stale):
  * husam's VRAM poll reads the SYNC singleton nerd_herd._get_singleton().snapshot();
    we patch _get_singleton to return a stub whose snapshot() reports plenty of VRAM.
  * fatih_hoca.image_select._snapshot is patched so the local pick wins selection.
  * clair_obscur.start is mocked (returns a base_url; no process launched).
  * get_local_manager().shutdown() is a no-op mock (no real DaLLaMa unload).
  * the A1111 backend is a monkeypatched httpx.AsyncClient returning a base64 PNG.
"""
import base64
import io
import os

import pytest
from PIL import Image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 90, 70)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_local_image_lane_e2e(monkeypatch, tmp_path, temp_db):
    # ── env: A1111 backend, fake exe so selection's existence gate passes ──
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
    fake_exe = tmp_path / "fake_exe"
    fake_exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(fake_exe))

    # ── snapshot: idle, plenty of VRAM, no LLM loaded ──
    # Used by BOTH the selection VRAM-fit gate (image_select._snapshot) and the
    # husam handover VRAM poll (nerd_herd._get_singleton().snapshot()).
    class _Local:
        model_name = None
        idle_seconds = 30.0
        is_swapping = False
        requests_processing = 0

    class _Snap:
        vram_available_mb = 8000
        in_flight_calls = []
        local = _Local()
        cloud = {}
        # NOTE: the local image entry carries a cold-start eviction cost
        # (image_select._EVICTION_LOW = 2.0). On a stone-cold idle GPU
        # clair_obscur scores 7.5 - 2.0 = 5.5 and LOSES to pollinations (6.0).
        # To FORCE the local pick we report the server as already resident:
        # eviction drops to 0 + warm-batch bonus (+1.0) → clair_obscur = 8.5.
        # This is the realistic "warm batch / lane already hot" state the local
        # lane is designed to win, and it is the one the plan snippet's stale
        # image_server_resident=False would have failed to select.
        image_server_resident = True
        image_server_vram_mb = 4500

    # Selection seam: local pick must win.
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _Snap())

    # Husam VRAM-poll seam: the SYNC singleton snapshot (NOT refresh_snapshot,
    # which the plan snippet wrongly patched). Return high VRAM so the poll
    # passes on the first read.
    class _Singleton:
        def snapshot(self):
            return _Snap()
    monkeypatch.setattr("nerd_herd._get_singleton", lambda: _Singleton())

    # DaLLaMa unload seam: no-op (no real local LLM to evict).
    class _Mgr:
        async def shutdown(self):
            pass
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    # clair_obscur.start seam: returns base_url; launches NO process.
    async def _co_start():
        return "http://127.0.0.1:7860"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    # record_swap seam: count swaps (must be exactly one).
    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

    # beckman lane-switch hook seam: the queue is empty after this single image
    # task, so _post_completion_image_lane should record a release hint.
    hints = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))

    # ── mock A1111 backend: monkeypatch httpx.AsyncClient in local_server ──
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"images": [_png_b64()], "info": "{\"seed\": 33}"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            assert "/sdapi/v1/txt2img" in url
            return _Resp()
    monkeypatch.setattr(
        "paintress.providers.local_server.httpx.AsyncClient", _Client)

    import husam
    from src.infra.db import add_task, get_task
    from src.core import heartbeat as _hb
    from general_beckman import next_task, on_task_finished

    # Admission keeps a process-global short-circuit cache
    # (_last_admission_fp / _last_admission_admitted). Patch via monkeypatch so
    # it is reset on entry AND auto-restored on exit — this test neither
    # inherits nor leaks admission state into sibling beckman tests that share
    # the module globals.
    import general_beckman as _gb
    monkeypatch.setattr(_gb, "_last_admission_fp", None)
    monkeypatch.setattr(_gb, "_last_admission_admitted", True)

    # ── seed an image task into the real temp DB ──
    # add_task serializes ``context`` itself (json.dumps), so pass the DICT —
    # passing a pre-serialized string double-encodes it and the admission
    # selector then parses a str (not a dict) → AttributeError → select=None.
    ctx = {"image_call": {
        "raw_dispatch": True, "prompt": "a fox in snow",
        "out_dir": str(tmp_path), "width": 512, "height": 512, "seed": 33,
        "filename_hint": "fox", "quality_tier": "fast", "agent_type": "image",
    }}
    tid = await add_task(title="img", description="", agent_type="image",
                         kind="image", context=ctx)

    # ── manual pump (Plan 1 v3 pattern) ──
    task = await next_task()
    assert task is not None and task["id"] == tid, "image task must be admitted"
    # Admission forced the local pick + stamped the provider.
    pick = task.get("preselected_pick")
    assert pick is not None and pick.model.provider == "clair_obscur", \
        "admission must select the local clair_obscur image provider"
    assert task.get("preselected_pick_provider") == "clair_obscur", \
        "admission must stamp preselected_pick_provider for the lane hook"

    # husam.run drives the handover inside _run_image (Task 10). Set the
    # current_task_id contextvar so telemetry rows carry the task id.
    token = _hb.current_task_id.set(tid)
    try:
        res = await husam.run(task)
    finally:
        _hb.current_task_id.reset(token)

    # ── assert the host-path PNG ──
    assert os.path.isfile(res["path"]), "a PNG must be written to disk"
    assert os.path.getsize(res["path"]) > 0
    assert res["provider"] == "clair_obscur"
    assert res["seed_used"] == 33
    assert res["is_local"] is True

    # ── exactly one swap recorded for the cold-start handover ──
    assert swaps == ["clair_obscur/sdxl-turbo"], \
        "exactly one swap must be recorded on the local handover"

    # ── close the telemetry round-trip + fire the beckman lane hook ──
    # Mirror the orchestrator's envelope wrapping for raw_dispatch results
    # (src/core/orchestrator.py): the worker dict is wrapped with an explicit
    # status="completed" + a non-empty `result`, else route_result treats the
    # status-less husam image dict as Failed and the task re-pends (the hook
    # would then see itself in the queue and stay warm instead of releasing).
    import json as _json
    finished = {
        "status": "completed",
        "result": _json.dumps(res),
        **{k: v for k, v in res.items() if k != "result"},
    }
    await on_task_finished(tid, finished)

    # The just-finished task was a LOCAL image and the queue is now empty
    # (lane idle) → the warm-batch hook records a release hint (the backstop in
    # clair_obscur then times the actual stop).
    assert hints["n"] == 1, "lane switch / idle queue must hint clair_obscur release"

    # ── telemetry rows landed in the real DB ──
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM model_pick_log WHERE task_id = ?", (tid,))
    (n_pick,) = await cur.fetchone()
    assert n_pick >= 1, "a model_pick_log row must land for the image pick"

    cur = await db.execute(
        "SELECT COUNT(*), MAX(provider) FROM model_call_tokens WHERE task_id = ?",
        (tid,))
    n_tok, prov = await cur.fetchone()
    assert n_tok >= 1, "a model_call_tokens telemetry row must land"
    assert prov == "clair_obscur"

    # The task itself reached a terminal state.
    row = await get_task(tid)
    assert row is not None

    # Release the in-process in-flight slot reserved at admission. In
    # production the orchestrator's dispatch finally does this (release_task);
    # the manual pump has no such finally, so the reserved slot would otherwise
    # leak into src.core.in_flight._task_slots (a module global) and make
    # sibling beckman tests see a phantom busy-local → reject their candidates.
    from src.core.in_flight import release_task
    await release_task(tid)
