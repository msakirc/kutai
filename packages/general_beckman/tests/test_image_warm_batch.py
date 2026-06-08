"""Plan 2 Task 11 — beckman warm-batch hook drives clair_obscur's backstop.

On image-task completion, beckman peeks the next admittable candidate:
  - next is ANOTHER local image  → keep clair_obscur warm (NO hint, NO stop)
  - lane switch (next is LLM/none) → record_release_hint() (NOT a direct stop;
    the backstop in clair_obscur.server fires the actual stop after idle)
  - the just-finished task was a CLOUD image (preselected_pick_provider !=
    "clair_obscur") → never touch clair_obscur at all.

The hook calls ``clair_obscur.record_release_hint()`` (module-level, sync) and
must never crash beckman if clair_obscur is absent (guarded import).
"""
import pytest


@pytest.mark.asyncio
async def test_local_image_followed_by_image_no_hint(monkeypatch):
    """Two local image tasks back-to-back → NO release hint → clair_obscur
    stays warm. The next image dispatch will idempotently clear any pending
    hint (in husam `_run_image`, Task 10)."""
    import general_beckman as gb

    hints = {"n": 0}
    stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    async def _peek():
        return {"id": 2, "kind": "image", "agent_type": "image",
                "context": '{"image_call": {"raw_dispatch": true}}'}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert hints["n"] == 0, "warm-batch must NOT hint release"
    assert stops["n"] == 0, "warm-batch must NOT direct-stop"


@pytest.mark.asyncio
async def test_local_image_followed_by_llm_hints_release(monkeypatch):
    """Lane switch → record_release_hint (NOT direct stop). Backstop times
    the actual stop after idle_release_seconds."""
    import general_beckman as gb
    hints = {"n": 0}
    stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    async def _peek():
        return {"id": 3, "kind": "llm", "agent_type": "coder",
                "context": "{}"}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert hints["n"] == 1
    assert stops["n"] == 0, "lane switch hints; backstop fires the stop"


@pytest.mark.asyncio
async def test_local_image_followed_by_empty_queue_hints_release(monkeypatch):
    """No next admittable task → lane is idle → hint release (backstop stops)."""
    import general_beckman as gb
    hints = {"n": 0}
    stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    async def _peek():
        return None
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert hints["n"] == 1
    assert stops["n"] == 0


@pytest.mark.asyncio
async def test_cloud_image_never_touches_clair_obscur(monkeypatch):
    """A cloud-image task must not call record_release_hint or stop."""
    import general_beckman as gb
    hints = {"n": 0}; stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    await gb._post_completion_image_lane({
        "id": 4, "kind": "image", "agent_type": "image",
        "context": "{}", "preselected_pick_provider": "pollinations",
    }, {"status": "completed"})
    assert hints["n"] == 0 and stops["n"] == 0


@pytest.mark.asyncio
async def test_non_image_task_is_noop(monkeypatch):
    """A non-image task completing never touches clair_obscur."""
    import general_beckman as gb
    hints = {"n": 0}; stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    await gb._post_completion_image_lane({
        "id": 5, "kind": "main_work", "agent_type": "coder", "context": "{}",
    }, {"status": "completed"})
    assert hints["n"] == 0 and stops["n"] == 0


@pytest.mark.asyncio
async def test_clair_obscur_absent_does_not_crash(monkeypatch):
    """If record_release_hint raises (clair_obscur absent/misconfigured), the
    hook must swallow the error — never crash the on_task_finished hot path."""
    import general_beckman as gb

    def _boom():
        raise RuntimeError("clair_obscur gone")
    monkeypatch.setattr("clair_obscur.record_release_hint", _boom)

    async def _peek():
        return {"id": 3, "kind": "llm", "agent_type": "coder", "context": "{}"}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    # Must not raise.
    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
