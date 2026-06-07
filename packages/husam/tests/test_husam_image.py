import pytest


def _img_pick(tmp_path):
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick
    model = ImageModelInfo(
        name="pollinations/flux", provider="pollinations", location="cloud",
        endpoint="https://x/", cost_per_image=0.0,
    )
    return Pick(model=model, min_time_seconds=0.0, score=6.0, top_summary="t")


@pytest.mark.asyncio
async def test_husam_image_routes_to_paintress_and_writes_telemetry(monkeypatch, tmp_path):
    import husam
    from src.core.llm_dispatcher import get_dispatcher, CallCategory

    pick = _img_pick(tmp_path)

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"), provider="pollinations",
                           model="pollinations/flux", cost=0.0, seed_used=5, latency=0.1)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    tele = {"begin": 0, "end": 0, "pick_log": 0, "tokens": 0, "cost": 0}

    async def _bc(**kw):
        tele["begin"] += 1
        assert kw["category"] == "image"
        assert kw["model_name"] == "pollinations/flux"
        return "call-1"
    async def _ec(call_id): tele["end"] += 1
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)

    async def _rct(**kw):
        tele["tokens"] += 1
        assert kw["model"] == "pollinations/flux"
        assert kw["call_category"] == "image"
        assert kw["prompt_tokens"] == 0 and kw["completion_tokens"] == 0
    async def _rcc(task_id, cost_usd): tele["cost"] += 1
    monkeypatch.setattr("src.infra.db.record_call_tokens", _rct)
    monkeypatch.setattr("src.infra.db.record_call_cost", _rcc)

    async def _rp(**kw):
        tele["pick_log"] += 1
        assert kw["success"] is True
        assert getattr(kw["pick"].model, "name", "") == "pollinations/flux"
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a cat", "out_dir": str(tmp_path),
            "width": 512, "height": 512,
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await husam.run(task)
    assert res["path"].endswith("out.png")
    assert res["provider"] == "pollinations"
    assert CallCategory.IMAGE.value == "image"
    assert tele["begin"] == 1 and tele["end"] == 1
    assert tele["pick_log"] == 1 and tele["tokens"] == 1
    assert tele["cost"] == 0  # free provider (cost 0.0) → cost row guarded out
    # FIX 7: result must carry BOTH "cost" (legacy readers) and "cost_usd"
    # (on_task_finished reads result.get("cost_usd") for mission spend tracking).
    assert res["cost"] == 0.0
    assert res["cost_usd"] == 0.0
    assert res["cost_usd"] == res["cost"]


@pytest.mark.asyncio
async def test_husam_image_result_carries_cost_usd_for_paid_provider(monkeypatch, tmp_path):
    """FIX 7: the cost_usd key must equal the provider's cost so a future PAID
    image provider accrues spend to the mission via on_task_finished."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher

    pick = _img_pick(tmp_path)

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"), provider="pollinations",
                           model="pollinations/flux", cost=0.042, seed_used=5, latency=0.1)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    async def _bc(**kw): return "c"
    async def _ec(call_id): pass
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    async def _rct(**kw): pass
    async def _rcc(task_id, cost_usd): pass
    monkeypatch.setattr("src.infra.db.record_call_tokens", _rct)
    monkeypatch.setattr("src.infra.db.record_call_cost", _rcc)
    async def _rp(**kw): pass
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a cat", "out_dir": str(tmp_path),
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await husam.run(task)
    assert res["cost"] == 0.042
    assert res["cost_usd"] == 0.042


@pytest.mark.asyncio
async def test_husam_image_unknown_provider_uses_recognized_category(monkeypatch, tmp_path):
    """FIX 5: an unknown_provider ImageResult error must raise ModelCallFailed
    with a beckman-RECOGNIZED category, NOT the unregistered "fatal" string
    (which decide_retry/TRANSIENT_CATEGORIES don't know → wasteful backoff-retry
    of a permanent misconfig). We use "availability" (a recognized category)."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher
    from src.core.router import ModelCallFailed
    from general_beckman.retry import TRANSIENT_CATEGORIES

    pick = _img_pick(tmp_path)

    async def _unknown_provider_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(provider="pollinations", model="pollinations/flux",
                           error="unknown_provider: no such provider 'xyz'")
    monkeypatch.setattr("paintress.generate", _unknown_provider_generate)

    async def _bc(**kw): return "c"
    async def _ec(call_id): pass
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    monkeypatch.setattr("src.infra.db.record_call_tokens", lambda **kw: None)
    monkeypatch.setattr("src.infra.db.record_call_cost", lambda *a, **kw: None)
    async def _rp(**kw): pass
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}

    with pytest.raises(ModelCallFailed) as ei:
        await husam.run(task)
    cat = ei.value.error_category
    assert cat != "fatal"
    # Must be a category beckman's retry policy actually recognizes.
    assert cat in (TRANSIENT_CATEGORIES | {"quality", "budget"})
    assert cat == "availability"


@pytest.mark.asyncio
async def test_husam_image_provider_quality_failure_stays_availability(monkeypatch, tmp_path):
    """FIX 5 guard: the NON-unknown_provider failure path (e.g. quality_failure)
    keeps raising "availability" — unchanged behavior."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher
    from src.core.router import ModelCallFailed

    pick = _img_pick(tmp_path)

    async def _quality_fail(p, spec):
        from paintress import ImageResult
        return ImageResult(provider="pollinations", model="pollinations/flux",
                           error="quality_failure:blank")
    monkeypatch.setattr("paintress.generate", _quality_fail)

    async def _bc(**kw): return "c"
    async def _ec(call_id): pass
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    monkeypatch.setattr("src.infra.db.record_call_tokens", lambda **kw: None)
    monkeypatch.setattr("src.infra.db.record_call_cost", lambda *a, **kw: None)
    async def _rp(**kw): pass
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}

    with pytest.raises(ModelCallFailed) as ei:
        await husam.run(task)
    assert ei.value.error_category == "availability"


@pytest.mark.asyncio
async def test_husam_image_reselect_without_preselected_pick(monkeypatch, tmp_path):
    """FIX 8a: when no preselected_pick is attached (and not an ImageModelInfo),
    husam must call fatih_hoca.select(needs_image=True, ...) itself. Exercises
    the REAL selector path inside husam._run_image."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        # echo the selected provider so we prove the real scorer ran.
        prov = getattr(p.model, "provider", "")
        return ImageResult(path=str(tmp_path / "out.png"), provider=prov,
                           model=getattr(p.model, "name", ""), cost=0.0,
                           seed_used=5, latency=0.1)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    async def _bc(**kw): return "c"
    async def _ec(call_id): pass
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    monkeypatch.setattr("src.infra.db.record_call_tokens", lambda **kw: None)
    monkeypatch.setattr("src.infra.db.record_call_cost", lambda *a, **kw: None)
    async def _rp(**kw): pass
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    # No HF_TOKEN → scorer must pick the free pollinations provider.
    monkeypatch.delenv("HF_TOKEN", raising=False)

    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a lake", "out_dir": str(tmp_path),
            "quality_tier": "fast",
        }},
        "kind": "image",
        # NO preselected_pick — forces the real fatih_hoca.select(needs_image=True).
    }
    res = await husam.run(task)
    assert res["provider"] == "pollinations"
    assert res["model"] == "pollinations/flux"


@pytest.mark.asyncio
async def test_husam_image_cost_row_skipped_for_free_provider(monkeypatch, tmp_path):
    """FIX 8b: with a LIVE task_id (contextvar set) and cost 0.0, record_call_cost
    must NOT fire — proving the cost>0.0 guard (not merely the task_id-None guard
    that the original assertion accidentally tested)."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher
    from src.core.heartbeat import current_task_id

    pick = _img_pick(tmp_path)

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"), provider="pollinations",
                           model="pollinations/flux", cost=0.0, seed_used=5, latency=0.1)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    cost_calls = {"n": 0}
    async def _bc(**kw): return "c"
    async def _ec(call_id): pass
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    async def _rct(**kw): pass
    async def _rcc(task_id, cost_usd): cost_calls["n"] += 1
    monkeypatch.setattr("src.infra.db.record_call_tokens", _rct)
    monkeypatch.setattr("src.infra.db.record_call_cost", _rcc)
    async def _rp(**kw): pass
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a cat", "out_dir": str(tmp_path),
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    tok = current_task_id.set(42)
    try:
        res = await husam.run(task)
    finally:
        current_task_id.reset(tok)
    assert res["provider"] == "pollinations"
    # task_id was LIVE (42) but cost was 0.0 → cost row guarded out by cost>0.0.
    assert cost_calls["n"] == 0


@pytest.mark.asyncio
async def test_husam_image_failure_writes_pick_log_and_ends_call(monkeypatch, tmp_path):
    import husam
    from src.core.llm_dispatcher import get_dispatcher

    pick = _img_pick(tmp_path)

    async def _fail_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(provider="pollinations", model="pollinations/flux",
                           error="quality_failure:blank")
    monkeypatch.setattr("paintress.generate", _fail_generate)

    counts = {"end": 0, "pick_log_fail": 0}
    async def _bc(**kw): return "c"
    async def _ec(call_id): counts["end"] += 1
    monkeypatch.setattr("src.core.in_flight.begin_call", _bc)
    monkeypatch.setattr("src.core.in_flight.end_call", _ec)
    monkeypatch.setattr("src.infra.db.record_call_tokens", lambda **kw: None)
    monkeypatch.setattr("src.infra.db.record_call_cost", lambda *a, **kw: None)

    async def _rp(**kw):
        if not kw["success"]:
            counts["pick_log_fail"] += 1
    monkeypatch.setattr(get_dispatcher(), "_record_pick", _rp)

    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}

    from src.core.router import ModelCallFailed
    with pytest.raises(ModelCallFailed):
        await husam.run(task)
    assert counts == {"end": 1, "pick_log_fail": 1}
