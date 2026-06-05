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
