import pytest
from paintress import generate, ImageSpec, ImageResult


class _FakeProvider:
    name = "fake"
    def available(self): return True
    async def generate(self, spec, *, base_url=None):
        return b"PNGBYTES", {"seed_used": 7}


@pytest.mark.asyncio
async def test_generate_routes_by_provider_name(monkeypatch, tmp_path):
    monkeypatch.setattr("paintress._PROVIDERS", {"fake": _FakeProvider()})
    monkeypatch.setattr("paintress.assess", lambda b: type("V", (), {"ok": True, "reason": ""})())

    class Pick:
        class model:
            provider = "fake"; endpoint = ""; api_base = None
            cost_per_image = 0.0; name = "fake/model"; is_local = False
    res = await generate(Pick(), ImageSpec(prompt="a cat", out_dir=str(tmp_path),
                                            filename_hint="cat"))
    assert isinstance(res, ImageResult) and res.error is None
    assert res.provider == "fake" and res.path.endswith(".png")
    assert res.seed_used == 7


@pytest.mark.asyncio
async def test_unknown_provider_returns_error(tmp_path):
    class Pick:
        class model:
            provider = "nope"; endpoint = ""; api_base = None
            cost_per_image = 0.0; name = "nope"; is_local = False
    res = await generate(Pick(), ImageSpec(prompt="x", out_dir=str(tmp_path)))
    assert res.error is not None and "unknown_provider" in res.error


@pytest.mark.asyncio
async def test_quality_failure_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr("paintress._PROVIDERS", {"fake": _FakeProvider()})
    monkeypatch.setattr("paintress.assess", lambda b: type("V", (), {"ok": False, "reason": "blank"})())

    class Pick:
        class model:
            provider = "fake"; endpoint = ""; api_base = None
            cost_per_image = 0.0; name = "fake/model"; is_local = False
    res = await generate(Pick(), ImageSpec(prompt="x", out_dir=str(tmp_path)))
    assert res.error is not None and "quality_failure:blank" in res.error
