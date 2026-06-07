import os
import pytest
from paintress import generate, ImageSpec, ImageResult
from paintress import _safe_filename


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


# --- FIX 6b: out_dir confinement tests ---

def test_safe_filename_strips_separators():
    """_safe_filename must never produce a path component with / or \\ or .."""
    name = _safe_filename("../../etc/passwd")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name.endswith(".png")


@pytest.mark.asyncio
async def test_generated_file_stays_inside_out_dir(monkeypatch, tmp_path):
    """Even with a tricky filename_hint, the written file must live inside out_dir."""
    monkeypatch.setattr("paintress._PROVIDERS", {"fake": _FakeProvider()})
    monkeypatch.setattr("paintress.assess", lambda b: type("V", (), {"ok": True, "reason": ""})())

    class Pick:
        class model:
            provider = "fake"; endpoint = ""; api_base = None
            cost_per_image = 0.0; name = "fake/model"; is_local = False

    res = await generate(
        Pick(),
        ImageSpec(prompt="x", out_dir=str(tmp_path), filename_hint="../../evil"),
    )
    assert res.error is None
    assert res.path is not None
    real_out = os.path.realpath(str(tmp_path))
    real_file = os.path.realpath(res.path)
    assert real_file.startswith(real_out + os.sep) or real_file == real_out


@pytest.mark.asyncio
async def test_path_escape_returns_error(monkeypatch, tmp_path):
    """If the confinement guard fires, the result carries error='path_escape'."""
    import paintress as _paintress

    monkeypatch.setattr("paintress._PROVIDERS", {"fake": _FakeProvider()})
    monkeypatch.setattr("paintress.assess", lambda b: type("V", (), {"ok": True, "reason": ""})())

    # Force _safe_filename to return an absolute path outside out_dir
    monkeypatch.setattr(_paintress, "_safe_filename",
                        lambda hint: "/tmp/evil_escape.png")

    class Pick:
        class model:
            provider = "fake"; endpoint = ""; api_base = None
            cost_per_image = 0.0; name = "fake/model"; is_local = False

    res = await generate(
        Pick(),
        ImageSpec(prompt="x", out_dir=str(tmp_path), filename_hint="whatever"),
    )
    assert res.error == "path_escape"
