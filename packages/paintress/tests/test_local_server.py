import base64
import io

import pytest
from PIL import Image

from paintress.providers import local_server as ls_mod
from paintress.providers.local_server import LocalServerProvider
from paintress.types import ImageSpec


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (50, 100, 150)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_a1111_post_returns_bytes(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")

    class _Resp:
        status_code = 200
        def json(self): return {"images": [_png_b64()], "info": "{\"seed\": 99}"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            assert "/sdapi/v1/txt2img" in url
            return _Resp()

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)

    prov = LocalServerProvider()
    data, meta = await prov.generate(
        ImageSpec(prompt="a cat", out_dir="/tmp", seed=99, width=512, height=512),
        base_url="http://127.0.0.1:7860",
    )
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 99


@pytest.mark.asyncio
async def test_a1111_accepts_data_uri_prefix(monkeypatch):
    """A1111 response with data:image/...;base64, prefix must decode correctly."""
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    raw_b64 = _png_b64()
    data_uri = f"data:image/png;base64,{raw_b64}"

    class _Resp:
        status_code = 200
        def json(self): return {"images": [data_uri], "info": "{}"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw): return _Resp()

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    data, _ = await LocalServerProvider().generate(
        ImageSpec(prompt="x", out_dir="/tmp", seed=1, width=32, height=32),
        base_url="http://127.0.0.1:7860",
    )
    assert data.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_comfyui_prompt_then_history_poll(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "comfyui")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "8188")

    state = {"posted": False, "history_calls": 0, "viewed": False}

    class _PromptResp:
        status_code = 200
        def json(self): return {"prompt_id": "abc-123"}
        def raise_for_status(self): pass
    class _HistEmpty:
        status_code = 200
        def json(self): return {}
        def raise_for_status(self): pass
    class _HistDone:
        status_code = 200
        def json(self):
            return {"abc-123": {"outputs": {"9": {"images": [{
                "filename": "out.png", "subfolder": "", "type": "output",
            }]}}}}
        def raise_for_status(self): pass

    _VIEW_BODY = b"\x89PNG\r\n\x1a\nFAKE"

    class _ViewResp:
        status_code = 200
        def raise_for_status(self): pass
        async def aiter_bytes(self, chunk_size=None):
            yield _VIEW_BODY

    class _ViewCtx:
        async def __aenter__(self):
            state["viewed"] = True
            return _ViewResp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            state["posted"] = True
            return _PromptResp()
        async def get(self, url, **kw):
            if "/history/" in url:
                state["history_calls"] += 1
                return _HistEmpty() if state["history_calls"] < 2 else _HistDone()
            raise AssertionError(url)
        def stream(self, method, url, **kw):
            if "/view" in url:
                return _ViewCtx()
            raise AssertionError(url)

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    monkeypatch.setattr(ls_mod, "_PROMPT_POLL_INTERVAL", 0.01)

    data, meta = await LocalServerProvider().generate(
        ImageSpec(prompt="a dog", out_dir="/tmp", seed=7, width=512, height=512),
        base_url="http://127.0.0.1:8188",
    )
    assert state["posted"] and state["viewed"]
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 7


@pytest.mark.asyncio
async def test_comfyui_view_raises_when_response_exceeds_cap(monkeypatch):
    """ComfyUI /view stream that exceeds _MAX_BYTES → RuntimeError('response_too_large')."""
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "comfyui")
    monkeypatch.setattr(ls_mod, "_MAX_BYTES", 10)
    monkeypatch.setattr(ls_mod, "_PROMPT_POLL_INTERVAL", 0.01)

    class _PromptResp:
        status_code = 200
        def json(self): return {"prompt_id": "cap-test"}
        def raise_for_status(self): pass

    class _HistDone:
        status_code = 200
        def json(self):
            return {"cap-test": {"outputs": {"9": {"images": [{
                "filename": "big.png", "subfolder": "", "type": "output",
            }]}}}}
        def raise_for_status(self): pass

    class _BigViewResp:
        status_code = 200
        def raise_for_status(self): pass
        async def aiter_bytes(self, chunk_size=None):
            yield b"X" * 20  # 20 > 10

    class _BigViewCtx:
        async def __aenter__(self): return _BigViewResp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw): return _PromptResp()
        async def get(self, url, **kw): return _HistDone()
        def stream(self, method, url, **kw): return _BigViewCtx()

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError, match="response_too_large"):
        await LocalServerProvider().generate(
            ImageSpec(prompt="x", out_dir="/tmp", seed=1),
            base_url="http://127.0.0.1:8188",
        )


def test_available_reflects_exe_present(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(tmp_path / "no_such_exe"))
    assert LocalServerProvider().available() is False
    exe = tmp_path / "exe"; exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(exe))
    assert LocalServerProvider().available() is True
