import base64
import io

import pytest
from PIL import Image

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
    class _Bytes:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nFAKE"
        def raise_for_status(self): pass

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
            if "/view" in url:
                state["viewed"] = True; return _Bytes()
            raise AssertionError(url)

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    import paintress.providers.local_server as ls
    monkeypatch.setattr(ls, "_PROMPT_POLL_INTERVAL", 0.01)

    data, meta = await LocalServerProvider().generate(
        ImageSpec(prompt="a dog", out_dir="/tmp", seed=7, width=512, height=512),
        base_url="http://127.0.0.1:8188",
    )
    assert state["posted"] and state["viewed"]
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 7


def test_available_reflects_exe_present(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(tmp_path / "no_such_exe"))
    assert LocalServerProvider().available() is False
    exe = tmp_path / "exe"; exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(exe))
    assert LocalServerProvider().available() is True
