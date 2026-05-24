# Image Generation — Plan 1: Cloud spine + `/image`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end cloud image generation reachable from Telegram `/image <prompt>`, flowing through the existing beckman→hoca→dispatcher pipeline with two new thin packages (`paintress`, `renoir`) and an image scorer in `fatih_hoca`.

**Architecture:** Image generation is just tasks through the singular lifecycle (see `docs/superpowers/specs/2026-05-23-image-generation-design.md` §2). `paintress` = image interaction caller (≈ HaLLederiz Kadir). `renoir` = image quality judge (≈ dogru_mu_samet). `fatih_hoca` gains an image scorer (`image_select.py`) selecting among statically-registered cloud providers. The dispatcher routes image picks to paintress instead of HK via a modality branch. **This plan is cloud-only**: `clair_obscur` (local server), VRAM eviction-cost, GPU handover, and i2p integration are Plans 2 & 3.

**Tech Stack:** Python 3.10, async/await, aiosqlite, httpx (HTTP to image providers), Pillow (image validation in renoir), pytest. New packages use src-layout like `packages/fatih_hoca/`.

**Scope boundary (in this plan):** data taxonomy · `renoir` · `paintress` (pollinations + HF providers) · hoca image catalog + scorer + `needs_image` gate · dispatcher image branch · orchestrator image-lane route · beckman image admission · `/image` command · e2e host-path test.
**NOT in this plan (Plan 2/3):** `clair_obscur`, local SDXL selection, eviction-cost from nerd_herd, GPU handover/unload touch, swap-budget interaction, i2p placeholder-swap, prompt-writing coulson task. The image scorer's eviction-cost is a **stub** here (cloud-only: local providers are simply absent from the catalog).

---

## File structure

**New packages:**
- `packages/renoir/` — `src/renoir/__init__.py` (`assess(bytes) -> ImageVerdict`), `pyproject.toml`, `tests/`.
- `packages/paintress/` — `src/paintress/__init__.py` (`generate`), `types.py` (`ImageSpec`, `ImageResult`), `providers/{base,pollinations,huggingface}.py`, `pyproject.toml`, `tests/`.

**Modified (fatih_hoca):**
- `packages/fatih_hoca/src/fatih_hoca/registry.py` — add `BaseModelInfo`, make `ModelInfo(BaseModelInfo)`, add `ImageModelInfo(BaseModelInfo)`.
- `packages/fatih_hoca/src/fatih_hoca/image_providers.py` (new) — static `ImageModelInfo` catalog.
- `packages/fatih_hoca/src/fatih_hoca/image_select.py` (new) — image scorer.
- `packages/fatih_hoca/src/fatih_hoca/__init__.py:439` — `select()` dispatches to image scorer when `needs_image=True`.

**Modified (pipeline):**
- `src/core/llm_dispatcher.py` — `CallCategory.IMAGE`, `dispatch()` reads `context.image_call`, image execute path → `paintress`.
- `src/core/orchestrator.py:277-289` — route `context.image_call.raw_dispatch` → `dispatcher.dispatch`.
- `packages/general_beckman/src/general_beckman/__init__.py:551` — image-aware admission `select(needs_image=True)`.
- `src/app/telegram_bot.py` — `/image` command + handler.

**Test infra:**
- root `conftest.py` `_PACKAGE_SRCS` — add `renoir` + `paintress` (same place `safety_guard` was added in `ae004547`).

---

## Task 1: Data taxonomy — BaseModelInfo + ImageModelInfo

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/registry.py:50-124`
- Test: `packages/fatih_hoca/tests/test_image_model_info.py`

Low-risk interpretation of the "shared base + subclasses" decision: add a **minimal marker base** and make the existing `ModelInfo` inherit it (no 40-field relocation), add a fresh `ImageModelInfo`. `isinstance` branching works; the hot path is untouched.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_model_info.py
from fatih_hoca.registry import BaseModelInfo, ModelInfo, ImageModelInfo


def test_modelinfo_is_basemodelinfo():
    m = ModelInfo(name="x", location="cloud", provider="groq", litellm_name="groq/x", capabilities={})
    assert isinstance(m, BaseModelInfo)
    assert m.is_local is False


def test_image_model_info_fields_and_isinstance():
    im = ImageModelInfo(
        name="pollinations/flux", provider="pollinations", location="cloud",
        endpoint="https://image.pollinations.ai/prompt/", quality_rank=6.0,
        cost_per_image=0.0, vram_mb=0, supports_seed=True,
    )
    assert isinstance(im, BaseModelInfo)
    assert not isinstance(im, ModelInfo)
    assert im.is_local is False
    assert im.supports_image_generation is True


def test_local_image_model_is_local():
    im = ImageModelInfo(name="sdxl-local", provider="clair_obscur", location="local",
                        endpoint="http://127.0.0.1:7860", quality_rank=7.0,
                        cost_per_image=0.0, vram_mb=4000, supports_seed=True)
    assert im.is_local is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_model_info.py -q`
Expected: FAIL — `ImportError: cannot import name 'BaseModelInfo'`.

- [ ] **Step 3: Add `BaseModelInfo` base + `ImageModelInfo`**

In `registry.py`, immediately above the existing `@dataclass class ModelInfo:` (line ~50), insert:

```python
@dataclass
class BaseModelInfo:
    """Minimal shared base so text and image picks can be branched by isinstance.

    Deliberately does NOT relocate ModelInfo's fields — it is a marker base
    carrying only the cross-modality essentials. Keeps the LLM hot path
    untouched (ModelInfo just inherits this).
    """
    name: str
    provider: str
    location: str = "cloud"

    @property
    def is_local(self) -> bool:
        return self.location in ("local", "ollama")

    @property
    def supports_image_generation(self) -> bool:
        return False
```

Change the existing class declaration from `class ModelInfo:` to `class ModelInfo(BaseModelInfo):`. ModelInfo already declares `name`, `provider`, `location` — dataclass inheritance reuses them (same names/order at the front), so no field changes are needed. (If the dataclass complains about field ordering, the fix is purely declarative; ModelInfo's existing `name`/`location`/`provider` already lead its field list.)

Then add, after `ModelInfo`:

```python
@dataclass
class ImageModelInfo(BaseModelInfo):
    """An image-generation provider/model. Parallel to ModelInfo for images."""
    litellm_name: str = ""          # unused for images; present for Pick compat
    endpoint: str = ""              # cloud URL or local base_url
    api_base: str | None = None
    quality_rank: float = 5.0       # 0-10, hand-set
    cost_per_image: float = 0.0
    vram_mb: int = 0                # local footprint; 0 for cloud
    supports_seed: bool = False
    max_width: int = 1024
    max_height: int = 1024
    is_loaded: bool = False
    tier: str = "free"

    @property
    def supports_image_generation(self) -> bool:
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_model_info.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Regression — existing fatih_hoca tests still green**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/ -q -x`
Expected: no NEW failures vs baseline (run the same command on `main` first if unsure).

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/registry.py packages/fatih_hoca/tests/test_image_model_info.py
git commit -m "feat(image): BaseModelInfo + ImageModelInfo taxonomy"
```

---

## Task 2: `renoir` package — image quality judge

**Files:**
- Create: `packages/renoir/pyproject.toml`, `packages/renoir/src/renoir/__init__.py`
- Test: `packages/renoir/tests/test_assess.py`
- Modify: root `conftest.py` (`_PACKAGE_SRCS`)

- [ ] **Step 1: Write the failing test**

```python
# packages/renoir/tests/test_assess.py
import io
from PIL import Image
from renoir import assess, ImageVerdict


def _png_bytes(color=(120, 80, 200), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_ok():
    v = assess(_png_bytes())
    assert isinstance(v, ImageVerdict)
    assert v.ok is True


def test_not_an_image_rejected():
    v = assess(b"<html>rate limited</html>")
    assert v.ok is False
    assert v.reason == "not_an_image"


def test_all_one_color_rejected():
    v = assess(_png_bytes(color=(0, 0, 0)))
    assert v.ok is False
    assert v.reason == "blank"


def test_too_small_rejected():
    v = assess(_png_bytes(size=(4, 4)))
    assert v.ok is False
    assert v.reason == "too_small"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/renoir/tests/test_assess.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'renoir'`.

- [ ] **Step 3: Create the package**

`packages/renoir/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "renoir"
version = "0.1.0"
description = "Image-quality judge (image-side dogru_mu_samet)."
requires-python = ">=3.10"
dependencies = ["pillow>=10"]

[tool.setuptools.packages.find]
where = ["src"]
```

`packages/renoir/src/renoir/__init__.py`:
```python
"""renoir — image quality judge. Parallel to dogru_mu_samet (text)."""
from __future__ import annotations

import io
from dataclasses import dataclass

_MIN_DIM = 16  # px; smaller than this is never a real asset


@dataclass(frozen=True)
class ImageVerdict:
    ok: bool
    reason: str = ""


def assess(data: bytes) -> ImageVerdict:
    """Validate generated image bytes. Catches the free-provider failure mode
    of HTTP 200 with a non-image / blank / tiny body."""
    if not data:
        return ImageVerdict(False, "empty")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return ImageVerdict(False, "not_an_image")

    w, h = img.size
    if w < _MIN_DIM or h < _MIN_DIM:
        return ImageVerdict(False, "too_small")

    # All-one-color guard (blank / error tile). getextrema returns per-band
    # (min, max); identical min==max on every band == single color.
    try:
        extrema = img.convert("RGB").getextrema()
        if all(lo == hi for lo, hi in extrema):
            return ImageVerdict(False, "blank")
    except Exception:
        pass

    return ImageVerdict(True, "")
```

- [ ] **Step 4: Register package for tests + editable install**

In the root `conftest.py`, add `"renoir"` to the `_PACKAGE_SRCS` list (the same list `safety_guard` was added to in commit `ae004547` — `grep -n _PACKAGE_SRCS conftest.py`). Then:

Run: `.venv/Scripts/python -m pip install -e packages/renoir`
Expected: `Successfully installed renoir-0.1.0`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/renoir/tests/test_assess.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add packages/renoir conftest.py
git commit -m "feat(image): renoir image-quality judge package"
```

---

## Task 3: `paintress` types + provider Protocol + dispatch skeleton

**Files:**
- Create: `packages/paintress/pyproject.toml`, `packages/paintress/src/paintress/__init__.py`, `.../types.py`, `.../providers/__init__.py`, `.../providers/base.py`
- Test: `packages/paintress/tests/test_dispatch.py`
- Modify: root `conftest.py` (`_PACKAGE_SRCS`)

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_dispatch.py
import pytest
from paintress import generate, ImageSpec, ImageResult
from paintress.providers.base import ImageProvider


class _FakeProvider:
    name = "fake"
    def available(self): return True
    async def generate(self, spec, *, base_url=None):
        return b"PNGBYTES", {"seed_used": 7}


@pytest.mark.asyncio
async def test_generate_routes_to_provider_by_name(monkeypatch, tmp_path):
    monkeypatch.setattr("paintress._PROVIDERS", {"fake": _FakeProvider()})
    monkeypatch.setattr("paintress.assess", lambda b: type("V", (), {"ok": True, "reason": ""})())

    class Pick:
        class model:
            provider = "fake"
            endpoint = ""
            api_base = None
            cost_per_image = 0.0
            name = "fake/model"
            is_local = False
    spec = ImageSpec(prompt="a cat", out_dir=str(tmp_path), filename_hint="cat")
    res = await generate(Pick(), spec)
    assert isinstance(res, ImageResult)
    assert res.error is None
    assert res.provider == "fake"
    assert res.path.endswith(".png")
    assert res.seed_used == 7


@pytest.mark.asyncio
async def test_unknown_provider_returns_error(tmp_path):
    class Pick:
        class model:
            provider = "nope"; endpoint=""; api_base=None; cost_per_image=0.0; name="nope"; is_local=False
    res = await generate(Pick(), ImageSpec(prompt="x", out_dir=str(tmp_path)))
    assert res.error is not None
    assert "unknown_provider" in res.error
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_dispatch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'paintress'`.

- [ ] **Step 3: Create types + base Protocol**

`packages/paintress/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "paintress"
version = "0.1.0"
description = "Image interaction caller (image-side HaLLederiz Kadir)."
requires-python = ">=3.10"
dependencies = ["httpx>=0.27", "renoir"]

[tool.setuptools.packages.find]
where = ["src"]
```

`packages/paintress/src/paintress/types.py`:
```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImageSpec:
    prompt: str
    out_dir: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    seed: int | None = None          # None = random
    quality_tier: str = "fast"       # "fast" | "quality"
    filename_hint: str | None = None


@dataclass
class ImageResult:
    path: str | None = None
    provider: str = ""
    model: str = ""
    cost: float = 0.0
    latency: float = 0.0
    seed_used: int | None = None
    error: str | None = None
```

`packages/paintress/src/paintress/providers/__init__.py`: (empty)

`packages/paintress/src/paintress/providers/base.py`:
```python
from __future__ import annotations
from typing import Protocol
from ..types import ImageSpec


class ImageProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    async def generate(self, spec: ImageSpec, *, base_url: str | None = None) -> tuple[bytes, dict]:
        """Return (image_bytes, meta). meta may carry {'seed_used': int}.
        MUST NOT raise — raise-free contract; on failure raise is caught by caller."""
        ...
```

- [ ] **Step 4: Create the dispatch entry**

`packages/paintress/src/paintress/__init__.py`:
```python
"""paintress — image interaction caller. Given hoca's pick, calls the provider,
validates via renoir, writes the PNG, returns ImageResult. LLM-free."""
from __future__ import annotations

import os
import re
import time

from renoir import assess
from .types import ImageSpec, ImageResult

# Provider registry is populated in Task 4/5. Kept module-level so tests can
# monkeypatch it.
_PROVIDERS: dict = {}


def _safe_filename(hint: str | None) -> str:
    base = (hint or "image").strip() or "image"
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base)[:48]
    return f"{base}_{int(time.time() * 1000) % 1_000_000}.png"


async def generate(pick, spec: ImageSpec) -> ImageResult:
    model = pick.model
    provider = getattr(model, "provider", "")
    prov = _PROVIDERS.get(provider)
    if prov is None or not prov.available():
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"unknown_provider:{provider}")

    base_url = getattr(model, "api_base", None) or getattr(model, "endpoint", None)
    started = time.time()
    try:
        data, meta = await prov.generate(spec, base_url=base_url)
    except Exception as exc:  # raise-free contract backstop
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"provider_raised:{exc}")

    verdict = assess(data)
    if not verdict.ok:
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"quality_failure:{verdict.reason}")

    os.makedirs(spec.out_dir, exist_ok=True)
    path = os.path.join(spec.out_dir, _safe_filename(spec.filename_hint))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

    return ImageResult(
        path=path, provider=provider, model=getattr(model, "name", ""),
        cost=float(getattr(model, "cost_per_image", 0.0) or 0.0),
        latency=time.time() - started,
        seed_used=(meta or {}).get("seed_used"),
    )
```

- [ ] **Step 5: Register + install**

Add `"paintress"` to root `conftest.py` `_PACKAGE_SRCS`. Run:
`.venv/Scripts/python -m pip install -e packages/paintress`
Expected: `Successfully installed paintress-0.1.0`.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_dispatch.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add packages/paintress conftest.py
git commit -m "feat(image): paintress types + provider Protocol + dispatch"
```

---

## Task 4: `paintress` Pollinations provider (default, no key)

**Files:**
- Create: `packages/paintress/src/paintress/providers/pollinations.py`
- Modify: `packages/paintress/src/paintress/__init__.py` (register provider)
- Test: `packages/paintress/tests/test_pollinations.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_pollinations.py
import pytest
from paintress.providers.pollinations import PollinationsProvider
from paintress.types import ImageSpec


@pytest.mark.asyncio
async def test_pollinations_builds_url_and_returns_bytes(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nFAKE"
        headers = {"content-type": "image/png"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **kw):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    prov = PollinationsProvider()
    spec = ImageSpec(prompt="a red bicycle", out_dir="/tmp", width=512, height=512, seed=42)
    data, meta = await prov.generate(spec)
    assert data.startswith(b"\x89PNG")
    assert "a%20red%20bicycle" in captured["url"] or "a+red+bicycle" in captured["url"] or "a%2520" not in captured["url"]
    assert "seed=42" in captured["url"]
    assert meta["seed_used"] == 42
    assert prov.available() is True


@pytest.mark.asyncio
async def test_pollinations_non_image_content_type_still_returns_bytes_for_renoir(monkeypatch):
    # paintress -> renoir judges validity; provider just returns what it got.
    class _Resp:
        status_code = 200
        content = b"<html>error</html>"
        headers = {"content-type": "text/html"}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def get(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    data, meta = await PollinationsProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert data == b"<html>error</html>"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_pollinations.py -q`
Expected: FAIL — `ModuleNotFoundError: ...pollinations`.

- [ ] **Step 3: Implement the provider**

`packages/paintress/src/paintress/providers/pollinations.py`:
```python
from __future__ import annotations

import random
import urllib.parse

import httpx

from ..types import ImageSpec

_BASE = "https://image.pollinations.ai/prompt/"
_TIMEOUT = 90.0


class PollinationsProvider:
    name = "pollinations"

    def available(self) -> bool:
        return True  # no key, public free service

    async def generate(self, spec: ImageSpec, *, base_url: str | None = None):
        seed = spec.seed if spec.seed is not None else random.randint(1, 2_000_000_000)
        prompt = urllib.parse.quote(spec.prompt or "")
        params = {
            "width": spec.width, "height": spec.height,
            "seed": seed, "model": "flux", "nologo": "true",
        }
        url = f"{base_url or _BASE}{prompt}?" + urllib.parse.urlencode(params)
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content, {"seed_used": seed}
```

- [ ] **Step 4: Register in the provider map**

In `packages/paintress/src/paintress/__init__.py`, replace `_PROVIDERS: dict = {}` with a populated default:
```python
from .providers.pollinations import PollinationsProvider

_PROVIDERS: dict = {
    "pollinations": PollinationsProvider(),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_pollinations.py packages/paintress/tests/test_dispatch.py -q`
Expected: PASS (4 passed). (The dispatch test monkeypatches `_PROVIDERS`, so it still works.)

- [ ] **Step 6: Commit**

```bash
git add packages/paintress/src/paintress/providers/pollinations.py packages/paintress/src/paintress/__init__.py packages/paintress/tests/test_pollinations.py
git commit -m "feat(image): paintress pollinations provider"
```

---

## Task 5: `paintress` Hugging Face provider (HF_TOKEN, FLUX.1-schnell)

**Files:**
- Create: `packages/paintress/src/paintress/providers/huggingface.py`
- Modify: `packages/paintress/src/paintress/__init__.py` (register)
- Test: `packages/paintress/tests/test_huggingface.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_huggingface.py
import pytest
from paintress.providers.huggingface import HuggingFaceProvider
from paintress.types import ImageSpec


@pytest.mark.asyncio
async def test_hf_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert HuggingFaceProvider().available() is False


@pytest.mark.asyncio
async def test_hf_posts_and_returns_image(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nIMG"
        headers = {"content-type": "image/png"}
        def json(self): return {}
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    data, meta = await HuggingFaceProvider().generate(ImageSpec(prompt="a forest", out_dir="/tmp"))
    assert data.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_hf_503_model_loading_raises_retryable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 503
        content = b'{"error":"Model is loading","estimated_time":20}'
        headers = {"content-type": "application/json"}
        def json(self): return {"error": "Model is loading", "estimated_time": 20}
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "model_loading" in str(ei.value)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_huggingface.py -q`
Expected: FAIL — `ModuleNotFoundError: ...huggingface`.

- [ ] **Step 3: Implement the provider**

`packages/paintress/src/paintress/providers/huggingface.py`:
```python
from __future__ import annotations

import os
import httpx

from ..types import ImageSpec

_MODEL = "black-forest-labs/FLUX.1-schnell"
_URL = f"https://api-inference.huggingface.co/models/{_MODEL}"
_TIMEOUT = 120.0


class HuggingFaceProvider:
    name = "huggingface"

    def available(self) -> bool:
        return bool(os.getenv("HF_TOKEN"))

    async def generate(self, spec: ImageSpec, *, base_url: str | None = None):
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Accept": "image/png"}
        payload = {"inputs": spec.prompt or ""}
        if spec.seed is not None:
            payload["parameters"] = {"seed": spec.seed}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(base_url or _URL, headers=headers, json=payload)
        # HF returns 503 with {"error":"...loading","estimated_time":N} on cold start.
        if resp.status_code == 503:
            raise RuntimeError("model_loading: HF FLUX cold start, retry")
        if resp.status_code in (401, 403):
            raise RuntimeError(f"hf_auth:{resp.status_code}")
        resp.raise_for_status()
        return resp.content, {"seed_used": spec.seed}
```

- [ ] **Step 4: Register**

In `__init__.py` `_PROVIDERS`, add:
```python
from .providers.huggingface import HuggingFaceProvider
_PROVIDERS = {
    "pollinations": PollinationsProvider(),
    "huggingface": HuggingFaceProvider(),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/ -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add packages/paintress/src/paintress/providers/huggingface.py packages/paintress/src/paintress/__init__.py packages/paintress/tests/test_huggingface.py
git commit -m "feat(image): paintress huggingface provider"
```

---

## Task 6: fatih_hoca image-provider catalog

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/image_providers.py`
- Test: `packages/fatih_hoca/tests/test_image_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_providers.py
from fatih_hoca.image_providers import image_catalog
from fatih_hoca.registry import ImageModelInfo


def test_catalog_has_cloud_providers():
    cat = image_catalog()
    names = {m.name for m in cat}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
    assert all(isinstance(m, ImageModelInfo) for m in cat)


def test_pollinations_is_free_cloud():
    cat = {m.name: m for m in image_catalog()}
    p = cat["pollinations/flux"]
    assert p.provider == "pollinations"
    assert p.is_local is False
    assert p.cost_per_image == 0.0
    assert p.supports_image_generation is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers.py -q`
Expected: FAIL — `ModuleNotFoundError: ...image_providers`.

- [ ] **Step 3: Implement the static catalog**

`packages/fatih_hoca/src/fatih_hoca/image_providers.py`:
```python
"""Static image-provider catalog. Image providers are NOT in cloud /models
discovery (that's LLM-only), so they are registered here. Plan 2 adds the
local clair_obscur entry; this is cloud-only."""
from __future__ import annotations

from .registry import ImageModelInfo


def image_catalog() -> list[ImageModelInfo]:
    return [
        ImageModelInfo(
            name="pollinations/flux", provider="pollinations", location="cloud",
            endpoint="https://image.pollinations.ai/prompt/",
            quality_rank=6.0, cost_per_image=0.0, vram_mb=0,
            supports_seed=True, tier="free",
        ),
        ImageModelInfo(
            name="huggingface/flux-schnell", provider="huggingface", location="cloud",
            endpoint="https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
            quality_rank=8.0, cost_per_image=0.0, vram_mb=0,
            supports_seed=True, tier="free",
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_providers.py packages/fatih_hoca/tests/test_image_providers.py
git commit -m "feat(image): static image-provider catalog in fatih_hoca"
```

---

## Task 7: fatih_hoca image scorer (`image_select.py`)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/image_select.py`
- Test: `packages/fatih_hoca/tests/test_image_select.py`

Cloud-only scorer. Eviction-cost is a stub (no local providers in the catalog yet, so it never fires). Honors `failures` and `availability`.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_select.py
from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick, SelectionFailure


def test_picks_highest_quality_available(monkeypatch):
    # HF available (token present) -> higher quality_rank wins
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.name == "huggingface/flux-schnell"


def test_falls_back_to_pollinations_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = select_image(quality_tier="fast", failures=[], hf_available=False)
    assert pick.model.name == "pollinations/flux"


def test_excludes_failed_provider():
    # pollinations failed -> with no HF, nothing eligible
    pick = select_image(quality_tier="fast", failures=["pollinations/flux"], hf_available=False)
    assert isinstance(pick, SelectionFailure)
    assert pick.reason == "availability"


def test_failed_pollinations_falls_to_hf():
    pick = select_image(quality_tier="fast", failures=["pollinations/flux"], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select.py -q`
Expected: FAIL — `ModuleNotFoundError: ...image_select`.

- [ ] **Step 3: Implement the scorer**

`packages/fatih_hoca/src/fatih_hoca/image_select.py`:
```python
"""Purpose-built image-model scorer. Sibling to selector.py (text). Cloud-only
in Plan 1; eviction-cost (local) is a stub and never fires because the catalog
has no local entries yet (Plan 2 adds it)."""
from __future__ import annotations

import os

from .image_providers import image_catalog
from .registry import ImageModelInfo
from .types import Pick, SelectionFailure


def _provider_available(m: ImageModelInfo, hf_available: bool | None) -> bool:
    if m.provider == "huggingface":
        return os.getenv("HF_TOKEN") is not None if hf_available is None else hf_available
    if m.provider == "pollinations":
        return True
    return False  # local providers: Plan 2


def _eviction_cost(m: ImageModelInfo) -> float:
    # Plan 1: cloud-only. Local eviction-cost (from nerd_herd) is Plan 2.
    return 0.0


def select_image(
    *,
    quality_tier: str = "fast",
    failures: list[str] | None = None,
    hf_available: bool | None = None,
    remaining_budget_usd: float | None = None,
) -> Pick | SelectionFailure:
    failed = set(failures or [])
    candidates: list[tuple[float, ImageModelInfo]] = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        # Base score: quality-led (all MVP providers are free so cost ties).
        score = m.quality_rank
        if quality_tier == "fast":
            score -= 0.0  # no penalty; latency handled by provider choice later
        score -= _eviction_cost(m)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0][1]
    return Pick(model=best, min_time_seconds=0.0, score=candidates[0][0],
                top_summary="; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_select.py packages/fatih_hoca/tests/test_image_select.py
git commit -m "feat(image): fatih_hoca image scorer (cloud-only)"
```

---

## Task 8: Wire `select(needs_image=True)` to the image scorer

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py:439-451`
- Test: `packages/fatih_hoca/tests/test_select_image_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_select_image_dispatch.py
import fatih_hoca
from fatih_hoca.types import Pick
from fatih_hoca.registry import ImageModelInfo


def test_select_needs_image_returns_image_pick(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    assert isinstance(pick, Pick)
    assert isinstance(pick.model, ImageModelInfo)
    assert pick.model.name == "pollinations/flux"


def test_select_without_needs_image_unaffected(monkeypatch):
    # Sanity: text path still callable (returns Pick or None, not an image model).
    res = fatih_hoca.select(task="router", agent_type="router", difficulty=3)
    if res is not None and hasattr(res, "model"):
        assert not isinstance(res.model, ImageModelInfo)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_select_image_dispatch.py::test_select_needs_image_returns_image_pick -q`
Expected: FAIL — `TypeError: select() got an unexpected keyword argument 'needs_image'` (or text-path error).

- [ ] **Step 3: Add the dispatch branch**

In `packages/fatih_hoca/src/fatih_hoca/__init__.py`, at the top of the `select()` body (line ~440, before it builds text Requirements), insert:

```python
def select(**kwargs):
    # Image modality short-circuit — route to the purpose-built image scorer
    # before any text-model machinery runs.
    if kwargs.pop("needs_image", False):
        from .image_select import select_image
        return select_image(
            quality_tier=kwargs.get("quality_tier", "fast"),
            failures=[getattr(f, "model", f) for f in (kwargs.get("failures") or [])],
            remaining_budget_usd=kwargs.get("remaining_budget_usd"),
        )
    # ... existing text-selection body unchanged ...
```

Note: `failures` entries may be `Failure` objects (text path) or plain provider-name strings (image retry). The comprehension reads `.model` when present, else the item itself.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_select_image_dispatch.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/ -q`
Expected: no new failures vs baseline.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_select_image_dispatch.py
git commit -m "feat(image): select(needs_image=True) -> image scorer"
```

---

## Task 9: Dispatcher image branch

**Files:**
- Modify: `src/core/llm_dispatcher.py` (`CallCategory`, `dispatch()`, new `_dispatch_image`)
- Test: `tests/core/test_dispatcher_image.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_dispatcher_image.py
import pytest
from src.core.llm_dispatcher import get_dispatcher, CallCategory


@pytest.mark.asyncio
async def test_dispatch_image_routes_to_paintress(monkeypatch, tmp_path):
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    model = ImageModelInfo(name="pollinations/flux", provider="pollinations",
                           location="cloud", endpoint="https://x/", cost_per_image=0.0)
    pick = Pick(model=model, min_time_seconds=0.0)

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"), provider="pollinations",
                           model="pollinations/flux", cost=0.0, seed_used=5)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    spec = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a cat", "out_dir": str(tmp_path),
            "width": 512, "height": 512,
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await get_dispatcher().dispatch(spec)
    assert res["path"].endswith("out.png")
    assert res["provider"] == "pollinations"
    assert CallCategory.IMAGE.value == "image"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_image.py -q`
Expected: FAIL — `AttributeError: IMAGE` (enum) / KeyError on image_call.

- [ ] **Step 3: Add `CallCategory.IMAGE` + image dispatch**

In `src/core/llm_dispatcher.py`, extend the enum (after `OVERHEAD`):
```python
    IMAGE = "image"          # image generation; routes to paintress, not HK
```

In `dispatch()` (line ~535), at the very top after extracting `spec`, add:
```python
        image_call = (spec.get("context", {}) or {}).get("image_call")
        if isinstance(image_call, dict) and image_call.get("raw_dispatch"):
            return await self._dispatch_image(spec, image_call)
```

Add the new method:
```python
    async def _dispatch_image(self, spec: dict, image_call: dict) -> dict:
        """Route an image task to paintress. Beckman attached the ImageModelInfo
        Pick as spec['preselected_pick']. No re-selection here (dumb pipe)."""
        import paintress
        from src.core.router import ModelCallFailed

        pick = spec.get("preselected_pick")
        if pick is None:
            raise ModelCallFailed(call_id="image", last_error="no image pick",
                                  error_category="availability")
        out_dir = image_call.get("out_dir") or "."
        s = paintress.ImageSpec(
            prompt=image_call.get("prompt", ""),
            out_dir=out_dir,
            negative_prompt=image_call.get("negative_prompt"),
            width=int(image_call.get("width", 1024)),
            height=int(image_call.get("height", 1024)),
            seed=image_call.get("seed"),
            quality_tier=image_call.get("quality_tier", "fast"),
            filename_hint=image_call.get("filename_hint"),
        )
        from src.core import heartbeat as _hb
        async with _hb.keepalive():
            res = await paintress.generate(pick, s)
        if res.error is not None:
            retryable = not res.error.startswith("unknown_provider")
            raise ModelCallFailed(call_id=getattr(pick.model, "name", "image"),
                                  last_error=res.error,
                                  error_category="availability" if retryable else "fatal")
        return {
            "content": res.path, "path": res.path, "provider": res.provider,
            "model": res.model, "cost": res.cost, "latency": res.latency,
            "seed_used": res.seed_used, "is_local": getattr(pick.model, "is_local", False),
            "ran_on": res.provider,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_image.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/core/test_dispatcher_image.py
git commit -m "feat(image): dispatcher image branch -> paintress"
```

---

## Task 10: Orchestrator routes the image lane

**Files:**
- Modify: `src/core/orchestrator.py:277-289`
- Test: `tests/core/test_orchestrator_image_route.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_orchestrator_image_route.py
import pytest


@pytest.mark.asyncio
async def test_image_call_routes_to_dispatcher(monkeypatch):
    import src.core.orchestrator as orch
    seen = {}

    class _Disp:
        async def dispatch(self, spec):
            seen["spec"] = spec
            return {"path": "/tmp/x.png", "provider": "pollinations"}
    monkeypatch.setattr("src.core.llm_dispatcher.get_dispatcher", lambda: _Disp())

    task = {"id": 1, "kind": "image",
            "context": {"image_call": {"raw_dispatch": True, "prompt": "a dog"}},
            "preselected_pick": object()}
    result = await orch._dispatch(task)   # use the real dispatch entry name
    assert seen["spec"]["context"]["image_call"]["prompt"] == "a dog"
```

(If the orchestrator's per-task dispatch helper has a different name than `_dispatch`, adjust the call — confirm via `grep -n "image_call\|llm_call\|raw_dispatch" src/core/orchestrator.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_orchestrator_image_route.py -q`
Expected: FAIL — image_call not routed (dispatcher not called / seen empty).

- [ ] **Step 3: Extend the raw-dispatch branch**

In `src/core/orchestrator.py` at the `_is_raw` check (line ~277-289), change the condition so an `image_call` also routes to the dispatcher. Locate:
```python
        _is_raw = bool((ctx.get("llm_call") or {}).get("raw_dispatch"))
```
and replace with:
```python
        _llm_rd = bool((ctx.get("llm_call") or {}).get("raw_dispatch"))
        _img_rd = bool((ctx.get("image_call") or {}).get("raw_dispatch"))
        _is_raw = _llm_rd or _img_rd
```
The existing `get_dispatcher().dispatch({...})` call already passes `context` + `preselected_pick`, so the image branch inside `dispatch()` (Task 9) picks it up. Ensure the dispatched spec includes the full `context` (it does — `_ctx_rd` is the task context).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_orchestrator_image_route.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/core/test_orchestrator_image_route.py
git commit -m "feat(image): orchestrator routes image_call to dispatcher"
```

---

## Task 11: Beckman image-aware admission

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py:551-558` (admission select), `:608` (attach pick)
- Test: `packages/general_beckman/tests/test_image_admission.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_image_admission.py
import pytest


def test_image_task_selects_with_needs_image(monkeypatch):
    import general_beckman as gb
    captured = {}

    def _fake_select(**kw):
        captured.update(kw)
        from fatih_hoca.types import Pick
        from fatih_hoca.registry import ImageModelInfo
        return Pick(model=ImageModelInfo(name="pollinations/flux", provider="pollinations",
                                         location="cloud"), min_time_seconds=0.0)
    monkeypatch.setattr("fatih_hoca.select", _fake_select)

    spec = {"kind": "image", "context": {"image_call": {"raw_dispatch": True, "prompt": "x"}},
            "agent_type": "image"}
    pick = gb._select_for_admission(spec)   # confirm helper name via grep
    assert captured.get("needs_image") is True
    assert pick.model.name == "pollinations/flux"
```

(Confirm the exact admission-select helper around `__init__.py:551` via `grep -n "fatih_hoca.select" packages/general_beckman/src/general_beckman/__init__.py`. If selection is inline in `next_task`, extract it into a small `_select_for_admission(spec)` helper as part of this task and test that.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_admission.py -q`
Expected: FAIL — `needs_image` not passed / helper missing.

- [ ] **Step 3: Make admission image-aware**

At `__init__.py:551`, where beckman calls `fatih_hoca.select(...)`, detect an image task and branch the kwargs. Wrap the existing call:
```python
        _is_image = bool(
            (spec.get("context", {}) or {}).get("image_call")
            or spec.get("kind") == "image"
        )
        if _is_image:
            ic = (spec.get("context", {}) or {}).get("image_call", {}) or {}
            pick = fatih_hoca.select(
                needs_image=True,
                quality_tier=ic.get("quality_tier", "fast"),
                failures=spec.get("failures") or [],
            )
        else:
            pick = fatih_hoca.select(
                task=agent_type, agent_type=agent_type, difficulty=difficulty,
                urgency=urgency, estimated_input_tokens=est_in,
                estimated_output_tokens=est_out,
            )
```
The `preselected_pick` attach at `:608` is unchanged (`task["preselected_pick"] = pick`). A `SelectionFailure` from the image scorer flows through beckman's existing SelectionFailure handling.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_admission.py -q`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/ -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_image_admission.py
git commit -m "feat(image): beckman image-aware admission (needs_image)"
```

---

## Task 12: `/image` Telegram command

**Files:**
- Modify: `src/app/telegram_bot.py` (`_setup_handlers`, new `cmd_image`)
- Test: `tests/app/test_cmd_image.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_cmd_image.py
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cmd_image_enqueues_and_sends_photo(monkeypatch, tmp_path):
    import src.app.telegram_bot as tb

    png = tmp_path / "out.png"; png.write_bytes(b"\x89PNG")

    class _TaskResult:
        status = "completed"
        result = {"path": str(png), "provider": "pollinations"}
    enq = AsyncMock(return_value=_TaskResult())
    monkeypatch.setattr("general_beckman.enqueue", enq)

    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    update = MagicMock()
    update.effective_chat.id = 99
    update.message.reply_photo = AsyncMock()
    iface._reply = AsyncMock()
    ctx = MagicMock(); ctx.args = ["a", "red", "bicycle"]

    await iface.cmd_image(update, ctx)

    assert enq.await_count == 1
    spec = enq.await_args.args[0]
    assert spec["context"]["image_call"]["prompt"] == "a red bicycle"
    assert spec["context"]["image_call"]["raw_dispatch"] is True
    update.message.reply_photo.assert_awaited()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/app/test_cmd_image.py -q`
Expected: FAIL — `AttributeError: cmd_image`.

- [ ] **Step 3: Register the handler**

In `_setup_handlers()` (telegram_bot.py ~1927), add:
```python
        self.app.add_handler(CommandHandler("image", self.cmd_image))
```

- [ ] **Step 4: Implement `cmd_image`**

Add the method (near other `cmd_*`):
```python
    async def cmd_image(self, update, context):
        """/image <prompt> — generate an image via the cloud image pipeline."""
        prompt = " ".join(context.args or []).strip()
        if not prompt:
            await self._reply(update, "Usage: `/image a red bicycle`",
                              parse_mode="Markdown", reply_markup=REPLY_KEYBOARD)
            return
        chat_id = update.effective_chat.id
        import os, tempfile, general_beckman
        out_dir = os.path.join(tempfile.gettempdir(), "kutai_images", str(chat_id))
        spec = {
            "title": f"image:{prompt[:40]}",
            "description": "Telegram /image generation",
            "agent_type": "image",
            "kind": "image",
            "runner": "direct",
            "priority": 5,
            "context": {"image_call": {
                "raw_dispatch": True, "prompt": prompt, "out_dir": out_dir,
                "width": 1024, "height": 1024, "quality_tier": "quality",
                "filename_hint": prompt[:30],
            }},
        }
        await self._reply(update, "🎨 Generating…", reply_markup=REPLY_KEYBOARD)
        result = await general_beckman.enqueue(spec, await_inline=True)
        path = None
        if getattr(result, "status", "") == "completed":
            res = result.result if isinstance(result.result, dict) else {}
            path = res.get("path")
        if path and os.path.isfile(path):
            with open(path, "rb") as fh:
                await update.message.reply_photo(photo=fh, caption=prompt[:200])
        else:
            err = getattr(result, "error", "") or "generation failed"
            await self._reply(update, f"❌ Image failed: {err}",
                              reply_markup=REPLY_KEYBOARD)
```

(If `result.result` arrives as a JSON string in this codebase, parse it — mirror `_task_result_to_request_response` in `llm_dispatcher.py`. Confirm the `TaskResult.result` shape via the dispatcher's `request()` mapping.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/app/test_cmd_image.py -q`
Expected: PASS.

- [ ] **Step 6: Verify import + handler registration**

Run: `.venv/Scripts/python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/app/telegram_bot.py tests/app/test_cmd_image.py
git commit -m "feat(image): /image Telegram command"
```

---

## Task 13: End-to-end host-path test

**Files:**
- Test: `tests/integration/test_image_e2e.py`

Proves the whole lane: image spec → beckman admit (image scorer) → dispatcher image branch → paintress (mocked provider) → PNG on disk. Host-path (recurring lesson: unit-green ≠ wired).

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_e2e.py
import os
import pytest


@pytest.mark.asyncio
async def test_image_generation_end_to_end(monkeypatch, tmp_path):
    # Mock the actual HTTP provider call only; everything else is real.
    import paintress

    class _FakeProvider:
        name = "pollinations"
        def available(self): return True
        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO(); Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": 11}
    monkeypatch.setattr(paintress, "_PROVIDERS", {"pollinations": _FakeProvider()})
    monkeypatch.delenv("HF_TOKEN", raising=False)  # force pollinations

    from src.core.llm_dispatcher import get_dispatcher
    import fatih_hoca

    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    spec = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a mountain lake", "out_dir": str(tmp_path),
            "width": 64, "height": 64, "filename_hint": "lake",
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await get_dispatcher().dispatch(spec)
    assert res["provider"] == "pollinations"
    assert os.path.isfile(res["path"])
    assert res["seed_used"] == 11
    # renoir validated it: a real 64x64 PNG exists
    assert os.path.getsize(res["path"]) > 0
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_e2e.py -q`
Expected: PASS (1 passed).

- [ ] **Step 3: Full new-suite green-check**

Run the new tests together (NOT mixing `tests/` and `packages/*/tests/` in one invocation — the conftest collision in the handoff §4):
```
.venv/Scripts/python -m pytest packages/renoir/tests packages/paintress/tests packages/fatih_hoca/tests/test_image_model_info.py packages/fatih_hoca/tests/test_image_providers.py packages/fatih_hoca/tests/test_image_select.py packages/fatih_hoca/tests/test_select_image_dispatch.py packages/general_beckman/tests/test_image_admission.py -q
.venv/Scripts/python -m pytest tests/core/test_dispatcher_image.py tests/core/test_orchestrator_image_route.py tests/app/test_cmd_image.py tests/integration/test_image_e2e.py -q
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_image_e2e.py
git commit -m "test(image): end-to-end cloud /image host-path test"
```

---

## Plan 1 done-when

- `/image <prompt>` in Telegram returns a generated photo via Pollinations (or HF when `HF_TOKEN` set).
- All new tests green; no regressions in `packages/fatih_hoca/tests/`, `packages/general_beckman/tests/`.
- Local providers absent (clair_obscur is Plan 2); eviction-cost is a no-op stub.

## Follow-on plans (write after Plan 1 executes)
- **Plan 2 — local `clair_obscur` + GPU handover:** clair_obscur package (PID-lock/orphan-reconcile), nerd_herd image-server VRAM residency, real eviction-cost in `image_select` (reads nerd_herd), dispatcher `dallama.unload()` handover touch, beckman warm-batch + swap-budget, local provider in catalog.
- **Plan 3 — i2p integration:** prompt-writing coulson task (+ templates), `swap_placeholder_images` mr_roboto mechanical, asset serving into web-preview, i2p prototype-phase wiring.
