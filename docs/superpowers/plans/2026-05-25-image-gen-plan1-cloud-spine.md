# Image Generation — Plan 1 (v3): Cloud spine + `/image`

> **v3 (2026-06-05)** corrects three execution-blockers found validating v2 against live code: (1) image execution moves to **husam** (no `dispatcher.dispatch()` exists); (2) failed-model propagation wraps names in `Failure(...)` for the text path (raw strings crashed the selector); (3) `/image` + e2e drop **`await_inline`** for the durable CPS `on_complete` continuation / manual pump-cycle tests. See "Audit findings" below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end cloud image generation reachable from Telegram `/image <prompt>`, flowing through the existing beckman→hoca→**husam** pipeline with two new thin packages (`paintress`, `renoir`) and an image scorer in `fatih_hoca`, **with full telemetry round-trip (pick_log, in-flight registry, token/cost) and a working failures-propagation across retries**. Result delivery is **CPS** (durable `on_complete` continuation) — **no `await_inline`** (per `feedback`/CPS-migration: await_inline is being retired).

**Architecture:** Image generation is just tasks through the singular lifecycle (see `docs/superpowers/specs/2026-05-23-image-generation-design.md` §2). `paintress` = image interaction caller (≈ HaLLederiz Kadir). `renoir` = image quality judge (≈ dogru_mu_samet). `fatih_hoca` gains a static image catalog + a purpose-built scorer (`image_select.py`) routed via `select(needs_image=True)`. **`husam`** (`packages/husam/`, the non-agentic raw-dispatch worker) routes image picks to paintress via a modality branch **at the top of `husam.worker.run()`** — NOT the dispatcher. There is **no `LLMDispatcher.dispatch()` method** (it was removed); the raw path is `orchestrator._dispatch` → `husam.run()` → (text) `dispatcher.execute()` / (image) `paintress.generate()`. Husam is the only caller permitted to reach the dispatcher (alongside coulson). The image path writes pick_log, in-flight, and token/cost telemetry mirroring husam's text envelope (`get_dispatcher()._record_pick` + `in_flight.begin_call/end_call` + `db.record_call_tokens/record_call_cost`). **This plan is cloud-only**: `clair_obscur` (local server), real eviction-cost from nerd_herd, GPU handover, and i2p integration are Plans 2 & 3.

**Tech Stack:** Python 3.10, async/await, aiosqlite, httpx (HTTP to image providers), Pillow (image validation in renoir), pytest. New packages use src-layout like `packages/fatih_hoca/`.

**Scope (Plan 1 only):** data type · `renoir` · `paintress` (pollinations + HF providers) · hoca image catalog + scorer + `needs_image` dispatch · benchmark-enrichment guard · dispatcher image branch with full telemetry · orchestrator image-lane route · **shared inter-task failures-propagation fix in beckman (benefits text AND image)** · beckman image admission with SelectionFailure handling · `/image` command with stringified-result parse · **real e2e covering beckman→orchestrator→dispatcher→paintress** · telemetry round-trip verification.

**NOT in this plan (Plan 2/3):** `clair_obscur`, local SDXL selection, real eviction-cost reading nerd_herd, GPU handover/unload, swap-budget interaction, i2p placeholder-swap, prompt-writing coulson task. In Plan 1 the image scorer's `_eviction_cost` is a no-op stub; local providers are absent from the catalog so the stub never fires.

---

## Audit findings this rewrite addresses

This is the **v3 correction pass** (2026-06-05). The v2 plan was validated against live code and had **three execution-blocking errors** plus the original v1 issues. v3 fixes all of them at the structural level:

**v2 → v3 corrections (the blockers):**
1. **No `dispatch()` method exists.** v2's Task 11/12 added an image branch to `LLMDispatcher.dispatch()` and asserted `get_dispatcher().dispatch(spec)`. Grep `def dispatch` = zero hits. The raw path is `orchestrator._dispatch` → **`husam.run()`** (`src/core/orchestrator.py:364-377`) → `dispatcher.execute()` (`packages/husam/src/husam/worker.py:273-288`). **v3 puts the image branch in `husam.worker.run()`**, where text raw-dispatch already lives. (The 2026-06-05 dispatcher de-accretion also moved the telemetry body to `src/telemetry/pick_recorder.py`, keeping `_record_pick` as a thin delegator — v3 reuses that delegator from husam.)
2. **v2's failures-propagation crashes the TEXT path.** `Failure` is a dataclass with `model: str` (`packages/fatih_hoca/src/fatih_hoca/types.py:29-33`); the selector does `{f.model for f in failures}` (`selector.py:175`). v2 forwarded raw **strings** as `failures=`, so `str.model` → `AttributeError` on the first text retry. v2's unit test never caught it (it monkeypatched `fatih_hoca.select`). **v3 wraps names in `Failure(model=name, reason="prior_admission")` for the text path** (image path keeps strings — the image scorer normalizes). v3 also fixes the test's name-extractor to read `f.model` (string), matching reality.
3. **`await_inline` is being retired and cannot drive retries inline.** `await_inline=True` inserts to DB and blocks on an `asyncio.Future` that resolves ONLY on terminal state (`general_beckman/__init__.py:986-994, 1245-1252`); execution + retry are driven by the orchestrator pump + `on_task_finished` re-enqueue across cycles. v2's `/image` and e2e tests used `await_inline`. **v3 delivers `/image` via the durable CPS `on_complete` continuation** (registry at `general_beckman/continuations.py:25-50`; Telegram singleton at `telegram_bot.py:304-353`) and **drives e2e tests via manual pump cycles** (`next_task()` → `husam.run(task)` → `on_task_finished(tid, result)`).

**Original v1 issues (already fixed in v2, retained):** dataclass-inheritance compile break; scorer `failures` string-vs-object bug; JSON-string vs dict mismatch in `/image` result; missing image telemetry; "e2e" tests skipping beckman+orchestrator.

Recon confirmed (corrected file:line, 2026-06-05):
- `ModelInfo` required fields with NO defaults: `name, location, provider, litellm_name` at `registry.py:50-56`. Any defaulted dataclass parent would crash compile. **`ImageModelInfo` is independent; husam branches on `isinstance(pick.model, ImageModelInfo)` (and on `context.image_call`).**
- Orchestrator stringifies husam's return: `"result": json.dumps(_dispatch_result)` at `orchestrator.py:378-382`. The continuation handler / `TaskResult.result` therefore sees a JSON string — confirmed; both `/image` handler and e2e parse it.
- `_record_pick` (thin delegator, `llm_dispatcher.py` — sig `pick, task, category, success, error_category, agent_type, difficulty`), `begin_call/end_call` (`in_flight.py:199-281`, sig `category, model_name, provider, is_local, task_id, est_tokens` / `call_id`), `record_call_tokens` (`db.py:5814`, keyword-only, tolerates zero tokens), `record_call_cost` (`db.py:7547`, `(task_id, cost_usd)`) — all **generic**, recon-verified. husam's image path calls them all.
- Beckman has **no** SelectionFailure handler at admission (grep empty). v3 adds one at the real `next_task()` selection site.
- Inter-task failures-propagation does NOT exist: `on_task_finished` writes `task.context["failed_models"]` as a list of **name strings** (`general_beckman/__init__.py:892-896`) but `next_task()`'s `fatih_hoca.select(**_sel_kwargs)` (`:637-650`) never sets `failures=`. v3 adds the shared mechanism — text retries get the win too (correctly, via `Failure` objects).

---

## File structure

**New packages:**
- `packages/renoir/` — `assess(bytes) -> ImageVerdict`. Pillow-backed validity check.
- `packages/paintress/` — `generate(pick, spec) -> ImageResult`; `types.py`; `providers/{base,pollinations,huggingface}.py`.

**Modified (fatih_hoca):**
- `packages/fatih_hoca/src/fatih_hoca/registry.py` — add `ImageModelInfo` dataclass (independent, NOT a subclass).
- `packages/fatih_hoca/src/fatih_hoca/image_providers.py` (new) — static `ImageModelInfo` catalog.
- `packages/fatih_hoca/src/fatih_hoca/image_select.py` (new) — image scorer.
- `packages/fatih_hoca/src/fatih_hoca/__init__.py:439` — `select()` modality dispatch (image short-circuit before text engine; benchmark enrichment skipped for image entries).

**Modified (pipeline):**
- `src/core/llm_dispatcher.py` — add `CallCategory.IMAGE = "image"` only (telemetry helpers already generic; **no `dispatch()`/`_dispatch_image`** — that method does not exist).
- `packages/husam/src/husam/worker.py` — image modality branch at the top of `run()`; new `_run_image(task, image_call)` with the telemetry envelope (in_flight + `get_dispatcher()._record_pick` + `record_call_tokens/cost`) wrapped in `heartbeat.keepalive()`; calls `paintress.generate`.
- `src/core/orchestrator.py:355-363` — extend `_is_raw` to also match `image_call.raw_dispatch` (routes the task to the existing `husam.run` path; no other orchestrator change).
- `packages/general_beckman/src/general_beckman/__init__.py` — add `_select_for_admission(spec, sel_kwargs)` helper that reads `task.context["failed_models"]`, wraps them as `Failure(...)` for the text path / forwards strings for the image path, and adds the `needs_image=True` branch; wire it at the existing `next_task()` selection site (`:637-650`); add SelectionFailure handling there.
- `src/app/telegram_bot.py` — `/image` command (enqueue + CPS `on_complete`, **no await_inline**) + `image_delivery.resume`/`image_delivery.err` continuation handlers registered in the existing `register_continuations()`; resume handler parses the JSON-stringified result and sends the photo via the module singleton.

**Test infra:**
- root `conftest.py` `_PACKAGE_SRCS` — add `renoir` + `paintress` (same place `safety_guard` was added in commit `ae004547`; `husam` is already registered).
- e2e/integration tests use the **manual pump-cycle** pattern (`_fresh_db` fixture → `add_task`/`enqueue` (no await_inline) → `next_task()` → `husam.run(task)` → `on_task_finished(tid, result)`); see `tests/test_beckman_next_task.py` / `tests/integration/conftest.py::temp_db` for the canonical fixtures.

---

## Task 1: `ImageModelInfo` (independent dataclass — no inheritance)

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/registry.py` (add new class after `ModelInfo`)
- Test: `packages/fatih_hoca/tests/test_image_model_info.py`

The previous plan attempted a `BaseModelInfo` parent class with `location: str = "cloud"`. Per recon (`registry.py:53-56`), `ModelInfo` has FOUR required fields with no defaults at the top, so dataclass inheritance from a defaulted base would raise `TypeError: non-default argument ... follows default argument`. **v2 sidesteps this entirely:** `ImageModelInfo` is an independent dataclass; the dispatcher branches on `isinstance(pick.model, ImageModelInfo)`. No `ModelInfo` refactor. Hot path untouched.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_model_info.py
from fatih_hoca.registry import ModelInfo, ImageModelInfo


def test_image_model_info_basic_fields():
    m = ImageModelInfo(
        name="pollinations/flux", provider="pollinations", location="cloud",
        endpoint="https://image.pollinations.ai/prompt/",
        quality_rank=6.0, cost_per_image=0.0, vram_mb=0, supports_seed=True,
    )
    assert m.name == "pollinations/flux"
    assert m.is_local is False
    assert m.supports_image_generation is True
    assert m.tier == "free"


def test_image_model_info_local_flag():
    m = ImageModelInfo(name="x", provider="clair_obscur", location="local",
                       endpoint="http://127.0.0.1:7860", quality_rank=7.0,
                       cost_per_image=0.0, vram_mb=4000, supports_seed=True)
    assert m.is_local is True


def test_image_and_text_branch_by_isinstance():
    im = ImageModelInfo(name="im", provider="p", location="cloud",
                       endpoint="", quality_rank=5.0)
    tm = ModelInfo(name="tm", location="cloud", provider="p", litellm_name="p/tm")
    assert isinstance(im, ImageModelInfo)
    assert not isinstance(tm, ImageModelInfo)
    assert isinstance(tm, ModelInfo)
    assert not isinstance(im, ModelInfo)


def test_modelinfo_still_constructs_unchanged():
    # Regression — ModelInfo is NOT touched by this task.
    m = ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x")
    assert m.supports_function_calling is False  # default preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_model_info.py -q`
Expected: FAIL — `ImportError: cannot import name 'ImageModelInfo'`.

- [ ] **Step 3: Add `ImageModelInfo` (independent dataclass)**

In `packages/fatih_hoca/src/fatih_hoca/registry.py`, AFTER the closing of `class ModelInfo` (find via `grep -n "^class ModelInfo" registry.py`), insert:

```python
from dataclasses import dataclass


@dataclass
class ImageModelInfo:
    """An image-generation provider/model.

    Independent from ModelInfo by design. Sharing a dataclass parent would
    push defaulted fields above ModelInfo's required (name/location/provider/
    litellm_name) fields and break compile. The dispatcher branches on
    isinstance(pick.model, ImageModelInfo) at the call site.
    """
    name: str
    provider: str
    location: str
    endpoint: str = ""
    api_base: str | None = None
    quality_rank: float = 5.0    # 0-10, hand-set per provider
    cost_per_image: float = 0.0
    vram_mb: int = 0              # local footprint; 0 for cloud
    supports_seed: bool = False
    max_width: int = 1024
    max_height: int = 1024
    is_loaded: bool = False
    tier: str = "free"
    # Carried for telemetry parity with ModelInfo (read by _record_pick).
    litellm_name: str = ""

    @property
    def is_local(self) -> bool:
        return self.location in ("local", "ollama")

    @property
    def supports_image_generation(self) -> bool:
        return True
```

(`dataclass` is already imported at the top of `registry.py` for `ModelInfo`; the import line is illustrative — verify and reuse.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_model_info.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Regression — full fatih_hoca suite**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/ -q -x`
Expected: no NEW failures vs `main` baseline. (Run the same command on `main` first to establish baseline if unsure.)

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/registry.py packages/fatih_hoca/tests/test_image_model_info.py
git commit -m "feat(image): ImageModelInfo (independent dataclass, no inheritance)"
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


def _png(color=(120, 80, 200), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_ok():
    v = assess(_png())
    assert isinstance(v, ImageVerdict) and v.ok is True


def test_not_an_image_rejected():
    v = assess(b"<html>rate limited</html>")
    assert v.ok is False and v.reason == "not_an_image"


def test_empty_rejected():
    v = assess(b"")
    assert v.ok is False and v.reason == "empty"


def test_all_one_color_rejected():
    v = assess(_png(color=(0, 0, 0)))
    assert v.ok is False and v.reason == "blank"


def test_too_small_rejected():
    v = assess(_png(size=(4, 4)))
    assert v.ok is False and v.reason == "too_small"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/renoir/tests/test_assess.py -q`
Expected: FAIL — `ModuleNotFoundError: renoir`.

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

_MIN_DIM = 16


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

    try:
        extrema = img.convert("RGB").getextrema()
        if all(lo == hi for lo, hi in extrema):
            return ImageVerdict(False, "blank")
    except Exception:
        pass

    return ImageVerdict(True, "")
```

- [ ] **Step 4: Register + install**

In root `conftest.py`, add `"renoir"` to the `_PACKAGE_SRCS` list (`grep -n _PACKAGE_SRCS conftest.py` — same list `safety_guard` was added to in commit `ae004547`).

Run: `.venv/Scripts/python -m pip install -e packages/renoir`
Expected: `Successfully installed renoir-0.1.0`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/renoir/tests/test_assess.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add packages/renoir conftest.py
git commit -m "feat(image): renoir image-quality judge package"
```

---

## Task 3: `paintress` types + provider Protocol + dispatch skeleton

**Files:**
- Create: `packages/paintress/pyproject.toml`, `src/paintress/__init__.py`, `types.py`, `providers/__init__.py`, `providers/base.py`
- Test: `packages/paintress/tests/test_dispatch.py`
- Modify: root `conftest.py` (`_PACKAGE_SRCS`)

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_dispatch.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_dispatch.py -q`
Expected: FAIL — `ModuleNotFoundError: paintress`.

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
    seed: int | None = None
    quality_tier: str = "fast"
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

        Raise-tolerant contract: providers SHOULD return on error rather than
        raise, but the caller (paintress.generate) catches and maps to
        ImageResult.error as a backstop.
        """
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

_PROVIDERS: dict = {}  # populated by Tasks 4/5


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
    except Exception as exc:
        return ImageResult(provider=provider, model=getattr(model, "name", ""),
                           error=f"provider_raised:{exc.__class__.__name__}:{exc}")

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
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add packages/paintress conftest.py
git commit -m "feat(image): paintress types + provider Protocol + dispatch"
```

---

## Task 4: `paintress` Pollinations provider

**Files:**
- Create: `packages/paintress/src/paintress/providers/pollinations.py`
- Modify: `packages/paintress/src/paintress/__init__.py` (`_PROVIDERS`)
- Test: `packages/paintress/tests/test_pollinations.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_pollinations.py
import pytest
from paintress.providers.pollinations import PollinationsProvider
from paintress.types import ImageSpec


@pytest.mark.asyncio
async def test_pollinations_builds_url_returns_bytes(monkeypatch):
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
    # quote(safe="") encodes spaces as %20 AND would encode "/" too.
    assert "a%20red%20bicycle" in captured["url"]
    assert "seed=42" in captured["url"]
    assert meta["seed_used"] == 42
    assert prov.available() is True


@pytest.mark.asyncio
async def test_pollinations_escapes_slashes(monkeypatch):
    # Without safe="" the URL would break on prompts containing "/".
    captured = {}
    class _Resp:
        status_code = 200; content = b"\x89PNG"; headers = {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def get(self, url, **kw):
            captured["url"] = url
            return _Resp()
    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    await PollinationsProvider().generate(
        ImageSpec(prompt="ai/ml art", out_dir="/tmp", seed=1)
    )
    assert "/prompt/ai%2Fml%20art" in captured["url"]


@pytest.mark.asyncio
async def test_pollinations_returns_bytes_for_renoir_to_judge(monkeypatch):
    # Provider returns raw response bytes; renoir judges validity.
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
        # safe="" encodes "/" too — pollinations puts the prompt in the path.
        prompt = urllib.parse.quote(spec.prompt or "", safe="")
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

- [ ] **Step 4: Register in `_PROVIDERS`**

In `packages/paintress/src/paintress/__init__.py`, replace `_PROVIDERS: dict = {}` with:
```python
from .providers.pollinations import PollinationsProvider

_PROVIDERS: dict = {
    "pollinations": PollinationsProvider(),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_pollinations.py packages/paintress/tests/test_dispatch.py -q`
Expected: PASS (6 passed). Dispatch test monkeypatches `_PROVIDERS`, so it still works.

- [ ] **Step 6: Commit**

```bash
git add packages/paintress/src/paintress/providers/pollinations.py packages/paintress/src/paintress/__init__.py packages/paintress/tests/test_pollinations.py
git commit -m "feat(image): paintress pollinations provider (URL-safe encoding)"
```

---

## Task 5: `paintress` Hugging Face provider

**Files:**
- Create: `packages/paintress/src/paintress/providers/huggingface.py`
- Modify: `packages/paintress/src/paintress/__init__.py` (`_PROVIDERS`)
- Test: `packages/paintress/tests/test_huggingface.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_huggingface.py
import pytest
from paintress.providers.huggingface import HuggingFaceProvider
from paintress.types import ImageSpec


def test_hf_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert HuggingFaceProvider().available() is False


def test_hf_available_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    assert HuggingFaceProvider().available() is True


@pytest.mark.asyncio
async def test_hf_posts_and_returns_image(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nIMG"
        headers = {"content-type": "image/png"}
        def json(self): return {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    data, meta = await HuggingFaceProvider().generate(
        ImageSpec(prompt="a forest", out_dir="/tmp", seed=99)
    )
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 99


@pytest.mark.asyncio
async def test_hf_503_model_loading_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 503
        content = b'{"error":"Model is loading","estimated_time":20}'
        headers = {"content-type": "application/json"}
        def json(self): return {"error": "Model is loading", "estimated_time": 20}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "model_loading" in str(ei.value)


@pytest.mark.asyncio
async def test_hf_403_gated_raises_auth(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 403; content = b''
        headers = {}
        def json(self): return {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "hf_auth:403" in str(ei.value)
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
        if resp.status_code == 503:
            raise RuntimeError("model_loading: HF FLUX cold start, retry")
        if resp.status_code in (401, 403):
            raise RuntimeError(f"hf_auth:{resp.status_code}")
        resp.raise_for_status()
        return resp.content, {"seed_used": spec.seed}
```

- [ ] **Step 4: Register**

In `__init__.py` `_PROVIDERS`:
```python
from .providers.huggingface import HuggingFaceProvider
_PROVIDERS = {
    "pollinations": PollinationsProvider(),
    "huggingface": HuggingFaceProvider(),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/ -q`
Expected: PASS (10 passed).

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


def test_catalog_has_pollinations_and_hf():
    cat = image_catalog()
    names = {m.name for m in cat}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
    assert all(isinstance(m, ImageModelInfo) for m in cat)


def test_catalog_entries_are_cloud_free():
    cat = {m.name: m for m in image_catalog()}
    p = cat["pollinations/flux"]
    assert p.provider == "pollinations" and not p.is_local
    assert p.cost_per_image == 0.0 and p.tier == "free"
    assert p.supports_image_generation is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers.py -q`
Expected: FAIL — `ModuleNotFoundError: image_providers`.

- [ ] **Step 3: Implement**

`packages/fatih_hoca/src/fatih_hoca/image_providers.py`:
```python
"""Static image-provider catalog. Image providers are NOT in cloud /models
discovery (LLM-only), so they're registered here.

Plan 2 will append the local clair_obscur entry; Plan 1 ships cloud only.
"""
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

- [ ] **Step 4: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers.py -q`
Expected: PASS (2 passed).

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_providers.py packages/fatih_hoca/tests/test_image_providers.py
git commit -m "feat(image): static image-provider catalog"
```

---

## Task 7: fatih_hoca image scorer (`image_select.py`) — with correct `failures` handling

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/image_select.py`
- Test: `packages/fatih_hoca/tests/test_image_select.py`

Cloud-only scorer. Eviction-cost is a stub (no local providers, so it never fires). `failures` is a list of **provider/model names** (strings) — the prior plan compared model objects against names which always missed. v2 enforces string-name comparison and explicitly tests it.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_select.py
from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick, SelectionFailure


def test_picks_highest_quality_available(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.name == "huggingface/flux-schnell"


def test_falls_back_to_pollinations_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = select_image(quality_tier="fast", failures=[], hf_available=False)
    assert pick.model.name == "pollinations/flux"


def test_excludes_failed_name_string():
    # failures is a list of STRING names — not model objects.
    pick = select_image(quality_tier="fast",
                        failures=["pollinations/flux"], hf_available=False)
    assert isinstance(pick, SelectionFailure)
    assert pick.reason == "availability"


def test_failed_pollinations_falls_to_hf():
    pick = select_image(quality_tier="fast",
                        failures=["pollinations/flux"], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_failures_with_unknown_names_does_not_crash():
    # Spurious failure entries are ignored without raising.
    pick = select_image(quality_tier="fast",
                        failures=["something/unrelated"], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_pick_carries_top_summary():
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert "huggingface/flux-schnell" in pick.top_summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select.py -q`
Expected: FAIL — `ModuleNotFoundError: image_select`.

- [ ] **Step 3: Implement**

`packages/fatih_hoca/src/fatih_hoca/image_select.py`:
```python
"""Purpose-built image-model scorer. Sibling to selector.py (text). Cloud-only
in Plan 1; eviction-cost is a stub and never fires because the catalog has no
local entries (Plan 2 adds it)."""
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
    return False  # local providers ineligible in Plan 1


def _eviction_cost(m: ImageModelInfo) -> float:
    """Stub — replaced in Plan 2 with the real formula reading nerd_herd.
    Plan 1 is cloud-only; cloud providers always score 0 here."""
    return 0.0


def select_image(
    *,
    quality_tier: str = "fast",
    failures: list[str] | None = None,
    hf_available: bool | None = None,
    remaining_budget_usd: float | None = None,
) -> Pick | SelectionFailure:
    # IMPORTANT: failures is a list of STRING provider/model names. Callers that
    # have rich Failure objects must extract .model.name (or equivalent) BEFORE
    # passing here. See fatih_hoca.select() dispatch for the conversion.
    failed = set(failures or [])
    candidates: list[tuple[float, ImageModelInfo]] = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        score = m.quality_rank - _eviction_cost(m)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_summary = "; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5])
    best_score, best = candidates[0]
    return Pick(model=best, min_time_seconds=0.0, score=best_score, top_summary=top_summary)
```

- [ ] **Step 4: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select.py -q`
Expected: PASS (6 passed).

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_select.py packages/fatih_hoca/tests/test_image_select.py
git commit -m "feat(image): hoca image scorer with name-based failures handling"
```

---

## Task 8: `select(needs_image=True)` dispatch + benchmark-enrichment guard

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py:439`
- Test: `packages/fatih_hoca/tests/test_select_image_dispatch.py`

The image branch short-circuits BEFORE the text engine. It converts a caller-supplied `failures` list (which may contain `Failure` objects from the text retry world, OR plain strings from beckman's image retry) to a uniform list of name strings before handing to `select_image`. Benchmark-enrichment must NOT be invoked on `ImageModelInfo` entries — verified by a side test.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_select_image_dispatch.py
import fatih_hoca
from fatih_hoca.types import Pick
from fatih_hoca.registry import ImageModelInfo, ModelInfo


def test_select_needs_image_returns_image_pick(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    assert isinstance(pick, Pick)
    assert isinstance(pick.model, ImageModelInfo)
    assert pick.model.name == "pollinations/flux"


def test_select_needs_image_accepts_failure_objects():
    # Caller may pass Failure-like objects with .model.name (text retry shape).
    class _F:
        class model:
            name = "pollinations/flux"
    pick = fatih_hoca.select(needs_image=True, failures=[_F()])
    # With pollinations excluded by failures and no HF_TOKEN, scorer returns SelectionFailure.
    from fatih_hoca.types import SelectionFailure
    assert isinstance(pick, SelectionFailure)


def test_select_needs_image_accepts_plain_name_strings():
    from fatih_hoca.types import SelectionFailure
    pick = fatih_hoca.select(needs_image=True, failures=["pollinations/flux"])
    assert isinstance(pick, SelectionFailure)


def test_text_select_unchanged(monkeypatch):
    # Regression: non-image callers see no behavior change.
    res = fatih_hoca.select(task="router", agent_type="router", difficulty=3)
    if res is not None and hasattr(res, "model"):
        assert not isinstance(res.model, ImageModelInfo)


def test_benchmark_enrichment_skips_image_entries():
    # Verifies the guard: enrichment must not crash on entries that look like
    # ImageModelInfo (no benchmark fields). We invoke the guard helper directly
    # so the test doesn't depend on a full hoca init().
    from fatih_hoca.image_providers import image_catalog
    from fatih_hoca import _is_image_entry
    for m in image_catalog():
        assert _is_image_entry(m) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_select_image_dispatch.py -q`
Expected: FAIL — `TypeError: select() got an unexpected keyword argument 'needs_image'` and `ImportError: cannot import name '_is_image_entry'`.

- [ ] **Step 3: Add dispatch branch + enrichment guard**

In `packages/fatih_hoca/src/fatih_hoca/__init__.py`, BEFORE the existing `def select(**kwargs):` body, add:

```python
def _is_image_entry(m) -> bool:
    """True if m is an image-modality catalog entry; used by benchmark
    enrichment to skip image rows (which lack LLM benchmark fields)."""
    from .registry import ImageModelInfo
    return isinstance(m, ImageModelInfo)
```

Then inside `select(**kwargs)`, at the very top (before any text-engine work), insert:

```python
def select(**kwargs):
    # Image modality short-circuit. Runs BEFORE any text-model machinery.
    if kwargs.pop("needs_image", False):
        from .image_select import select_image
        raw_failures = kwargs.get("failures") or []
        # Normalize to plain name strings. Callers may pass:
        #   - plain strings (beckman image branch forwards context.failed_models)
        #   - real fatih_hoca Failure dataclasses where `.model` is a STRING
        #     (types.py:29 — NOT `.model.name`)
        #   - hypothetical objects with `.model.name`
        # Handle all three; never raise on a spurious entry.
        failures: list[str] = []
        for f in raw_failures:
            if isinstance(f, str):
                name = f
            else:
                m = getattr(f, "model", None)
                name = m if isinstance(m, str) else getattr(m, "name", None)
            if name:
                failures.append(name)
        return select_image(
            quality_tier=kwargs.get("quality_tier", "fast"),
            failures=failures,
            remaining_budget_usd=kwargs.get("remaining_budget_usd"),
        )
    # ... existing text-selection body unchanged ...
```

> **Note (v3):** the real `Failure` dataclass stores `model` as a STRING (`packages/fatih_hoca/src/fatih_hoca/types.py:29`). The normalize above handles the string case first so a real `Failure` resolves to `f.model` directly; the `getattr(m, "name", None)` arm only covers the synthetic `_F.model.name` test object and is harmless for strings.

Also find the benchmark-enrichment site (`grep -n "benchmark" packages/fatih_hoca/src/fatih_hoca/__init__.py` — the call that iterates the catalog at `init()`). Wrap the iteration with a guard: `if _is_image_entry(m): continue` so image entries never enter the LLM benchmark pipeline. If the enrichment is inside a registry method rather than `__init__`, place the guard at the relevant iteration site and document the exact path in the commit.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_select_image_dispatch.py packages/fatih_hoca/tests/test_image_model_info.py packages/fatih_hoca/tests/test_image_providers.py packages/fatih_hoca/tests/test_image_select.py -q`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/ -q -x`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_select_image_dispatch.py
git commit -m "feat(image): select(needs_image=True) dispatch + benchmark-enrichment guard"
```

---

## Task 9: Shared inter-task failures-propagation in beckman.next_task

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (admission flow)
- Test: `packages/general_beckman/tests/test_failures_propagation.py`

**Why this is its own task and why it benefits text too:** Recon discovered the LLM path also lacks inter-task failures-propagation. `on_task_finished` writes `task.context["failed_models"]` (orchestrator `:824-828`) but `next_task()` never reads it back when re-admitting a retried task. Without this, a re-admitted text task can re-pick the just-failed model; for image tasks (which retry single-shot via beckman, no ReAct), the missing propagation is fatal. v2 fixes it at the beckman level — both text and image retries get the win.

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_failures_propagation.py
import pytest


def test_admission_select_forwards_failed_models(monkeypatch):
    """next_task admission must read task.context['failed_models'] and pass
    them as failures= to fatih_hoca.select. Applies to text AND image tasks."""
    import general_beckman as gb
    captured = {}

    def _fake_select(**kw):
        captured.update(kw)
        from fatih_hoca.types import Pick
        from fatih_hoca.registry import ModelInfo
        return Pick(
            model=ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x"),
            min_time_seconds=0.0,
        )
    monkeypatch.setattr("fatih_hoca.select", _fake_select)

    spec = {
        "kind": "main_work",
        "agent_type": "coder",
        "context": {"failed_models": ["groq/oss-120b", "gemini/2.5-flash"]},
    }
    gb._select_for_admission(spec)  # helper added in this task
    raw = captured.get("failures") or []
    # TEXT path forwards real Failure dataclasses where `.model` is a STRING
    # (types.py:29). Extract the name from `.model`, not `.model.name`.
    names = []
    for f in raw:
        n = getattr(f, "model", None) if not isinstance(f, str) else f
        if isinstance(n, str) and n:
            names.append(n)
    assert "groq/oss-120b" in names
    assert "gemini/2.5-flash" in names
    # And they must be Failure objects, not bare strings (the bug v3 fixes).
    from fatih_hoca.types import Failure
    assert all(isinstance(f, Failure) for f in raw)


def test_admission_select_no_failed_models_passes_empty(monkeypatch):
    import general_beckman as gb

    def _fake_select(**kw):
        # Should still receive failures=[] (not None) for the image branch's normalize.
        assert "failures" in kw
        assert kw["failures"] == []
        from fatih_hoca.types import Pick
        from fatih_hoca.registry import ModelInfo
        return Pick(
            model=ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x"),
            min_time_seconds=0.0,
        )
    monkeypatch.setattr("fatih_hoca.select", _fake_select)
    gb._select_for_admission({"kind": "main_work", "agent_type": "coder",
                              "context": {}})
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_failures_propagation.py -q`
Expected: FAIL — `AttributeError: _select_for_admission` (extracted in next step).

- [ ] **Step 3: Add `_select_for_admission` helper that reads failed_models — without regressing text selection**

> **v3 correction:** the v2 helper rebuilt the text-selection kwargs from scratch, **dropping** `call_category`, `needs_thinking`, and `prefer_speed` that `next_task()` already computes richly at `:637-650`. That is a silent text-selection regression. v3's helper **accepts the kwargs `next_task()` already built** and only layers on `failures` + the image branch. And it wraps failed-model **names as `Failure(...)` objects** for the text path (the selector does `{f.model for f in failures}` where `.model` is a string — passing raw strings crashes; passing `Failure(model=name)` works).

In `packages/general_beckman/src/general_beckman/__init__.py`, add the helper:

```python
def _select_for_admission(spec: dict, sel_kwargs: dict | None = None):
    """Single admission-time selection point. Reads failed_models from the task
    context and forwards them so a re-admitted retry never re-picks the
    just-failed provider. Applies to text AND image tasks.

    `sel_kwargs` is the rich text-selection kwargs dict next_task() already
    computed (task/agent_type/difficulty/urgency/call_category/needs_thinking/
    prefer_speed/est tokens). For the TEXT path we preserve it verbatim and only
    add `failures`. For the IMAGE path we ignore it and call the image scorer.
    """
    import fatih_hoca
    from fatih_hoca.types import Failure

    ctx = spec.get("context") or {}
    if isinstance(ctx, str):
        import json as _json
        try:
            ctx = _json.loads(ctx) or {}
        except Exception:
            ctx = {}
    failed_models = list(ctx.get("failed_models") or [])

    is_image = bool(ctx.get("image_call")) or spec.get("kind") == "image"
    if is_image:
        ic = ctx.get("image_call") or {}
        # Image scorer takes plain name strings (select() normalizes); forward as-is.
        return fatih_hoca.select(
            needs_image=True,
            quality_tier=ic.get("quality_tier", "fast"),
            failures=failed_models,
        )

    # Text path: preserve next_task()'s rich kwargs; only add failures.
    kw = dict(sel_kwargs or {})
    kw.setdefault("agent_type", spec.get("agent_type", ""))
    kw.setdefault("task", kw.get("agent_type", spec.get("agent_type", "")))
    # CRITICAL: selector does `{f.model for f in failures}` — `.model` is a
    # STRING field on Failure (types.py:29). Raw strings would crash; wrap them.
    kw["failures"] = [Failure(model=n, reason="prior_admission") for n in failed_models]
    return fatih_hoca.select(**kw)
```

Then at the existing `next_task()` selection site (`grep -n "fatih_hoca.select" packages/general_beckman/src/general_beckman/__init__.py` — currently `pick = fatih_hoca.select(**_sel_kwargs)` around `:650`), REPLACE that single call with:
```python
pick = _select_for_admission(task, _sel_kwargs)
```
(`task` is the candidate row being admitted in the `for task in candidates:` loop — it carries `context`, `kind`, `agent_type`; `_sel_kwargs` is the dict already built just above. Do NOT delete the `_sel_kwargs` construction — the helper consumes it.)

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_failures_propagation.py -q`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/ -q -x`
Expected: no new failures. (Existing tests that hit `next_task` should keep working because `_select_for_admission` preserves the prior text-path kwargs.)

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_failures_propagation.py
git commit -m "feat(beckman): admission propagates failed_models to hoca (text + image)"
```

---

## Task 10: Beckman SelectionFailure handler at admission

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (admission flow, around the `_select_for_admission` call site)
- Test: `packages/general_beckman/tests/test_admission_selection_failure.py`

Recon confirmed: beckman has **no** SelectionFailure handler today; only the dispatcher catches it (`src/core/llm_dispatcher.py:412-431`). For an image task whose admission can't find any eligible provider, beckman receives a `SelectionFailure` from `_select_for_admission` and must surface it cleanly — mark the task failed with the failure reason — instead of crashing on `pick.model.name`.

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_admission_selection_failure.py
import pytest


@pytest.mark.asyncio
async def test_selection_failure_marks_task_failed(monkeypatch):
    """When _select_for_admission returns SelectionFailure, the task must
    be marked failed with status='failed' and a clear error — not crash."""
    import general_beckman as gb
    from fatih_hoca.types import SelectionFailure

    monkeypatch.setattr(
        "general_beckman._select_for_admission",
        lambda spec: SelectionFailure(reason="availability",
                                      detail="no eligible image provider"),
    )
    spec = {"kind": "image", "agent_type": "image",
            "context": {"image_call": {"prompt": "x"}}}
    outcome = await gb._handle_admission_pick(spec, pick=None)
    assert outcome["status"] == "failed"
    assert "availability" in outcome.get("error", "")


@pytest.mark.asyncio
async def test_budget_failure_marks_paused(monkeypatch):
    import general_beckman as gb
    from fatih_hoca.types import SelectionFailure

    monkeypatch.setattr(
        "general_beckman._select_for_admission",
        lambda spec: SelectionFailure(reason="budget", detail="exceeded"),
    )
    spec = {"kind": "image", "mission_id": 7,
            "context": {"image_call": {"prompt": "x"}}}
    outcome = await gb._handle_admission_pick(spec, pick=None)
    assert outcome["status"] in ("paused", "failed")  # either is fine; "paused" preferred
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_admission_selection_failure.py -q`
Expected: FAIL — `AttributeError: _handle_admission_pick`.

- [ ] **Step 3: Add the handler**

In `packages/general_beckman/src/general_beckman/__init__.py`, add a small handler function next to `_select_for_admission`:

```python
async def _handle_admission_pick(spec: dict, pick) -> dict:
    """Convert a SelectionFailure (or a None pick) into a task-status outcome
    instead of letting beckman crash downstream on pick.model.name.

    Returns {'status': 'failed'|'paused'|'ok', 'error': str|None, 'pick': Pick|None}.
    """
    from fatih_hoca.types import SelectionFailure
    import fatih_hoca

    if pick is None:
        pick = _select_for_admission(spec)

    if isinstance(pick, SelectionFailure):
        # Budget failures pause the mission (mirror dispatcher's behavior).
        if pick.reason == "budget":
            try:
                from general_beckman.lifecycle_events import emit_pause
                mid = spec.get("mission_id")
                if mid is not None:
                    await emit_pause(int(mid),
                                     reason="no_model_fits_budget",
                                     triggered_by="auto:budget")
                return {"status": "paused", "error": f"{pick.reason}:{pick.detail}",
                        "pick": None}
            except Exception:
                pass
        return {"status": "failed",
                "error": f"selection_failure:{pick.reason}:{pick.detail}",
                "pick": None}

    if pick is None:
        return {"status": "failed",
                "error": "selection_failure:no_pick", "pick": None}

    return {"status": "ok", "error": None, "pick": pick}
```

At the existing call site (where `_select_for_admission` was wired in Task 9), pass the **already-computed pick** so prod never re-selects (the `pick is None` fallback inside the helper exists only for the unit test):
```python
pick = _select_for_admission(task, _sel_kwargs)   # Task 9
_outcome = await _handle_admission_pick(task, pick=pick)
if _outcome["status"] != "ok":
    # Mark task failed/paused; do NOT proceed to attach preselected_pick.
    await _mark_admission_failed(task, _outcome["status"], _outcome["error"])
    continue  # match the surrounding `for task in candidates:` loop semantics
pick = _outcome["pick"]
task["preselected_pick"] = pick
```

**`_mark_admission_failed`** — no such helper exists today (recon confirmed). Reuse the real terminal-write path: `_dlq_write` from `general_beckman.apply` (the pattern used at `:461-466`) for the `failed` case, and `update_task(task["id"], status="paused", error=...)` for `paused`. Concretely:
```python
async def _mark_admission_failed(task: dict, status: str, error: str) -> None:
    if status == "failed":
        from general_beckman.apply import _dlq_write
        await _dlq_write(task, error=error, category="availability",
                         attempts=int(task.get("worker_attempts") or 0) + 1)
    else:  # paused
        from src.infra.db import update_task
        await update_task(task["id"], status=status, error=error[:500])
```
(Confirm `_dlq_write`'s current kwargs with `grep -n "def _dlq_write" packages/general_beckman/src/general_beckman/apply.py` before wiring.)

- [ ] **Step 4: Run tests + regression**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_admission_selection_failure.py packages/general_beckman/tests/test_failures_propagation.py -q`
Expected: PASS.

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/ -q -x`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_admission_selection_failure.py
git commit -m "feat(beckman): SelectionFailure handler at admission (image safe)"
```

---

## Task 11: Husam image branch with FULL telemetry

**Files:**
- Modify: `src/core/llm_dispatcher.py` (add `CallCategory.IMAGE` only)
- Modify: `packages/husam/src/husam/worker.py` (image branch at top of `run()`; new `_run_image`)
- Test: `packages/husam/tests/test_husam_image.py`

> **v3 correction:** the image lane lives in **husam**, not a (non-existent) `dispatcher.dispatch()`. Husam is the raw-dispatch worker the orchestrator already routes to (`orchestrator.py:364-377`), and the only caller (besides coulson) permitted to touch the dispatcher. The text path calls `dispatcher.execute()`, whose telemetry envelope is `begin_call → … → _record_pick(success/failure) → end_call(finally)`; token/cost rows land inside `hallederiz_kadir`. The image path **bypasses** `dispatcher.execute` (it calls `paintress.generate`), so `_run_image` must reproduce that envelope itself: `in_flight.begin_call` → `paintress.generate` (wrapped in `heartbeat.keepalive()`) → `get_dispatcher()._record_pick` (success/failure) → `in_flight.end_call` (finally) → `db.record_call_tokens` (zero tokens) → `db.record_call_cost`. `_record_pick` is the thin delegator the 2026-06-05 de-accretion kept precisely as the monkeypatch surface — husam reuses it. task_id comes from the `heartbeat.current_task_id` contextvar (same as `execute()`), since the orchestrator does not pass a task id into `husam.run`.

- [ ] **Step 1: Write the failing test**

```python
# packages/husam/tests/test_husam_image.py
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
    """husam.run on an image task must (a) return the path dict and (b) write
    pick_log + begin_call/end_call + record_call_tokens. Telemetry holes are
    the bug this test exists to catch."""
    import husam
    from src.core.llm_dispatcher import get_dispatcher, CallCategory

    pick = _img_pick(tmp_path)

    async def _fake_generate(p, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"), provider="pollinations",
                           model="pollinations/flux", cost=0.0, seed_used=5,
                           latency=0.1)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    tele = {"begin": 0, "end": 0, "pick_log": 0, "tokens": 0, "cost": 0}

    async def _bc(**kw):
        tele["begin"] += 1
        assert kw["category"] == "image"
        assert kw["model_name"] == "pollinations/flux"
        return "call-1"
    async def _ec(call_id): tele["end"] += 1
    # husam._run_image lazy-imports begin_call/end_call from src.core.in_flight —
    # monkeypatch at the source module so the call-time lookup resolves the fake.
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
    # _record_pick is the thin delegator husam borrows; patch it on the singleton.
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
    # Free provider (cost == 0.0) → record_call_cost is guarded out, no cost row.
    assert tele["cost"] == 0


@pytest.mark.asyncio
async def test_husam_image_failure_writes_pick_log_and_ends_call(monkeypatch, tmp_path):
    """On paintress error, pick_log fires success=False, end_call runs in finally,
    husam raises ModelCallFailed."""
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/husam/tests/test_husam_image.py -q`
Expected: FAIL — `AttributeError: IMAGE` (no `CallCategory.IMAGE`) and/or husam runs the text path on the image task (no telemetry asserts hit).

- [ ] **Step 3a: Add `CallCategory.IMAGE` to the enum**

In `src/core/llm_dispatcher.py`, extend `CallCategory` (after `OVERHEAD`):
```python
    IMAGE = "image"     # Image generation via husam → paintress (single-shot)
```
That is the ONLY change to `llm_dispatcher.py`. (Do NOT add a `dispatch()`/`_dispatch_image` — they don't exist; image execution lives in husam.)

- [ ] **Step 3b: Add the image branch + `_run_image` to `husam.worker`**

In `packages/husam/src/husam/worker.py`, at the **very top of `run()`** (before the text-only `llm_call` rehydration), add the modality detection:
```python
async def run(task: dict) -> dict:
    import json as _json
    _ctx = task.get("context")
    if isinstance(_ctx, str):
        try:
            _ctx = _json.loads(_ctx)
        except Exception:
            _ctx = {}
    _image_call = _ctx.get("image_call") if isinstance(_ctx, dict) else None
    if isinstance(_image_call, dict) and _image_call.get("raw_dispatch"):
        return await _run_image({**task, "context": _ctx}, _image_call)
    # ── existing text path unchanged below ──
    ...
```

Then add `_run_image` (module-level function in `worker.py`). It reproduces `dispatcher.execute()`'s telemetry envelope but calls paintress. Husam is permitted to reach the dispatcher, but here it only borrows the `_record_pick` delegator:
```python
async def _run_image(task: dict, image_call: dict) -> dict:
    """Image-modality lane. Single-shot. Mirrors dispatcher.execute()'s telemetry
    envelope (begin_call → call → _record_pick success/failure → end_call finally
    → record_call_tokens → record_call_cost) but calls paintress, not the LLM
    dispatcher primitive. All hooks are generic per recon: begin/end_call accept
    any category/model strings; _record_pick reads only pick.model.name/.is_local/
    .provider; record_call_tokens tolerates zero tokens; record_call_cost no-ops
    on zero (and we guard it out for free providers)."""
    import time as _time
    import paintress
    import fatih_hoca
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import SelectionFailure
    from src.core.in_flight import begin_call, end_call
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    from src.core.router import ModelCallFailed
    from src.core import heartbeat as _hb

    # Honor Beckman's admission pick; only re-select if it's missing/wrong-modality.
    pick = task.get("preselected_pick")
    if pick is None or not isinstance(getattr(pick, "model", None), ImageModelInfo):
        ctx = task.get("context") if isinstance(task.get("context"), dict) else {}
        pick = fatih_hoca.select(
            needs_image=True,
            quality_tier=image_call.get("quality_tier", "fast"),
            failures=list((ctx or {}).get("failed_models") or []),
        )
    if pick is None or isinstance(pick, SelectionFailure):
        raise ModelCallFailed(call_id="image",
                              last_error="no eligible image provider",
                              error_category="availability")
    model = pick.model

    # task_id from the heartbeat contextvar (orchestrator sets it, same as execute()).
    task_id = None
    try:
        from src.core.heartbeat import current_task_id
        task_id = current_task_id.get()
    except Exception:
        pass

    agent_type = image_call.get("agent_type", "image")
    difficulty = int(image_call.get("difficulty", 5) or 5)
    disp = get_dispatcher()

    call_id = await begin_call(
        category=CallCategory.IMAGE.value,
        model_name=getattr(model, "name", ""),
        provider=getattr(model, "provider", ""),
        is_local=bool(getattr(model, "is_local", False)),
        task_id=task_id,
        est_tokens=0,
    )

    started = _time.time()
    result = None
    try:
        s = paintress.ImageSpec(
            prompt=image_call.get("prompt", ""),
            out_dir=image_call.get("out_dir") or ".",
            negative_prompt=image_call.get("negative_prompt"),
            width=int(image_call.get("width", 1024)),
            height=int(image_call.get("height", 1024)),
            seed=image_call.get("seed"),
            quality_tier=image_call.get("quality_tier", "fast"),
            filename_hint=image_call.get("filename_hint"),
        )
        try:
            async with _hb.keepalive():
                result = await paintress.generate(pick, s)
        except Exception as exc:
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=False, error_category="raw_exception",
                                    agent_type=agent_type, difficulty=difficulty)
            raise ModelCallFailed(call_id=getattr(model, "name", "image"),
                                  last_error=f"{exc.__class__.__name__}:{exc}",
                                  error_category="raw_exception")
        if result.error is None:
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=True, error_category="",
                                    agent_type=agent_type, difficulty=difficulty)
        else:
            # unknown_provider is fatal; quality/provider failures are retryable.
            err_cat = "fatal" if result.error.startswith("unknown_provider") else "availability"
            await disp._record_pick(pick=pick, task="image", category=CallCategory.IMAGE,
                                    success=False, error_category=err_cat,
                                    agent_type=agent_type, difficulty=difficulty)
            raise ModelCallFailed(call_id=getattr(model, "name", "image"),
                                  last_error=result.error, error_category=err_cat)
    finally:
        await end_call(call_id)

    # Per-call token telemetry (zero tokens for image; row exists for rollup).
    duration_ms = int((_time.time() - started) * 1000)
    try:
        from src.infra.db import record_call_tokens, record_call_cost
        await record_call_tokens(
            task_id=task_id, agent_type=agent_type,
            workflow_step_id=image_call.get("workflow_step_id"),
            workflow_phase=image_call.get("workflow_phase"),
            call_category=CallCategory.IMAGE.value,
            model=getattr(model, "name", ""), provider=getattr(model, "provider", ""),
            is_streaming=False, prompt_tokens=0, completion_tokens=0,
            reasoning_tokens=0, total_tokens=0, duration_ms=duration_ms,
            iteration_n=int(image_call.get("iteration_n", 0) or 0), success=True,
        )
        if result.cost > 0.0 and task_id is not None:
            await record_call_cost(task_id, float(result.cost))
    except Exception:
        pass  # telemetry best-effort

    return {
        "content": result.path, "path": result.path, "provider": result.provider,
        "model": result.model, "cost": result.cost, "latency": result.latency,
        "seed_used": result.seed_used,
        "is_local": getattr(model, "is_local", False),
        "ran_on": result.provider,
    }
```

> **Confirm at impl time:** husam's existing `ModelCallFailed(...)` raise (`grep -n "ModelCallFailed" packages/husam/src/husam/worker.py`) for the exact kwarg names — match them (the text path already raises it). `_record_pick`'s `category` param is typed `CallCategory`; `CallCategory.IMAGE` satisfies it and `.value == "image"` lands in the `model_pick_log.category` column.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/husam/tests/test_husam_image.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression — husam suite**

Run: `.venv/Scripts/python -m pytest packages/husam/tests/ -q -x`
Expected: no new failures vs `main` (the text path is untouched — the image branch only fires on `context.image_call.raw_dispatch`).

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py packages/husam/src/husam/worker.py packages/husam/tests/test_husam_image.py
git commit -m "feat(image): husam image branch (_run_image) with full telemetry envelope"
```

---

## Task 12: Orchestrator routes the image lane (to husam)

**Files:**
- Modify: `src/core/orchestrator.py:355-363` (`_is_raw` inside `_dispatch`'s `_run`)
- Test: `tests/core/test_orchestrator_image_route.py`

Recon confirmed: when `_is_raw` is True, `_dispatch` calls **`husam.run(...)`** (`orchestrator.py:364-377`) — not a dispatcher. Extending `_is_raw` to also match `image_call.raw_dispatch` is the ONLY orchestrator change; husam (Task 11) detects `context.image_call` and runs the image lane.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_orchestrator_image_route.py
import json
import pytest


@pytest.mark.asyncio
async def test_image_call_routes_to_husam(monkeypatch):
    """_dispatch must detect image_call.raw_dispatch and route the task through
    husam.run (the SAME raw path llm_call.raw_dispatch uses) — there is no
    dispatcher.dispatch()."""
    import src.core.orchestrator as orch
    import husam

    seen = {}
    async def _fake_husam_run(spec):
        seen["spec"] = spec
        return {"path": "/tmp/x.png", "provider": "pollinations",
                "cost": 0.0, "latency": 0.1, "seed_used": 1}
    monkeypatch.setattr(husam, "run", _fake_husam_run)

    finished = {}
    async def _fake_finish(task_id, result=None):
        finished["task_id"] = task_id
        finished["result"] = result
    monkeypatch.setattr("general_beckman.on_task_finished", _fake_finish)

    o = orch.Orchestrator.__new__(orch.Orchestrator)
    task = {
        "id": 1, "kind": "image",
        "context": {"image_call": {"raw_dispatch": True, "prompt": "a dog"}},
        "preselected_pick": object(),
    }
    await o._dispatch(task)

    # husam saw the image context → _is_raw matched image_call.
    assert seen["spec"]["context"]["image_call"]["prompt"] == "a dog"
    # The husam dict round-tripped to on_task_finished, JSON-stringified under "result".
    raw = (finished["result"] or {}).get("result")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["path"] == "/tmp/x.png"
```

> If the bare `Orchestrator.__new__` + `_dispatch` call trips on context-injection or metric helpers, monkeypatch those the way the existing raw-dispatch orchestrator test does (`grep -n "raw_dispatch" tests/core/ tests/` for the closest precedent) — the assertion that matters is "husam.run saw the image context."

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_orchestrator_image_route.py -q`
Expected: FAIL — husam.run not invoked for image_call (the text `_is_raw` ignores `image_call`).

- [ ] **Step 3: Extend the `_is_raw` branch**

In `src/core/orchestrator.py` (around `:355-363`, inside `_dispatch`'s nested `_run`), locate:
```python
        _llm_call_rd = _ctx_rd.get("llm_call") if isinstance(_ctx_rd, dict) else None
        _is_raw = isinstance(_llm_call_rd, dict) and _llm_call_rd.get("raw_dispatch") is True
```
and REPLACE with:
```python
        _llm_call_rd = _ctx_rd.get("llm_call") if isinstance(_ctx_rd, dict) else None
        _img_call_rd = _ctx_rd.get("image_call") if isinstance(_ctx_rd, dict) else None
        _llm_rd = isinstance(_llm_call_rd, dict) and _llm_call_rd.get("raw_dispatch") is True
        _img_rd = isinstance(_img_call_rd, dict) and _img_call_rd.get("raw_dispatch") is True
        _is_raw = _llm_rd or _img_rd
```
The existing `husam.run({"context": _ctx_rd, "kind": ..., "preselected_pick": ..., "mission_id": ...})` call already forwards `context` + `preselected_pick`; husam (Task 11) detects `context.image_call` → `_run_image`. No further orchestrator change needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_orchestrator_image_route.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/core/test_orchestrator_image_route.py
git commit -m "feat(image): orchestrator routes image_call.raw_dispatch to husam"
```

---

## Task 13: `/image` Telegram command — CPS delivery (no await_inline)

**Files:**
- Modify: `src/app/telegram_bot.py` (`_setup_handlers` + new `cmd_image`; `_image_delivery_resume`/`_image_delivery_err` handlers; register them in the existing `register_continuations()`; add a `_send_telegram_photo_via_resume` helper)
- Test: `tests/app/test_cmd_image.py`

> **v3 correction:** **no `await_inline`.** `/image` enqueues the image task and returns immediately; the photo is delivered by a durable **CPS `on_complete` continuation** — exactly the SP2 pattern `_handle_casual` / `_casual_reply_resume` already use (`telegram_bot.py:8346-8370`, `:356-384`). The resume handler runs with no live `Update`, so it reaches the bot via the module singleton (`get_telegram()`, `:304-353`) and receives `chat_id` from `cont_state`. Recon confirmed the orchestrator JSON-stringifies husam's return (`orchestrator.py:378-382`), so the continuation's `result["result"]` is a JSON **string** — the handler parses it (and tolerates the flat restart-reconcile shape), mirroring `_casual_reply_resume`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_cmd_image.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cmd_image_enqueues_cps_no_await_inline(monkeypatch):
    """/image must enqueue with the CPS on_complete/on_error continuation and
    carry chat_id + prompt in cont_state. It must NOT use await_inline."""
    import src.app.telegram_bot as tb

    enq = AsyncMock(return_value=123)  # enqueue returns a task_id (no await_inline)
    monkeypatch.setattr("general_beckman.enqueue", enq)

    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    update = MagicMock()
    update.effective_chat.id = 99
    ctx = MagicMock(); ctx.args = ["a", "red", "bicycle"]

    await iface.cmd_image(update, ctx)

    assert enq.await_count == 1
    spec = enq.await_args.args[0]
    kwargs = enq.await_args.kwargs
    assert spec["context"]["image_call"]["prompt"] == "a red bicycle"
    assert spec["context"]["image_call"]["raw_dispatch"] is True
    assert spec["kind"] == "image"
    assert kwargs["on_complete"] == "image_delivery.resume"
    assert kwargs["on_error"] == "image_delivery.err"
    assert kwargs["cont_state"]["chat_id"] == 99
    assert kwargs["cont_state"]["prompt"] == "a red bicycle"
    assert "await_inline" not in kwargs  # explicitly NOT used


@pytest.mark.asyncio
async def test_image_delivery_resume_parses_json_string_and_sends_photo(monkeypatch, tmp_path):
    """CPS resume: result['result'] is a JSON STRING (orchestrator stringified
    husam's dict). Handler must json.loads it, then send the photo via the
    module singleton — there is no live Update in a resume."""
    import src.app.telegram_bot as tb

    png = tmp_path / "out.png"; png.write_bytes(b"\x89PNG")

    sent = {}
    class _Bot:
        async def send_photo(self, chat_id, photo, caption=None):
            sent["chat_id"] = chat_id
            sent["bytes"] = photo.read()
            sent["caption"] = caption
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())

    result = {"result": json.dumps({"path": str(png), "provider": "pollinations"})}
    await tb._image_delivery_resume(123, result, {"chat_id": 99, "prompt": "a red bicycle"})

    assert sent["chat_id"] == 99
    assert sent["bytes"].startswith(b"\x89PNG")
    assert "bicycle" in (sent["caption"] or "")


@pytest.mark.asyncio
async def test_image_delivery_resume_handles_dict_result_too(monkeypatch, tmp_path):
    """Defensive: tolerate an already-dict result (flat restart-reconcile shape)."""
    import src.app.telegram_bot as tb
    png = tmp_path / "out.png"; png.write_bytes(b"\x89PNG")

    sent = {}
    class _Bot:
        async def send_photo(self, chat_id, photo, caption=None):
            sent["ok"] = True
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())

    await tb._image_delivery_resume(123, {"path": str(png)}, {"chat_id": 99, "prompt": "x"})
    assert sent.get("ok") is True


@pytest.mark.asyncio
async def test_image_delivery_err_reports_failure(monkeypatch):
    import src.app.telegram_bot as tb
    msgs = {}
    class _Bot:
        async def send_message(self, chat_id, text, **kw):
            msgs["chat_id"] = chat_id; msgs["text"] = text
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())

    await tb._image_delivery_err(123, {"error": "selection_failure:availability"},
                                 {"chat_id": 99})
    assert msgs["chat_id"] == 99
    assert "failed" in msgs["text"].lower() or "❌" in msgs["text"]


@pytest.mark.asyncio
async def test_cmd_image_no_args_replies_usage():
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    update = MagicMock(); ctx = MagicMock(); ctx.args = []
    await iface.cmd_image(update, ctx)
    msg = iface._reply.await_args.args[1]
    assert "Usage" in msg or "/image" in msg
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/app/test_cmd_image.py -q`
Expected: FAIL — `AttributeError: cmd_image` / `_image_delivery_resume`.

- [ ] **Step 3: Register the command handler + the CPS continuation handlers**

In `_setup_handlers()` (telegram_bot.py, near the other `CommandHandler(...)` lines — `grep -n 'CommandHandler("' src/app/telegram_bot.py`):
```python
        self.app.add_handler(CommandHandler("image", self.cmd_image))
```

In the module-level `register_continuations()` (telegram_bot.py:472-490, where `telegram.casual_reply_resume` is registered), add:
```python
        register_resume("image_delivery.resume", _image_delivery_resume)
        register_resume("image_delivery.err",    _image_delivery_err)
```
(`src.app.telegram_bot` is already in `continuations._HANDLER_MODULES`, so these get picked up at startup + restart-reconcile.)

- [ ] **Step 4: Implement `cmd_image` (CPS) + the resume/err handlers**

Add the command method near the other `cmd_*` methods:
```python
    async def cmd_image(self, update, context):
        """/image <prompt> — generate via the cloud image pipeline.

        CPS: enqueue + on_complete continuation; the photo is delivered from
        _image_delivery_resume. NO await_inline (it is being retired and can't
        drive cross-cycle retries anyway)."""
        prompt = " ".join(context.args or []).strip()
        if not prompt:
            await self._reply(update, "Usage: `/image a red bicycle`",
                              parse_mode="Markdown", reply_markup=REPLY_KEYBOARD)
            return
        import os, tempfile
        from general_beckman import enqueue
        chat_id = update.effective_chat.id
        out_dir = os.path.join(tempfile.gettempdir(), "kutai_images", str(chat_id))
        spec = {
            "title": f"image:{prompt[:40]}",
            "description": "Telegram /image generation",
            "agent_type": "image",
            "kind": "image",
            "priority": 5,
            "context": {"image_call": {
                "raw_dispatch": True, "prompt": prompt, "out_dir": out_dir,
                "width": 1024, "height": 1024, "quality_tier": "quality",
                "filename_hint": prompt[:30],
            }},
        }
        await self._reply(update, "🎨 Generating…", reply_markup=REPLY_KEYBOARD)
        await enqueue(
            spec,
            on_complete="image_delivery.resume",
            on_error="image_delivery.err",
            cont_state={"chat_id": chat_id, "prompt": prompt},
        )
```

Add the module-level continuation handlers + photo helper near `_casual_reply_resume` / `_send_telegram_via_resume`:
```python
async def _send_telegram_photo_via_resume(chat_id: int, path: str, *, caption=None) -> bool:
    """Send a photo from a CPS resume context (no live Update)."""
    try:
        tg = get_telegram()
    except RuntimeError:
        return False
    if tg is None or not getattr(tg, "app", None):
        return False
    try:
        with open(path, "rb") as fh:
            await tg.app.bot.send_photo(chat_id=chat_id, photo=fh,
                                        caption=(caption or "")[:1000])
        return True
    except Exception as exc:
        logger.debug("resume telegram photo send failed", chat_id=chat_id, error=str(exc))
        return False


async def _image_delivery_resume(child_task_id: int, result: dict, state: dict) -> None:
    """CPS resume for /image. state = {'chat_id': int, 'prompt': str}.

    result envelope: result['result'] is husam's image dict, JSON-stringified by
    the orchestrator. Tolerate dict + flat restart-reconcile shapes (mirrors
    _casual_reply_resume)."""
    import json, os
    chat_id = (state or {}).get("chat_id")
    if chat_id is None:
        return
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:
            inner = {}
    if isinstance(inner, dict):
        payload = inner
    elif isinstance(result.get("path"), str):
        payload = result   # flat reconcile shape
    else:
        payload = {}
    path = payload.get("path")
    prompt = (state.get("prompt") or "")[:200]
    if path and os.path.isfile(path):
        await _send_telegram_photo_via_resume(chat_id, path, caption=prompt)
    else:
        await _send_telegram_via_resume(chat_id, "❌ Image failed: no image produced")


async def _image_delivery_err(child_task_id: int, result: dict, state: dict) -> None:
    """CPS on_error for /image."""
    chat_id = (state or {}).get("chat_id")
    if chat_id is None:
        return
    err = (result or {}).get("error") if isinstance(result, dict) else ""
    await _send_telegram_via_resume(chat_id, f"❌ Image failed: {err or 'generation error'}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/app/test_cmd_image.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Verify import + handler registration**

Run: `.venv/Scripts/python -c "from src.app.telegram_bot import TelegramInterface, _image_delivery_resume; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/app/telegram_bot.py tests/app/test_cmd_image.py
git commit -m "feat(image): /image command + CPS photo-delivery continuation (no await_inline)"
```

---

## Task 14: Real end-to-end test — beckman.enqueue → orchestrator → dispatcher → paintress

**Files:**
- Create: `tests/integration/test_image_e2e.py`

The prior plans' "e2e" tests either called a non-existent `dispatcher.dispatch` or relied on `await_inline` (which inserts to DB and blocks on a Future resolved by a separate pump — it does NOT execute inline). **v3 drives one real pump cycle manually**, no await_inline: `enqueue` (returns task_id) → `next_task()` (admission runs `_select_for_admission` image branch → hoca image scorer → `ImageModelInfo` pick attached) → `husam.run(task)` (image lane → paintress, mocked at the provider layer so renoir + PNG-write are real) → `on_task_finished(tid, result)` (the way `orchestrator._dispatch` finishes — JSON-stringifying husam's dict).

> **Verify at impl:** confirm `next_task()`'s post-selection admission gating (pool pressure / reserve / `_handle_admission_pick`) does not crash on an `ImageModelInfo` pick (it carries `name/provider/location/is_local/vram_mb` but no text-only fields like tok/s or context). If any gate reads a text-only field, guard the image branch around it. This is the one integration seam the unit tests can't prove.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_e2e.py
import json
import os
import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_task
import general_beckman
from general_beckman import enqueue, next_task, on_task_finished
import husam


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated DB (mirrors tests/test_beckman_next_task.py::_fresh)."""
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    _db_mod._db_connection = None
    await init_db()


def _fake_pollinations(seed=11):
    class _FakeProvider:
        name = "pollinations"
        def available(self): return True
        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": seed}
    return _FakeProvider()


@pytest.mark.asyncio
async def test_image_generation_full_pipeline(monkeypatch, tmp_path):
    """One manual pump cycle, NO await_inline. beckman admission → husam image
    lane → paintress (mocked at provider layer; renoir + PNG-write real) →
    on_task_finished round-trip."""
    await _fresh_db(tmp_path, monkeypatch)
    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS", {"pollinations": _fake_pollinations()})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    spec = {
        "title": "e2e", "description": "e2e image",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a mountain lake",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
            "filename_hint": "lake",
        }},
    }
    tid = await enqueue(spec)                       # returns task_id (no await_inline)
    assert isinstance(tid, int)

    task = await next_task()                        # admission + image pick
    assert task is not None and task["id"] == tid

    result = await husam.run(task)                  # image lane
    assert result["provider"] == "pollinations"
    assert result["seed_used"] == 11
    assert os.path.isfile(result["path"]) and os.path.getsize(result["path"]) > 0

    # Finish the way orchestrator._dispatch does (json.dumps the husam dict).
    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})
    row = await get_task(tid)
    assert row["status"] == "completed"
    raw = row.get("result")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["path"] == result["path"]
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_e2e.py -q`
Expected: PASS (1 passed). If `enqueue`'s spec schema differs (e.g. `kind` must be a top-level kwarg), match an existing `enqueue` caller in `telegram_bot.py`; if `next_task()` needs cron-seed/paused-pattern resets, copy them from `_fresh_db` in `tests/test_beckman_next_task.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_image_e2e.py
git commit -m "test(image): e2e through beckman+orchestrator+dispatcher+paintress"
```

---

## Task 15: Telemetry round-trip verification

**Files:**
- Create: `tests/integration/test_image_telemetry_roundtrip.py`

`feedback_verify_verdict_roundtrip`: "a post-hook is only wired when its verdict reaches the consumer; registry-shape tests pass while source stays ungraded forever." Same risk for image telemetry: pick_log/in-flight/cost can all be called and STILL not produce observable rows if the DB layer rejects/ignores them. This task verifies the actual rows land.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_telemetry_roundtrip.py
import json
import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_db
from general_beckman import enqueue, next_task, on_task_finished
import husam


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    _db_mod._db_connection = None
    await init_db()


def _fake_pollinations():
    class _FakeProvider:
        name = "pollinations"
        def available(self): return True
        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (50, 60, 70)).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": 3}
    return _FakeProvider()


@pytest.mark.asyncio
async def test_image_call_writes_pick_log_and_token_rows(monkeypatch, tmp_path):
    """After a successful husam image call, model_pick_log AND model_call_tokens
    each gain a row for this call. Anything less = telemetry pipeline lying."""
    await _fresh_db(tmp_path, monkeypatch)
    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS", {"pollinations": _fake_pollinations()})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    db = await get_db()
    cur = await db.execute("SELECT COUNT(*) FROM model_pick_log")
    base_pick = (await cur.fetchone())[0]
    cur = await db.execute("SELECT COUNT(*) FROM model_call_tokens")
    base_tok = (await cur.fetchone())[0]

    spec = {
        "title": "telemetry-roundtrip",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "telemetry test",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
            "filename_hint": "tel",
        }},
    }
    tid = await enqueue(spec)
    task = await next_task()
    assert task["id"] == tid

    # husam._run_image reads the task-id from this contextvar (orchestrator sets
    # it in prod); set it so the token row attributes to the task.
    from src.core.heartbeat import current_task_id
    _tok = current_task_id.set(tid)
    try:
        result = await husam.run(task)
    finally:
        current_task_id.reset(_tok)
    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})

    cur = await db.execute("SELECT COUNT(*) FROM model_pick_log")
    new_pick = (await cur.fetchone())[0]
    cur = await db.execute("SELECT COUNT(*) FROM model_call_tokens")
    new_tok = (await cur.fetchone())[0]
    assert new_pick == base_pick + 1, \
        "model_pick_log did NOT gain a row — pick_log wiring is broken"
    assert new_tok == base_tok + 1, \
        "model_call_tokens did NOT gain a row — token telemetry is broken"

    cur = await db.execute(
        "SELECT picked_model, category FROM model_pick_log ORDER BY rowid DESC LIMIT 1"
    )
    row = await cur.fetchone()
    assert row[0] == "pollinations/flux"
    assert row[1] == "image"   # CallCategory.IMAGE.value

    cur = await db.execute(
        "SELECT call_category, model FROM model_call_tokens ORDER BY rowid DESC LIMIT 1"
    )
    row = await cur.fetchone()
    assert row[0] == "image"
    assert row[1] == "pollinations/flux"
```

> **Confirm column names** against the live schema (`grep -n "model_pick_log" src/infra/db.py` / `model_call_tokens`): the asserts assume `picked_model`/`category` and `call_category`/`model`. Adjust if the de-accretion renamed them.

- [ ] **Step 2: Run + verify**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_telemetry_roundtrip.py -q`
Expected: PASS (1 passed). If "model_pick_log did NOT gain a row," husam's `disp._record_pick` call (Task 11) fired but didn't reach the DB — investigate `pick_recorder.record_pick`'s write path.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_image_telemetry_roundtrip.py
git commit -m "test(image): telemetry round-trip — pick_log + model_call_tokens"
```

---

## Task 16: Retry / failures-propagation integration test

**Files:**
- Create: `tests/integration/test_image_retry_propagation.py`

Combines Tasks 9 + 11: a paintress failure must propagate `failed_models` into the next admission via `_select_for_admission`, so the second try picks a DIFFERENT provider. Driven as **two manual pump cycles** (no await_inline). **Note the pick order:** the image scorer ranks HF (`quality_rank=8.0`) ABOVE pollinations (`6.0`), so with `HF_TOKEN` set HF is picked FIRST. So the test makes **HF the flaky one** and pollinations the fallback — cycle 1 picks HF and fails, cycle 2 excludes HF and succeeds on pollinations.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_retry_propagation.py
import json
import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_task, update_task
from general_beckman import enqueue, next_task, on_task_finished
import husam
from src.core.router import ModelCallFailed


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    _db_mod._db_connection = None
    await init_db()


@pytest.mark.asyncio
async def test_failed_provider_excluded_on_retry(monkeypatch, tmp_path):
    """Cycle 1: HF (top-ranked) is picked and fails → on_task_finished records it
    in context.failed_models. Cycle 2: _select_for_admission forwards failed_models
    → image scorer excludes HF → pollinations picked → success. Without Task 9's
    propagation, cycle 2 would re-pick the dead HF — the silent rot the audit warned of."""
    await _fresh_db(tmp_path, monkeypatch)

    seen = {"hf": 0}
    class _FlakyHF:
        name = "huggingface"
        def available(self): return True
        async def generate(self, spec, *, base_url=None):
            seen["hf"] += 1
            raise RuntimeError("simulated provider down")

    class _FakePollinations:
        name = "pollinations"
        def available(self): return True
        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 100, 100)).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": 2}

    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS",
                        {"pollinations": _FakePollinations(), "huggingface": _FlakyHF()})
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")

    spec = {
        "title": "retry-test", "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "retry test",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
            "filename_hint": "ret", "quality_tier": "quality",
        }},
    }
    tid = await enqueue(spec)

    # ── CYCLE 1: HF picked (top rank) → fails.
    task1 = await next_task()
    assert task1["id"] == tid
    picked1 = task1["preselected_pick"].model.name
    assert picked1 == "huggingface/flux-schnell"
    with pytest.raises(ModelCallFailed):
        await husam.run(task1)
    # Finish as orchestrator._dispatch would on failure.
    await on_task_finished(tid, {"status": "failed", "error": "simulated provider down",
                                 "error_category": "availability", "model": picked1})
    row1 = await get_task(tid)
    assert row1["status"] == "pending"
    ctx1 = json.loads(row1.get("context") or "{}")
    assert picked1 in ctx1.get("failed_models", [])
    await update_task(tid, next_retry_at=None)   # clear backoff so cycle 2 admits now

    # ── CYCLE 2: HF excluded → pollinations picked → success.
    task2 = await next_task()
    assert task2 is not None and task2["id"] == tid
    assert task2["preselected_pick"].model.name == "pollinations/flux"
    result = await husam.run(task2)
    assert result["provider"] == "pollinations"
    assert seen["hf"] >= 1   # HF was attempted before exclusion
    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})
    assert (await get_task(tid))["status"] == "completed"
```

- [ ] **Step 2: Run + verify**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_retry_propagation.py -q`
Expected: PASS. If cycle 2 re-picks HF, `_select_for_admission` is not reading `task.context["failed_models"]` (Task 9) — debug there. If cycle 2's `next_task()` returns `None`, the retry backoff (`next_retry_at`) wasn't cleared, or availability didn't re-pend (check `_retry_or_dlq`).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_image_retry_propagation.py
git commit -m "test(image): retry excludes failed provider via failed_models propagation"
```

---

## Task 17: Documentation + done-when sweep

**Files:**
- Modify: `docs/superpowers/specs/2026-05-23-image-generation-design.md` (small status banner)
- Test (smoke): run the full new-test set together

- [ ] **Step 1: Add a "Plan 1 v3 status" line to the spec header**

In `docs/superpowers/specs/2026-05-23-image-generation-design.md`, under the `Status:` line, add:
```
> Plan 1 v3 shipped <YYYY-MM-DD>: cloud spine + /image working (image lane in husam, CPS photo delivery — no await_inline), telemetry round-trip verified, failures-propagation fix shared with text path. Plans 2 (clair_obscur) and 3 (i2p) pending.
```

- [ ] **Step 2: Full new-suite smoke**

Run in two passes (NOT mixed — conftest collision per `2026-05-21` handoff §4): pass 1 = package tests, pass 2 = root `tests/`.
```
.venv/Scripts/python -m pytest packages/renoir/tests packages/paintress/tests packages/husam/tests/test_husam_image.py packages/fatih_hoca/tests/test_image_model_info.py packages/fatih_hoca/tests/test_image_providers.py packages/fatih_hoca/tests/test_image_select.py packages/fatih_hoca/tests/test_select_image_dispatch.py packages/general_beckman/tests/test_failures_propagation.py packages/general_beckman/tests/test_admission_selection_failure.py -q
.venv/Scripts/python -m pytest tests/core/test_orchestrator_image_route.py tests/app/test_cmd_image.py tests/integration/test_image_e2e.py tests/integration/test_image_telemetry_roundtrip.py tests/integration/test_image_retry_propagation.py -q
```
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-23-image-generation-design.md
git commit -m "docs(image): mark Plan 1 v3 shipped (cloud spine working)"
```

---

## Plan 1 v3 done-when

- `/image <prompt>` in Telegram **delivers a generated photo via a CPS `on_complete` continuation** (Pollinations, or HF when `HF_TOKEN` set) — **no `await_inline`** anywhere.
- The image lane executes inside **husam** (`_run_image`), not a dispatcher method — and husam is the only non-coulson caller of the dispatcher.
- A failed provider on the first attempt is **excluded from the second attempt** via the shared `failed_models`-propagation in `_select_for_admission` (text path wraps names in `Failure(...)`; image path forwards strings) — which also fixes text retries.
- After every successful image call, `model_pick_log` AND `model_call_tokens` each contain a row attributable to the call (verified by Task 15).
- A `SelectionFailure` at admission (no eligible provider) marks the task failed cleanly — beckman does not crash on `pick.model.name`.
- All new tests green; no regressions in `packages/husam/tests/`, `packages/fatih_hoca/tests/`, `packages/general_beckman/tests/`.
- Local providers absent from the image catalog (`clair_obscur` is Plan 2); eviction-cost is a 0-returning stub.

## Follow-on plans (assume Plan 1 v3 has merged)
- **Plan 2 — local `clair_obscur` + GPU handover:** clair_obscur package (PID-lock/orphan-reconcile), nerd_herd image-server VRAM residency, real `_eviction_cost` (reads nerd_herd) replacing Plan 1's stub, the `dallama.unload()` handover touch inside `husam._run_image` (local branch), beckman warm-batch + swap-budget, local provider in `image_providers.py`.
- **Plan 3 — i2p integration:** prompt-writing coulson task (+ templates), `swap_placeholder_images` mr_roboto mechanical, asset serving into web-preview, i2p prototype-phase wiring.
