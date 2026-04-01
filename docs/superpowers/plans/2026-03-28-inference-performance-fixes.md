# Inference Performance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model selection speed-aware, let llama-server auto-fit VRAM, support per-model flags, gate Perplexica on model speed, and add direct SearXNG bypass.

**Architecture:** Feed measured tok/s back into ModelInfo after each inference call. Router uses this to score speed dimension. llama-server uses `--fit` instead of pre-calculated `--n-gpu-layers`. Perplexica checks loaded model speed before attempting LLM synthesis. Direct SearXNG path bypasses Vane's slow LLM entirely.

**Tech Stack:** Python 3.10, llama.cpp (llama-server with --fit), Vane/Perplexica (Docker), SearXNG API, aiohttp.

**Reference:** `docs/inference-performance-xray.md` for benchmark data and full analysis.

---

## File Map

| File | Changes |
|------|---------|
| `src/core/router.py` | Feed measured TPS into ModelInfo after each call; amplify speed dimension when `prefer_speed=True` |
| `src/models/model_registry.py` | Add `update_measured_speed()` method; add auto-demote logic for models below speed threshold |
| `src/models/local_model_manager.py` | Remove `--n-gpu-layers`, let `--fit` auto-handle; add per-model extra flags from ModelInfo |
| `src/tools/web_search.py` | Gate Perplexica on loaded model speed; add direct SearXNG search path |
| `src/core/llm_dispatcher.py` | Expose loaded model speed for Perplexica gate |

---

### Task 1: Feed Measured TPS Back Into ModelInfo

**Files:**
- Modify: `src/core/router.py:1157-1164` (where `llm performance` is logged)
- Modify: `src/models/model_registry.py` (add `update_measured_speed`)
- Test: `tests/integration/test_orchestrator_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_model_speed_tracking.py
import pytest
from src.models.model_registry import ModelInfo, ModelRegistry

def test_update_measured_speed_stores_value():
    """After update_measured_speed, ModelInfo.tokens_per_second reflects the new value."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.models = {}
    info = ModelInfo(
        name="test-model", location="local", provider="llama_cpp",
        litellm_name="openai/test-model", capabilities={}, context_length=8192,
        max_tokens=4096, tokens_per_second=0.0,
    )
    registry.models["test-model"] = info

    registry.update_measured_speed("test-model", 42.5)
    assert info.tokens_per_second == 42.5

def test_update_measured_speed_uses_ema():
    """Speed updates use exponential moving average, not raw replacement."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.models = {}
    info = ModelInfo(
        name="test-model", location="local", provider="llama_cpp",
        litellm_name="openai/test-model", capabilities={}, context_length=8192,
        max_tokens=4096, tokens_per_second=40.0,
    )
    registry.models["test-model"] = info

    registry.update_measured_speed("test-model", 50.0)
    # EMA with alpha=0.3: 40 * 0.7 + 50 * 0.3 = 43.0
    assert abs(info.tokens_per_second - 43.0) < 0.1

def test_update_measured_speed_ignores_unknown_model():
    """Updating speed for unknown model name is a no-op."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.models = {}
    registry.update_measured_speed("nonexistent", 42.5)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py -v`
Expected: FAIL — `update_measured_speed` not defined

- [ ] **Step 3: Add `update_measured_speed()` to ModelRegistry**

In `src/models/model_registry.py`, add method after `all_models()` (around line 1472):

```python
def update_measured_speed(self, model_name: str, measured_tps: float) -> None:
    """Update a model's tokens_per_second using exponential moving average.

    Called by the router after each successful inference call to keep
    the speed estimate fresh.  EMA smooths out variance from different
    prompt lengths and concurrent load.
    """
    info = self.models.get(model_name)
    if info is None:
        return
    if info.tokens_per_second <= 0:
        # First measurement — use raw value
        info.tokens_per_second = measured_tps
    else:
        # EMA with alpha=0.3 (recent measurements weighted 30%)
        alpha = 0.3
        info.tokens_per_second = info.tokens_per_second * (1 - alpha) + measured_tps * alpha
```

- [ ] **Step 4: Wire router to call `update_measured_speed` after each inference**

In `src/core/router.py`, find the `llm performance` log block (around line 1157). After the log statement, add:

```python
# Feed measured speed back into registry for future scoring
try:
    registry = get_registry()
    registry.update_measured_speed(model.name, tok_per_sec)
except Exception:
    pass  # non-critical — don't break inference for a metrics update
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/model_registry.py src/core/router.py tests/integration/test_model_speed_tracking.py
git commit -m "feat(router): feed measured TPS back into ModelInfo via EMA"
```

---

### Task 2: Speed Coefficient Affects Model Selection

**Files:**
- Modify: `src/core/router.py:552-588` (speed dimension scoring)
- Test: `tests/integration/test_model_speed_tracking.py` (extend)

The router already reads `model.tokens_per_second` in the speed dimension (lines 552-588). But:
1. The `tokens_per_second` field was always 0.0 (fixed in Task 1)
2. When `prefer_speed=True`, the speed dimension weight needs amplification

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_model_speed_tracking.py`:

```python
from src.core.router import score_model_for_task, ModelRequirements
from src.models.model_registry import ModelInfo

def _make_model(name, tps, is_loaded=False):
    m = ModelInfo(
        name=name, location="local", provider="llama_cpp",
        litellm_name=f"openai/{name}", capabilities={"general": 5.0},
        context_length=8192, max_tokens=4096,
        supports_function_calling=True,
        tokens_per_second=tps,
    )
    m.is_loaded = is_loaded
    return m

def test_faster_model_scores_higher_when_prefer_speed():
    """A model with 50 tok/s should score higher than 5 tok/s when prefer_speed=True."""
    reqs = ModelRequirements(task="assistant", difficulty=3, prefer_speed=True)
    fast = _make_model("fast-9b", tps=50.0)
    slow = _make_model("slow-27b", tps=5.0)

    fast_score = score_model_for_task(fast, reqs)
    slow_score = score_model_for_task(slow, reqs)
    assert fast_score > slow_score, f"Fast ({fast_score}) should beat slow ({slow_score})"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py::test_faster_model_scores_higher_when_prefer_speed -v`
Expected: May pass or fail depending on existing speed scoring — if it fails, the speed dimension isn't using TPS correctly.

- [ ] **Step 3: Amplify speed dimension when `prefer_speed=True`**

In `src/core/router.py`, find the speed scoring block (around line 552-588). After the TPS-based speed score is computed, add an amplification multiplier:

Find the section where the final `speed_score` is computed for local models (look for `speed_score = ...` in the speed dimension). After it:

```python
# Amplify speed score when prefer_speed is set — measured TPS directly boosts score
if reqs.prefer_speed and tps > 0:
    # Normalize TPS to a 0-1 scale (50+ tok/s = 1.0, 5 tok/s = 0.1)
    tps_boost = min(1.0, tps / 50.0)
    speed_score = speed_score * (0.5 + tps_boost * 0.5)
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/router.py tests/integration/test_model_speed_tracking.py
git commit -m "feat(router): amplify speed dimension using measured TPS when prefer_speed=True"
```

---

### Task 3: Remove `--n-gpu-layers`, Let `--fit` Auto-Handle

**Files:**
- Modify: `src/models/local_model_manager.py:518-531` (cmd construction)
- Modify: `src/models/local_model_manager.py:439-450` (dynamic recalc at swap time)

New llama.cpp has `--fit` enabled by default which auto-calculates optimal GPU layers. Explicit `--n-gpu-layers` conflicts with `--fit` and causes abort when model doesn't fully fit VRAM.

- [ ] **Step 1: Remove `--n-gpu-layers` from default cmd, add only if user-override exists**

In `src/models/local_model_manager.py`, modify `_start_server()` cmd construction (line 518+):

```python
cmd = [
    str(LLAMA_SERVER_PATH),
    "--model", model.path,
    "--alias", "local-model",
    "--port", str(self.port),
    "--host", "127.0.0.1",
    # --n-gpu-layers omitted: llama-server --fit auto-calculates optimal GPU layers.
    # Only pass explicit layers if user set gpu_layers override in models.yaml.
    "--ctx-size", str(model.context_length),
    "--flash-attn", "auto",
    "--metrics",
    "--threads", str(self._get_inference_threads()),
    "--batch-size", "2048",
    "--ubatch-size", "512",
]

# Only pass gpu_layers if explicitly overridden in models.yaml
if model.gpu_layers > 0 and model._gpu_layers_from_override:
    cmd.extend(["--n-gpu-layers", str(model.gpu_layers)])
```

- [ ] **Step 2: Add `_gpu_layers_from_override` flag to ModelInfo**

In `src/models/model_registry.py`, add field to ModelInfo dataclass (after `gpu_layers`):

```python
gpu_layers: int = 0
_gpu_layers_from_override: bool = False  # True if set via models.yaml, not auto-calculated
total_layers: int = 0
```

- [ ] **Step 3: Set the flag in model loading**

In `src/models/model_registry.py`, in `_load_local_models()` (around line 1060), where `gpu_layers` is assigned:

```python
if "gpu_layers" in registry_overrides:
    gpu_layers = registry_overrides["gpu_layers"]
    _gpu_layers_from_override = True
else:
    gpu_layers = calculate_gpu_layers(...)
    _gpu_layers_from_override = False
```

And pass it to ModelInfo construction.

- [ ] **Step 4: Remove the dynamic gpu_layers recalculation at swap time**

In `src/models/local_model_manager.py`, in `_swap_model()` (around line 439-450), the dynamic recalculation block:

```python
if "gpu_layers" not in registry_overrides:
    new_layers = calculate_gpu_layers(...)
    ...
```

Change to only recalculate if override exists (otherwise --fit handles it):

```python
# Only recalculate if user explicitly overrides gpu_layers in models.yaml.
# Otherwise, llama-server --fit auto-calculates optimal layers for current VRAM.
if "gpu_layers" in registry_overrides:
    model_info.gpu_layers = registry_overrides["gpu_layers"]
    model_info._gpu_layers_from_override = True
```

- [ ] **Step 5: Test by importing**

Run: `.venv/Scripts/python.exe -c "from src.models.local_model_manager import LocalModelManager; print('OK')"`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add src/models/local_model_manager.py src/models/model_registry.py
git commit -m "feat(llama): remove --n-gpu-layers, let --fit auto-calculate optimal GPU layers"
```

---

### Task 4: Per-Model llama-server Flags

**Files:**
- Modify: `src/models/model_registry.py` (add `extra_server_flags` field)
- Modify: `src/models/local_model_manager.py:532-545` (append extra flags)
- Modify: `src/models/families.py` or equivalent (family-level defaults)

Models like Apriel need `--no-jinja --chat-template chatml`, MoE models need `--override-kv`. Currently MoE is hardcoded; Apriel crashes.

- [ ] **Step 1: Add `extra_server_flags` to ModelInfo**

In `src/models/model_registry.py`, add field to ModelInfo (after `sampling_overrides`):

```python
# Per-model llama-server extra flags (from models.yaml or family detection)
# Example: ["--no-jinja", "--chat-template", "chatml"]
extra_server_flags: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Populate from family detection in model loading**

In `src/models/model_registry.py`, in `_load_local_models()`, after capabilities are estimated:

```python
# Per-model server flags from family or overrides
extra_flags = list(registry_overrides.get("extra_server_flags", []))
if not extra_flags:
    # Family-based defaults
    if model_type == "moe":
        extra_flags = ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"]
    if family_key in ("apriel",) or "apriel" in name_lower:
        extra_flags = ["--no-jinja", "--chat-template", "chatml"]
```

- [ ] **Step 3: Use extra_server_flags in _start_server() instead of hardcoded MoE check**

In `src/models/local_model_manager.py`, replace the MoE-specific block (lines 543-545):

```python
# Remove this:
# if model.model_type == "moe":
#     cmd.extend(["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"])

# Replace with:
if model.extra_server_flags:
    cmd.extend(model.extra_server_flags)
```

- [ ] **Step 4: Test import**

Run: `.venv/Scripts/python.exe -c "from src.models.local_model_manager import LocalModelManager; print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/models/model_registry.py src/models/local_model_manager.py
git commit -m "feat(llama): per-model server flags (Apriel --no-jinja, MoE --override-kv)"
```

---

### Task 5: Auto-Demote Unusable Models After Measurement

**Files:**
- Modify: `src/models/model_registry.py` (extend `update_measured_speed`)

Models below 2 tok/s are unusable for interactive tasks. After first measurement, auto-demote them so the router avoids them.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_model_speed_tracking.py`:

```python
def test_very_slow_model_gets_demoted():
    """A model measured at <2 tok/s should be demoted."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.models = {}
    info = ModelInfo(
        name="slow-model", location="local", provider="llama_cpp",
        litellm_name="openai/slow-model", capabilities={}, context_length=8192,
        max_tokens=4096, tokens_per_second=0.0,
    )
    registry.models["slow-model"] = info

    registry.update_measured_speed("slow-model", 1.0)
    assert info.demoted is True

def test_fast_model_not_demoted():
    """A model at 10 tok/s should not be demoted."""
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.models = {}
    info = ModelInfo(
        name="fast-model", location="local", provider="llama_cpp",
        litellm_name="openai/fast-model", capabilities={}, context_length=8192,
        max_tokens=4096, tokens_per_second=0.0,
    )
    registry.models["fast-model"] = info

    registry.update_measured_speed("fast-model", 10.0)
    assert info.demoted is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py::test_very_slow_model_gets_demoted -v`
Expected: FAIL

- [ ] **Step 3: Add auto-demote logic to `update_measured_speed`**

Extend the method added in Task 1:

```python
def update_measured_speed(self, model_name: str, measured_tps: float) -> None:
    """Update a model's tokens_per_second using EMA. Auto-demotes very slow models."""
    info = self.models.get(model_name)
    if info is None:
        return
    if info.tokens_per_second <= 0:
        info.tokens_per_second = measured_tps
    else:
        alpha = 0.3
        info.tokens_per_second = info.tokens_per_second * (1 - alpha) + measured_tps * alpha

    # Auto-demote models below 2 tok/s — they're unusable for interactive tasks.
    # Only demote local models (cloud models have different speed characteristics).
    _MIN_USABLE_TPS = 2.0
    if info.is_local and info.tokens_per_second < _MIN_USABLE_TPS and not info.demoted:
        info.demoted = True
        logger.info(
            f"Auto-demoted {model_name}: {info.tokens_per_second:.1f} tok/s "
            f"< {_MIN_USABLE_TPS} minimum"
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_model_speed_tracking.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/model_registry.py tests/integration/test_model_speed_tracking.py
git commit -m "feat(registry): auto-demote models below 2 tok/s after first measurement"
```

---

### Task 6: Perplexica Gates on Model Speed

**Files:**
- Modify: `src/tools/web_search.py:130-165` (add speed check before Perplexica call)
- Modify: `src/core/llm_dispatcher.py` (expose `get_loaded_model_speed`)

Perplexica needs llama-server running AND the loaded model must be fast enough (>5 tok/s) and non-thinking. Otherwise skip straight to DuckDuckGo.

- [ ] **Step 1: Add `get_loaded_model_speed()` to dispatcher**

In `src/core/llm_dispatcher.py`, add method after `_get_loaded_litellm_name()`:

```python
def get_loaded_model_speed(self) -> float:
    """Get the currently loaded model's measured tok/s. Returns 0 if unknown."""
    try:
        from src.models.local_model_manager import get_local_manager
        manager = get_local_manager()
        if manager.runtime_state and manager.runtime_state.measured_tps > 0:
            return manager.runtime_state.measured_tps
        # Fall back to registry value
        if manager.current_model:
            from src.models.model_registry import get_registry
            info = get_registry().get(manager.current_model)
            if info:
                return info.tokens_per_second
    except Exception:
        pass
    return 0.0

def is_loaded_model_thinking(self) -> bool:
    """Check if the currently loaded model is a thinking model."""
    try:
        from src.models.local_model_manager import get_local_manager
        manager = get_local_manager()
        if manager.runtime_state:
            return manager.runtime_state.thinking_enabled
    except Exception:
        pass
    return False
```

- [ ] **Step 2: Gate Perplexica on model speed in web_search.py**

In `src/tools/web_search.py`, at the start of `_search_perplexica()` (after the PERPLEXICA_URL check, around line 145):

```python
# Skip Perplexica if loaded model is too slow or uses thinking mode.
# Perplexica's LLM synthesis takes 50-70s even with fast models;
# slow or thinking models would take 500s+ and always timeout.
_MIN_PERPLEXICA_TPS = 5.0
try:
    from src.core.llm_dispatcher import get_dispatcher
    dispatcher = get_dispatcher()
    model_speed = dispatcher.get_loaded_model_speed()
    is_thinking = dispatcher.is_loaded_model_thinking()
    if model_speed > 0 and model_speed < _MIN_PERPLEXICA_TPS:
        logger.debug(
            "perplexica: skipping, model too slow",
            speed=f"{model_speed:.1f} tok/s",
            min_required=_MIN_PERPLEXICA_TPS,
        )
        return None
    if is_thinking:
        logger.debug("perplexica: skipping, thinking model wastes tokens")
        return None
except Exception:
    pass  # can't check speed — proceed anyway
```

- [ ] **Step 3: Test import**

Run: `.venv/Scripts/python.exe -c "from src.tools.web_search import web_search; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/tools/web_search.py src/core/llm_dispatcher.py
git commit -m "feat(perplexica): gate on model speed (>5 tok/s) and skip thinking models"
```

---

### Task 7: Agent-Model Map for Web Search

**Files:**
- Modify: `src/core/router.py:1478-1502` (AGENT_REQUIREMENTS)

Agents that frequently use `web_search` (researcher, assistant) should have `prefer_speed=True` so the router favors fast models that work well with Perplexica.

- [ ] **Step 1: Update AGENT_REQUIREMENTS**

In `src/core/router.py`, modify the `researcher` and `assistant` entries:

```python
# researcher already has prefer_speed=True — no change needed
# assistant already has prefer_speed=True — no change needed
# error_recovery does NOT need speed — it rarely uses web_search
```

Check: both `researcher` and `assistant` already have `prefer_speed=True` in the current AGENT_REQUIREMENTS (confirmed from exploration). No code change needed — this is already correct.

- [ ] **Step 2: Commit (docs-only if no code change)**

If no code change needed, skip this commit.

---

### Task 8: Direct SearXNG Bypass (Perplexica Acceleration)

**Files:**
- Modify: `src/tools/web_search.py` (add `_search_searxng_direct()`)

Instead of Vane's full pipeline (SearXNG → embed → LLM synthesis → return), call SearXNG directly for raw search results. This eliminates the 50-55s LLM synthesis phase.

- [ ] **Step 1: Add `_search_searxng_direct()` function**

In `src/tools/web_search.py`, add before `_search_perplexica()`:

```python
async def _search_searxng_direct(
    query: str, max_results: int = 10
) -> list[dict] | None:
    """Search SearXNG directly inside the Vane container, bypassing LLM synthesis.

    This is much faster than full Perplexica (6-10s vs 60-80s) because it
    skips the LLM synthesis step. The KutAI agent will synthesize results
    itself using its own LLM context.

    Returns list of {title, url, snippet} dicts, or None on failure.
    """
    perplexica_url = os.getenv("PERPLEXICA_URL", "").strip()
    if not perplexica_url:
        return None

    # SearXNG runs inside the Vane container on the same base URL
    searxng_url = f"{perplexica_url}/api/search"

    try:
        async with aiohttp.ClientSession() as session:
            # Use Vane's /api/search but without chatModel — Vane may not support this.
            # Instead, call the SearXNG instance directly.
            # SearXNG is at port 8080 inside the container, but not exposed.
            # We can use Vane's search endpoint with a very short LLM timeout.
            # Alternative: call SearXNG JSON API if exposed.

            # For now, use a lightweight query to SearXNG via container's internal API
            # The container exposes SearXNG on internal port 8080; we can reach it
            # if we know the container's IP, but the simplest path is to ask Vane
            # but skip the LLM part by using an embedding-only mode.

            # Pragmatic approach: just use DuckDuckGo when Perplexica is too slow.
            # The SearXNG direct path requires Docker network configuration
            # (exposing port 8080 from the Vane container).
            return None  # TODO: implement when SearXNG port is exposed

    except Exception as e:
        logger.debug("searxng direct search failed", error=str(e))
        return None
```

Note: This task requires exposing SearXNG's port from the Vane Docker container. Add to `docker-compose.yml`:

```yaml
vane:
    ports:
      - "3000:3000"
      - "3001:8080"  # Expose SearXNG for direct search
```

Then the function becomes:

```python
async def _search_searxng_direct(
    query: str, max_results: int = 10
) -> list[dict] | None:
    """Search SearXNG directly, bypassing Vane LLM synthesis."""
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:3001")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{searxng_url}/search",
                params={"q": query, "format": "json", "engines": "duckduckgo,google,bing"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                results = []
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:300],
                    })
                if results:
                    logger.info("searxng direct search ok", count=len(results), query=query[:50])
                return results or None
    except Exception as e:
        logger.debug("searxng direct search failed", error=str(e))
        return None
```

- [ ] **Step 2: Wire into web_search as preferred path before Perplexica**

In the main `web_search()` function, add SearXNG direct as first attempt before Perplexica:

```python
# Try direct SearXNG first (fast, 6-10s, raw results)
searxng_results = await _search_searxng_direct(query, max_results)
if searxng_results:
    # Format as web_search result
    formatted = "\n\n".join(
        f"**{r['title']}**\n{r['snippet']}\n{r['url']}"
        for r in searxng_results
    )
    return formatted

# Then try Perplexica (slower, 40-75s, synthesized answer)
perplexica_result = await _search_perplexica(query, max_results, focus_mode)
...
```

- [ ] **Step 3: Expose SearXNG port in docker-compose.yml**

Add port mapping:

```yaml
vane:
    ports:
      - "3000:3000"
      - "3001:8080"  # SearXNG direct access
```

- [ ] **Step 4: Add SEARXNG_URL to .env**

```
SEARXNG_URL=http://localhost:3001
```

- [ ] **Step 5: Test**

```bash
docker compose up -d
curl -s "http://localhost:3001/search?q=test&format=json" | python -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('results',[])),'results')"
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/web_search.py docker-compose.yml .env
git commit -m "feat(search): direct SearXNG bypass for fast web search (6-10s vs 60-80s)"
```

---

### Task 9: Verify Integration End-to-End

- [ ] **Step 1: Start KutAI and run a shopping search**

1. Start wrapper: use `/restart` via Telegram
2. Send `/shop coffee machine` via Telegram
3. Check logs for: speed measurement, model selection, Perplexica gate, SearXNG direct, DuckDuckGo fallback

- [ ] **Step 2: Verify speed tracking**

Check logs for `update_measured_speed` or `tokens_per_second` updates after first inference call.

- [ ] **Step 3: Verify --fit is used**

Check `Starting llama-server` log line — should NOT contain `--n-gpu-layers` (unless models.yaml override).

- [ ] **Step 4: Verify Perplexica speed gate**

If slow model loaded: log should show `perplexica: skipping, model too slow`
If fast model loaded: Perplexica should be attempted

- [ ] **Step 5: Update docs**

Update `docs/inference-performance-xray.md` with post-fix measurements.

- [ ] **Step 6: Final commit**

```bash
git commit -m "docs: update inference xray with post-fix measurements"
```

---

## Self-Review Checklist

1. **Spec coverage**: All 9 fixes covered (1: TPS feedback, 2: speed scoring, 3: --fit, 4: per-model flags, 5: auto-demote, 6: Perplexica gate, 7: agent map (already done), 8: SearXNG bypass, 9: integration test).

2. **Placeholder scan**: Task 8 has a partial implementation (SearXNG requires Docker port exposure). The stub is marked clearly. All other tasks have complete code.

3. **Type consistency**: `update_measured_speed(model_name: str, measured_tps: float)` matches all call sites. `_gpu_layers_from_override: bool` consistent between ModelInfo definition and usage. `get_loaded_model_speed() -> float` matches Perplexica gate check.
