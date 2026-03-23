# Phase 3: GPU Load Control & Auto-Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the load mode system functional — 4 modes (full/heavy/shared/minimal) that are actually enforced in model selection, with automatic GPU external usage detection that dynamically switches between ANY mode based on external VRAM pressure.

**Architecture:** Enhance `load_manager.py` with 4 modes + VRAM budgets. Add external process detection to `gpu_monitor.py` using pynvml process enumeration. Create a background auto-detect loop that monitors external GPU usage every 30s and dynamically selects the appropriate mode. Wire load mode enforcement into `router.py`'s `select_model()` as hard filters + scoring modifiers. Expose all state via `/metrics` Prometheus endpoint.

**Tech Stack:** pynvml (GPU process detection), asyncio (background loop), existing Prometheus text format in api.py

---

### Task 1: Add Heavy Mode + VRAM Budgets to load_manager.py

**Files:**
- Modify: `src/infra/load_manager.py`

**Step 1: Update LOAD_MODES and budget fractions**

Replace the entire `load_manager.py` with 4-mode support:

```python
LOAD_MODES = ("full", "heavy", "shared", "minimal")

VRAM_BUDGETS = {
    "full": 1.0,
    "heavy": 0.9,
    "shared": 0.5,
    "minimal": 0.0,
}

DESCRIPTIONS = {
    "full":    "Full GPU — all local capacity available",
    "heavy":   "Heavy GPU — 90% VRAM cap, slight headroom for OS/desktop",
    "shared":  "Shared GPU — 50% VRAM cap, prefer cloud for heavy tasks",
    "minimal": "Minimal GPU — local inference disabled, cloud only",
}
```

Update `set_load_mode()` to accept "heavy", update `is_local_inference_allowed()` to return False only for minimal, update `get_vram_budget_fraction()` to use the VRAM_BUDGETS dict.

Add `_auto_mode: bool = False` flag to track whether mode was set by auto-detection (vs user override). Add `set_load_mode(mode, source="user"|"auto")` — when source is "user", disable auto-management. Add `is_auto_managed() -> bool` helper.

**Step 2: Verify**

Run: `python -c "from src.infra.load_manager import LOAD_MODES, VRAM_BUDGETS; print(LOAD_MODES, VRAM_BUDGETS)"`

**Step 3: Commit**

```bash
git add src/infra/load_manager.py
git commit -m "feat(phase3): add heavy mode + VRAM budgets to load_manager"
```

---

### Task 2: GPU External Process Detection

**Files:**
- Modify: `src/models/gpu_monitor.py`

**Step 1: Add ExternalGPUUsage dataclass and detect method**

Add to `gpu_monitor.py`:

```python
@dataclass
class ExternalGPUUsage:
    detected: bool = False
    external_vram_mb: int = 0
    external_process_count: int = 0
    our_vram_mb: int = 0
    total_vram_mb: int = 0

    @property
    def external_vram_fraction(self) -> float:
        if self.total_vram_mb == 0:
            return 0.0
        return self.external_vram_mb / self.total_vram_mb
```

Add `detect_external_gpu_usage(self) -> ExternalGPUUsage` to `GPUMonitor`:
- Use `pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)` to get all GPU processes
- Get our PID via `os.getpid()` and parent PID tree
- Sum VRAM of processes NOT in our PID tree
- Flag as detected if external VRAM > 2048 MB OR external fraction > 0.30

**Step 2: Verify**

Run: `python -c "from src.models.gpu_monitor import get_gpu_monitor; m = get_gpu_monitor(); print(m.detect_external_gpu_usage())"`

**Step 3: Commit**

```bash
git add src/models/gpu_monitor.py
git commit -m "feat(phase3): add external GPU process detection via pynvml"
```

---

### Task 3: Auto-Detect Loop with Dynamic Mode Switching

**Files:**
- Modify: `src/infra/load_manager.py` (add auto-detect loop function)

**Step 1: Add `suggest_mode_for_external_usage()` function**

Maps external VRAM fraction to the appropriate mode:
- external < 10% → "full"
- external < 30% → "heavy"
- external < 60% → "shared"
- external >= 60% → "minimal"

**Step 2: Add `run_gpu_autodetect_loop()` async function**

```python
async def run_gpu_autodetect_loop(notify_fn=None):
    """Background loop: check external GPU usage every 30s, auto-switch mode."""
    # Track consecutive readings for stability (avoid flapping)
    # Downgrade immediately on detection, upgrade after 5 min of sustained lower usage
    # Never auto-restore if user manually set mode (check _auto_managed flag)
    # Call notify_fn(message) to send Telegram alerts on mode changes
```

Logic:
- Poll `gpu_monitor.detect_external_gpu_usage()` every 30s
- On increased external usage: switch DOWN immediately, notify
- On decreased external usage: wait 5 minutes of sustained decrease, then switch UP, notify
- If user manually set mode via `/load`, stop auto-managing until they run `/load auto`

**Step 3: Commit**

```bash
git add src/infra/load_manager.py
git commit -m "feat(phase3): auto-detect loop with dynamic GPU mode switching"
```

---

### Task 4: Router Enforcement of Load Modes

**Files:**
- Modify: `src/core/router.py` (in `select_model()`)

**Step 1: Add load mode hard filter in select_model()**

After existing hard filters (line ~359), add:

```python
# ── Load mode enforcement ──
from src.infra.load_manager import is_local_inference_allowed, get_vram_budget_fraction

if model.is_local and not is_local_inference_allowed():
    _skip("load_mode_minimal"); continue

if model.is_local:
    vram_budget = get_vram_budget_fraction()
    if vram_budget < 1.0 and vram_budget > 0:
        # Check if model fits within VRAM budget
        model_vram = getattr(model, 'vram_required_mb', 0) or 0
        from src.models.gpu_monitor import get_gpu_monitor
        gpu_state = get_gpu_monitor().get_state().gpu
        if gpu_state.available and model_vram > 0:
            budget_mb = int(gpu_state.vram_total_mb * vram_budget)
            if model_vram > budget_mb:
                _skip(f"vram_budget({model_vram}>{budget_mb})"); continue
```

**Step 2: Add load mode scoring modifier**

In the COST EFFICIENCY section, when mode is "shared" or "heavy", reduce local model cost_score proportionally to penalize local preference:

```python
if model.is_local:
    vram_budget = get_vram_budget_fraction()
    if vram_budget < 1.0:
        # Scale down local preference — cloud becomes more attractive
        cost_score = int(cost_score * (0.5 + vram_budget * 0.5))
        reasons.append(f"load_pen={vram_budget:.1f}")
```

**Step 3: Commit**

```bash
git add src/core/router.py
git commit -m "feat(phase3): enforce load modes in router model selection"
```

---

### Task 5: Prometheus Metrics for Load System

**Files:**
- Modify: `src/app/api.py` (in `prometheus_metrics()`)

**Step 1: Add load mode + GPU detection metrics**

After the existing auto-tuner metrics block, add:

```python
# ── GPU load mode metrics ──
try:
    from src.infra.load_manager import get_load_mode, get_vram_budget_fraction, is_auto_managed
    mode = await get_load_mode()
    budget = get_vram_budget_fraction()
    auto = 1 if is_auto_managed() else 0

    lines.append("# HELP kutay_gpu_load_mode Current GPU load mode (0=minimal,1=shared,2=heavy,3=full)")
    lines.append("# TYPE kutay_gpu_load_mode gauge")
    mode_val = {"minimal": 0, "shared": 1, "heavy": 2, "full": 3}.get(mode, 3)
    lines.append(f"kutay_gpu_load_mode {mode_val}")

    lines.append("# HELP kutay_gpu_load_mode_info GPU load mode label")
    lines.append("# TYPE kutay_gpu_load_mode_info gauge")
    lines.append(f'kutay_gpu_load_mode_info{{mode="{mode}"}} 1')

    lines.append("# HELP kutay_gpu_vram_budget_fraction VRAM budget fraction (0.0-1.0)")
    lines.append("# TYPE kutay_gpu_vram_budget_fraction gauge")
    lines.append(f"kutay_gpu_vram_budget_fraction {budget:.2f}")

    lines.append("# HELP kutay_gpu_auto_managed Whether GPU mode is auto-managed")
    lines.append("# TYPE kutay_gpu_auto_managed gauge")
    lines.append(f"kutay_gpu_auto_managed {auto}")
except Exception as e:
    logger.debug(f"Load mode metrics unavailable: {e}")

# ── External GPU usage metrics ──
try:
    from src.models.gpu_monitor import get_gpu_monitor
    ext = get_gpu_monitor().detect_external_gpu_usage()
    lines.append("# HELP kutay_gpu_external_vram_mb External process VRAM usage in MB")
    lines.append("# TYPE kutay_gpu_external_vram_mb gauge")
    lines.append(f"kutay_gpu_external_vram_mb {ext.external_vram_mb}")

    lines.append("# HELP kutay_gpu_external_processes Number of external GPU processes")
    lines.append("# TYPE kutay_gpu_external_processes gauge")
    lines.append(f"kutay_gpu_external_processes {ext.external_process_count}")

    lines.append("# HELP kutay_gpu_external_vram_fraction External VRAM usage as fraction")
    lines.append("# TYPE kutay_gpu_external_vram_fraction gauge")
    lines.append(f"kutay_gpu_external_vram_fraction {ext.external_vram_fraction:.4f}")
except Exception as e:
    logger.debug(f"External GPU metrics unavailable: {e}")
```

**Step 2: Commit**

```bash
git add src/app/api.py
git commit -m "feat(phase3): add GPU load mode + external usage Prometheus metrics"
```

---

### Task 6: Update Telegram /load Command

**Files:**
- Modify: `src/app/telegram_bot.py` (around line 1192)

**Step 1: Update cmd_load to support 4 modes + auto**

Update the help text and accept "heavy" and "auto" as valid arguments:

```python
async def cmd_load(self, update, context):
    """/load full|heavy|shared|minimal|auto — set GPU load mode"""
    args = context.args or []
    if not args:
        from src.infra.load_manager import get_load_mode, is_auto_managed
        current = await get_load_mode()
        auto_str = " (auto-managed)" if is_auto_managed() else " (manual)"
        await update.message.reply_text(
            f"Current load mode: *{current}*{auto_str}\n\n"
            "Usage: `/load full|heavy|shared|minimal|auto`\n"
            "• *full* — all GPU available\n"
            "• *heavy* — 90% VRAM cap\n"
            "• *shared* — 50% VRAM cap\n"
            "• *minimal* — cloud only\n"
            "• *auto* — enable auto-detection",
            parse_mode="Markdown",
        )
        return

    choice = args[0].lower()
    if choice == "auto":
        from src.infra.load_manager import enable_auto_management
        await enable_auto_management()
        await update.message.reply_text("GPU load mode set to *auto-managed*. Will adjust based on external GPU usage.", parse_mode="Markdown")
        return

    from src.infra.load_manager import set_load_mode
    msg = await set_load_mode(choice, source="user")
    await update.message.reply_text(msg, parse_mode="Markdown")
```

**Step 2: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(phase3): update /load command for 4 modes + auto"
```

---

### Task 7: Wire Auto-Detect Loop into Startup

**Files:**
- Modify: `src/app/run.py` (after monitoring loop creation, ~line 310)

**Step 1: Add auto-detect task creation**

```python
# Phase 3: Start GPU auto-detect loop
gpu_detect_task = None
try:
    from src.infra.load_manager import run_gpu_autodetect_loop

    async def _notify_telegram(msg: str):
        try:
            from src.app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
                return
            import aiohttp
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with aiohttp.ClientSession() as s:
                await s.post(url, json={
                    "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                    "text": msg,
                    "parse_mode": "Markdown",
                }, timeout=aiohttp.ClientTimeout(total=5))
        except Exception:
            pass

    gpu_detect_task = asyncio.create_task(
        run_gpu_autodetect_loop(notify_fn=_notify_telegram),
        name="gpu_autodetect_loop",
    )
    _log.info("GPU auto-detect loop started")
except Exception as exc:
    _log.debug("GPU auto-detect loop not started", reason=str(exc))
```

Don't forget to cancel it at shutdown alongside the other tasks.

**Step 2: Commit**

```bash
git add src/app/run.py
git commit -m "feat(phase3): wire GPU auto-detect loop into startup"
```
