# KutAI Package Extraction Report v2

**Date:** 2026-04-12
**Status:** 3 packages extracted (yazbunu, vecihi, yasar_usta), 2 candidates remaining

---

## Completed Extractions

### 1. Yazbunu (logging)
- **Repo:** github.com/msakirc/yazbunu (topic: kutai)
- **Location:** Separate workspace at `C:\Users\sakir\Dropbox\Workspaces\yazbunu`
- **Install:** `pip install git+https://github.com/msakirc/yazbunu.git` or editable from local path
- **KutAI shim:** `src/infra/logging_config.py` re-exports `get_logger`, `init_logging`, `YazFormatter`
- **What it does:** Zero-dep structured JSONL logging with a mobile-friendly web viewer (port 9880)
- **Health endpoint:** `/health` (returns 200), NOT `/` (returns 404)
- **Version:** 0.2.0
- **Status:** Stable, no changes needed

### 2. Vecihi (tiered scraper)
- **Repo:** Not yet pushed (lives in `packages/vecihi/`)
- **Package:** `packages/vecihi/` with src layout
- **KutAI shim:** `src/tools/scraper.py` re-exports everything from vecihi
- **Install:** `-e ./packages/vecihi` in requirements.txt
- **What it does:** Auto-escalating web scraper: HTTP -> TLS fingerprint (curl_cffi) -> Stealth (Camoufox) -> Browser (Playwright). Escalates on WAF/Cloudflare blocks (403, 429, challenge pages), stops on timeouts.
- **Public API:** `scrape_url(url, max_tier)`, `scrape_urls(urls, max_tier, max_concurrent)`, `ScrapeTier`, `ScrapeResult`, `detect_block()`
- **Dependencies:** aiohttp (required), curl_cffi (optional [tls]), scrapling (optional [stealth])
- **Tests:** 26 standalone + 25 via KutAI shim
- **Backward compat:** Shim provides underscore-prefixed aliases (`_fetch_http`, `_detect_block`) for old consumers
- **Consumers in KutAI:** `src/tools/page_fetch.py`, `src/shopping/scrapers/base.py`, `src/shopping/scrapers/trendyol.py`, `src/shopping/resilience/redirect_resolver.py`, `src/tools/pharmacy.py`
- **Status:** Stable, ready for separate repo. Tag with `kutai` topic.

### 3. Yasar Usta (process manager)
- **Repo:** github.com/msakirc/yasar-usta (topic: kutai)
- **Package:** `packages/yasar_usta/` with src layout
- **KutAI consumer:** `kutai_wrapper.py` (108 lines of config, down from 1,440)
- **Install:** `-e ./packages/yasar_usta` in requirements.txt
- **What it does:** Telegram-controlled process manager with heartbeat watchdog, escalating backoff, Claude Code remote trigger, sidecar management, full i18n
- **Public API:** `ProcessGuard`, `GuardConfig`, `Messages`, `SidecarConfig`, `HeartbeatWriter`, `write_heartbeat()`, `EXIT_RESTART` (42), `EXIT_STOP` (0)
- **Dependencies:** aiohttp (for Telegram API)
- **Tests:** 74
- **Status:** Active development, several post-extraction fixes applied

#### Yasar Usta Architecture
```
ProcessGuard (guard.py)
  ├── SubprocessManager (subprocess_mgr.py)  — start/stop/wait, output piping, heartbeat watchdog
  ├── TelegramAPI (telegram.py)              — send/edit/poll, non-destructive offset
  ├── BackoffTracker (backoff.py)            — escalating delays with stability reset
  ├── SidecarManager (sidecar.py)            — detached subprocess with PID file + health URL
  ├── commands.py                            — keyboard builders, log formatter
  ├── remote.py                              — Claude Code remote trigger (feature-detected)
  ├── status.py                              — status panel builder
  ├── heartbeat.py                           — HeartbeatWriter, write_heartbeat(), EXIT_* constants
  ├── lock.py                                — cross-platform single-instance (msvcrt/fcntl)
  └── config.py                              — GuardConfig, Messages (i18n), SidecarConfig
```

#### Yasar Usta Post-Extraction Fixes (lessons learned)
1. **Dual-polling bug:** When app starts after crash/backoff, must stop the wrapper's Telegram poller first. Otherwise both wrapper and orchestrator consume updates simultaneously. Fixed: all app starts go through `_start_app()` which stops poller.
2. **Graceful shutdown on Windows:** `send_signal(SIGINT)` raises ValueError with `CREATE_NEW_PROCESS_GROUP`. Must use `os.kill(pid, CTRL_BREAK_EVENT)` instead. `CTRL_BREAK_EVENT` triggers `KeyboardInterrupt` in the child, allowing graceful shutdown.
3. **Backward-compat commands:** Old Telegram commands (`/kutai_start`, `/kutai_status`, `/restart_usta`) and callback data (`usta_refresh`, `restart_yazbunu`) must be accepted alongside new generic names.
4. **Sidecar health URL:** Must match the actual endpoint (`/health` not `/`). yazbunu returns 404 on `/`.
5. **Sidecar ownership:** Yasar Usta owns the sidecar lifecycle, not the orchestrator. Orchestrator's yazbunu restart button kills the process; wrapper's periodic health check (every ~30s) notices it's dead and restarts it.
6. **Always-on Telegram poller:** The current architecture runs the Telegram poller for the entire guard lifetime (not just when app is down). The poller handles `/restart` and `/stop` commands even while the app is running, setting intent flags that the main loop reads after `wait_for_exit()`.
7. **Separate bot token:** Yasar Usta uses `YASAR_USTA_BOT_TOKEN` env var (separate from KutAI's `TELEGRAM_BOT_TOKEN`) to avoid polling conflicts.

#### KutAI-Specific Config (stays in kutai_wrapper.py)
- Turkish message strings via `Messages(...)`
- `_kill_orphan_processes(exit_code)` hook — kills llama-server.exe on crash (skips on exit 42)
- Yazbunu sidecar config (port 9880, `--log-dir ./logs`)
- Claude remote with `--name Kutay`
- `extra_processes=[{"exe": "llama-server.exe", "label": "llama-server"}]` for status panel

---

## Remaining Extraction Candidates

### 4. Local LLM Manager (NOT YET STARTED)
- **Source files:** `src/models/local_model_manager.py` (1,194 lines), `src/models/gpu_monitor.py` (251 lines), `src/models/gpu_scheduler.py` (234 lines)
- **What it does:** llama-server process lifecycle (start/stop/swap models), GPU VRAM monitoring, priority-based inference queue, circuit breaker for restart failures, dynamic context window calculation
- **Why it's valuable:** Nothing on PyPI does programmatic llama-server management. Ollama hides process control. LM Studio is GUI-only. This fills a real gap.
- **Estimated effort:** 3-4 days
- **Coupling analysis:**
  - `local_model_manager.py` — Moderate coupling. Imports: `get_registry()` (model metadata), `get_dispatcher().swap_budget` (swap tracking), `get_db()` (task counts), `accelerate_retries()`. Needs ~5 abstract interfaces.
  - `gpu_monitor.py` — **Zero coupling.** Only imports `get_logger`. Uses pynvml for NVIDIA GPU stats, psutil for system stats. Can extract as-is.
  - `gpu_scheduler.py` — Minimal coupling. Only imports `get_logger` + one optional DB hook (`accelerate_retries`). Priority-based async queue with preemption logging.
- **Extractable bundle:** These three form a coherent "local LLM toolkit":
  - GPU monitoring (VRAM, utilization, temp, external process detection)
  - Inference queue (priority-based, timeout-aware, preemption-logged)
  - Server lifecycle (start, stop, swap, health check, circuit breaker)
- **Key capabilities that don't exist elsewhere:**
  - Model swapping orchestration (atomicity under load, inference drain before kill, dynamic context recalculation)
  - Health watchdog (dual-mode: crash + hang detection, auto-restart with circuit breaker)
  - External GPU usage detection (identifies non-orchestrator GPU processes, VRAM tracking)
  - Measured throughput tracking (tokens/sec from actual inference, not benchmarks)
- **Model registry** (`src/models/model_registry.py`) could pair with this but is more tightly coupled to KutAI's 14-dimension capability scoring. Consider extracting just the GGUF scanning + metadata reading part.

### 5. Turkish Shopping Scrapers (NOT YET STARTED)
- **Source files:** `src/shopping/scrapers/` (19 scrapers, ~8,500 LOC), `src/shopping/models.py`, `src/shopping/text_utils.py`, `src/shopping/cache.py`, `src/shopping/request_tracker.py`
- **What it does:** 19 async scrapers for Turkish e-commerce sites with shared base class, rate limiting, UA rotation, TLS fingerprint bypass, SQLite caching with TTL
- **Why it's valuable:** Zero competition on PyPI. Anyone building Turkish price comparison, shopping bot, or market research tool would install this.
- **Estimated effort:** 2-3 days
- **Coupling analysis (5 KutAI imports, all replaceable):**
  - `src.infra.logging_config` → replace with stdlib logging
  - `src.shopping.config` → extract alongside (rate limits, cache TTL)
  - `src.shopping.cache` → extract alongside (SQLite with TTL)
  - `src.shopping.models` → extract alongside (Product, Review dataclasses)
  - `src.tools.scraper` → now `vecihi` (already extracted!)
- **Sites covered:**
  - Marketplaces: Trendyol (990 LOC), Hepsiburada (685 LOC), Amazon TR (665 LOC)
  - Price aggregators: Akakce (415 LOC), Epey (499 LOC)
  - Grocery: Getir, Migros, Aktuel Katalog
  - Home/furniture: Koctas, IKEA
  - Forums/reviews: Technopat, DonanımHaber, Sikayetvar, Eksisozluk
  - Books: Kitapyurdu
  - Electronics: Direnc.net
  - Sports: Decathlon
  - Vehicles: Arabam
  - Classifieds: Sahibinden (disabled — aggressive blocking)
  - Fallback: Google CSE
- **Turkish text utilities (non-trivial):**
  - `parse_turkish_price()` — handles "1.299,99 TL" format
  - `normalize_turkish()` — correct I/İ/ı casing
  - `normalize_product_name()` — removes 13 Turkish marketing fillers
  - Mojibake/encoding fixes for Turkish characters
  - Bilingual search variant generation (TR↔EN)

---

## Rejected Candidates (with reasons)

| Component | Why Not |
|---|---|
| **retry_engine** (`src/core/retry.py`) | Dataclass + if/else. Tenacity exists. No standalone value. |
| **agent_actions** (`src/models/models.py`) | Trivial Pydantic models. KutAI-specific validation rules. |
| **scaffolder** (`src/tools/scaffolder.py`) | Dict of string templates. Cookiecutter exists. 167 lines. |
| **free_apis** (`src/tools/free_apis.py`) | Dict of URLs + httpx wrappers. Not a library. |
| **ReAct framework** (`src/agents/base.py`) | Crowded space (LangGraph, CrewAI, smolagents). Heavy refactor for marginal differentiation. |
| **LLM task router** (`src/core/llm_dispatcher.py` + `router.py`) | 15-dim scoring is powerful but too opinionated. Configuration surface too large for adoption. |
| **Embedding + vector store** (`src/memory/`) | Thin wrappers around sentence-transformers + ChromaDB. No value-add. |
| **Workflow engine** (`src/workflows/engine/`) | Tightly coupled to DB, blackboard, tools. Not extractable. |
| **Skills system** (`src/memory/skills.py`) | Hard-coupled to DB persistence. |
| **Web search pipeline** (`src/tools/web_search.py`) | 10+ internal imports, deeply integrated. |

---

## Infrastructure Notes

### Package Structure Pattern
All packages follow the same layout (established by yazbunu):
```
packages/<name>/
├── pyproject.toml          # setuptools, src layout
├── .gitignore              # *.egg-info/, __pycache__/
├── src/<name>/
│   ├── __init__.py         # public API exports
│   └── ...modules...
└── tests/
    └── test_*.py
```

### KutAI Integration Pattern
1. Package lives in `packages/<name>/` during active development
2. Editable install via `-e ./packages/<name>` in requirements.txt
3. KutAI's original module becomes a thin re-export shim (e.g., `src/tools/scraper.py`)
4. All existing `from src.module import ...` imports continue working
5. When stable, push to separate GitHub repo, tag with `kutai` topic
6. Production install switches to `git+https://github.com/msakirc/<name>.git`

### GitHub Discovery
- All KutAI ecosystem repos tagged with topic `kutai`
- Discovery page: github.com/topics/kutai
- Repos under `msakirc/` (personal account, not org)

### Development Workflow
- Develop in `packages/` with editable installs for rapid iteration
- Run package tests: `pytest packages/<name>/tests/ -v`
- Run KutAI shim tests: `pytest tests/test_<name>.py -v`
- When stable, sync to separate repo and update requirements.txt
