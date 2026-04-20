# KutAI — Claude Code Instructions

## Project Overview
KutAI is an autonomous AI agent system controlled via Telegram. It manages missions, tasks, shopping, todos, and workflows using local LLMs (llama-server/Ollama) and a modular agent architecture.

## Architecture
- **Entry point**: `kutai_wrapper.py` → `packages/yasar_usta/` (Yaşar Usta) → `src/app/run.py` → `src/core/orchestrator.py`
- **Telegram interface**: `src/app/telegram_bot.py` (TelegramInterface class, ~5800 lines)
- **Agents**: `src/agents/` — base.py (ReAct loop), specialized agents (coder, researcher, planner, etc.)
- **Model selection**: `packages/fatih_hoca/` (Fatih Hoca) — 15-dimension scoring, task profiles, swap budget, failure adaptation
- **Mechanical dispatcher**: `packages/salako/` (Salako) — non-LLM executors (workspace snapshot, git auto-commit, clarify, notify_user) invoked via `salako.run(task)` when `agent_type == "mechanical"`
- **Task master**: `packages/general_beckman/` (General Beckman) — task queue, lifecycle, look-ahead against cloud quota. Public API: `next_task()`, `on_task_finished()`, `enqueue()`.
- **LLM dispatch**: `src/core/llm_dispatcher.py` — thin ask→load→call→retry loop, delegates selection to Fatih Hoca
- **LLM execution**: `packages/hallederiz_kadir/` (HaLLederiz Kadir) — litellm calls, streaming, retries, response parsing, quality checks
- **Model management**: `packages/dallama/` (DaLLaMa) — llama-server process lifecycle, swap orchestration, health polling
- **System state**: `packages/nerd_herd/` (Nerd Herd) — GPU, VRAM, inference metrics, system snapshots for Fatih Hoca
- **Quality checks**: `packages/dogru_mu_samet/` (Doğru mu Samet) — degenerate/repetitive output detection
- **Cloud capacity**: `packages/kuleden_donen_var/` — provider rate limit tracking
- **Web scraping**: `packages/vecihi/` (Vecihi) — auto-escalating HTTP→TLS→Stealth→Browser
- **Database**: SQLite via `src/infra/db.py` (aiosqlite, WAL mode)
- **Vector store**: ChromaDB via `src/memory/vector_store.py`
- **Shopping**: `src/shopping/` — product search, comparison, price watching
- **Workflows**: `src/workflows/` — multi-phase mission pipelines
- **Shims**: `src/core/router.py`, `src/models/model_registry.py`, `src/models/capabilities.py`, `src/models/quota_planner.py`, `src/models/model_profiles.py`, `src/models/local_model_manager.py` — thin re-exports from extracted packages for backward compatibility

## Critical Rules

### ⚠️⚠️⚠️ REMOTE CONTROL SESSION RECOVERY (READ THIS FIRST) ⚠️⚠️⚠️
**If you receive the SAME message from the user 2-3 times, or detect signs of connection instability (repeated content, garbled input, user mentioning disconnects), the Remote Control session is likely broken.** The user is AFK during Remote Control — they CANNOT act on instructions. DO NOT keep trying to respond normally. Instead, IMMEDIATELY recover the session yourself:

1. List recent sessions: `ls -lt /c/Users/sakir/.claude/projects/C--Users-sakir-Dropbox-Workspaces-kutay/*.jsonl | head -5`
2. Identify the dropped session by peeking at the first user message: `head -5 <session>.jsonl | python -c "import sys,json; [print(json.loads(l).get('message',{}).get('content','')[:200]) for l in sys.stdin if 'user' in l]"`
3. Read the session JSONL to understand what was being worked on and what progress was made
4. **Continue the work in the current session** using the recovered context — pick up where the dropped session left off
5. If you cannot determine what was in progress, summarize what you found and wait for the user

**The old PowerShell window showing "active" is a lie — it's a dead connection. Killing that PowerShell is safe (does NOT affect llama-server or KutAI).**

### Process Management
- **NEVER use taskkill on llama-server** — it corrupts model state and VRAM
- **NEVER force-kill KutAI** when Telegram is responsive — use `/restart` or `/stop` via Telegram, or exit code 42. However, if the bot is hung and `/restart` doesn't work, killing the **orchestrator process** (NOT llama-server) is acceptable — Yaşar Usta will auto-restart it.
- **Yaşar Usta** (`packages/yasar_usta/`, entry point `kutai_wrapper.py`) is the process manager. It manages the orchestrator lifecycle, auto-restarts on crashes with escalating backoff (5→15→60→300s), detects hung orchestrators via heartbeat, and has its own dedicated Telegram bot (`YASAR_USTA_BOT_TOKEN`) for commands when KutAI is down.
- Yaşar Usta uses a two-file lock: `logs/guard.lock` (PID, always readable) + `logs/guard.lk` (msvcrt exclusive lock sentinel). After power failures, the lock can become stale — the lock mechanism reads the PID and checks if it's alive before refusing to start.
- **Interface naming**: The bot is displayed as "Kutay" in Telegram (user-facing name). The codebase uses "KutAI" internally. Never change internal module/class names, only Telegram-facing strings.

### Testing
- **ALWAYS test changes before committing** — run `python -c "from src.module import X"` at minimum
- **ALWAYS verify Telegram commands work** after modifying `telegram_bot.py`
- **NEVER run pytest without a timeout** — use `timeout 30 pytest tests/...` (targeted) or `timeout 120 pytest tests/` (full suite). Zombie pytest processes hold SQLite write locks and crash-loop KutAI.
- Integration tests: `timeout 60 pytest tests/integration/ -m "not llm"` (no LLM required)

### Code Style
- Async throughout — use `async/await`, not sync blocking
- Lazy imports for cross-module dependencies to avoid circular imports
- `src/infra/db.py` imports from `src/memory/` lazily (inside functions)
- All agents inherit from `BaseAgent` in `src/agents/base.py`
- Use `get_logger("component.name")` from `src/infra/logging_config.py`

### Telegram Bot
- **Two bots**: KutAI bot (`TELEGRAM_BOT_TOKEN`) for normal operation, Yaşar Usta bot (`YASAR_USTA_BOT_TOKEN`) for when KutAI is down
- Reply keyboard must be sent with messages for persistent buttons
- `python-telegram-bot` library (v20+, async)
- Command handlers registered in `_setup_handlers()`
- Inline menus use callback queries handled in `handle_callback()`

### LLM Dispatch & Model Routing
- **All LLM calls go through `LLMDispatcher`** (`src/core/llm_dispatcher.py`) — NEVER call `call_model()` directly from agents, classifiers, or graders
- **Four layers**: Dispatcher (ask→load→call→retry) → Fatih Hoca (model selection, scoring) → DaLLaMa (llama-server management) + HaLLederiz Kadir (litellm execution). `call_model()` is a legacy shim that routes through dispatcher.
- **Dispatcher** is a thin loop: calls `fatih_hoca.select()`, loads via DaLLaMa, calls HaLLederiz Kadir, retries with failure feedback (max 5 attempts). Owns message preparation (secret redaction, thinking adaptation) and timeout floors.
- **Fatih Hoca** owns all model knowledge: catalog (YAML+GGUF), benchmark enrichment (AA + 9 sources, cached in `.benchmark_cache/`, wired into `init()` as of 2026-04-17), 15-dimension scoring, task profiles, swap budget (max 3/5min), failure adaptation, quota planning. Queries Nerd Herd for system state via `snapshot()`. Every pick is persisted to `model_pick_log` for offline weight tuning.
- Two call categories: `MAIN_WORK` (agent execution, can trigger model swaps) and `OVERHEAD` (classifier, grader, self-reflection). Both go through the same `select→load→call` path; Fatih Hoca handles the distinction via scoring weights (loaded model gets massive stickiness for overhead).
- **Thinking/reasoning control**: llama.cpp v8668+ uses `--reasoning off --reasoning-budget 0` to disable thinking. The old `--chat-template-kwargs {"enable_thinking": false}` is deprecated and ignored. Always-on models (gpt-oss, Apriel) skip reasoning flags.
- See `docs/architecture-modularization.md` for full architecture documentation

### Common Pitfalls
- Missing `import asyncio` in `base.py` — agents use asyncio.wait_for extensively
- `BLOCKED_PATTERNS` must be defined before `LOCAL_BLOCKED_PATTERNS` in `shell.py`
- ChromaDB is a required dependency — don't make it optional
- Embedding model is `intfloat/multilingual-e5-base` (768 dims) — don't mix with other models
- The `memory` table uses exact key lookup — use `semantic_recall()` for fuzzy search
- Shopping after `/shop` must route to `shopping_advisor` agent, NOT `i2p` workflow
- **Datetime format for scheduled_tasks**: NEVER use `datetime.isoformat()` when storing to `scheduled_tasks.next_run` or `last_run` — use `strftime("%Y-%m-%d %H:%M:%S")` because SQLite's `datetime('now')` returns space-separated format. ISO format (with `T`) causes string comparison failures.
- **Shopping agents must NOT have file tools**: `shopping_advisor`, `product_researcher`, and `deal_analyst` must NOT have `read_file`, `write_file`, or `file_tree` in their `allowed_tools` — these cause the LLM to waste iterations browsing the filesystem instead of searching products.
- **Never call `call_model()` directly** — always use `LLMDispatcher.request()`. `call_model()` is a legacy shim; direct calls bypass failure tracking and retry logic.
- **LLM call bugs go to `packages/hallederiz_kadir/`** — timeout, retry, streaming, response parsing, quality check issues live there. Don't touch dispatcher for call execution bugs.
- **Model selection bugs go to `packages/fatih_hoca/`** — scoring, task profiles, swap budget, eligibility filtering. Don't touch dispatcher for selection bugs.
- **Mechanical executor (salako)**: steps with `agent: "mechanical"` in a workflow are routed to `salako.run()` before the LLM path. Expander propagates `executor` + `payload` into task context so they survive the DB round-trip. Auto-commit is now an **explicit** i2p step (3.git_commit siblings after key coder milestones) — ad-hoc `/task` with a coder agent no longer auto-commits; add a mechanical sibling step if needed.
- **Benchmark cache staleness**: `.benchmark_cache/_bulk_*.json` TTL is 48h. When stale, `BenchmarkCache.load()` returns None with a WARNING log (no silent fallback). Refresh with `python -m src.models.benchmark.benchmark_cli benchmarks`. Check `model_pick_log.snapshot_summary` if selection quality drops; query `SELECT picked_model, AVG(picked_score), COUNT(*) FROM model_pick_log GROUP BY picked_model ORDER BY 3 DESC` to see usage.
- **Never pass `--n-gpu-layers` to llama-server** — it overrides `--fit` (default-on since v8000+). `--fit` auto-calculates optimal GPU layer allocation. Forcing `--n-gpu-layers 99` causes VRAM thrashing for models that don't fully fit (e.g. Apriel 8.7GB on 8GB GPU: 3.7→6.8 tok/s). Only pass `--n-gpu-layers` when `models.yaml` has an explicit `gpu_layers` override.
- **Phase 2d utilization equilibrium (2026-04-20)**: `scarcity.py` computes signed scarcity ∈ [-1, +1] per pool (local / time_bucketed / per_call). `capability_curve.py` holds `CAP_NEEDED_BY_DIFFICULTY`. `ranking.py::_apply_utilization_layer` applies `composite *= 1 + K * scarcity * fit_dampener` with `K = UTILIZATION_K = 1.0`. No gate — the dampener handles "don't waste capability" naturally (symmetric for positive scarcity, over-qual-only for negative). Stickiness dialed 1.4→1.10 (main) / 2.0→1.50 (overhead) and fades when loaded local is under-qualified — it was overpowering cloud on hard tasks. Per_call has an abundance arm (+1 when budget >15% + task d≥7), depletion arm (<15%), and queue-pressure arm (easy tasks + hard queue ahead). Time_bucketed uses continuous `exp(-reset_in / 24h)` decay — 24h reset still a meaningful +0.37 signal. Validated by 7 stateful-simulator scenarios + real-registry swap-storm verification. To tune, always re-run `packages/fatih_hoca/tests/sim/run_scenarios.py` and `run_swap_storm_check.py`. Design reasoning at `docs/architecture/fatih-hoca-phase2d-equilibrium.md`.

### Telegram Bot Patterns
- **`_pending_action` flow**: When a command is called without args (e.g. `/shop`), it stores `_pending_action[chat_id]` and prompts the user. The NEXT message MUST be handled by checking `_pending_action` BEFORE calling the message classifier — otherwise it gets misclassified (e.g. "Coffee machine" routed to a workflow instead of shopping).
- **`REPLY_KEYBOARD` on every reply**: Every `reply_text` call must include `reply_markup=REPLY_KEYBOARD` or the persistent keyboard buttons disappear. The `_reply()` helper method handles this automatically — always prefer `_reply()` over raw `reply_text`.

### Git
- Commit messages follow conventional commits: `feat()`, `fix()`, `docs:`, `test:`
- Push to `main` branch directly (no PR workflow currently)

## Environment
- Windows 11, Python 3.10 (venv at `.venv/`)
- GPU: NVIDIA (shared between llama-server and optional Ollama)
- Embedding: sentence-transformers on CPU (multilingual-e5-base, 768d)
- DB path: configured in `.env` via `DB_PATH`
- Logs: `logs/` directory (guard.log, guard.jsonl, orchestrator.jsonl, dallama.jsonl, plus per-package logs)

## Key Files
| File | Purpose |
|------|---------|
| `kutai_wrapper.py` | Thin entry point → delegates to `packages/yasar_usta/` |
| `packages/yasar_usta/` | **Yaşar Usta** — process manager, auto-restart, heartbeat watchdog, own Telegram bot when KutAI is down |
| `packages/fatih_hoca/` | **Fatih Hoca** — model selection: scoring, task profiles, swap budget, failure adaptation |
| `packages/salako/` | **Salako** — mechanical dispatcher: workspace snapshot + git auto-commit + clarify + notify_user (non-LLM executors) |
| `packages/general_beckman/` | **General Beckman** — task master: queue selection, lifecycle (apply/retry/sweep/rewrite), quota look-ahead (Phase 2b Task 13) |
| `packages/dallama/` | **DaLLaMa** — llama-server process lifecycle, swap orchestration, health polling |
| `packages/hallederiz_kadir/` | **HaLLederiz Kadir** — LLM call execution: litellm, streaming, retries, quality checks |
| `packages/nerd_herd/` | **Nerd Herd** — system state: GPU, VRAM, inference metrics, snapshots for Fatih Hoca |
| `packages/dogru_mu_samet/` | **Doğru mu Samet** — degenerate/repetitive output detection |
| `packages/kuleden_donen_var/` | Cloud provider rate limit and capacity tracking |
| `packages/vecihi/` | **Vecihi** — auto-escalating web scraper (HTTP→TLS→Stealth→Browser) |
| `src/app/run.py` | Orchestrator startup, health checks |
| `src/app/telegram_bot.py` | All Telegram UI — commands, buttons, callbacks (~5800 lines) |
| `src/core/orchestrator.py` | Thin pump loop (~30 lines): beckman.next_task + asyncio.create_task dispatch (366 lines total including startup/shutdown) |
| `src/core/llm_dispatcher.py` | Thin ask→load→call→retry loop, delegates to Fatih Hoca + HaLLederiz Kadir |
| `src/core/router.py` | Shim — re-exports from fatih_hoca, keeps `call_model()` legacy shim |
| `docs/architecture-modularization.md` | Architecture doc: package boundaries, data flow, troubleshooting |
| `src/agents/base.py` | ReAct agent loop, tool execution, context building |
| `src/infra/db.py` | Database schema, queries, memory storage |
| `src/memory/rag.py` | RAG pipeline for agent context injection |
| `src/memory/embeddings.py` | Embedding generation (multilingual-e5-base) |
| `src/memory/vector_store.py` | ChromaDB collections and queries |
| `src/memory/skills.py` | Skill library with regex + vector matching |
| `src/memory/seed_skills.py` | 23 curated routing skills seeded at startup |
| `src/shopping/scrapers/` | 15 Turkish e-commerce scrapers |
| `src/app/price_watch_checker.py` | Daily price watch re-scraper + alert system |

## Strategic Context
- **Competitive edge**: Local GPU management (swap budgets, affinity), Turkish shopping intelligence (15 scrapers), self-improving skills
- **Don't compete on breadth** — depth in chosen domains beats framework ecosystems
- **Detailed analysis**: `docs/research/2026-03-30-competitive-analysis.md`
- **Known refactoring need**: `telegram_bot.py` (~5800 lines) should be split into modules
- **Skill system**: Auto-captures from 2+ iteration tasks, prunes bad skills (<30% success), ranks by Bayesian effectiveness. `/skillstats` shows A/B lift. See `src/memory/skills.py`
- **i2p v3 workflow**: 170 steps (was 328 in v2), difficulty routing (easy/medium/hard → model selection), artifact schema validation with retry, tools_hint per step, skip_when conditions, cross-feature dependencies, auto integration tests. See `src/workflows/i2p/i2p_v3.json`
- **Search pipeline**: 4-tier scraper (HTTP→TLS→Stealth→Browser), Brave+GCSE fallbacks, source quality tracking. See `docs/web-search-xray.md`
- **Shopping missions**: Two-tier — simple queries use single agent, complex queries create 3-task mission (researcher→analyst→advisor)
- **Free API registry**: 13 static APIs + auto-growth from public-apis/free-apis. See `src/tools/free_apis.py`

## Todo Module
- Table: `todo_items` in main DB
- Commands: `/todo`, `/todos`, `/cleartodos`
- Requirements: collect todos, remind every 2h, AI suggestions, easy mark-done via inline buttons
- Reminder system in `src/app/reminders.py`
- Orchestrator has todo scheduling logic in `_check_todo_reminders()`
