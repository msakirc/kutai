# KutAI — Claude Code Instructions

## Project Overview
KutAI is an autonomous AI agent system controlled via Telegram. It manages missions, tasks, shopping, todos, and workflows using local LLMs (llama-server/Ollama) and a modular agent architecture.

## Architecture
- **Entry point**: `kutai_wrapper.py` (Yaşar Usta) → `src/app/run.py` → `src/core/orchestrator.py`
- **Telegram interface**: `src/app/telegram_bot.py` (TelegramInterface class, ~3000 lines)
- **Agents**: `src/agents/` — base.py (ReAct loop), specialized agents (coder, researcher, planner, etc.)
- **LLM routing**: `src/core/router.py` — pure model scoring (15-dimension capability vectors), no I/O
- **LLM dispatch**: `src/core/llm_dispatcher.py` — candidate iteration, swap budget, model selection policy
- **LLM execution**: `packages/hallederiz_kadir/` (HaLLederiz Kadir) — litellm calls, streaming, retries, response parsing, quality checks
- **Model management**: `src/models/local_model_manager.py` — manages llama-server lifecycle
- **Database**: SQLite via `src/infra/db.py` (aiosqlite, WAL mode)
- **Vector store**: ChromaDB via `src/memory/vector_store.py`
- **Shopping**: `src/shopping/` — product search, comparison, price watching
- **Workflows**: `src/workflows/` — multi-phase mission pipelines

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
- **Yaşar Usta** (`kutai_wrapper.py`) is the process manager. It manages the orchestrator lifecycle, auto-restarts on crashes with escalating backoff (5→15→60→300s), detects hung orchestrators via heartbeat, and provides Telegram commands when KutAI is down.
- Yaşar Usta uses a two-file lock: `logs/wrapper.lock` (PID, always readable) + `logs/wrapper.lk` (msvcrt exclusive lock sentinel). After power failures, the lock can become stale — the lock mechanism reads the PID and checks if it's alive before refusing to start.
- **Interface naming**: The bot is displayed as "Kutay" in Telegram (user-facing name). The codebase uses "KutAI" internally. Never change internal module/class names, only Telegram-facing strings.

### Testing
- **ALWAYS test changes before committing** — run `python -c "from src.module import X"` at minimum
- **ALWAYS verify Telegram commands work** after modifying `telegram_bot.py`
- Run `pytest tests/` when modifying core logic
- Integration tests: `pytest tests/integration/ -m "not llm"` (no LLM required)

### Code Style
- Async throughout — use `async/await`, not sync blocking
- Lazy imports for cross-module dependencies to avoid circular imports
- `src/infra/db.py` imports from `src/memory/` lazily (inside functions)
- All agents inherit from `BaseAgent` in `src/agents/base.py`
- Use `get_logger("component.name")` from `src/infra/logging_config.py`

### Telegram Bot
- Bot token and admin chat ID come from `.env`
- Reply keyboard must be sent with messages for persistent buttons
- `python-telegram-bot` library (v20+, async)
- Command handlers registered in `_setup_handlers()`
- Inline menus use callback queries handled in `handle_callback()`
- **Yaşar Usta polls Telegram when KutAI is down** using non-destructive mode (never advances offset past non-wrapper updates, preserving them for the orchestrator)

### LLM Dispatch & Model Routing
- **All LLM calls go through `LLMDispatcher`** (`src/core/llm_dispatcher.py`) — NEVER call `call_model()` directly from agents, classifiers, or graders
- **Three layers**: Dispatcher (policy + candidate iteration) → Router (pure scoring) → HaLLederiz Kadir (litellm execution). `call_model()` is a legacy shim that routes through dispatcher.
- Two call categories: `MAIN_WORK` (agent execution, can trigger model swaps) and `OVERHEAD` (classifier, grader, self-reflection — CANNOT trigger swaps, uses loaded model or cloud)
- **Deferred grading**: Non-urgent tasks (priority < 8) defer grading to `GradeQueue` instead of swapping models. Grading happens when: model naturally swaps for main work, cloud has headroom, or queue exceeds threshold
- **Proactive GPU loading**: If queue has ANY tasks a local model can handle, load one — don't wait for local_only/prefer_local flags. Local inference is free.
- **Model-aware task ordering**: After fetching ready tasks, reorder by loaded model affinity (up to +0.9 priority boost, never overrides 2+ priority gap)
- **Loaded model runtime state**: `ModelRuntimeState` tracks actual thinking_enabled, context_length, gpu_layers, measured_tps — scorer uses these instead of static ModelInfo
- **Swap budget**: Max 3 swaps per 5 minutes. Exemptions: local_only, priority>=9
- **Thinking/reasoning control**: llama.cpp v8668+ uses `--reasoning off --reasoning-budget 0` to disable thinking. The old `--chat-template-kwargs {"enable_thinking": false}` is deprecated and ignored. Always-on models (gpt-oss, Apriel) skip reasoning flags.
- See `docs/orchestrator-xray.md` for full architecture documentation

### Common Pitfalls
- Missing `import asyncio` in `base.py` — agents use asyncio.wait_for extensively
- `BLOCKED_PATTERNS` must be defined before `LOCAL_BLOCKED_PATTERNS` in `shell.py`
- ChromaDB is a required dependency — don't make it optional
- Embedding model is `intfloat/multilingual-e5-base` (768 dims) — don't mix with other models
- The `memory` table uses exact key lookup — use `semantic_recall()` for fuzzy search
- Shopping after `/shop` must route to `shopping_advisor` agent, NOT `i2p` workflow
- **Datetime format for scheduled_tasks**: NEVER use `datetime.isoformat()` when storing to `scheduled_tasks.next_run` or `last_run` — use `strftime("%Y-%m-%d %H:%M:%S")` because SQLite's `datetime('now')` returns space-separated format. ISO format (with `T`) causes string comparison failures.
- **Shopping agents must NOT have file tools**: `shopping_advisor`, `product_researcher`, and `deal_analyst` must NOT have `read_file`, `write_file`, or `file_tree` in their `allowed_tools` — these cause the LLM to waste iterations browsing the filesystem instead of searching products.
- **Never call `call_model()` directly** — always use `LLMDispatcher.request()`. `call_model()` is a legacy shim; direct calls bypass swap protection, quota management, and deferred grading.
- **LLM call bugs go to `packages/hallederiz_kadir/`** — timeout, retry, streaming, response parsing, quality check issues live there. Don't touch router or dispatcher for call execution bugs.
- **`shopping_advisor` task profile** must exist in `TASK_PROFILES` (capabilities.py) — without it, shopping tasks fall back to a flat adhoc profile with bad scoring.
- **Never pass `--n-gpu-layers` to llama-server** — it overrides `--fit` (default-on since v8000+). `--fit` auto-calculates optimal GPU layer allocation. Forcing `--n-gpu-layers 99` causes VRAM thrashing for models that don't fully fit (e.g. Apriel 8.7GB on 8GB GPU: 3.7→6.8 tok/s). Only pass `--n-gpu-layers` when `models.yaml` has an explicit `gpu_layers` override.

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
- Logs: `logs/` directory (wrapper_meta.log, wrapper.log, orchestrator.jsonl)

## Key Files
| File | Purpose |
|------|---------|
| `kutai_wrapper.py` | **Yaşar Usta** — process manager, auto-restart, heartbeat watchdog, Telegram polling when down |
| `src/app/run.py` | Orchestrator startup, health checks |
| `src/app/telegram_bot.py` | All Telegram UI — commands, buttons, callbacks |
| `src/core/orchestrator.py` | Main loop, task processing, agent dispatch |
| `src/core/router.py` | Pure model scoring — 15-dimension capability vectors, no I/O |
| `src/core/llm_dispatcher.py` | LLM call coordinator — candidate iteration, swap budget, calls HaLLederiz Kadir |
| `packages/hallederiz_kadir/` | HaLLederiz Kadir — LLM call execution hub: litellm, streaming, retries, quality checks |
| `docs/orchestrator-xray.md` | Architecture X-ray: routing, concurrency, resource management |
| `src/agents/base.py` | ReAct agent loop, tool execution, context building |
| `src/infra/db.py` | Database schema, queries, memory storage |
| `src/memory/rag.py` | RAG pipeline for agent context injection |
| `src/memory/embeddings.py` | Embedding generation (multilingual-e5-base) |
| `src/memory/vector_store.py` | ChromaDB collections and queries |
| `src/app/config.py` | Environment config constants |
| `requirements.txt` | Python dependencies |
| `src/tools/scraper.py` | Tiered web scraper (HTTP→TLS→Stealth→Browser) |
| `src/tools/free_apis.py` | Free API registry with auto-growth discovery |
| `src/tools/pharmacy.py` | Pharmacy on duty finder with distance calc |
| `src/tools/play_store.py` | Google Play Store search/reviews/similar |
| `src/tools/github_search.py` | GitHub repo/code search via REST API |
| `src/tools/scaffolder.py` | Project scaffolding templates (fastapi/nextjs/expo/flask) |
| `src/memory/skills.py` | Skill library with regex + vector matching |
| `src/memory/seed_skills.py` | 23 curated routing skills seeded at startup |
| `src/shopping/scrapers/` | 15 Turkish e-commerce scrapers |
| `src/app/price_watch_checker.py` | Daily price watch re-scraper + alert system |
| `mcp.yaml` | MCP server config (Fetch, Sequential Thinking, GitHub) |

## Strategic Context
- **Competitive edge**: Local GPU management (swap budgets, affinity), Turkish shopping intelligence (15 scrapers), self-improving skills
- **Don't compete on breadth** — depth in chosen domains beats framework ecosystems
- **Detailed analysis**: `docs/research/2026-03-30-competitive-analysis.md`
- **Known refactoring need**: `telegram_bot.py` (~3400 lines) should be split into modules
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
