# KutAI — Claude Code Instructions

## Project Overview
KutAI is an autonomous AI agent system controlled via Telegram. It manages missions, tasks, shopping, todos, and workflows using local LLMs (llama-server/Ollama) and a modular agent architecture.

## Architecture
- **Entry point**: `kutai_wrapper.py` → `src/app/run.py` → `src/core/orchestrator.py`
- **Telegram interface**: `src/app/telegram_bot.py` (TelegramInterface class, ~3000 lines)
- **Agents**: `src/agents/` — base.py (ReAct loop), specialized agents (coder, researcher, planner, etc.)
- **LLM routing**: `src/core/router.py` — routes tasks to best available local model
- **LLM dispatch**: `src/core/llm_dispatcher.py` — centralized LLM call coordinator (MAIN_WORK vs OVERHEAD)
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
- **NEVER force-kill KutAI** when Telegram is responsive — use `/restart` or `/stop` via Telegram, or exit code 42. However, if the bot is hung and `/restart` doesn't work, killing the **orchestrator process** (NOT llama-server) is acceptable — the wrapper will auto-restart it.
- The wrapper (`kutai_wrapper.py`) manages the orchestrator lifecycle
- The wrapper has a file lock (`logs/wrapper.lock`) to prevent duplicates. After power failures, the lock can become stale — the lock mechanism uses zero-padded PIDs and stale-lock recovery (checks if PID is alive before refusing to start).

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
- **The wrapper polls Telegram when KutAI is down** using non-destructive mode (never advances offset past non-wrapper updates, preserving them for the orchestrator)

### LLM Dispatch & Model Routing
- **All LLM calls go through `LLMDispatcher`** (`src/core/llm_dispatcher.py`) — NEVER call `call_model()` directly from agents, classifiers, or graders
- Two call categories: `MAIN_WORK` (agent execution, can trigger model swaps) and `OVERHEAD` (classifier, grader, self-reflection — CANNOT trigger swaps, uses loaded model or cloud)
- **Deferred grading**: Non-urgent tasks (priority < 8) defer grading to `GradeQueue` instead of swapping models. Grading happens when: model naturally swaps for main work, cloud has headroom, or queue exceeds threshold
- **Proactive GPU loading**: If queue has ANY tasks a local model can handle, load one — don't wait for local_only/prefer_local flags. Local inference is free.
- **Model-aware task ordering**: After fetching ready tasks, reorder by loaded model affinity (up to +0.9 priority boost, never overrides 2+ priority gap)
- **Loaded model runtime state**: `ModelRuntimeState` tracks actual thinking_enabled, context_length, gpu_layers, measured_tps — scorer uses these instead of static ModelInfo
- **Swap budget**: Max 3 swaps per 5 minutes. Exemptions: local_only, priority>=9
- See `docs/orchestrator-xray.md` for full architecture documentation

### Common Pitfalls
- Missing `import asyncio` in `base.py` — agents use asyncio.wait_for extensively
- `BLOCKED_PATTERNS` must be defined before `LOCAL_BLOCKED_PATTERNS` in `shell.py`
- ChromaDB is a required dependency — don't make it optional
- Embedding model is `intfloat/multilingual-e5-small` (384 dims) — don't mix with other models
- The `memory` table uses exact key lookup — use `semantic_recall()` for fuzzy search
- Shopping after `/shop` must route to `shopping_advisor` agent, NOT `idea_to_product` workflow
- **Datetime format for scheduled_tasks**: NEVER use `datetime.isoformat()` when storing to `scheduled_tasks.next_run` or `last_run` — use `strftime("%Y-%m-%d %H:%M:%S")` because SQLite's `datetime('now')` returns space-separated format. ISO format (with `T`) causes string comparison failures.
- **Shopping agents must NOT have file tools**: `shopping_advisor`, `product_researcher`, and `deal_analyst` must NOT have `read_file`, `write_file`, or `file_tree` in their `allowed_tools` — these cause the LLM to waste iterations browsing the filesystem instead of searching products.
- **Never call `call_model()` directly** — always use `LLMDispatcher.request()`. Direct calls bypass swap protection, quota management, and deferred grading.
- **`shopping_advisor` task profile** must exist in `TASK_PROFILES` (capabilities.py) — without it, shopping tasks fall back to a flat adhoc profile with bad scoring.

### Telegram Bot Patterns
- **`_pending_action` flow**: When a command is called without args (e.g. `/shop`), it stores `_pending_action[chat_id]` and prompts the user. The NEXT message MUST be handled by checking `_pending_action` BEFORE calling the message classifier — otherwise it gets misclassified (e.g. "Coffee machine" routed to a workflow instead of shopping).
- **`REPLY_KEYBOARD` on every reply**: Every `reply_text` call must include `reply_markup=REPLY_KEYBOARD` or the persistent keyboard buttons disappear. The `_reply()` helper method handles this automatically — always prefer `_reply()` over raw `reply_text`.

### Git
- Commit messages follow conventional commits: `feat()`, `fix()`, `docs:`, `test:`
- Push to `main` branch directly (no PR workflow currently)

## Environment
- Windows 11, Python 3.10 (venv at `.venv/`)
- GPU: NVIDIA (shared between llama-server and optional Ollama)
- Embedding: sentence-transformers on CPU (primary), Ollama on GPU (fallback)
- DB path: configured in `.env` via `DB_PATH`
- Logs: `logs/` directory (wrapper_meta.log, wrapper.log, orchestrator.jsonl)

## Key Files
| File | Purpose |
|------|---------|
| `kutai_wrapper.py` | Process manager, auto-restart, Telegram polling when down |
| `src/app/run.py` | Orchestrator startup, health checks |
| `src/app/telegram_bot.py` | All Telegram UI — commands, buttons, callbacks |
| `src/core/orchestrator.py` | Main loop, task processing, agent dispatch |
| `src/core/router.py` | LLM model selection and routing |
| `src/core/llm_dispatcher.py` | Centralized LLM call coordinator — all LLM calls go through here |
| `docs/orchestrator-xray.md` | Architecture X-ray: routing, concurrency, resource management |
| `src/agents/base.py` | ReAct agent loop, tool execution, context building |
| `src/infra/db.py` | Database schema, queries, memory storage |
| `src/memory/rag.py` | RAG pipeline for agent context injection |
| `src/memory/embeddings.py` | Embedding generation (multilingual-e5-small) |
| `src/memory/vector_store.py` | ChromaDB collections and queries |
| `src/app/config.py` | Environment config constants |
| `requirements.txt` | Python dependencies |

## Strategic Context
- **Competitive edge**: Local GPU management (swap budgets, affinity), Turkish shopping intelligence (15 scrapers), self-improving skills
- **Don't compete on breadth** — depth in chosen domains beats framework ecosystems
- **Detailed analysis**: `docs/research/2026-03-30-competitive-analysis.md`
- **Known refactoring need**: `telegram_bot.py` (~3400 lines) should be split into modules
- **Skill system**: Working but needs instrumentation to prove value. See `src/memory/skills.py`
- **Search pipeline**: 4-tier scraper (HTTP→TLS→Stealth→Browser), Brave+GCSE fallbacks, source quality tracking. See `docs/web-search-xray.md`
- **Shopping missions**: Two-tier — simple queries use single agent, complex queries create 3-task mission (researcher→analyst→advisor)
- **Free API registry**: 13 static APIs + auto-growth from public-apis/free-apis. See `src/tools/free_apis.py`

## Todo Module
- Table: `todo_items` in main DB
- Commands: `/todo`, `/todos`, `/cleartodos`
- Requirements: collect todos, remind every 2h, AI suggestions, easy mark-done via inline buttons
- Reminder system in `src/app/reminders.py`
- Orchestrator has todo scheduling logic in `_check_todo_reminders()`
