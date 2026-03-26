# KutAI — Claude Code Instructions

## Project Overview
KutAI is an autonomous AI agent system controlled via Telegram. It manages missions, tasks, shopping, todos, and workflows using local LLMs (llama-server/Ollama) and a modular agent architecture.

## Architecture
- **Entry point**: `kutai_wrapper.py` → `src/app/run.py` → `src/core/orchestrator.py`
- **Telegram interface**: `src/app/telegram_bot.py` (TelegramInterface class, ~3000 lines)
- **Agents**: `src/agents/` — base.py (ReAct loop), specialized agents (coder, researcher, planner, etc.)
- **LLM routing**: `src/core/router.py` — routes tasks to best available local model
- **Model management**: `src/models/local_model_manager.py` — manages llama-server lifecycle
- **Database**: SQLite via `src/infra/db.py` (aiosqlite, WAL mode)
- **Vector store**: ChromaDB via `src/memory/vector_store.py`
- **Shopping**: `src/shopping/` — product search, comparison, price watching
- **Workflows**: `src/workflows/` — multi-phase mission pipelines

## Critical Rules

### Process Management
- **NEVER use taskkill on llama-server** — it corrupts model state and VRAM
- **NEVER force-kill KutAI** — use `/restart` or `/stop` via Telegram, or exit code 42
- The wrapper (`kutai_wrapper.py`) manages the orchestrator lifecycle
- The wrapper has a file lock (`logs/wrapper.lock`) to prevent duplicates

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
- **The wrapper also polls Telegram when KutAI is down** — be careful about update consumption conflicts

### Common Pitfalls
- Missing `import asyncio` in `base.py` — agents use asyncio.wait_for extensively
- `BLOCKED_PATTERNS` must be defined before `LOCAL_BLOCKED_PATTERNS` in `shell.py`
- ChromaDB is a required dependency — don't make it optional
- Embedding model is `intfloat/multilingual-e5-small` (384 dims) — don't mix with other models
- The `memory` table uses exact key lookup — use `semantic_recall()` for fuzzy search
- Shopping after `/shop` must route to `shopping_advisor` agent, NOT `idea_to_product` workflow

### Git
- Commit messages follow conventional commits: `feat()`, `fix()`, `docs:`, `test:`
- Always include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`
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
| `src/agents/base.py` | ReAct agent loop, tool execution, context building |
| `src/infra/db.py` | Database schema, queries, memory storage |
| `src/memory/rag.py` | RAG pipeline for agent context injection |
| `src/memory/embeddings.py` | Embedding generation (multilingual-e5-small) |
| `src/memory/vector_store.py` | ChromaDB collections and queries |
| `src/app/config.py` | Environment config constants |
| `requirements.txt` | Python dependencies |

## Todo Module
- Table: `todo_items` in main DB
- Commands: `/todo`, `/todos`, `/cleartodos`
- Requirements: collect todos, remind every 2h, AI suggestions, easy mark-done via inline buttons
- Reminder system in `src/app/reminders.py`
- Orchestrator has todo scheduling logic in `_check_todo_reminders()`
