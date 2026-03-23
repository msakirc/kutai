Phase 1: Debugging, Logging & Notifications

1.1 — ntfy Configuration & Topics
Verify src/app/config.py exposes NTFY_URL, NTFY_USER, NTFY_PASS from .env; add them if missing. Define two topic constants: NTFY_TOPIC_ERRORS = "orchestrator-errors" (immediate push, phone notifications ON — only ERROR/CRITICAL) and NTFY_TOPIC_LOGS = "orchestrator-logs" (batched, phone notifications OFF — INFO/WARNING/ERROR for browsing). Errors appear in both topics so the logs topic tells the full story without switching. Add both topic names to config.

1.2 — ntfy Core Send Function
Create src/infra/notifications.py. Implement send_ntfy(topic, title, message, priority, tags) that POSTs to {NTFY_URL}/{topic} with basic auth from config. Include headers: Title, Priority, Tags. Return the response status. On network failure: log the failure to file logger (never silently swallow), do not raise. On auth failure: log as ERROR to file. Never except: pass — every failure path logs what happened and why.

1.3 — NtfyAlertHandler (Immediate Errors)
In src/infra/notifications.py, add a logging.Handler subclass NtfyAlertHandler with level=ERROR. On every emit(): POST immediately to orchestrator-errors. Priority mapping: CRITICAL → 5 (max, persistent notification with sound), ERROR → 4 (high, sound). Format the message to include: originating component name, task_id if bound to the record, one-line summary as title, full message + traceback (if present) as body. Tag with the level name lowercase. On send failure: log to file via a separate file-only logger — never recurse into itself, never silently drop.

1.4 — NtfyBatchHandler (Buffered Logs)
In src/infra/notifications.py, add a logging.Handler subclass NtfyBatchHandler with level=INFO. Maintain a threading.Lock-protected list of formatted log lines. Start a daemon threading.Timer that fires every 30 seconds. On each flush: join all buffered lines into a single message formatted as [HH:MM:SS] LEVEL component: message (one per line), POST to orchestrator-logs as one ntfy message. Set the batch message priority based on highest severity in the buffer: contains ERROR → priority 3, WARNING-only → priority 2, INFO-only → priority 1. If buffer exceeds 50 records before the timer fires, flush early to avoid memory buildup. On flush failure: retry once after 5 seconds, then log warning to file and keep the buffer for next cycle (do not drop). Register an atexit hook that performs a final flush on interpreter shutdown. Errors in this handler are logged to file, never swallowed.

1.5 — Structured Logger Configuration
Create src/infra/logging.py. Configure structlog with JSON output. Define the log schema: every entry must contain {timestamp, level, component, message} as mandatory fields, plus optional context fields {task_id, goal_id, agent_type, model, action, duration_ms} that appear when bound and are omitted (not null) when not. Set up structlog processors: ISO8601 timestamper, level name, stack info renderer for exceptions, JSON serializer for file output, key-value colorized renderer for console output. Export a get_logger(component: str) function that returns a structlog bound logger with component pre-set. This is the only way any module in the project should obtain a logger.

1.6 — Log Sink Wiring
In src/infra/logging.py, implement init_logging() that creates the logs/ directory if absent, then attaches exactly four handlers to the Python root logger. Sink 1: console — StreamHandler(stdout), level DEBUG, human-readable key=value format. Sink 2: file — RotatingFileHandler("logs/orchestrator.jsonl", maxBytes=50_000_000, backupCount=5), level DEBUG, JSON-lines format. Sink 3: NtfyBatchHandler from 1.4, level INFO. Sink 4: NtfyAlertHandler from 1.3, level ERROR. Set root logger level to DEBUG. After attaching all handlers, emit one INFO log "Logging initialized, all sinks active" to confirm the pipeline works end to end. This function must be called in src/app/run.py as the very first action before any other import that might log.

1.7 — Startup Logging Initialization
Modify src/app/run.py. Move or add init_logging() call as the first executable line after minimal imports. Ensure no module-level code in any import triggers logging before init_logging() has run. If that's unavoidable, configure a basic StreamHandler fallback in src/infra/logging.py at import time that gets replaced when init_logging() runs. After init, log the Python version, OS, working directory, and config file path at INFO level.

1.8 — Instrument Agents
Modify every file in src/agents/: analyst.py, architect.py, assistant.py, base.py, coder.py, error_recovery.py, executor.py, fixer.py, implementer.py, planner.py, researcher.py, reviewer.py, summarizer.py, test_generator.py, visual_reviewer.py, writer.py. Each file gets from src.infra.logging import get_logger and logger = get_logger("agents.<filename>") near the top. In base.py: log agent instantiation, method entry/exit for the main run/execute method with task_id and agent_type bound. In each concrete agent: log when invoked (DEBUG), which model was selected (INFO), prompt sent (DEBUG, truncate to 500 chars), response received (DEBUG, truncate), token count and duration_ms (INFO), any exception (ERROR with full traceback via logger.exception()). Remove all print() statements.

1.9 — Instrument Core
Modify src/core/orchestrator.py, router.py, state_machine.py, task_classifier.py. Each gets its own get_logger("core.<name>"). orchestrator.py: log task received (INFO), routing decision (INFO), agent dispatched (INFO), result received (INFO), errors (ERROR). router.py: log classification input (DEBUG), route chosen (INFO), confidence score (DEBUG), fallback triggers (WARNING). state_machine.py: log every state transition as INFO with from_state and to_state fields. task_classifier.py: log classification result and scores (INFO). Remove all print() statements.

1.10 — Instrument Tools
Modify every file in src/tools/: apply_diff.py, ast_tools.py, code_runner.py, codebase_index.py, deploy.py, deps.py, download.py, edit_file.py, git_ops.py, http_client.py, linting.py, patch_file.py, shell.py, web_search.py, workspace.py. Each gets get_logger("tools.<name>"). Log: tool invoked with parameters at DEBUG (redact any secrets — file paths OK, tokens/passwords replaced with ***), result summary at INFO (e.g., "edited 3 files", "shell returned exit code 0"), duration_ms at DEBUG, errors at ERROR with full traceback. For shell.py and code_runner.py: log the command executed (DEBUG) and exit code (INFO). For web_search.py: log query (INFO) and result count (DEBUG). Remove all print().

1.11 — Instrument Workflows
Modify all files in src/workflows/engine/: artifacts.py, conditions.py, dispatch.py, expander.py, hooks.py, loader.py, pipeline_artifacts.py, pipeline_bridge.py, policies.py, quality_gates.py, runner.py, status.py. And src/workflows/pipeline/: pipeline.py, pipeline_context.py, pipeline_utils.py. Each gets get_logger("workflows.engine.<name>") or get_logger("workflows.pipeline.<name>"). Log: step started (INFO), step completed with duration (INFO), quality gate pass/fail (INFO for pass, WARNING for fail), artifact produced (DEBUG), pipeline stage transitions (INFO), context mutations (DEBUG), policy decisions (INFO). Remove all print().

1.12 — Instrument Memory
Modify all files in src/memory/: conversations.py, decay.py, embeddings.py, episodic.py, ingest.py, preferences.py, rag.py, vector_store.py. Each gets get_logger("memory.<name>"). Log: store operations (DEBUG), retrieve operations (DEBUG), cache hits vs misses (DEBUG), embedding generation calls with token count (INFO), RAG query and result count (INFO), decay operations (DEBUG), errors (ERROR). Remove all print().

1.13 — Instrument Models
Modify all files in src/models/: capabilities.py, gpu_monitor.py, gpu_scheduler.py, header_parser.py, local_model_manager.py, model_families.yaml (skip), model_profiles.py, model_registry.py, models.py, quota_planner.py, rate_limiter.py, and src/models/benchmark/benchmark_cli.py, benchmark_fetcher.py. Each gets get_logger("models.<name>"). Log: model load/unload (INFO), GPU VRAM allocation and release (INFO), rate limit hits (WARNING), swap requests (INFO), benchmark runs (INFO), quota decisions (DEBUG), header parse results (DEBUG). Remove all print().

1.14 — Instrument Infra, Integrations, Context, Collaboration, Parsing, Security
Modify src/infra/backpressure.py, db.py, dead_letter.py, error_policy.py — logger "infra.<name>". Log: DB queries at DEBUG, backpressure triggers at WARNING, dead letter enqueues at WARNING, error policy decisions at INFO. Modify src/integrations/base.py, http_integration.py, registry.py — logger "integrations.<name>". Log: external API calls at INFO (URL, method, status code), response time at DEBUG, failures at ERROR. Modify src/context/assembler.py, onboarding.py, repo_map.py — logger "context.<name>". Log: context assembly at DEBUG, onboarding steps at INFO, repo map generation at INFO with file count. Modify src/collaboration/blackboard.py, plan_verification.py — logger "collaboration.<name>". Log: blackboard reads/writes at DEBUG, plan verification pass/fail at INFO. Modify src/parsing/code_embeddings.py, tree_sitter_parser.py — logger "parsing.<name>". Log: parse invocations at DEBUG, results at DEBUG. Modify src/security/credential_store.py, sensitivity.py — logger "security.<name>". Log: credential access at INFO (log the key name, NEVER the value), sensitivity classifications at DEBUG. Remove all print() across all these files.

1.15 — Instrument App Layer
Modify src/app/telegram_bot.py: logger "app.telegram_bot". Log: message received (INFO, log user and command, not full message body), command parsed (DEBUG), response sent (DEBUG), errors (ERROR). Modify src/app/config.py: logger "app.config". Log: config loaded successfully (INFO), each missing optional var (WARNING), missing required var (CRITICAL). Modify src/app/run.py: logger "app.run". Log: startup sequence begin (INFO), shutdown signal received (INFO), main loop iteration count periodically (DEBUG, every 100 iterations or every 5 minutes). Remove all print().

1.16 — Exception Discipline Pass
Search the entire src/ directory for every except clause. Eliminate all banned patterns: bare except:, except Exception: pass, except Exception as e: pass, any except block that returns None or continues without logging. Every except block must call logger.exception() or logger.error() or at minimum logger.warning() with the error message and relevant context. If a function intentionally suppresses an error (optional cleanup, best-effort operations), it must log at WARNING with a clear reason why suppression is acceptable. No exception may vanish silently. Create a grep-verifiable rule: every except line must have a corresponding logger. call within the next 5 lines of the same block.

1.17 — print() Elimination Pass
Search the entire src/ directory for every print() call. Replace each with the appropriate logger call: print() used for debugging → logger.debug(), print() used for status → logger.info(), print() used for errors → logger.error(). After this step, zero print() calls should remain in src/. The only acceptable stdout output comes from the console log handler configured in 1.6.

1.18 — Startup Health Check
Modify src/app/run.py. After init_logging(), before entering the main orchestrator loop, run a health check sequence. Checks in order: (1) .env loaded with all required vars present — critical, abort if missing, (2) logs/ directory writable with test write — critical, (3) DB writable via test insert/delete through src/infra/db.py — critical, (4) ntfy reachable via GET to NTFY_URL — non-critical, (5) Docker sandbox alive via docker inspect on sandbox container — non-critical, (6) llama.cpp / llama-swap reachable via health endpoint — critical if zero models available, (7) Perplexica container up via health endpoint at PERPLEXICA_URL — non-critical, (8) Telegram bot reachable and send "🟢 System online" — non-critical, (9) Frontail up via GET to http://localhost:9001 — non-critical. Log each check result at INFO (pass), WARNING (degraded), or CRITICAL (fail) with check name, status, and duration_ms. On critical failure: send ntfy alert if ntfy is available, then abort with exit code 1. On non-critical failure: set degradation flag and continue.

1.19 — Runtime State & Degradation Flags
Create a runtime state dict in src/app/run.py (or a new src/infra/runtime_state.py if cleaner) accessible as a singleton to the orchestrator and all agents. Structure: {"web_search_available": bool, "sandbox_available": bool, "ntfy_available": bool, "telegram_available": bool, "frontail_available": bool, "degraded_capabilities": [str], "boot_time": str, "models_loaded": [str]}. Populate from health check results in 1.18. Modify src/core/orchestrator.py and any agent/tool that uses optional services to check the relevant flag before use. When a capability is degraded, log INFO explaining the fallback behavior (e.g., "web_search_degraded, skipping research step"). Never silently skip a step — always log why.

1.20 — Frontail Container for Debug Investigation
Add a frontail service to the existing sandbox/docker-compose.yml. Mount logs/orchestrator.jsonl as read-only. Expose port 9001. Set mem_limit: 50m. Command: --ui-highlight --number 1000 /logs/orchestrator.jsonl. Set restart: unless-stopped. This provides a mobile-friendly browser UI at http://<tailscale-ip>:9001 for investigating DEBUG-level logs around errors. Python's RotatingFileHandler manages rotation, not Frontail — on file rotation Frontail recovers on page refresh. With 50MB rotation and ~500-1000 logs/hr, rotation happens infrequently so this is acceptable.

1.21 — Log Hygiene & Maintenance
Add a logs/ entry to .gitignore if not present. Document the log file location and rotation policy in a comment block at the top of src/infra/logging.py: 5 rotated files × 50MB = 250MB max disk usage. Ensure RotatingFileHandler uses encoding="utf-8". Add a periodic log of system resource usage (RAM, CPU, disk) at INFO level every 15 minutes from src/app/run.py main loop, so resource trends are visible in the logs without external tooling.


Phase 2: Model Management — LlamaSwap, Dynamic Params & Benchmarks

2.1 — Replace Stop/Start with llama-swap
Major refactor of src/models/local_model_manager.py. Install llama-swap and create config/llama_swap.yaml defining all local models with their llama.cpp launch parameters. Modify local_model_manager.py: remove all process management code (kill, restart, PID tracking). Replace with HTTP calls to llama-swap's proxy endpoint. llama-swap handles model loading/unloading transparently. Set ttl per model (e.g., 300s for large models, 600s for small frequently-used ones) so idle models unload. The orchestrator just sends requests to localhost:<llama_swap_port>/v1/chat/completions with the desired model name — llama-swap does the rest. Modify src/app/config.py to add LLAMA_SWAP_URL and LLAMA_SWAP_CONFIG_PATH.

2.2 — Dynamic ngl from GPU State
Modify src/models/gpu_monitor.py — it likely already reads GPU stats, but doesn't compute optimal ngl. Add a function calculate_optimal_ngl(model_name: str) -> int that: reads current vram_free_mb from the monitor, looks up the model's per-layer VRAM cost from config/model_profiles.yaml (new field: vram_per_layer_mb), calculates floor((vram_free - 200MB buffer) / vram_per_layer), caps at the model's total layer count. Modify src/models/gpu_scheduler.py to call this before requesting a model load. When writing the llama-swap config or when making a load request, inject the computed ngl. Modify config/model_profiles.yaml (or create a new config/model_hardware.yaml) to include per-model fields: total_layers, vram_per_layer_mb, min_ngl (below which the model is too slow to bother).

2.3 — Model Parameter Profiles (YAML + Task Overrides)
Modify src/models/model_profiles.py and models.yaml. Ensure each model entry has:

yaml
qwen2.5-coder-7b:
  default_params:
    temperature: 0.2
    top_p: 0.9
    top_k: 40
    repeat_penalty: 1.1
  task_overrides:
    coding: {temperature: 0.1}
    creative: {temperature: 0.8}
    research: {temperature: 0.3}
    planning: {temperature: 0.4}
Modify model_profiles.py to expose get_params(model_name, task_type) -> dict that merges defaults with task-specific overrides. Modify src/core/router.py to call get_params() when selecting a model, passing the current task's classified type (from task_classifier.py). If a model has no profile entry, use sensible defaults and log a warning.

2.4 — Performance-Based Auto-Tuning
Create src/models/auto_tuner.py. After every task completion, log {model, task_type, params_used, quality_score, latency_ms, token_count} to a model_performance DB table (modify src/infra/db.py to add table). Expose a /tune Telegram command and a weekly scheduled trigger. When invoked: for each model+task_type combo with ≥20 data points, compute average quality per temperature bucket (0.0-0.2, 0.2-0.4, etc.). Recommend the bucket with highest quality. Write suggestions to config/model_profiles_suggested.yaml. Send diff to Telegram for approval before overwriting active profiles.

2.5 — Benchmark Redesign
Refactor src/models/benchmark/. Keep the CLI entry point (benchmark_cli.py) but rewrite benchmark_fetcher.py and create new benchmark suites: benchmark_coding.py (HumanEval subset: 20 problems, SWE-bench-lite-style: 10 fix tasks), benchmark_reasoning.py (MMLU subset: 20 multi-domain questions, GSM8K: 15 math problems), benchmark_instruction.py (IFEval: 15 format-compliance checks), benchmark_tools.py (10 scenarios requiring correct tool selection + parameter passing). Each suite: structured input, expected output, automated scorer. Store results in benchmark_results DB table. Modify benchmark_cli.py to orchestrate all suites, compare against prior runs, output a report. Add Telegram command: /benchmark [model_name]. Run on: new model addition, monthly schedule, or on-demand.

2.6 — Continuous Model Health Monitoring
Modify src/core/orchestrator.py. Every 100 tasks (or hourly, whichever comes first): compute per-model rolling metrics from model_performance table — success rate (last 50 tasks), avg quality, avg latency, error rate. If success rate < 60%: log warning, temporarily demote in src/core/router.py routing preferences, notify via Telegram. Auto-restore after next check window if metrics recover. Modify src/models/model_registry.py to support a demoted flag per model.

Phase 3: Resource Management — GPU Load Control & Cloud Hybrid

3.1 — User-Controlled GPU Load Modes
Create src/infra/load_manager.py. Four modes:

Full — use all available GPU/RAM (default when idle).
Heavy - Use max %90 available vram+ram. 
Shared — cap at 50% VRAM+RAM, reduce ngl (Phase 2.2 recalculates), prefer smaller models or cloud.
Minimal — zero local GPU. All inference offloaded to cloud. Local used only for orchestration.
Persist current mode in DB. On mode change: trigger gpu_scheduler to recalculate all model params, potentially unload current model via llama-swap API. Modify src/app/telegram_bot.py to add /load full|heavy|shared|minimal command. Also handle natural language via the message classifier: "I'm going to game" → switch to Minimal, send confirmation.

3.2 — Automatic Load Detection
Modify src/models/gpu_monitor.py. Add detect_external_gpu_usage() -> bool using pynvml or nvidia-smi: if non-orchestrator processes are using >30% GPU compute or >2GB VRAM, flag it. When detected: if mode is Full, auto-downgrade. Send Telegram: "Detected external GPU usage. Switched to Shared. Reply /load full when free." When external usage drops for 5 minutes, notify and offer to restore. Never auto-restore to Full — always ask.

3.3 — Cloud Provider Pool
Create src/models/cloud_pool.py and config/cloud_providers.yaml. Define providers:

yaml
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    rate_limits: {rpm: 60, rpd: 1000, tpm: 100000}
    models: [deepseek/deepseek-chat, qwen/qwen-2.5-coder-32b, ...]
  google:
    rate_limits: {rpm: 15, rpd: 1500, tpm: 1000000}
    models: [gemini-2.0-flash, gemini-2.5-pro]
Modify src/models/rate_limiter.py to support per-provider token-bucket limiting (it may already do per-model — extend to per-provider aggregates). Modify src/models/model_registry.py to unify local and cloud models under one registry, with location: local|cloud and provider fields.

3.4 — Hybrid Task Scheduling
Modify src/core/orchestrator.py and src/models/gpu_scheduler.py. When multiple tasks are ready: run one locally (if load mode allows) and dispatch others to cloud concurrently. Routing logic: check load_manager mode → check gpu_scheduler for local capacity → check rate_limiter for cloud headroom → assign task to best available. Track per-task ran_on: local|cloud|{provider} for cost analysis. Modify src/core/router.py to factor in location preference: cost-sensitive → local first, latency-sensitive → fastest available, quality-sensitive → best model regardless.

3.5 — Async Parallel Execution
Modify src/core/orchestrator.py. If the main loop isn't already fully async, refactor to use asyncio. When multiple tasks have no shared dependencies (different goals, or no file overlap — check via src/collaboration/blackboard.py), run concurrently up to MAX_CONCURRENT_TASKS (1 local + N cloud, dynamically calculated). Modify src/agents/base.py to support asyncio.gather for independent tool calls within a single agent execution (e.g., two read_file calls on different files).

Phase 4: Conversational Telegram Bot

4.1 — LLM-Based Message Classification
Modify src/app/telegram_bot.py. Replace keyword-based message handling. On every incoming non-command message, call cheapest available model:

text
Classify this user message. Context: previous messages if any.
Categories: goal, task, question, bug_report, feature_request, 
ui_note, progress_inquiry, feedback, followup, clarification_response, 
load_control, command, casual
Output: {"type": "...", "confidence": 0.0-1.0, "referenced_id": null|int}
Route by type. Fall back to keyword heuristic if LLM fails. Reuse src/core/task_classifier.py logic if applicable — it may already classify tasks, extend it for user-message classification.

4.2 — Conversation Context Tracking
Modify src/memory/conversations.py (already exists). Ensure it supports: multi-turn threaded conversations tied to a context_type (goal_discussion, task_followup, clarification, general, bug_report), a reference_id (goal or task ID), and an active/resolved/expired status. Modify src/app/telegram_bot.py: when the system asks for clarification, create a conversation record. When user replies, detect via LLM classifier (4.1) or Telegram reply-to-message that it's a followup, load the conversation, append the reply, route to handler. Expire conversations after 24h inactivity.

4.3 — True Two-Way Clarification Dialogue
Modify src/agents/base.py (the clarification-request mechanism) and src/app/telegram_bot.py. Currently when an agent asks for clarification, it probably blocks or continues without an answer. New flow: agent emits a needs_clarification action → orchestrator sends question to Telegram with task context → task enters awaiting_clarification state → user can reply with an answer or ask counter-questions ("what do you mean by X?") → system responds with more detail (re-query agent with user's counter-question) → conversation continues until user gives definitive answer or says "proceed"/"skip" → inject answer into agent context, resume task. Auto-proceed timeout: 30 min (warn at 25 min). Modify src/core/state_machine.py to add awaiting_clarification state if it doesn't exist.

4.4 — Bug Reports, Feature Requests & Ad-Hoc Feedback
Create src/infra/user_inputs.py. DB table: {id, type (bug|feature|ui_note|review|feedback), content, related_goal_id, priority (auto-assessed by LLM), status (new|triaged|in_progress|done), created_at}. Modify src/app/telegram_bot.py: when classifier (4.1) detects bug/feature/ui_note/feedback, store it, auto-detect related goal (fuzzy match on goal names + LLM), acknowledge: "Logged as bug #14. Linked to Goal #5 (Auth System)." Optionally auto-create a fix task for bugs. Add commands: /bugs, /features, /notes to list open items. /triage for admin to review and prioritize.

4.5 — Progress Inquiry (Natural Language)
Modify src/app/telegram_bot.py. When classifier returns progress_inquiry: extract the referenced project/goal/task (LLM parses "how's the auth system going?" → finds goal by name similarity), query DB for goal's subtask statuses, generate a human-readable summary via LLM: "Goal #5 (Auth System): 60% complete. 3/5 subtasks done. Currently running: JWT implementation (iteration 4/10). 1 blocked: needs DB schema decision. ETA: ~2 hours." Send to user. If no match found, ask for clarification.

4.6 — Proactive Progress Streaming
Modify src/agents/base.py. Add a progress_callback parameter to agent execution. After every iteration: if task has been running >30s, invoke callback with {task_id, iteration, max_iterations, last_action, summary}. Modify src/core/orchestrator.py to wire this callback to telegram_bot.send_notification. Throttle: max one update per 30s per task. User can reply /quiet <task_id> to mute. On task completion, always send final result regardless of quiet mode.

4.7 — Interactive Plan Approval
Modify the planner output handling in src/core/orchestrator.py (wherever _handle_subtasks or equivalent lives). When planner produces subtasks: format as a numbered Telegram message with inline keyboard buttons: ✅ Approve, ✏️ Modify, ❌ Reject. On Modify: user replies in natural language → LLM interprets changes → send corrections to planner as a new task. On Reject: cancel all subtasks, ask for guidance. On Approve or auto-approve timeout (10 min, configurable): proceed. Modify src/app/telegram_bot.py to handle inline button callbacks.

4.8 — Rich Results & File Exchange
Modify src/app/telegram_bot.py. On task result delivery: if >3000 chars, save full result as workspace/results/task_{id}.md, send summary + file attachment. Code → .py/.js file. Screenshots → photo. Handle incoming user files: save to workspace/uploads/, auto-detect type and purpose, create appropriate task or add to active context. Voice messages: transcribe via Whisper API → treat as text input through classifier (4.1).

4.9 — Pause, Resume & Control
Modify src/core/state_machine.py to add paused status if it doesn't exist. Modify src/app/telegram_bot.py: /pause <task_id>, /pause goal <goal_id> (pauses all pending/processing tasks), /resume <task_id|goal_id>, /priority <task_id> <high|normal|low>, /cancel <task_id>. Modify src/core/orchestrator.py to check for paused status before starting new iterations on a task.

Phase 5: Tool Upgrades

5.1 — Replace DDGS with Perplexica
Modify src/tools/web_search.py. The Perplexica Docker is already running. Add PERPLEXICA_URL to .env / src/app/config.py. Change the search implementation to call Perplexica's search API instead of DDGS. Support search types: web, academic, code (map to Perplexica's focus modes). Keep DDGS as a fallback if Perplexica is unreachable (check the degraded-capability flag from Phase 1.3). Log queries and result quality.

5.2 — URL Content Extraction
Create src/tools/web_extract.py. Implement extract_url(url) -> str using trafilatura (primary) with readability-lxml fallback. Much cleaner than raw curl for research. Register in the tool registry (wherever src/tools/__init__.py registers tools). Add to researcher agent's allowed tools in src/agents/researcher.py.

5.3 — Browser Automation
Create src/tools/browser.py. Implement: browser_navigate(url), browser_screenshot(url, selector?), browser_click(selector), browser_type(selector, text), browser_extract(css_selector), browser_pdf(url). Use Playwright with headless Chromium. Either run inside sandbox (add Playwright to sandbox/Dockerfile) or on host with network isolation. Register as tools. Add to src/agents/researcher.py and src/agents/executor.py allowed tools.

5.4 — Vision Tool
Create src/tools/vision.py. Implement analyze_image(filepath, question?) -> str. Route to a vision-capable model — modify src/models/capabilities.py to include a supports_vision flag. Try cloud models (GPT-4o, Gemini Flash) since local models likely don't support vision. Modify src/agents/visual_reviewer.py (already exists!) to use this tool instead of whatever it currently does.

5.5 — Document Processing
Create src/tools/documents.py. Implement: read_pdf(filepath) -> str (PyMuPDF), read_docx(filepath) -> str (python-docx), read_spreadsheet(filepath, sheet?) -> str (openpyxl), extract_text(filepath) -> str (auto-detect). Register as tools. Chunk large documents and return first N pages with page count + continuation option.

5.6 — Tool Result Caching
Modify src/tools/__init__.py (or wherever tool execution is centralized, possibly src/agents/base.py). Add per-agent-execution cache dict. Read-only tools (read_file, file_tree, git_status, web_search) cache results keyed by arguments. Cache invalidated when any write tool (write_file, edit_file, shell, patch_file, apply_diff) executes. Log cache hit rate per task. Simple functools-style wrapper — minimal code.

5.7 — MCP Client
Create src/tools/mcp_client.py. Implement MCP client protocol (JSON-RPC over stdio). Load server configs from config/mcp_servers.json. On startup: launch each server, call tools/list, register returned tools into the tool registry with auto-generated schemas. Proxy tools/call to the appropriate server. Handle crashes with restart + circuit breaker (3 failures in 5 min → disable server, retry in 10 min). Start with filesystem and GitHub MCP servers.

Phase 6: Collaboration & Context Enhancements

6.1 — Enrich Blackboard Schema
Modify src/collaboration/blackboard.py. Ensure the data structure includes: architecture (plan JSON), files (path → status mapping), decisions (what/why/by/timestamp), open_issues, constraints, dependency_map. Add optimistic locking (version field) to prevent concurrent write conflicts. Ensure read_blackboard and write_blackboard are registered as agent tools. If they're only used programmatically now, expose them so agents can read/write during execution.

6.2 — Strengthen Plan Verification
Modify src/collaboration/plan_verification.py. Ensure it checks: (1) every file referenced in subtask descriptions exists as a subtask target, (2) agent_type assignments are sensible (code tasks → coder, not writer), (3) dependency graph is acyclic, (4) estimated cost within budget, (5) no duplicate subtasks. On failure: create a high-priority correction task for the planner with specific feedback. Currently this module exists — audit what checks it performs and add any missing from this list.

6.3 — Agent-to-Agent Queries
Modify src/agents/base.py. Add an ask_agent action type. When the base agent loop encounters {"action": "ask_agent", "target": "researcher", "question": "..."}: create an inline high-priority subtask, await completion (5-min timeout), inject result back into the requesting agent's message history as a tool result. Modify src/core/orchestrator.py to handle this inline subtask creation and synchronous wait. Log all inter-agent queries.

6.4 — Ambient Context Injection
Modify src/context/assembler.py (already exists). Ensure every agent execution receives a compressed ## Current Context block: active projects + status, current goal progress, recent decisions (from blackboard), user preferences (from src/memory/preferences.py), system load mode, time of day. This gives agents awareness of the bigger picture. Keep it under 500 tokens — summarize aggressively.

Phase 7: Progress Tracking & Visible Artifacts

7.1 — Project Registry
Create src/infra/projects.py. DB table: {id, name, description, language, framework, repo_path, workspace_path, status, created_at, updated_at}. Every goal links to a project. Modify src/app/telegram_bot.py: /projects lists all with status badges, /project <id> shows details. Modify goal creation flow in orchestrator to associate goals with projects (auto-detect or ask user).

7.2 — Progress Notes
Create src/infra/progress.py. DB table: {id, project_id, goal_id, task_id, note_type (milestone|blocker|decision|artifact|log), content, created_at}. Modify src/core/orchestrator.py and src/workflows/engine/status.py: auto-generate notes on key events (goal created, subtask completed, tests run, review done). Modify src/app/telegram_bot.py: accept manual notes ("note for project X: decided to use JWT"), expose /progress <project_id> for timeline view.

7.3 — Artifact Registry Enhancement
Modify src/workflows/engine/artifacts.py. Ensure it tracks: {id, project_id, goal_id, task_id, type (code|test_result|screenshot|diagram|report|coverage), filepath, description, created_at}. Register artifacts automatically when agents produce files, test results, screenshots. /artifacts <project_id> via Telegram.

7.4 — Goal Completion Summary
Modify src/core/orchestrator.py (goal completion handler). When a goal completes: auto-generate summary (what requested, what delivered, files changed, test results, cost, time, decisions from blackboard). Save to workspace/results/goal_{id}_summary.md. Send condensed version via Telegram. Register as artifact.

Phase 8: Security Hardening

8.1 — Agent Permission Matrix
Create src/security/permissions.py. Define enforced per-agent-type allowed tools. Modify tool execution in src/agents/base.py: before running any tool, check against AGENT_PERMISSIONS[agent_type]. Reject unauthorized calls with a clear error. This overrides any per-agent allowed_tools lists with enforced boundaries.

8.2 — Shell Command Allowlist
Modify src/tools/shell.py. Replace any blocklist with a strict allowlist per agent type. Parse first token of each command, check against allowlist. Log all rejected commands. Coder: python, pip, npm, node, go, cargo, git, cat, ls, mkdir, cp, mv, grep, find, head, tail, wc, sort, curl. Reviewer: pytest, python -m py_compile, npm test.

8.3 — Secret Management
Create src/security/secrets.py. Encrypted secrets.enc file (Fernet, key from SECRETS_KEY env var). Agents reference secrets by name via get_secret(name) — injected into sandbox env vars, never into prompts. Scan outgoing model messages and Telegram output for known secret patterns; redact.

8.4 — Audit Trail
Create src/infra/audit.py. DB table (append-only): {id, timestamp, actor, action, target, details}. Modify src/agents/base.py, src/tools/__init__.py, src/core/orchestrator.py to log: every tool execution, model call, state transition, file modification, human approval. /audit <task_id> Telegram command.

Phase 9: Observability & Cost Control

9.1 — Metrics Collection
Create src/infra/metrics.py. In-memory counters: tasks_completed, tasks_failed, model_calls{model}, cost_total{model}, latency{model}, tool_calls{tool}, tokens{model}, queue_depth. Persist hourly to metrics DB table. Modify src/core/orchestrator.py to increment counters. /metrics Telegram command.

9.2 — Task Execution Trace
Create src/infra/tracing.py. Per-task ordered trace: [{type, timestamp, input_summary, output_summary, cost, duration}]. Store in task_traces DB table as JSON. Modify src/agents/base.py to append trace entries on every tool call and model call. /replay <task_id> Telegram command.

9.3 — Alerting
Create src/infra/alerting.py. Rules: >3 tasks fail in 60 min, daily cost > $X, model success rate < 50%, queue > 20. Check every orchestrator cycle. Configurable via config/alerts.yaml. Use ntfy (Phase 1.1) and Telegram for notifications.

9.4 — Cost Attribution
Modify src/models/quota_planner.py (already exists) to also track per-goal and per-project cost breakdowns. /costs → today's summary by model. /costs goal <id> → per-task breakdown. /costs week → weekly summary. Budget limits per goal and per day with warnings.

Phase 10: Coding Pipeline Quality

10.1 — Multi-Language Toolkit
Create src/languages/ directory: base.py (abstract), python.py (ruff, pytest, mypy), javascript.py (eslint, prettier, jest), typescript.py (extends JS + tsc), go.py, rust.py. Each implements: lint(), format(), test(), typecheck(), install_deps(). Modify src/tools/linting.py to delegate to the appropriate language toolkit based on project profile.

10.2 — Language-Aware Prompts
Modify src/context/assembler.py. On task execution, detect project language (from project registry or file extensions). Inject language-specific rules into agent system prompts: test commands, import conventions, idiomatic patterns. Store language prompt fragments in config/language_prompts.yaml. Don't create separate agents — inject context into existing ones.

10.3 — Structured Review Protocol
Modify src/agents/reviewer.py. Reviewer outputs structured JSON: {verdict, issues: [{severity, file, line, description, suggested_fix}]}. Modify src/agents/fixer.py to receive structured issues instead of prose. Modify pipeline in src/workflows/pipeline/pipeline.py to handle the structured format for pass/fail/fix routing.

10.4 — Type Checking & Coverage
Modify src/tools/linting.py or create src/tools/type_checker.py. Run mypy/tsc/go vet after implementation, before tests. Create src/tools/coverage.py: run pytest --cov / jest --coverage, parse reports, identify untested paths. /coverage <project_id> Telegram command.

10.5 — Multi-Runtime Sandbox
Modify sandbox/Dockerfile. Add: Node.js 20 LTS, Go 1.22, Rust (rustup). Modify src/tools/code_runner.py to accept a language parameter and route to the appropriate runtime. Modify src/tools/shell.py to be aware of available runtimes.

10.6 — PR Workflow
Create src/workflows/pr.py. On goal completion: generate PR description (diff summary, tests, coverage, reviews). Send to Telegram with approve/modify/reject buttons. Merge: squash-merge goal branch → main. If GitHub configured: create actual PR via API. Conflict detection with auto-rebase attempt.

Phase 11: Additional Workflows

11.1 — Research Pipeline
Create config/workflows/research.yaml. Stages: parse question → deep search via Perplexica → extract top sources → synthesize report with citations → deliver. Variants: competitive analysis, tech comparison, documentation lookup. Wire into workflow engine via src/workflows/engine/loader.py.

11.2 — Documentation Generator
Create config/workflows/documentation.yaml. Stages: analyze codebase structure → generate README, API docs, architecture overview, setup guide → review → deliver. Trigger: on goal completion or on-demand. Keep docs in sync: detect code changes via git diff → update affected sections.

11.3 — Bug Investigation Pipeline
Create config/workflows/bugfix.yaml. Stages: reproduce (run tests/check logs) → root cause analysis → implement fix → regression test → review → PR. Handle vague reports by asking clarifying questions (Phase 4.3 mechanism).

11.4 — Refactoring Pipeline
Create config/workflows/refactor.yaml. Stages: identify target (code smell or user request) → plan incremental changes → implement one file at a time → test after each → verify no regressions.

11.5 — Scheduled Maintenance
Create config/scheduled_tasks.yaml. Recurring: dependency update checks (weekly), dead code detection, test suite health, workspace cleanup, sandbox container pruning, log rotation. Modify src/core/orchestrator.py to read and execute scheduled tasks based on cron expressions.

Phase 12: API & Dashboard
12.1 — FastAPI Server
Create src/app/api.py. Endpoints: POST /goals, POST /tasks, GET /goals/{id}, GET /tasks/{id}, GET /queue, GET /stats, GET /models, GET /health, GET /projects, GET /artifacts/{id}. JSON responses. API key auth. Modify src/app/run.py to start FastAPI alongside orchestrator (background thread or co-hosted).

12.2 — WebSocket Live Streaming
Add /ws/stream/{task_id} WebSocket endpoint. Broadcast iteration updates, tool calls, partial results during execution. Modify src/agents/base.py progress callback (Phase 4.6) to also emit WebSocket events.

12.3 — Web Dashboard (Nice-to-Have)
HTMX + Jinja2 templates: project list, task queue (mermaid.js dependency graph), live execution view (WebSocket), cost chart (chart.js), artifact browser. Serve from FastAPI. Build only after API is stable.

Phase 13: Learning & Adaptation
13.1 — Prompt Versioning
Create prompts DB table. Modify all agents in src/agents/*.py to load system prompts from DB instead of hardcoded strings. Track quality scores per version. Auto-promote better versions after ≥10 tasks. /prompt <agent> to view/set.

13.2 — Skill Library
Create src/memory/skills.py. DB table: {name, description, trigger_pattern, tool_sequence, examples, success_count}. On successful novel task: extract approach, store as skill. On similar future tasks: inject skill as context via src/context/assembler.py. /skill add <name> to manually teach.

13.3 — Feedback Loop
Modify src/app/telegram_bot.py and src/memory/preferences.py. Track implicit feedback (no response = accepted, correction = partial rejection, explicit "wrong" = rejection). Score: accept=+1, partial=-0.5, reject=-1. Feed into: model stats, prompt versions, skill confidence. /feedback <task_id> <good|bad> [reason].

13.4 — Self-Improvement Proposals
Weekly scheduled task: analyze failures, cluster by category, generate improvement proposals via LLM. Send to Telegram for approval. On approval: auto-create implementation tasks. Modify src/core/orchestrator.py to schedule this.

Phase 14: Autonomous JARVIS Features
14.1 — Morning Briefing
Scheduled task: summarize overnight results, pending approvals, active goal progress, cost summary, system health. Single formatted Telegram message. Configurable time via .env.

14.2 — Proactive Monitoring
Create src/infra/monitoring.py. Background checks: GitHub repos (new issues/PRs), configured URLs (uptime), dependency advisories. Notify via Telegram with actionable suggestions. Configure in config/monitoring.yaml.

14.3 — Delegation Intelligence
Create src/security/risk_assessor.py. Per-task: assess reversibility, cost, external impact, novelty. Score 0-10. Below threshold → auto-execute. Above → approval. Learn from approval patterns. /autonomy <level> to adjust globally. Modify src/core/orchestrator.py to call risk assessor before task execution.

14.4 — Personal Knowledge Base
Modify src/memory/ modules (rag.py, vector_store.py, embeddings.py). Index all user conversations, uploaded docs, goal descriptions, feedback. /remember <text> → store with high importance. /recall <query> → semantic search and summarize. Auto-surface relevant memories during task execution via src/context/assembler.py.

14.5 — Multi-Step Goal Refinement
Modify goal creation flow in src/core/orchestrator.py. When user sends a vague goal: don't immediately plan. Start a clarification conversation (Phase 4.3 mechanism): LLM generates 2-3 targeted questions, wait for answers, refine, then plan. Auto-proceed after timeout with best-guess + disclaimer.

14.6 — Plugin Architecture
Create src/plugins/ directory with loader.py. Define Plugin interface: {name, version, tools, agents, prompts, mcp_servers}. plugins/ directory with plugin.json manifests. On startup: scan, validate, register. Start with 2 internal plugins as proof of concept (GitHub integration, deployment tools).
