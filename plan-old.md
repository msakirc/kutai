Phase 0: Critical Bug Fixes (Do First)
Fix pipeline status mismatch — CodingPipeline.run() returns "status": "completed" but orchestrator checks for "complete". Normalize to one value everywhere.

Fix task dedup hash — Include agent_type and parent_task_id in compute_task_hash. Currently different subtasks with identical titles get silently dropped.

Fix double git commits — Pipeline stage 7 commits AND _auto_commit fires for coder-type tasks. Add a flag or skip _auto_commit when task comes from pipeline.

Fix sequential processing — get_ready_tasks(limit=3) fetches 3 but processes them in a serial for-loop. Either fetch 1 or actually parallelize with asyncio.gather (requires workspace isolation first — for now just fetch 1).

Fix scheduled_tasks dead code — Table exists, nothing reads it. Either remove it or implement the scheduler (Phase 3).

Phase 1: Data Layer Hardening
SQLite WAL mode + connection pool — Enable WAL journal mode at init for concurrent reads. Create a shared aiosqlite connection pool (or switch to a wrapper like aiosqlite with a singleton connection + queue). Eliminate open/close on every call.

Task locking — Add UPDATE tasks SET status='processing' WHERE id=? AND status='pending' as an atomic claim. Check rowcount > 0 before proceeding. Prevents double-pickup.

Transaction safety — Wrap multi-step DB operations (add subtasks, update parent) in explicit transactions. Currently a crash between two await db.execute calls leaves inconsistent state.

Task checkpointing — Store intermediate agent state (current iteration, messages so far) in a task_state JSON column. On crash recovery, resume from last checkpoint instead of restarting.

Idempotency keys — For tool side effects (file writes, git commits, shell commands), store a hash of what was done. On retry, skip already-completed steps.

Phase 2: Structured Output & Parsing
Pydantic response models — Define ToolCallAction, FinalAnswerAction, ClarifyAction, DecomposeAction as Pydantic models. Replace all regex/brace-depth JSON parsing with model_validate_json + fallback.

Structured output for capable models — For models that support response_format (OpenAI, Gemini), pass the JSON schema directly. Only fall back to prompt-based JSON for models without it.

Retry with format correction — When JSON parsing fails, instead of falling through to final_answer with raw text, send a one-shot "fix your JSON" prompt with the exact parse error. Max 2 format retries before fallback.

Typed tool arguments — Use the existing TOOL_SCHEMAS to validate tool args before execution. Return clear error to the agent when arg types are wrong (e.g., start_line passed as string).

Phase 3: Scheduler & Task Engine
Cron scheduler — Implement a scheduler loop that reads scheduled_tasks, compares next_run to now, creates tasks when due, updates last_run and next_run. Run alongside main loop every 60s.

Task cancellation — Add cancelled status. Telegram command /cancel <id>. Orchestrator checks cancellation flag between agent iterations. Propagate cancellation to child tasks.

Task reprioritization — /priority <id> <level> Telegram command. Orchestrator respects dynamic priority changes on next cycle.

Task timeout per agent type — Add timeout_seconds column. Planner gets 120s, coder gets 300s, researcher gets 180s. Orchestrator enforces with asyncio.wait_for around process_task.

Parallel task execution — Run up to N independent tasks concurrently (tasks with no shared workspace paths). Requires workspace locking (Phase 6). Start with N=2.

Dependency graph visualization — /graph <goal_id> renders a text-based DAG of task dependencies and statuses. Helps you understand what's blocked and why.

Phase 4: Model Intelligence Layer
Response grading system — After every agent final_answer, run a model to score the result 1-5 on: relevance, completeness, correctness. Store grade in tasks.quality_score. Use for model performance tracking.

Model performance tracker — Table: model_stats(model, agent_type, avg_grade, avg_cost, avg_latency, success_rate, total_calls). Update after every call. Query when selecting models.

Performance-aware model selection — Modify select_model to factor in historical success rate and average grade for the specific agent type. Model that scores 4.5 on coding tasks ranks higher than one scoring 3.2, regardless of static quality number.

Mid-task model escalation — If an agent fails a tool call or gets a low self-assessment after iteration 3, automatically escalate remaining iterations to the next tier up. Log the escalation.

Cost budgets — Per-goal and daily cost caps in config. Check before each call_model. If budget exhausted, pause task and notify via Telegram. /budget command to view and adjust.

Streaming support — Use litellm.acompletion with stream=True for tasks where max_iterations > 4. Accumulate chunks, enables future real-time progress updates via Telegram.

Thinking/reasoning model support — Detect models with extended thinking (o1, Claude with thinking, QwQ). Don't set temperature for them. Increase timeout. Parse thinking blocks separately from output.

Phase 5: Agent Architecture Improvements
Agent execution patterns — Split BaseAgent.execute into strategies: ReactLoop (current), SingleShot (planner, classifier — no tool loop needed), ParallelSearch (researcher — fire multiple searches concurrently). Agent declares which pattern to use.

Inter-agent delegation — Add a delegate pseudo-tool: agent can invoke another agent inline and get its result back as tool output. Enables researcher calling coder for a quick script, or coder calling researcher for an API lookup.

Confidence-gated output — Agent must output a confidence score (1-5) with final_answer. If confidence < 3, automatically route to reviewer or escalate to human. If confidence < 2, reject and retry with stronger model.

Self-reflection step — Before accepting final_answer, inject a "review your own output" prompt asking the agent to check for errors, omissions, hallucinations. One extra LLM call, catches obvious mistakes.

Error recovery agent — Dedicated agent that receives failed tasks + error logs. Diagnoses root cause (bad prompt? missing tool? model too weak? missing dependency?). Either fixes and retries or escalates with a clear diagnosis. Replaces current "forward error to Telegram" pattern.

Agent specialization configs — Move system prompts to YAML/JSON files. Add per-agent config: retry_strategy, escalation_tier, required_tools, success_criteria. Makes agents configurable without code changes.

Phase 6: Workspace & Isolation
Per-goal workspace directories — workspace/goal_{id}/ isolation. Agents for different goals can't step on each other's files. Pipeline operates within goal workspace. Merge to main workspace on goal completion.

File locking — Advisory locks on files being edited. Agent acquires lock before write_file/edit_file, releases on tool completion. Prevents concurrent corruption.

Branch-per-goal git workflow — Auto-create goal/{id}-{slug} branch. All work happens on branch. On goal completion, prompt human for merge approval via Telegram. Enables rollback per goal.

Workspace snapshots — Before each coder/pipeline task, snapshot the workspace state (file hashes). On failure, offer one-click rollback to pre-task state via Telegram.

Multi-project support — Config file listing projects with their repo paths, languages, conventions. /project <name> switches active workspace. Agents load project-specific context.

Phase 7: Tool Ecosystem Expansion
Browser/scraper tool — Headless Chromium in Docker (Playwright). browse_url(url) → markdown. screenshot_url(url) → image path. Gives agents real web access beyond DuckDuckGo snippets.

Vision tool — analyze_image(path, question) — sends image to multimodal model (GPT-4o, Gemini). Enables UI verification, screenshot analysis, diagram understanding.

File download tool — download_file(url, save_as) — curl with progress, size limits, content-type validation. Agents can fetch assets, datasets, references.

Database tools — query_db(connection_string, sql) for read-only queries. execute_db(connection_string, sql) for writes (requires approval flag). Support SQLite, PostgreSQL, MySQL.

API client tool — http_request(method, url, headers, body) → response. Generic REST client. Agents can interact with any API.

MCP client — Implement Model Context Protocol client. list_mcp_servers(), call_mcp_tool(server, tool, args). Config file lists available MCP servers and their endpoints. Agents discover and use MCP tools like native tools.

Diff/patch tool — Replace line-number-based edit_file with a search-and-replace tool: patch_file(filepath, search_block, replace_block). Models are far better at specifying text blocks than line numbers.

Multi-language code runner — Extend run_code to support Node.js, Go, Rust, Bash. Language-specific Docker images or multi-runtime sandbox.

Multi-language linting — ESLint for JS/TS, rustfmt for Rust, gofmt for Go. Detect language from extension and route to correct linter.

Multi-language dependency manager — npm install for JS, cargo build for Rust, go mod tidy for Go. Extend verify_dependencies to detect and handle non-Python projects.

Calendar/email tools — Google Calendar read/write, Gmail send/read. Enables scheduling, meeting summaries, email drafts. Guard with approval for sends.

Notification channels — Slack, Discord, email as alternatives/additions to Telegram. Config-driven channel selection.

Phase 8: Coding Pipeline Maturity
Adaptive pipeline stages — Don't always run all 7 stages. Classify task complexity: one-liner → skip architect/test. Bug fix → skip architect, run fixer directly. New feature → full pipeline. Refactor → skip tests initially.

AST-aware code tools — get_function(filepath, function_name), replace_function(filepath, function_name, new_code), list_classes(filepath). Operate on code structure, not line numbers.

Codebase indexing — On project load, build an index: file → functions/classes/imports. Store in memory table. Agents query index instead of reading every file. Rebuild on file changes.

Context-aware code generation — Before coder/implementer starts, auto-inject: project conventions (detected from existing code), import patterns, naming style, error handling patterns. Agent generates code that matches the codebase.

Test-driven mode — Option to invert pipeline: write tests first (from spec), then implement until tests pass. More reliable for well-defined features.

PR workflow — On goal completion, generate a diff summary, list files changed, tests run, review notes. Send to Telegram as a PR-style review. Approve to merge, reject to iterate.

CI integration — After pipeline completes, trigger existing CI (GitHub Actions, etc.) via API. Wait for result. If CI fails, feed logs back to fixer agent.

Large codebase navigation — For projects >100 files: auto-generate a codebase map (module → purpose → key functions). Agents consult map before deciding which files to read. Prevents token waste from reading irrelevant files.

Language-agnostic pipeline — Detect project language. Route to appropriate tools (compiler, linter, test runner, package manager). Same pipeline logic, different tool bindings.

Incremental implementation — Track which files have been implemented. On retry or continuation, skip already-complete files. Resume from last incomplete file.

Phase 9: Memory & Learning
Vector memory store — Embed task results, decisions, and learnings using a local embedding model (or API). Store in ChromaDB or LanceDB. Query with semantic similarity instead of key-value lookup.

RAG over workspace — Index all workspace files into vector store. Agents query "find files related to authentication" instead of browsing the entire tree. Re-index on file changes.

RAG over documentation — Ingest uploaded PDFs, markdown docs, API references. Agents query them naturally. /ingest <url_or_file> Telegram command.

Episodic memory — Store full task execution traces (what worked, what failed, what was retried). On similar future tasks, retrieve relevant episodes and inject as "here's what worked last time".

Skill learning — When an agent successfully completes a novel task type, extract the approach as a reusable "skill" (prompt template + tool sequence + success criteria). Store in skills table. Future similar tasks load the skill.

User preference learning — Track your Telegram feedback (approvals, rejections, corrections). Build a preference profile: preferred code style, communication verbosity, risk tolerance, working hours. Inject into agent prompts.

Model grading feedback loop — Combine automated grading (Phase 4, step 21) with your explicit feedback. Over time, build a model ranking per task type that's empirically grounded, not hardcoded quality numbers.

Forgetting mechanism — Memory entries decay over time. Old, low-relevance, low-access memories get pruned. Prevents context pollution from outdated info.

Phase 10: Security & Privacy
Data classification — Classify task content as public, internal, sensitive, secret. Use keyword detection + LLM classification. Route: sensitive/secret → local models only. Never send to cloud.

PII detection — Regex + model-based PII scanner on all outgoing LLM calls. Detect credit cards, SSNs, API keys, passwords. Redact or block with notification.

Sandbox hardening — Drop all capabilities in Docker (--cap-drop ALL). No network access by default. Whitelist specific domains per task. Prevent data exfiltration via curl.

Network policies per tool — web_search gets internet access. shell gets none by default. http_request tool gets configurable allowlist. Implement via Docker network isolation or iptables rules in sandbox.

Secrets management — Encrypted secrets store. Agents reference secrets by name, never see raw values. Inject at execution time only. Rotate on schedule.

Audit log — Immutable append-only log of all: tool executions, file modifications, git operations, model calls, human approvals. Queryable via /audit command.

Encrypted DB — Switch to SQLCipher or encrypt sensitive columns. Task results and conversation logs contain potentially sensitive data.

Phase 11: Human Interface Upgrade
Conversation context — Maintain a sliding window of last 5 messages per Telegram chat. Inject into task context. Enables "do that again but with X" follow-ups that actually work.

Smart message classification — Replace keyword-based goal/task detection with an LLM call. Classify incoming message as: goal, task, question, feedback, clarification, command. Route accordingly.

Progress updates — For tasks running >60s, send periodic Telegram updates: "🔄 Task #42: Running tests (iteration 3/8)..." Pull from task checkpoint data.

Interactive plan approval — When planner creates subtasks, show them as numbered list with inline buttons: ✅ Approve All, ✏️ Modify, ❌ Reject. Modification lets you reply with changes.

Task result viewer — For long results, send a summary via Telegram + save full result to a file. Send file link or inline document. No more 3000-char truncation.

File/image exchange — Handle Telegram file uploads (store to workspace). Handle image uploads (save + optionally analyze with vision tool). Send generated files/images/screenshots back.

Cancel/pause from Telegram — /cancel <id>, /pause <id>, /resume <id>. Pause stops picking up subtasks but lets current one finish.

Multi-user support — Map Telegram user IDs to roles: admin, developer, viewer. Admins can do everything. Developers can create tasks. Viewers can only read status.

Voice message support — Transcribe Telegram voice messages via Whisper (local or API). Convert to text task. Respond with voice via TTS for hands-free operation.

Web dashboard — Simple FastAPI + HTMX dashboard: task list, goal progress, model stats, cost tracking, live logs. Telegram stays as mobile interface, dashboard for deep inspection.

Phase 12: Observability
Metrics collection — Track per-call: model, latency, tokens, cost, success/fail, agent type, tier. Store in metrics table. Aggregate hourly.

Performance dashboard — Add to web dashboard (step 85): model comparison charts, cost trends, success rates by agent type, average task duration, queue depth over time.

Alerting rules — Configurable alerts: "if >3 tasks fail in 1 hour", "if daily cost exceeds $X", "if model X fails >50%". Send via Telegram.

Health check endpoint — HTTP /health returning: orchestrator running, DB accessible, Docker sandbox alive, Telegram connected, models reachable. Enables external monitoring.

Distributed tracing — Assign trace IDs to goals. All subtasks, agent calls, tool executions carry the trace ID. Enables end-to-end debugging of a goal's execution path.

Phase 13: Scalability
Redis task queue — Replace SQLite polling with Redis-backed queue (or PostgreSQL with SKIP LOCKED). Enables multiple worker processes.

PostgreSQL migration — Replace SQLite with PostgreSQL for concurrent access, better JSON support, proper locking, and production reliability.

Worker pool architecture — Separate orchestrator (scheduler + Telegram) from workers (agent execution). Workers pull from queue. Scale workers independently.

Per-task Docker containers — Spawn a fresh container per task (or per goal). Full isolation. Destroy on completion. Slightly slower but eliminates all cross-task contamination.

Workspace as volume — Mount goal-specific volumes into task containers. Persist across task retries. Clean up on goal completion.

Phase 14: Jarvis Features
Proactive monitoring — Agent that periodically checks: your GitHub repos for new issues/PRs, server health, news in topics you care about. Surfaces relevant info proactively via Telegram.

Morning briefing — Scheduled daily task: summarize overnight work, pending decisions, calendar for today, important emails, goal progress. Send at your configured wake time.

Ambient awareness — Continuously updated context object: current projects, recent decisions, your preferences, active goals, time of day. Injected into every agent for consistent Jarvis-like behavior.

Multi-step conversation planning — When you describe a vague goal, Jarvis asks clarifying questions BEFORE planning. Back-and-forth Telegram conversation to refine the goal, then auto-triggers planning.

Delegation intelligence — Jarvis decides what to do autonomously vs. what to confirm with you, based on: task risk, reversibility, cost, your historical preferences. Low-risk → just do it. High-risk → ask first.

Personal knowledge base — Everything you tell Jarvis is indexed and searchable. "What was that API key format we discussed last week?" → retrieves from conversation history.

Cross-goal intelligence — When working on Goal B, Jarvis notices a conflict or synergy with Goal A. Flags it. Enables coherent long-term project management.

Plugin architecture — Define a standard interface for new capabilities: Plugin(name, tools, agents, prompts). Drop a plugin folder in, Jarvis discovers and integrates it. Enables community/marketplace.

Self-improvement loop — Jarvis periodically reviews its own failure logs, identifies patterns, proposes system improvements (new tool, better prompt, config change). Sends proposals to you for approval.
