Phase 9: Reliability Foundation

9.1 — Task State Machine
Create state_machine.py. Define enum of valid states and a TRANSITIONS dict mapping each state to its allowed next states. Add transition_task(task_id, to_state) that loads current state, validates the transition, updates DB with updated_at, raises InvalidTransition on illegal moves. Replace every raw update_task(status=...) call across orchestrator, base agent, and telegram_bot with transition_task. Add error_category column to tasks table (values: model_error, tool_error, timeout, budget_exceeded, invalid_output, dependency_failed, cancelled).

9.2 — Structured Output Enforcement
Refactor base.py response handling. New priority order: (1) if model supports function calling, use it — parse tool_calls from response, (2) if model supports response_format: json_schema, pass the schema from models.py get_action_json_schema(), (3) one retry with the parse error injected, (4) fail the iteration explicitly — no silent fallback to treating raw text as final_answer. Delete the 5-layer regex cascade in _parse_agent_response. Keep only: try json.loads → try fence extraction → one brace-depth scan → explicit failure. Add per-task-type output validators: code tasks must contain a file path or code block, research tasks must contain at least one URL or source reference, planner tasks must contain subtasks array.

9.3 — Error Taxonomy & Retry Policies
Create error_policy.py. Define retry policies per error category: model_error → retry with next fallback model, tool_error → retry same model (tool might be transient), timeout → retry with increased timeout, invalid_output → retry with correction prompt, budget_exceeded → pause task and notify, dependency_failed → fail immediately. Refactor process_task in orchestrator to catch exceptions, classify them into categories, and apply the matching policy instead of the generic retry_count < max_retries check.

9.4 — Atomic DB Operations
Wrap all multi-statement DB writes in explicit BEGIN/COMMIT (extend the pattern from add_subtasks_atomically to process_task completion updates, memory storage, cost recording). Make write_file tool write to a temp file then os.replace() for atomic file creation. Add PRAGMA wal_checkpoint(TRUNCATE) call in close_db.

9.5 — Per-Task Token & Cost Budgets
Add max_cost column to tasks table (nullable, default from goal or global config). Before each call_model in the agent loop, calculate expected cost from input token count × model pricing. If expected cost would exceed remaining budget for that task, downgrade tier or fail with budget_exceeded. Add per-goal cost cap — in _handle_complete, sum all task costs for the goal; if over limit, cancel remaining pending subtasks and notify.

Phase 10: Model Backend Expansion & Intelligent Router
Use every available model intelligently.

10.1 — llama.cpp / Custom Endpoint Support
Add to config.py: scan for LLAMA_CPP_ENDPOINTS env var (comma-separated name=url pairs, e.g. qwen3-8b=http://localhost:8080). For each endpoint, probe /v1/models to confirm it's alive, add to MODEL_POOL with provider: "llamacpp", litellm_name: "openai/<name>" with api_base set. Add similar support for CUSTOM_OPENAI_ENDPOINTS (covers vLLM, TGI, LocalAI, LM Studio — anything OpenAI-compatible). Add api_base field to MODEL_POOL entries and pass it through to litellm.acompletion().

10.2 — Semantic Task Classifier
Replace keyword-based classify_task in router.py. Use a local embedding model (Ollama nomic-embed-text or all-minilm via llama.cpp) to embed the task description. Maintain a small labeled reference set of ~50 example tasks across categories: {simple_qa, code_simple, code_complex, research, writing, planning, action_required, sensitive}. Classify by cosine similarity to nearest cluster centroid. Fall back to existing keyword heuristic if embedding model unavailable. Cache embeddings for identical task descriptions.

10.3 — Sensitivity Detection
Create security/sensitivity.py. Regex scanner for: API keys (common prefixes like sk-, ghp_, AKIA), emails, credit card patterns, SSN patterns, private key blocks. Run on task title + description + context before model selection. Return sensitivity level: public, private, secret. In router.py select_model: if sensitivity is private or secret, filter candidates to provider in ("ollama", "llamacpp") only. If no local model available, block the task and notify via Telegram: "This task contains sensitive data but no local model is available."

10.4 — Capability-Based Model Matching
Add capability requirements to tasks: needs_vision, needs_long_context, needs_function_calling, needs_code. In select_model, filter MODEL_POOL by required capabilities before scoring. Estimate input token count before selection — skip models whose max_input_tokens is too small. Add context_length field to MODEL_POOL entries (query from litellm model info or hardcode).

10.5 — Model Pinning & Override
Allow task context to contain {"model_override": "exact_litellm_name"}. When present, skip tier selection entirely and use that model. Expose via Telegram: /task Fix the login bug --model claude-sonnet-4-20250514. Useful for debugging model-specific issues or forcing expensive model on critical tasks.

Phase 11: Memory & Knowledge System
Stop being a goldfish.

11.1 — Vector Store Setup
Add ChromaDB (pip install chromadb, runs in-process, no server needed, Windows-compatible). Create memory/vector_store.py with: init_store(), embed_and_store(text, metadata, collection), query(text, collection, top_k), delete(ids). Collections: episodic (task results), semantic (facts/preferences), codebase (code chunks), errors (failure patterns), conversations (user interactions). Use Ollama or llama.cpp embedding model. Fallback: sentence-transformers all-MiniLM-L6-v2 running locally via Python.

11.2 — Episodic Memory (Task History)
After every completed task, embed and store: {task_title, task_description, agent_type, result_summary (first 500 chars), model_used, success, error_if_any, duration}. After every failed task + recovery, store: {error_signature, root_cause, fix_applied, prevention_hint}. Before executing any task, query episodic memory for top-3 similar past tasks. Inject as ## Relevant Past Experience in agent context: "A similar task succeeded using approach X" or "Warning: similar task failed due to Y."

11.3 — RAG Pipeline for Agent Context
Create memory/rag.py. Function retrieve_context(task, agent_type, max_tokens) that: (1) queries episodic memory for similar tasks, (2) queries semantic memory for relevant facts, (3) queries error patterns for matching warnings, (4) ranks by recency × relevance × importance, (5) deduplicates, (6) returns formatted text block within token budget. Call this from BaseAgent._build_context — inject RAG results between task description and tool descriptions.

11.4 — Conversation Continuity
Embed every Telegram user message and AI response into conversations collection. On new message, retrieve last 5 exchanges + top-3 semantically similar past conversations. Replace user_last_task_id hack with embedding-based follow-up detection: embed "list them" → finds that it's most similar to the previous task about "find all Python files" → correctly resolves the reference.

11.5 — Document Ingestion
Add /ingest <url_or_filepath> Telegram command. For URLs: download page content, extract text (using trafilatura or readability), chunk into ~500-token pieces, embed and store in semantic collection with source URL. For files: support PDF (PyMuPDF), DOCX (python-docx), Markdown, plain text. Same chunk-and-embed flow. Agents can then query ingested knowledge naturally. Add ingest_document tool so agents can also ingest docs during task execution.

11.6 — Memory Forgetting & Decay
Add access_count and last_accessed fields to vector store metadata. On every retrieval hit, increment access_count and update last_accessed. Weekly background task: compute relevance score = access_count × recency_weight. Delete memories below threshold. Cap total memories per collection (e.g., 10,000). Always keep user preferences and error patterns (high importance floor).

11.7 — User Preference Learning
Track every Telegram interaction: task accepted (no follow-up), task modified (follow-up correction), task rejected (explicit negative). Store patterns in semantic collection with category user_preference. Detect: preferred languages, frameworks, naming conventions, verbosity, risk tolerance. Inject into system prompts: "User preferences: prefers concise output, uses snake_case, favors FastAPI over Flask, always wants tests."

Phase 12: Large Codebase Engine
Handle real projects, not toy scripts.

12.1 — Tree-sitter Multi-Language Parsing
Replace Python-only ast module in ast_tools.py and codebase_index.py with tree-sitter (pip install tree-sitter + language grammars). Support: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++. Create parsing/tree_sitter_parser.py with unified interface: parse_file(filepath) → {functions, classes, imports, exports, line_count}. Update codebase_index.build_index to use tree-sitter, falling back to regex for unsupported languages.

12.2 — Code Embedding Index
After structural parsing (12.1), embed each function/class docstring + signature + first 5 lines of body. Store in codebase vector collection with metadata: {filepath, symbol_name, symbol_type, line_start, line_end, language}. Incremental re-indexing: compare file hashes from workspace snapshots, only re-embed changed files. Trigger re-index after any write_file/edit_file/patch_file tool call.

12.3 — Intelligent Context Assembly
Create context/assembler.py. Given a task description: (1) embed the description, (2) query code embedding index for top-k relevant symbols, (3) for each matched symbol, also pull its imports and dependents (from structural index), (4) include related test files, (5) include recent git changes to matched files. Assemble into a focused code context that fits within a configurable token budget (default: 50% of model's context window). Replace the current "dump entire file_tree + read entire files" approach in agent context building.

12.4 — Repository Map
Create context/repo_map.py. Auto-generate on project onboarding or first workspace scan: module dependency graph (who imports whom), entry points (main.py, app.py, index.js, etc.), test mapping (which test files test which modules), config files and their roles, directory purpose summary. Store as structured JSON + a compressed text version for prompt injection. Inject compressed version into planner and architect system prompts.

12.5 — Diff-First Editing
Add apply_diff tool: accepts standard unified diff format, applies it with patch command or Python difflib. Modify coder and fixer agent system prompts to prefer patch_file/edit_file/apply_diff over write_file for existing files. For files >200 lines: never send entire file to model — extract relevant function/section via AST, send only that, accept patch back, apply to full file. Add validation: after any edit tool, re-parse the file with tree-sitter to confirm it's still syntactically valid.

12.6 — Project Onboarding
Enhance /project add <path> command. On registration: (1) detect language and framework, (2) run tree-sitter index, (3) build code embeddings, (4) generate repo map, (5) detect conventions (naming, test framework, linter, etc.), (6) store project profile in DB. Every task touching this project auto-loads the profile. Add project_profile injection into all agent system prompts when working within a registered project.

Phase 13: Agent Collaboration
Agents that work together, not in isolation.

13.1 — Shared Blackboard
Create collaboration/blackboard.py. Per-goal structured state store with typed entries:

text
{
  "architecture": {plan_json},
  "files": {"path": {"status": "implemented|planned|failed", "interface_hash": "..."}},
  "decisions": [{"what": "...", "why": "...", "by": "architect"}],
  "open_issues": [...],
  "constraints": [...]
}
Backed by a blackboards DB table (goal_id, data JSON). Add read_blackboard and write_blackboard as agent tools. Agents read/write structured data instead of parsing prior_steps text blobs.

13.2 — Plan Verification
After planner creates subtasks (in _handle_subtasks), run a verification pass: (1) every file mentioned in descriptions exists as a subtask target, (2) agent_type assignments are sensible (code tasks → coder/pipeline, not writer), (3) dependency graph is acyclic, (4) total estimated cost fits within goal budget, (5) no duplicate subtasks. On failure, send feedback to planner as a new high-priority task: "Your plan has issues: [specifics]. Revise."

13.3 — Agent-to-Agent Queries
Add {"action": "ask_agent", "target": "researcher", "question": "..."} as a new action type in models.py. When base agent loop encounters this action: create an inline high-priority subtask with the question, wait for its completion (with timeout), inject the result back into the requesting agent's message history as a tool result. Prevents architects guessing at things researchers could find.

13.4 — Parallel Independent Tasks
In orchestrator run_loop: when multiple tasks are ready and have no shared dependencies (different goal_ids, or no file overlap), run them truly concurrently. Increase MAX_CONCURRENT_TASKS dynamically based on available model capacity (check rate limiter headroom). Within a single agent execution, detect independent tool calls (e.g., two read_file calls) and execute them with asyncio.gather.

13.5 — Interactive Plan Approval
When planner produces subtasks, send to Telegram as a numbered list with inline buttons: ✅ Approve All, ✏️ Modify, ❌ Reject. On Modify: user replies with changes (e.g., "skip step 3, combine 4 and 5"), create a correction task for planner. On Reject: cancel all subtasks. On Approve or after 10 minutes with no response: proceed (configurable auto-approve timeout).

Phase 14: Multi-Language Coding
Real projects aren't Python-only.

14.1 — Language Toolkit Interface
Create languages/base.py with abstract LanguageToolkit: lint(file), format(file), test(path), typecheck(path), detect_imports(file), install_deps(path), compile(path), run(file). Implement languages/python.py (ruff, pytest, mypy, pip), languages/javascript.py (eslint, prettier, jest/vitest, npm/yarn), languages/typescript.py (extends JS + tsc), languages/go.py (go vet, go test, go build), languages/rust.py (cargo clippy, cargo test, cargo build). also make sure you support java/kotlin fully. 

14.2 — Language-Aware Agent Prompts
On task execution, detect project language from project profile (12.6) or file extensions. Dynamically append language-specific rules to agent system prompts: test runner commands, import conventions, common pitfalls, idiomatic patterns. Don't create separate agents per language — reuse existing agents with injected language context.

14.3 — Multi-Runtime Sandbox
Extend sandbox Dockerfile to include: Node.js 20 LTS, Go 1.22, Rust (rustup), Java 21. Add language parameter to run_code tool — route to appropriate interpreter/compiler. Alternatively, maintain separate lightweight container images per language stack and select at task time.

14.4 — Language-Aware Dependency Detection
Extend deps.py to handle: package.json (npm), go.mod (go), Cargo.toml (cargo), build.gradle (gradle), build.gradle.kts (kotlin) and pom.xml (maven). Detect language from project indicators, run appropriate package manager inside sandbox. Same flow: parse imports → detect missing → auto-install.

Phase 15: MCP & Tool Ecosystem
Plug into the wider world.

15.1 — MCP Client
Create tools/mcp_client.py. Implement MCP client protocol (JSON-RPC over stdio or HTTP SSE). Load server configs from mcp_servers.json: [{"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /path"}, ...]. On startup: launch each server, call tools/list, register returned tools into TOOL_REGISTRY with auto-generated schemas. Handle tool invocation by proxying tools/call to the appropriate MCP server. Handle server crashes with restart + circuit breaker.

15.2 — Browser Automation
Add Playwright to sandbox (or run on host). Create tools/browser.py: browser_navigate(url) → cleaned page text, browser_screenshot(url, selector?) → saves PNG to workspace, returns path, browser_click(selector), browser_type(selector, text), browser_extract(css_selector) → text content, browser_pdf(url) → saves PDF. Run headless Chromium. Add to allowed_tools for researcher and executor agents.

15.3 — Vision Tool
Create tools/vision.py: analyze_image(filepath, question?) → description. Sends image to a vision-capable model (GPT-4o, Gemini, Claude). Auto-detect vision-capable models in MODEL_POOL by checking capabilities. If no vision model available, return error. Use cases: analyze screenshots from browser tool, read diagrams, verify UI output.

15.4 — Document Processing
Create tools/documents.py: read_pdf(filepath) → extracted text (PyMuPDF), read_docx(filepath) → text (python-docx), read_spreadsheet(filepath, sheet?) → CSV text (openpyxl), extract_text(filepath) → auto-detect format and extract. Register all as tools. Chunk large documents and return first N pages with total page count.

15.5 — Email Integration
Create tools/email.py: read_inbox(limit?, filter?) → list of {subject, from, date, preview} (IMAP), read_email(id) → full body, send_email(to, subject, body, attachments?) (SMTP). Config: EMAIL_IMAP_HOST, EMAIL_SMTP_HOST, EMAIL_USER, EMAIL_PASS env vars. send_email always requires approval (requires_approval=True). Add calendar_events(days?) if Google Calendar credentials provided.

15.6 — Database Tools
Create tools/database.py: db_query(connection_string, query) → result table as text, db_schema(connection_string) → tables and columns listing. Support SQLite (local), PostgreSQL, MySQL via appropriate Python drivers. Write queries (INSERT/UPDATE/DELETE) require requires_approval=True. Connection strings stored in encrypted config, referenced by alias name.

15.7 — URL Content Extraction
Create tools/web_extract.py: extract_url(url) → cleaned article text. Use trafilatura or readability-lxml for content extraction. Much better than raw curl for research tasks. Add as preferred alternative to shell curl in researcher agent's allowed_tools.

15.8 — Tool Result Caching
In tools/__init__.py, add a per-agent-execution cache. Read-only tools (file_tree, read_file, project_info, git_status, git_log) cache their results keyed by arguments. Cache invalidated when any write tool (write_file, edit_file, patch_file, shell) is called. Saves redundant tool calls across iterations within one task.


Phase 16: Security & Privacy
Required before trusting with real data.

16.1 — Per-Goal Docker Isolation
Each goal gets its own container instance. Modify shell.py to accept a container_name parameter. In process_task, resolve which container to use based on goal_id. On goal creation: docker run -d --name goal_{id}_sandbox ... mounting only workspace/goal_{id}/. On goal completion: docker rm -f goal_{id}_sandbox. Shared tasks (no goal_id) use the default sandbox.

16.2 — Agent Permission Matrix
Create security/permissions.py. Define per-agent-type allowed tools:

python
AGENT_PERMISSIONS = {
    "planner": {"file_tree", "project_info", "read_file", "read_blackboard", "web_search"},
    "researcher": {"web_search", "read_file", "file_tree", "extract_url", "browser_navigate"},
    "reviewer": {"read_file", "file_tree", "shell", "git_diff", "read_blackboard"},
    "coder": ALL_TOOLS,
    ...
}
Enforce in execute_tool: check AGENT_PERMISSIONS[agent_type] before running. Override current per-agent allowed_tools lists (which are suggestions) with enforced permissions.

16.3 — Command Allowlist
Replace shell blocklist with allowlist per agent type. Coder: python, pip, npm, node, go, cargo, git, cat, ls, mkdir, cp, mv, grep, find, head, tail, wc, sort, curl (specific domains). Researcher: curl only. Reviewer: pytest, python -m py_compile, npm test, go test. Parse the first token of each shell command and check against allowlist before execution.

16.4 — Secret Management
Create security/secrets.py. Load secrets from encrypted secrets.enc file (using Fernet symmetric encryption, key from SECRETS_KEY env var). Agents reference secrets by name via get_secret(name) tool — returns value but it's injected into sandbox env vars, never into prompts. Scan all outgoing model messages for known secret patterns; redact before sending. Scan Telegram output similarly.

16.5 — Audit Trail
Create audit table: {id, timestamp, actor (agent_type/user/system), action (tool_call/model_call/state_change/file_write/approval), target, details, sensitivity_level}. Append-only — no UPDATE or DELETE. Log every: tool execution with args, model call with tier and cost, task state transition, file modification, human approval/rejection. Add /audit <task_id> Telegram command to view.

16.6 — Encrypted Sensitive Columns
Add column-level encryption for tasks.result, conversations.content when task sensitivity > public. Use Fernet with key from env var. Decrypt on read, encrypt on write. Transparent to the rest of the codebase via helper functions in db.py.

Phase 17: Human Interface Upgrade
Telegram that actually works.

17.1 — LLM-Based Message Classification
Replace keyword-based handle_message in telegram_bot.py. Send user message to cheapest model with prompt: "Classify this message as: goal (complex multi-step project), task (single actionable item), question (wants information), feedback (commenting on previous result), followup (references previous interaction), command (system operation). Respond with JSON: {type, confidence}." Route accordingly. Fall back to keyword heuristic if LLM call fails.

17.2 — Progress Streaming
In base.py agent loop, after every iteration: if task has been running >30s, send Telegram update: "🔄 Task #{id}: iteration {n}/{max}, last action: {tool_name or 'thinking'}...". Use a callback mechanism — pass a progress_callback to agent execute() that orchestrator wires to telegram.send_notification. Throttle to max one update per 30s.

17.3 — Rich Result Delivery
When task result > 3000 chars: save full result to workspace/results/task_{id}.md, send summary (first 500 chars) via Telegram with the file attached as a document. For code results: send as .py/.js file attachment. For image results (screenshots): send as photo.

17.4 — File & Image Exchange
Handle Telegram file uploads: save to workspace/uploads/, create a task "Process uploaded file: {filename}" or add to current task context. Handle image uploads: save to workspace, optionally pass to vision tool (15.3). Handle voice messages: transcribe with Whisper (OpenAI API or local whisper.cpp), convert to text task.

17.5 — Pause & Resume
Add task statuses paused to state machine. /pause <id> — sets status to paused, current iteration completes but no new iterations start. /resume <id> — sets status back to pending. /pause goal <id> — pauses all pending/processing tasks under that goal.

17.6 — Multi-User Support
Add users table: {telegram_id, username, role (admin/developer/viewer), created_at}. Admin: full access. Developer: create goals/tasks, view results. Viewer: /status, /queue, /goals only. Check update.effective_user.id against users table on every command. First registered user becomes admin. /adduser <telegram_id> <role> command for admin.

Phase 18: Observability
Can't fix what you can't see.

18.1 — Structured Logging
Replace logging.info with structured JSON logs. Each log entry: {timestamp, level, component, task_id, goal_id, agent_type, model, action, duration_ms, cost, tokens, message}. Use structlog or custom JSON formatter. Output to file + console. Makes log parsing and searching possible.

18.2 — Metrics Collection
Create metrics.py. Track in-memory counters (no external dependency): tasks_completed, tasks_failed, model_calls{model,tier}, cost_total{model}, latency_histogram{model}, tool_calls{tool,success}, tokens_used{model,direction}, queue_depth, active_tasks. Persist hourly snapshots to metrics DB table. Expose via API endpoint (Phase 19) and /metrics Telegram command.

18.3 — Task Replay
Store complete execution trace per task: ordered list of {type (model_call/tool_call/state_change), timestamp, input_summary, output_summary, cost, duration}. Store in task_traces table as JSON array. /replay <task_id> command: dumps the trace step-by-step to Telegram. Shows exact sequence: what the agent thought → what tool it called → what happened → next thought.

18.4 — Alerting
Create alerting.py. Configurable rules in DB: {condition, threshold, window_minutes, action}. Examples: "if >3 tasks fail in 60 min → notify", "if daily cost > $X → notify + pause new tasks", "if model X success rate < 50% in last hour → disable model temporarily". Check rules every cycle in orchestrator main loop.

18.5 — Health Checks
/health Telegram command. Check: DB writable (test write+read), Docker sandbox alive (docker inspect), each configured model reachable (lightweight ping call), disk space > 1GB, embedding model available. Return green/yellow/red status per component. If Docker is down: auto-disable shell-dependent agents, notify, continue with text-only agents.

18.6 — Cost Attribution
Break down costs by: model, agent_type, goal, task, hour/day. Store in existing cost_budgets + model_stats tables. /costs command: "Today: $0.42 | Goal #3: $0.28 (67%) | Top model: gemini-flash $0.31". /costs goal <id>: per-task breakdown.

Phase 19: API & Dashboard
Beyond Telegram.

19.1 — FastAPI Server
Create api.py. Run alongside orchestrator (same process, mounted as sub-application or separate thread). Endpoints: POST /goals, POST /tasks, GET /goals/{id}, GET /tasks/{id}, GET /queue, GET /stats, POST /tasks/{id}/cancel, POST /tasks/{id}/clarify, GET /models, GET /health. All return JSON. Auth via API key header.

19.2 — WebSocket Streaming
Add WebSocket /ws/stream/{task_id}. During agent execution, broadcast: iteration updates, tool call names, partial results. Clients subscribe to specific task or goal. Enables live-updating UI without polling.

19.3 — Web Dashboard
Minimal HTMX + FastAPI templates (no heavy JS framework needed): goal list with status badges, task queue with dependency arrows (mermaid.js), live task execution view (WebSocket-fed), cost chart (chart.js), model performance table, task replay viewer. Serve static files from FastAPI. This is a nice-to-have; API alone is sufficient.

Phase 20: Learning & Adaptation
Get smarter over time.

20.1 — Prompt Versioning
Create prompts table: {id, agent_type, version, system_prompt, created_at, total_uses, avg_quality_score}. Move all hardcoded get_system_prompt strings into DB. Load on agent initialization. /prompt <agent_type> — view current prompt. /prompt <agent_type> set <text> — create new version. Track quality scores per prompt version. Auto-promote: if new version scores higher over 10+ tasks, make it default.

20.2 — Skill Acquisition
Create skills table: {id, name, description, trigger_pattern, prompt_addon, tool_sequence, examples, success_count}. When an agent completes a novel task type successfully, extract the approach: which tools in what order, key prompt patterns. Store as a skill. On future similar tasks (detected via embedding similarity), inject the skill as additional context: "Recommended approach: [tool sequence]. Example: [past success]." Users can also teach skills: /skill add <name> <description> <example>.

20.3 — Feedback Loop
After every result delivered to user, track implicit feedback: no response (accepted), follow-up correction (partial rejection), explicit "wrong" or "redo" (full rejection), "thanks" or continued conversation (accepted). Score: accept=+1, partial=-0.5, reject=-1. Feed scores into: model performance stats (demote bad models), prompt versions (demote bad prompts), skill confidence (demote unreliable skills). Explicit /feedback <task_id> <good|bad> [reason] command for direct feedback.

20.4 — Self-Improvement Proposals
Weekly scheduled task: analyze all failures from the past week. Cluster by error category and root cause. Generate improvement proposals via medium-tier model: "Pattern: 40% of Docker-related tasks fail with 'command not found'. Suggestion: add common tool installation to sandbox Dockerfile." Send proposals to admin via Telegram for approval. On approval, create implementation tasks.

20.5 — Knowledge Distillation
When an expensive model (Claude, GPT-4o) produces a high-quality result (score ≥ 4.5), extract the reasoning pattern and decision framework. Store in semantic memory with high importance. On future similar tasks routed to cheaper models, inject this distilled knowledge: "For tasks like this, a proven approach is: [extracted pattern]."

Phase 21: Advanced Coding Pipeline
The "software company" features.

21.1 — Structured Review Protocol
Replace string-matching review pass/fail. Reviewer must output structured JSON: {"verdict": "pass|needs_fixes|fail", "issues": [{"severity": "critical|warning|nit", "file": "...", "line": N, "description": "...", "suggested_fix": "..."}]}. Fixer receives structured issues array, not prose. Track review pass rate per model → feed into router.

21.2 — Type Checking Integration
Add type checking to pipeline: Python (mypy/pyright), TypeScript (tsc --noEmit), Go (go vet). Run after implementation, before tests. Type errors fed to fixer as structured issues. Block commit if critical type errors remain.

21.3 — Test Coverage
Run pytest --cov (Python), jest --coverage (JS), go test -cover. Parse coverage report. Identify untested critical paths. Auto-generate targeted tests for uncovered functions. /coverage <goal_id> — show coverage summary. Optionally block merge below threshold.

21.4 — PR Workflow
On goal completion: generate PR description (diff summary, test results, coverage delta, review notes). Send to Telegram with ✅ Merge, 🔄 Iterate, ❌ Reject buttons. Merge: squash-merge goal branch into main, clean up branch and workspace. Conflict detection: if main has diverged, attempt auto-rebase, fail gracefully with notification.

21.5 — CI Integration
After pipeline commit, optionally trigger external CI (GitHub Actions via API, or local script). Wait for result (poll or webhook). On CI failure: parse logs, extract error, create fixer task. Configurable per project in project profile.

21.6 — Deployment Tool
Create tools/deploy.py: deploy(target) executes deployment script from project config. Targets: {"staging": "docker-compose -f staging.yml up -d", "production": "ssh deploy@server './deploy.sh'"}. Always requires_approval=True. Post-deploy: run health check URL if configured. On failure: auto-rollback if previous version is known.

Phase 22: Autonomous Jarvis
The end state.

22.1 — Proactive Monitoring
Background agents that periodically check: GitHub repos for new issues/PRs (via API), configured URLs for uptime, RSS feeds for news in configured topics. Surface findings as Telegram notifications: "New issue #42 on your repo: 'Login broken on Safari'. Shall I investigate?"

22.2 — Morning Briefing
Scheduled task at configured wake time. Summarize: overnight work results, pending decisions needing approval, today's calendar events (if email integration enabled), active goal progress, cost summary. Send as single formatted Telegram message.

22.3 — Ambient Context
Create context/ambient.py. Continuously updated context object: current active projects + status, recent decisions and their rationale, user preferences (from 11.7), time of day + day of week, current goals and their progress. Inject compressed version into every agent as ## Current Context. Makes agents aware of the bigger picture.

22.4 — Multi-Step Goal Refinement
When user sends a vague goal ("make my app faster"), don't immediately create a planning task. Instead, start a Telegram conversation: ask 2-3 clarifying questions (which app? what's slow? what's acceptable?), wait for answers, refine the goal description, then trigger planning. Use LLM to generate clarifying questions. Auto-proceed if no response after configurable timeout.

22.5 — Delegation Intelligence
Create delegation/risk_assessor.py. For each task, assess: reversibility (can undo?), cost (cheap = safe), external impact (sends email = high risk), novelty (first time doing this?), user history (user approved similar before?). Score 0-10. Below threshold (e.g., 3): auto-execute. Above: ask for approval. Threshold adjustable. Learn from user's approval patterns to refine thresholds.

22.6 — Cross-Goal Intelligence
When working on Goal B, check if any task overlaps with or conflicts with active Goal A. Examples: both goals modify the same file, Goal B's dependency contradicts Goal A's approach, shared utility code could benefit both. Notify user of conflicts. Suggest synergies: "Goal A already implemented a similar auth module — reuse it?"

22.7 — Plugin Architecture
Define Plugin interface: {name, version, tools: list[ToolDef], agents: list[AgentDef], prompts: list[PromptAddon], mcp_servers: list[MCPConfig]}. Create plugins/ directory. On startup, scan for plugin directories, load and register their tools/agents/prompts. Plugin manifest: plugin.json. Enables modular extension without touching core code. Document the interface so community can build plugins.

22.8 — Personal Knowledge Base
Everything the user tells Jarvis (via Telegram, API, or ingested docs) is indexed and searchable. "What was that API format we discussed last week?" → semantic search over conversation history → retrieve and summarize. /remember <text> — explicitly store important info. /recall <query> — search all knowledge.

22.9 — Self-Monitoring & Healing
Weekly self-assessment: compare this week's success rate, average quality score, cost per task against 4-week moving average. On degradation: identify cause (model API issues? prompt regression? new error pattern?). Auto-remediate: switch degraded models, rollback prompt changes if recent change correlates with quality drop. If can't self-heal: report to user with diagnosis and suggested actions.
