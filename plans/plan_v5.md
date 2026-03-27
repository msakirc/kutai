# Plan v5 — Long-Term Roadmap & Known Gaps

Generated: 2026-03-23
Context: After completing plan_v4 Phases 2-11 implementation pass.

---

## A. Deferred / Skipped Items from v4

### Phase 1: Logging Polish (Low Priority — Functional but Not Perfect)

1. **1.16 — Exception Discipline Pass**: Systematic audit of all `except: pass` blocks.
   Currently many try/except blocks use bare `pass` for best-effort operations (metrics,
   tracing, audit). These are intentional for non-critical paths, but a full audit should
   add at least `logger.debug()` to every suppression point.

2. **1.17 — print() Elimination**: Some `print()` calls may still exist in utility scripts
   and benchmark CLI. Low priority since the main runtime paths use structlog.

3. **1.20 — Frontail Container**: Browser-based log viewer. The docker-compose.yml has a
   placeholder but it's not fully wired. Nice-to-have for mobile debugging.

4. **1.21 — System resource logging**: Periodic CPU/RAM/disk metrics in logs. Currently
   GPU is monitored but host resources are not logged periodically.

### Phase 5: Tool Gaps

5. **5.3 — Browser Automation (Playwright)**: `src/tools/browser.py` does not exist.
   Would enable visual testing, screenshot-based review, and web scraping beyond text.
   Requires Playwright + Chromium in sandbox Dockerfile.

6. **5.5 — Document Processing**: `src/tools/documents.py` does not exist. PDF/DOCX/XLSX
   reading would improve research workflows. Libraries: PyMuPDF, python-docx, openpyxl.

7. **5.7 — MCP Client**: `src/tools/mcp_client.py` does not exist. Would allow dynamic
   tool registration from external MCP servers (filesystem, GitHub, etc.). Medium priority
   as the tool registry is already rich.

### Phase 10: Coding Pipeline Gaps

8. **10.2 — Language Prompt YAML**: Language-specific prompt hints are embedded in
   `LanguageToolkit.get_prompt_hints()` Python methods. A centralized
   `config/language_prompts.yaml` would be more maintainable but is not critical since
   the current approach works.

9. **10.4 — Unified Type Checker Tool**: `src/tools/type_checker.py` does not exist as a
   standalone tool. Type checking is available via language toolkits but agents can't
   invoke it directly as a named tool. Create a thin wrapper that delegates to
   `toolkit.typecheck_command()`.

10. **10.6 — PR Approval Flow via Telegram**: PR creation via `gh pr create` is wired
    into the pipeline, but the Telegram approve/modify/reject inline buttons for PR
    merging are not implemented. Currently PRs are created but must be merged manually.

### Phase 12: API & Dashboard (Skipped Entirely)

11. **12.1 — FastAPI Endpoints**: `src/app/api.py` exists with basic CRUD + Prometheus
    /metrics endpoint, but many planned endpoints are missing: /ws/stream, /artifacts,
    /projects API.

12. **12.2 — WebSocket Live Streaming**: Not implemented. Would allow real-time task
    execution viewing from a web dashboard.

13. **12.3 — Web Dashboard**: No HTMX/Jinja2 dashboard exists. Telegram is the sole UI.

---

## B. Weaknesses Found During Implementation Audit

### Architecture & Design

14. **Two `record_model_call` functions with same name**: `src/infra/metrics.py` has a
    sync in-memory counter version, `src/infra/db.py` has an async DB version. Both are
    called from different places (router vs base.py). Should consolidate or rename to
    avoid confusion. Suggested: rename metrics.py version to `increment_model_metric()`.

15. **Cost tracking fragmented across 3 systems**: Per-task cost in `tasks.cost` column,
    per-goal cost in blackboard `cost_tracking`, and in-memory counters in `metrics.py`.
    These can drift. Need a single source of truth with derived views.

16. **No graceful shutdown**: When the orchestrator is killed, in-progress tasks may be
    left in `running` status with no cleanup. Need an atexit handler that: persists
    in-memory metrics, marks running tasks as `interrupted`, flushes audit log.

17. **SQLite under concurrency**: The system uses aiosqlite for all persistence. Under
    heavy concurrent load (multiple agents writing), WAL mode helps but SQLite is not
    designed for high write concurrency. Monitor for `database is locked` errors. Long-
    term: consider PostgreSQL for production deployments.

### Agent Quality

18. **Fixer agent still receives prose context**: Even though pipeline.py now formats
    structured issues, the fixer agent's system prompt doesn't explicitly tell it to
    parse the `## Issues to Fix` section. It works because the format is clear, but
    adding explicit JSON parsing in the fixer prompt would improve reliability.

19. **Reviewer JSON output not validated**: The pipeline parses reviewer output as JSON
    but falls back to prose silently. No schema validation (e.g., required fields like
    `verdict`, `issues`). Should add a lightweight JSON schema check with retry on
    malformed output.

20. **Agent iteration limits vary without clear rationale**: base agent max=10, reviewer
    max=4, etc. These are tuned by feel. Should be informed by actual performance data
    from the tracing system.

### Security

21. **Secret redaction is regex-only**: The `redact_secrets()` function catches known
    patterns (API keys, card numbers, etc.) but cannot detect novel secret formats or
    generic base64-encoded credentials. Consider adding entropy-based detection for
    high-entropy strings in sensitive contexts.

22. **No prompt injection defense**: Agent system prompts don't include guards against
    prompt injection from user-supplied task descriptions. A malicious task description
    could instruct the agent to ignore its system prompt. Add a post-processing check
    or sandboxed execution boundary.

23. **Shell allowlist is command-level, not argument-level**: The allowlist checks the
    first token of shell commands (e.g., `git` is allowed) but doesn't restrict
    arguments (e.g., `git push --force` is permitted). For destructive commands, need
    argument-level restrictions.

### Observability

24. **Metrics not exposed for external scraping**: Prometheus metrics are at `/metrics`
    in the FastAPI app, but the app may not be running or accessible from a Prometheus
    instance. No Prometheus scrape config exists in the monitoring stack.

25. **No distributed tracing**: Task traces are per-task in the DB. When a task spawns
    subtasks or inline agent queries, there's no parent-child trace correlation. A
    trace_id propagation mechanism would help debug complex goal executions.

26. **Alert rules are hardcoded**: `src/infra/alerting.py` has 3 rules with hardcoded
    thresholds. Should load from `config/alerts.yaml` as plan_v4 specified. Easy fix.

### Workflow Engine

27. **Utility workflows (bugfix, research, documentation, refactor) untested**: These
    JSON workflow definitions exist but may not have been exercised end-to-end. Need
    integration tests or at least a manual test run for each.

28. **No workflow versioning/migration**: When a workflow JSON is updated, in-progress
    goals using the old version have no migration path. The runner resumes from
    checkpoints but new steps added to the JSON won't appear in running goals.

29. **Template expansion is one-shot**: The `feature_implementation_template` expands
    once when `implementation_backlog` is produced. If the backlog changes mid-execution
    (e.g., new features discovered), there's no re-expansion mechanism.

### Performance

30. **Context assembly does DB + embedding queries every iteration**: `assemble_context()`
    runs vector search, git queries, and DB lookups on every agent iteration. For agents
    with 10 iterations, this is 10x the cost. Should cache context per-task and only
    refresh on explicit invalidation.

31. **No model response streaming**: All model calls use `acompletion()` which waits for
    the full response. For long outputs (architecture docs, code generation), streaming
    would improve perceived latency and allow early cancellation.

32. **Sandbox cold start**: The Docker sandbox container starts fresh for each shell
    command. If the container isn't running, there's startup latency. Should ensure
    the sandbox stays warm and reuse between commands.

---

## C. Future Features (Beyond v4 Scope)

33. **Multi-user support**: Currently single-user (one Telegram admin). Support multiple
    users with separate goal spaces, cost budgets, and permission levels.

34. **Goal templates**: Pre-defined goal structures for common tasks ("build a REST API",
    "create a CLI tool"). User picks template, fills in params, system plans and executes.

35. **Model A/B testing**: Run the same task on two models, compare results, auto-select
    winner. Feed into auto-tuner for continuous improvement.

36. **Workspace snapshots**: Before destructive operations, snapshot the workspace (git
    stash or tarball). Enable rollback if a goal produces worse code than before.

37. **Cost prediction before execution**: Before starting a goal, estimate total cost
    based on similar past goals (from metrics DB). Show user: "This will cost ~$2.50
    and take ~45 minutes. Proceed?"

38. **External webhook triggers**: Start goals/tasks from GitHub webhooks (new issue →
    auto-investigate), CI failures (auto-create bugfix task), or cron schedules.

39. **Web Admin Dashboard**: Lightweight web dashboard (FastAPI + Jinja2 or simple SPA)
    showing real-time mission status, agent activity, model utilization, memory stats,
    and task queue. Accessible from any device without clogging Telegram. Should include:
    - Live task queue with status indicators
    - Agent execution timeline (which agent is running, iteration count)
    - Model manager state (loaded model, VRAM usage, swap queue)
    - Memory/vector store stats (collection sizes, recent queries)
    - System metrics (CPU, RAM, GPU utilization)
    - Quick actions (cancel task, restart orchestrator, force model swap)

40. **Per-Model Sampling Overrides**: Temperature, top_p, top_k, repeat_penalty and other
    sampling parameters configurable per model AND per agent type. Stored in models.yaml
    or a new `model_profiles.yaml`. Overrides flow: agent request → router → model call →
    llama-server API body. Also support llama-server-level parameters (like `--temp`,
    `--top-k`) that are passed at server startup for the loaded model. Architecture:
    - `ModelRegistry` holds per-model default sampling params
    - `BaseAgent` can specify agent-level sampling preferences
    - Router merges: agent prefs → model defaults → system defaults (in priority order)
    - Params passed in the `/chat/completions` request body to llama-server

41. **Local Accuracy Benchmarks**: Test GGUF models before deploying them in production
    using an LLM-as-judge grading system. Architecture:
    - Small curated test suite per capability dimension (reasoning, code gen, instruction
      following, Turkish language, JSON compliance, tool calling)
    - Run test prompts against the candidate model
    - Grade responses using a known-good model (e.g. best available local or cloud model)
    - Score feeds into `ModelRegistry.capability_scores` to update model profiles
    - Can run on-demand (`/benchmark <model>`) or automatically when new GGUF detected
    - Results stored in DB for historical comparison
    - Integration with existing `benchmark_fetcher.py` and auto-tuner

---

## D. Deferred Shopping Plan Items

Items from `plans/plans_shopping.md` that were not implemented in the initial pass.

### Phase 2: Remote Execution via GitHub Actions (Deferred Entirely)

39. **2.1 — GitHub Actions Scraper Workflow**: A GitHub Actions workflow that accepts
    scraper name + parameters, runs the scraper in a clean environment, and returns
    results. Avoids IP bans on the user's home IP. Trigger via workflow_dispatch.

40. **2.2 — Result Transfer**: Mechanism to get scraper results from GitHub Actions back
    to the local system. Options: artifact download, repository dispatch event with
    payload, or a shared cloud storage bucket.

41. **2.3 — Local/Remote Decision Logic**: In the search executor, decide per-request
    whether to run locally or remotely. Factors: domain risk level (Hepsiburada high →
    remote, Akakçe low → local), daily request count, time since last request, whether
    the user's IP was recently blocked.

42. **2.4 — GitHub Actions Rate Budget**: Track GitHub Actions minutes consumed. Free tier
    is 2,000 min/month. Each scraper run ~1-2 min. Budget accordingly.

### Phase 10: Missing Resilience Modules

43. **10.2 — Anti-Detection Monitoring** (`src/shopping/resilience/detection_monitor.py`):
    Per-domain success rate tracking, automatic cooldown when degraded, gradual resume
    after cooldown. Expose metrics via `src/infra/metrics.py`.

44. **10.3 — Response Validation / Cross-Source Price Verification**: When same product
    found on multiple sources, flag prices deviating >40% from median. Don't silently
    drop — mark as suspicious.

45. **10.4 — Turkish Encoding Edge Cases**: Handle mixed ISO-8859-9/UTF-8, HTML entities
    in product names, inconsistent Turkish character usage in specs. Test scrapers
    against real pages with Turkish characters.

46. **10.6 — Stale Data Detection** (`src/shopping/resilience/staleness.py`): Beyond TTL —
    monitor price volatility and reduce cache TTL for volatile products, detect flash
    sale indicators, warn user before purchase decisions based on old data.

### Phase 11: Missing Special Intelligence Modules

47. **11.2 — Cross-Category Bundle Detector** (`src/shopping/intelligence/bundle_detector.py`):
    Detect "al X öde Y", "set fiyatı", "çeyiz paketi", store cart discounts. Suggest
    combining items to cross free shipping thresholds.

48. **11.3 — Used Market Awareness** (`src/shopping/intelligence/used_market.py`): Check
    sahibinden, dolap.com for used alternatives. Safety rules per category (never suggest
    used baby/medical/safety products). Refurbished awareness.

49. **11.6 — Bulk / Wholesale Detection** (`src/shopping/intelligence/bulk_detector.py`):
    Compare unit prices at different quantities, detect fake bulk deals where per-unit
    price is higher, factor in shelf life and storage.

50. **11.8 — Import vs Domestic Advisor** (`src/shopping/intelligence/import_domestic.py`):
    Brand origin mapping (domestic vs imported), grey market / parallel import detection
    ("ithalatçı garantili"), BTK registration warnings for phones.

51. **11.9 — Counterfeit / Fraud Detection** (`src/shopping/intelligence/fraud_detector.py`):
    Red flags for fake goods (cosmetics, memory cards, chargers), safety warnings for
    counterfeit chargers/power banks, "A kalite" / "muadil" keyword detection.

52. **11.10 — Campaign Pattern Learner** (`src/shopping/intelligence/campaign_patterns.py`):
    Learn per-category discount patterns from observed price history. Predict savings
    for upcoming events. Initially empty, grows over time.

53. **11.11 — Complementary Product Suggester** (`src/shopping/intelligence/complementary.py`):
    LLM-generated complement suggestions (phone → case, printer → cartridges). Economic
    intelligence for consumable-heavy products.

54. **11.12 — Environmental / Efficiency Advisor** (`src/shopping/intelligence/environmental.py`):
    Energy efficiency, water efficiency, repairability scores, expected lifespan. Frame
    as cost savings, not abstract environmental benefits.

### Phase 12: Missing Test Coverage

55. **12.3 — End-to-End Scenario Tests** (`tests/shopping/test_scenarios.py`): 15+
    realistic scenarios with expected behavior (DDR5 RAM search, oven cupboard with
    dimensions, çeyiz hazırlığı, bayram hediyesi, etc.). Run against mocked scrapers.

56. **12.4 — Output Quality Evaluation** (`tests/shopping/test_output_quality.py`):
    LLM-as-judge evaluation of recommendation quality. Score on relevance, completeness,
    accuracy, helpfulness, Turkish market awareness.

57. **12.5 — Performance Benchmarks** (`tests/shopping/test_performance.py`): Measure
    end-to-end latency for quick search (<30s target), full workflow (<5min target),
    cache hit rates, LLM inference bottlenecks.

### Phase 13: Maintenance & Evolution (Not Started)

58. **13.1 — Scraper Health Dashboard**: Grafana panels for per-domain success rate,
    cache efficiency, shopping session metrics, knowledge freshness.

59. **13.2 — Knowledge Base Refresh Workflow**: Semi-automated monthly refresh via
    Perplexica, category-specific triggers, knowledge_gaps.log.

60. **13.3 — Prompt Optimization Pipeline**: Track prompt quality per intelligence module,
    A/B test prompt variants, category-specific prompt tuning.

61. **13.4 — Self-Improving Substitution Map**: Learn from accepted/rejected
    substitutions, grow the map over time.

62. **13.5 — Scraper Auto-Repair**: LLM-assisted selector repair when sites change
    structure. Present for human review, don't auto-deploy.

63. **13.6 — System Self-Assessment**: Weekly automated benchmark against predefined
    queries. Week-over-week quality tracking.

64. **13.7 — Feature Usage Analytics**: Track which intelligence features get used,
    prioritize maintenance accordingly.

### Known Inconsistencies / Tech Debt from Shopping Implementation

65. **Test isolation**: Shopping test files share DB singleton state. Running the full
    `tests/shopping/` suite together causes 7 flaky failures in test_phase0 due to
    rate_budget DB leaking into request_tracker tests. Each file passes individually.
    Fix: use unique temp DB paths per test class, or add proper cleanup of module-level
    singletons between test files.

66. **Separate shopping memory DB**: Shopping memory uses `data/shopping_memory.db`
    while the rest of the app uses the main `data/kutay.db`. This was intentional for
    isolation but means no transactional consistency between user profile data and
    main app data. Consider merging into main DB long-term.

67. **Scraper fixture coverage**: Only 5 of 15 scrapers have fixture-based tests
    (Akakçe, Trendyol, Hepsiburada, Migros, Technopat). Remaining 10 need fixtures:
    Amazon TR, Ekşi Sözlük, Şikayetvar, Getir, Aktüel Katalog, Sahibinden, Koçtaş,
    IKEA, Donanım Haber, Google CSE.

68. **Shopping blackboard usage**: Shopping workflow steps don't use the blackboard
    system for inter-agent state sharing. Main workflows use read_blackboard/write_blackboard
    tools for sharing architecture decisions, file lists, etc. Shopping workflows should
    use blackboard to share: analyzed intent, search results, product matches, user
    constraints between steps. Currently each step is independent.

69. **Cost tracking for shopping LLM calls**: The centralized `_llm.py` routes through
    `call_model()` but doesn't pass mission_id context. Shopping LLM costs aren't
    attributed to the correct mission/task. Fix: thread task_id through _llm_call so
    the router can associate costs.

70. **Duplicate Perplexica integration**: `src/shopping/integrations/perplexica.py` is a
    separate Perplexica client duplicating `src/tools/web_search.py` which already has
    Perplexica support. The shopping agents already have `web_search` in allowed_tools.
    Consider removing the standalone perplexica.py and using web_search exclusively, or
    merge the shopping-specific query formatting into the main web_search tool.

71. **Orchestrator shopping workflow hook**: The orchestrator (`src/core/orchestrator.py`)
    doesn't call `detect_shopping_intent()` or `should_start_shopping_workflow()` from
    the dispatch module. While shopping tasks created via /price /watch /compare work
    (they set agent_type directly), natural language shopping queries via normal chat
    still need the orchestrator to detect shopping intent and route to the right
    workflow. Wire `dispatch.detect_shopping_intent()` into the orchestrator's task
    classification → workflow routing path.

72. **Shopping workflow loader path**: The workflow loader resolves paths as
    `WORKFLOW_DIR / dir_name / workflow_name.json`. For shopping this means it looks for
    `src/workflows/shopping/shopping.json` when given workflow_name="shopping". This
    works for the main "shopping" workflow, but "quick_search" would look for
    `src/workflows/quick_search/quick_search.json` instead of
    `src/workflows/shopping/quick_search.json`. The sub-workflows need to be loadable
    either by adjusting the loader or by using the full path "shopping/quick_search".

73. **Import style inconsistency**: Shopping modules use absolute imports
    (`from src.shopping.xxx import`) while the rest of the codebase (agents, core) uses
    relative imports (`from .xxx import`, `from ..xxx import`). Both work but the style
    is inconsistent. Low priority — functional but not idiomatic.
