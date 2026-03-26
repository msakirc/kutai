# Coding Pipeline -- Issues & Fixes

## Current Capabilities

### Agent Roster (Coding-Related)
| Agent | Role | Max Iterations | Default Tier | Key Tools |
|-------|------|---------------|-------------|-----------|
| `planner` | Decomposes missions into subtasks | 3 | medium | file_tree, project_info, read_file, web_search |
| `architect` | Designs system architecture (ARCHITECTURE.md) | 4 | medium | file_tree, project_info, read_file, write_file, web_search |
| `coder` | Writes, runs, debugs, commits code | 8 | medium | shell, file_tree, read/write/edit/patch_file, apply_diff, get_function, query_codebase, lint, git_*, run_code, web_search |
| `implementer` | Implements one file per invocation | 6 | medium | file_tree, read_file, write_file, edit_file, shell, lint |
| `test_generator` | Writes and runs pytest tests | 6 | medium | file_tree, read_file, write_file, edit_file, shell, lint |
| `reviewer` | Structured JSON code review | 4 | medium | read_file, file_tree, project_info, shell, git_diff |
| `fixer` | Applies fixes from review/test feedback | 8 | medium | file_tree, read/write/edit/patch_file, apply_diff, get_function, query_codebase, shell, lint |
| `error_recovery` | Diagnoses and fixes failed tasks | 4 | medium | shell, read_file, write_file, file_tree, run_code |
| `visual_reviewer` | Analyzes screenshots/UI | 3 | medium | read_file, file_tree, web_search |

### CodingPipeline (Multi-Agent Orchestration)
- **Location**: `src/workflows/pipeline/pipeline.py`
- **Stages**: Architect -> Implement (per-file) -> Dependencies -> Test -> Coverage -> Review+Fix loop -> Commit -> PR
- **Complexity classifier**: oneliner / bugfix / refactor / tdd / feature (keyword-based)
- **Adaptive stages**: Simpler tasks skip architect/test stages
- **TDD mode**: Tests before implementation when explicitly requested
- **Review-fix loop**: Up to 5 QA iterations with model diversity enforcement
- **Coverage gate**: Runs pytest --cov after test stage
- **Incremental progress**: Saves/resumes file implementation progress
- **Context accumulation**: PipelineContext tracks architecture plan, implementations, test results, review feedback
- **Convention detection**: Injects naming style, async patterns, common imports into agent prompts
- **Codebase map**: Injects structural overview for larger codebases
- **PR creation**: Automatically pushes branch and creates GitHub PR on mission branches

### Tool Infrastructure
- **File editing**: 4 tools (write_file, edit_file, patch_file, apply_diff) -- well-differentiated
- **AST tools**: get_function, replace_function, list_classes, list_functions, get_imports (Python-only via ast module)
- **Codebase indexing**: build_index, query_codebase, codebase_map (tree-sitter + ast + regex fallback)
- **Git ops**: init, commit, branch, log, diff, rollback, status -- runs on host, not in Docker
- **Shell**: Docker sandbox with command blocklist and per-agent allowlists
- **Linting**: Language-toolkit dispatch (Python/JS/TS/Go/Rust), falls back to ruff for Python
- **Coverage**: pytest --cov and jest --coverage
- **Code runner**: Python-only execution via Docker sandbox
- **Tree-sitter**: Multi-language parsing (Python, JS, TS, Go, Rust, Java, C/C++) with ast/regex fallback
- **Security**: Agent permission matrix, path sandboxing, command blocklist

### Context Pipeline
- Task description + workspace snapshot
- Dependency results (completed sibling tasks)
- Prior steps from same mission (inline context chain)
- Project profile (language, framework, conventions from onboarding)
- Blackboard (structured per-mission state: architecture, files, decisions)
- Skills library (matching learned patterns)
- RAG context (retrieved from vector store)
- User preferences
- Project memories
- Ambient context assembly

---

## Critical Issues

### 1. Docker Sandbox is a Hard Dependency -- No Fallback
**Location**: `src/tools/shell.py` lines 227-234
**Impact**: ALL coding tasks fail when Docker is offline

When Docker is not running, `run_shell()` returns an error string:
```
"Docker sandbox is not available.\nRun:\n  docker build -t orchestrator-sandbox:latest .\n..."
```

This propagates up to every agent that uses `shell`, `run_code`, or `lint` (which delegates to shell). There is **no fallback** to host-local execution. Affected agents: coder, implementer, fixer, test_generator, reviewer, executor, error_recovery.

**Impact chain**:
- `shell` fails -> agents cannot run or test code
- `lint` fails -> apply_diff and auto-lint after file edits fail silently
- `run_code` fails -> code_runner.py returns error
- `coverage` fails -> quality gate has no data
- The entire CodingPipeline review+fix loop is crippled because fixer/reviewer cannot run tests

**Fix**: Add a `SANDBOX_MODE` config: `docker` (current), `local` (subprocess with security guards), `none` (tools return graceful skip). Implement local sandbox with cwd confinement, timeout, and resource limits. This is the highest-impact single fix.

### 2. `implementer` Agent Missing Critical Editing Tools
**Location**: `src/agents/implementer.py` line 22-28
**Impact**: Implementer cannot use patch_file, apply_diff, get_function, or query_codebase

The implementer's `allowed_tools` list is: `file_tree, read_file, write_file, edit_file, shell, lint`.

Missing compared to coder/fixer:
- `patch_file` -- the most reliable editing tool per the coder's own prompt
- `apply_diff` -- needed for multi-location edits
- `get_function` -- critical for reading exact function source before editing
- `query_codebase` -- needed to find related code
- `project_info` -- useful for understanding project structure
- `git_diff`, `git_commit` -- no git awareness

Since the pipeline runs implementer on **every file**, this forces it to use `write_file` (full file rewrite) or `edit_file` (fragile line-number-based) for all changes. For existing files, this is error-prone.

**Fix**: Add `patch_file`, `apply_diff`, `get_function`, `query_codebase`, and `project_info` to implementer's allowed_tools.

### 3. `reviewer` Cannot Run Tests (Missing `run_code` and `lint`)
**Location**: `src/agents/reviewer.py` line 22-28
**Impact**: Reviewer prompt says "run tests" but allowed_tools list limits actual capability

The reviewer's `allowed_tools` are: `read_file, file_tree, project_info, shell, git_diff`.

Missing tools:
- `lint` -- reviewer should be able to lint-check files
- `run_code` -- reviewer should be able to run quick verification
- `query_codebase` -- useful for cross-referencing reviewed code

Note: The security permissions in `security/permissions.py` line 49 **do** allow `lint` and `run_code` for reviewer. But the agent's `allowed_tools` list is the effective filter (base.py line 143-148). The permission matrix and allowed_tools are **redundant and inconsistent** -- this is a broader design issue (see Integration Issues #1).

**Fix**: Add `lint`, `run_code`, and `query_codebase` to reviewer's allowed_tools.

### 4. `code_runner` Only Supports Python
**Location**: `src/tools/code_runner.py` line 13
**Impact**: All non-Python code execution fails with a flat error message

```python
if language != "python":
    return f"Only Python execution is supported, got: {language}"
```

Since the pipeline supports multi-language projects (tree-sitter parses JS/TS/Go/Rust/Java/C), the code_runner is a bottleneck. Agents prompted to "run your code" will fail for any non-Python project.

**Fix**: Add Node.js execution at minimum (write to temp file, run `node /tmp/_run_code.js`). Go and Rust could compile+run. Should leverage the language toolkit infrastructure already in `src/languages/`.

---

## Integration Issues

### 1. Dual Permission System (allowed_tools vs. security/permissions.py)
**Location**: `src/agents/base.py` line 143, `src/security/permissions.py`
**Impact**: Confusing, inconsistent, hard to maintain

Two independent permission systems:
1. `BaseAgent.allowed_tools` (per-agent class attribute) -- filters which tools appear in the system prompt
2. `security/permissions.py` AGENT_PERMISSIONS -- checked at tool execution time via `_check_tool_permission()`

These are **not synchronized**. Examples of drift:
- `reviewer` allowed_tools lacks `lint` and `run_code`, but permissions.py allows them
- `implementer` has `allowed_tools = None` in permissions.py (full access) but a restricted list in the class
- `architect` allowed_tools includes `write_file` but permissions.py does NOT include it

When they conflict, the agent's prompt won't mention the tool (so the LLM won't try to use it), even if permissions would have allowed it. The reverse is also true: if a tool appears in the prompt but is blocked by permissions, the agent wastes an iteration on a rejected call.

**Fix**: Make one system the source of truth. Best approach: keep agent-class `allowed_tools` as the single list; remove or convert permissions.py to an audit log. Alternatively, auto-generate allowed_tools from permissions at agent init.

### 2. `pipeline_utils._load_progress()` Uses Wrong Import
**Location**: `src/workflows/pipeline/pipeline_utils.py` lines 84, 101
**Impact**: Incremental progress save/load silently fails

```python
import tools.workspace as _ws  # WRONG -- should be src.tools.workspace
```

This `import tools.workspace` will raise ImportError in the real runtime (the module path is `src.tools.workspace`). Both `_load_progress()` and `_save_progress()` have this bug. Since the exception is caught and swallowed, incremental progress is silently broken -- the pipeline will re-implement files that were already completed if it resumes.

**Fix**: Change to `from src.tools.workspace import WORKSPACE_DIR` or use the module-level `_ws` import already at the top of the file.

### 3. Coverage Stage References Non-Existent `pipe_ctx.workspace` Attribute
**Location**: `src/workflows/pipeline/pipeline.py` line 324
**Impact**: Coverage always fails with AttributeError (caught, so pipeline continues)

```python
cov_report = await get_coverage_summary(
    project_root=pipe_ctx.workspace or ".",  # PipelineContext has no 'workspace' attribute
```

`PipelineContext` (in pipeline_context.py) does not define a `workspace` attribute. The AttributeError is caught by the broad `except Exception`, so coverage is silently skipped every time.

**Fix**: Add `workspace: str = ""` to PipelineContext dataclass, or use `get_mission_workspace_relative(pipe_ctx.mission_id)`.

### 4. Coverage Tool Signature Mismatch
**Location**: `src/tools/coverage.py` line 17 vs. `pipeline.py` line 324
**Impact**: Even if the workspace issue is fixed, the call would fail

`get_coverage_summary()` accepts `project_path` and `language`, but pipeline.py calls it with `project_root` and `language`. Wrong keyword argument name.

**Fix**: Change pipeline.py call to use `project_path=` instead of `project_root=`.

### 5. Architect Prompt Says "MUST run write_file" but Has Typo
**Location**: `src/agents/architect.py` line 80
**Impact**: Minor -- "YouMUST" (missing space)

```
"- YouMUST run `write_file` to save `ARCHITECTURE.md` before finishing.\n"
```

**Fix**: Add space: "You MUST".

---

## Missing Features

### 1. No Host-Local Test Runner
**Impact**: When Docker is down, there is zero testing capability

Currently all test execution goes through Docker sandbox (`shell` -> `docker exec`). There is no way to run `pytest` or `npm test` on the host. This is the primary blocker when Docker is unavailable.

**Fix**: Add a `local_test_runner` tool that runs tests in a subprocess with cwd/timeout restrictions, activated when Docker is detected as offline.

### 2. No Dedicated Git Tool for Agents (Beyond Basic Ops)
**Impact**: Agents cannot do branch management, cherry-pick, stash, or rebase

`git_ops.py` provides: init, commit, branch, log, diff, rollback, status. Missing:
- `git_stash` / `git_stash_pop` -- useful for fixer to save WIP
- `git_cherry_pick` -- useful for selective changes
- `git_blame` -- useful for reviewer to understand change history
- `git_merge` / `git_rebase` -- needed for branch workflow

Agents must fall back to `shell` for these, which means running git inside Docker (where the repo may not be mounted the same way as on host). Since git_ops runs on the **host** while shell runs in **Docker**, there is a topology mismatch.

**Fix**: Add `git_stash`, `git_blame` to git_ops.py. For more complex ops, ensure shell can access the git repo correctly.

### 3. No Static Analysis Beyond Linting
**Impact**: Security vulnerabilities and type errors go undetected

The pipeline has lint (ruff/eslint) and coverage (pytest --cov), but no:
- Type checking (mypy, pyright)
- Security scanning (bandit, semgrep)
- Dependency vulnerability scanning (pip-audit, npm audit)

The quality_gates.py defines `security_scan_clean` for phase_10, but there is no tool that actually runs a security scan.

**Fix**: Add `type_check` tool (mypy for Python, tsc for TypeScript) and `security_scan` tool (bandit for Python). Wire into pipeline between test and review stages.

### 4. No Workspace Isolation Between Concurrent Tasks
**Impact**: Multiple agents editing the same workspace can corrupt each other's work

The pipeline runs implementer calls **sequentially** (good), but the orchestrator can run multiple tasks concurrently (`MAX_CONCURRENT_TASKS = 3`). If two different missions or two independent tasks write to the same workspace, they can conflict.

Git branching per mission (`create_mission_branch`) provides some isolation, but agents within the same mission working on different files could still clash.

**Fix**: Add file-level locking in workspace.py, or enforce sequential execution within a single mission's coding tasks.

### 5. No Language-Specific Test Framework Selection
**Impact**: test_generator prompt hardcodes pytest, fails for JS/Go/Rust projects

The test_generator's system prompt says "Use pytest for all Python testing" and "Use shell to run pytest". For non-Python projects, the agent has no guidance on which test framework to use.

**Fix**: Inject the detected project language and recommended test framework into the test_generator's task description. Use language toolkit's `test_command()` method.

### 6. No Rollback on Pipeline Failure
**Impact**: A failed pipeline leaves partially-implemented files in the workspace

If the pipeline fails mid-implementation (e.g., iteration 3 of 5 files), there is no automatic rollback. The workspace snapshot is taken before execution (line 1082-1096 in orchestrator.py), but there is no code that restores it on failure.

**Fix**: Add a rollback mechanism: on pipeline failure, revert to the pre-execution workspace snapshot or git commit.

---

## Agent Prompt Issues

### 1. Coder Prompt Claims Docker Has Pre-Installed Packages
**Location**: `src/agents/coder.py` lines 91-93
**Impact**: Agent assumes packages exist that may not be installed

```
"- The shell runs in a Docker sandbox with Python 3.12.\n"
"- Common packages are pre-installed (flask, fastapi, requests, pandas, pytest, etc.).\n"
```

If the Docker image does not actually include these packages, the agent will skip `pip install` and fail on imports. No runtime verification of what is actually installed.

**Fix**: Either ensure the Docker image matches the claim, or change the prompt to say "Install required packages with pip before using them."

### 2. Executor Prompt Mentions Docker Internals
**Location**: `src/agents/executor.py` lines 70-73
**Impact**: Leaks implementation details that confuse the LLM

```
"- Shell commands run inside a Docker container, NOT on the host.\n"
"- Host tools like `ollama`, `systemctl`, `docker` are NOT available in shell.\n"
```

While technically accurate, this bleeds infrastructure details into the agent's reasoning. The executor is a general-purpose agent (default_tier: cheap) -- it may receive non-coding tasks where Docker is irrelevant.

**Fix**: Move sandbox-specific guidance into a conditional block that only appears when shell tools are available.

### 3. Implementer Prompt References ARCHITECTURE.md Without Ensuring It Exists
**Location**: `src/agents/implementer.py` line 55
**Impact**: For non-pipeline invocations, ARCHITECTURE.md may not exist

```
"Ensure your code perfectly matches the interfaces designing in the `ARCHITECTURE.md`.\n"
```

When implementer is used outside the CodingPipeline (e.g., standalone task), there may be no ARCHITECTURE.md. The agent will waste iterations trying to find a non-existent file.

**Fix**: Conditionally inject ARCHITECTURE.md reference only when it was actually created by the architect stage.

### 4. Fixer Has No `run_code` or `git_commit` Tools
**Location**: `src/agents/fixer.py` line 21-32
**Impact**: Fixer cannot do quick code execution or commit fixes

The fixer's allowed_tools list is missing:
- `run_code` -- useful for quick verification without full shell commands
- `git_commit` -- fixer should be able to commit its fixes
- `git_diff` -- fixer should see what it changed
- `web_search` -- occasionally needed to look up error messages

**Fix**: Add `run_code`, `git_diff`, and optionally `git_commit` to fixer's allowed_tools.

### 5. No Tool Usage Examples in Agent Prompts
**Impact**: LLMs sometimes guess wrong argument formats

While the base agent injects tool descriptions and optional examples from TOOL_REGISTRY, the per-agent system prompts reference tools by name without showing exact argument structures. Models (especially cheaper ones) may use wrong arg names.

The TOOL_REGISTRY does include `example` keys for some tools, but these are generic. Agent-specific examples (e.g., how a reviewer should invoke `shell` to run pytest) would improve tool call accuracy.

**Fix**: Add 1-2 concrete tool call examples to each agent's system prompt, showing the exact JSON format for their most common tool invocations.

---

## Recommendations

### Priority 1 -- High Impact, Moderate Effort
1. **Add local shell fallback** when Docker is offline (subprocess with security constraints). This is the single highest-impact improvement. Without it, the entire coding pipeline is non-functional when Docker is down.
2. **Fix implementer's tool list** -- add patch_file, apply_diff, get_function, query_codebase. Immediate quality improvement for the most-called pipeline agent.
3. **Fix the broken import** in `pipeline_utils.py` (_load_progress / _save_progress). Currently incremental progress is silently broken.
4. **Fix coverage stage** -- add workspace attribute to PipelineContext, fix keyword arg name.

### Priority 2 -- Medium Impact, Low Effort
5. **Unify permission systems** -- either remove security/permissions.py or auto-sync with allowed_tools. Current dual system causes drift bugs.
6. **Add missing tools to reviewer and fixer** -- lint, run_code for reviewer; run_code, git_diff for fixer.
7. **Fix agent prompt issues** -- implementer ARCHITECTURE.md reference, Docker claims in coder, typo in architect.
8. **Add language-aware test framework selection** -- inject language/framework into test_generator via task description.

### Priority 3 -- High Impact, High Effort
9. **Add static analysis tools** -- mypy type checking, bandit security scanning. Wire into pipeline quality gates.
10. **Add workspace rollback** on pipeline failure -- use pre-execution snapshot.
11. **Add multi-language code_runner** -- at minimum Node.js support.
12. **Add workspace-level file locking** for concurrent task safety.

### Priority 4 -- Nice to Have
13. Add git_blame, git_stash to git_ops.py for more sophisticated agent workflows.
14. Add tool usage examples to agent system prompts for cheaper models.
15. Add dependency vulnerability scanning (pip-audit, npm audit).
