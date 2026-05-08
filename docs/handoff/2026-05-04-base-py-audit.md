# BaseAgent Audit — Kill Agents Phase 1

Anchor: 2026-05-04 (continues `2026-05-04-kill-agents.md`)

Caveman tone in body; full sentences where precision matters.

---

## Sizes

- `src/agents/base.py` — 4092 LOC (handoff said 5800, off but still huge).
- 20 subclass files — 1700 LOC. Each has 1 method (`get_system_prompt`), except 2 outliers (see §Outliers).
- Net target ≈ 4350 LOC across new modules vs. 5800 LOC today. ~1450 LOC drop. Real win: separable units, single retry layer.

## Two-retry-layer confirmation

Dispatcher already retries+mid-attempt-swaps (singular admission, shipped 2026-05-04). BaseAgent runs its own format/guard/validation/self-reflect retries on top. Past month's bug pile (envelope-unwrap, retry-staleness, grader-echo, schema-checklist-on-wrong-dict) all live at this seam. Kill collapses to: dispatcher = LLM-call retry, react.py = ReAct sub-iterations only.

## Block-by-block categorization

| Block (lines) | Category | Destination |
|---|---|---|
| `_unwrap_final_answer` (68-109) | response parsing | `src/core/response_parser.py` |
| `_partition_tool_calls` + `SIDE_EFFECT_TOOLS` + `CACHEABLE_READ_TOOLS` (47-123) | tool exec helpers | `src/core/tool_executor.py` |
| `BaseAgent` class attrs (151-188) | profile fields | `src/core/profiles.py` Profile dataclass |
| `get_system_prompt` (218-228) | profile field | `Profile.system_prompt` |
| `_get_available_tools_prompt` (233-330) | prompt build | `src/core/context_builder.py` |
| `_build_full_system_prompt` (331-389) | prompt build | `src/core/context_builder.py` |
| `_is_action_task` (391-448) | heuristic for hallucination guard | `src/core/react.py` |
| `_get_search_depth` (450-463) | tiny helper | `src/core/react.py` |
| `_check_sub_iteration_guards` (474-592) | ReAct guards | `src/core/react.py` |
| `_check_tool_permission` (598-607) | tool exec | `src/core/tool_executor.py` |
| `_trim_for_escalation` (609-671) | ReAct escalation | `src/core/react.py` |
| `_escalate_requirements` (673-678) | trivial wrapper | inline / delete |
| `_build_context` (683-1224, ~540 LOC) | context builder MEGA | `src/core/context_builder.py` |
| `_truncate_to_tokens` (1226-1231) | util | `src/core/context_builder.py` |
| `_fetch_deps` (1233-1369) | artifact-store fetch | `src/core/context_builder.py` |
| `_format_prior_steps` (1370-1385) | format util | `src/core/context_builder.py` |
| `_format_conversation` (1386-1408) | format util | `src/core/context_builder.py` |
| `_parse_agent_response` (1409-1475) | parse | `src/core/response_parser.py` |
| `_try_parse_json` (1477-1490) | parse | `src/core/response_parser.py` |
| `_normalize_action` (1492-1612) | parse alias map | `src/core/response_parser.py` |
| `_count_tokens` / `_get_context_window` (1617-1661) | ctx-window | `src/core/context_window.py` |
| `_trim_messages_if_needed` (1664-1736) | ctx-window | `src/core/context_window.py` |
| `_prune_tool_results_to_fit` (1738-1812) | ctx-window | `src/core/context_window.py` |
| `_build_litellm_tools` (1817-1843) | tool schema build | `src/core/tool_executor.py` |
| `_parse_function_call_response` (1845-1895) | parse | `src/core/response_parser.py` |
| `_validate_response` (1900-1934) | output validation | `src/core/react.py` (small enough to inline) |
| `execute` (1946-2154) | dispatch entry + tools_hint override + step-refresh + emit-pass | split: dispatcher entry + `src/core/react.py` setup |
| `_execute_react_loop` (2156-3454, ~1300 LOC) | THE loop | `src/core/react.py` |
| `_build_model_requirements` (3456-3643) | reqs assembly + step-refresh dup | `src/core/requirements_builder.py` (or fold into fatih_hoca) |
| `_maybe_constrained_emit` (3645-3843) | post-hoc structured emit | `src/workflows/engine/` post-hook (workflow-step-specific) |
| `execute_single_shot` (3845-3931) | thin wrapper around dispatcher | DELETE — caller uses `dispatcher.request()` + `parse_agent_response()` directly |
| `_self_reflect` (3933-4005) | optional post-final review | `src/core/react.py` |
| `_tool_idempotency_key` (4011-4019) | tool-exec idem | `src/core/tool_executor.py` |
| `_save_checkpoint` / `_clear_checkpoint_safe` (4024-4073) | ReAct state persist | `src/core/react.py` |
| `_safe_log` (4078-4092) | trivial fire-and-forget | inline at callsites |

## Profile dataclass (harvested from subclasses)

```python
@dataclass(frozen=True)
class Profile:
    name: str
    description: str = ""
    system_prompt: str = ""           # static; runtime override via prompt_versions DB stays
    allowed_tools: tuple[str, ...] | None = None  # None = all; () = none
    max_iterations: int = MAX_AGENT_ITERATIONS
    execution_pattern: str = "react_loop"         # "react_loop" | "single_shot"
    enable_self_reflection: bool = False
    min_confidence: int = 0
    can_create_subtasks: bool = False
    default_tier: str = "cheap"        # legacy; consider dropping if unused after migration
    min_tier: str = "cheap"            # legacy
```

`_suppress_clarification` is runtime, not profile — set from `task.context.may_need_clarification`.

`allowed_tools` mutation by skill-injection (`_build_context` skill block, line 1112-1117) needs careful port: profile must stay frozen, react-loop holds a per-execution mutable `tool_set` it can extend.

## Outliers — NOT ReAct agents (but still LLM-driven)

`grader.py` and `artifact_summarizer.py` override `execute()` entirely. No ReAct loop in the agent class. The actual LLM call is **enqueued through Beckman as an overhead sub-task** by the helper (`grade_task` in `src/core/grading.py` enqueues with `raw_dispatch: True`; `_llm_summarize` same pattern per commit `c174026`).

So the agent class is a 3-line glue wrapper:
1. fetch source task row
2. call helper (helper enqueues overhead sub-task → Beckman → dispatcher → result)
3. shape result into `posthook_verdict` dict for Beckman's rewrite layer

**Right destination is NOT mr_roboto (call IS LLM)** and NOT a profile (no ReAct). Right destination is **`src/workflows/engine/post_hooks/`** (or Beckman's terminal hook handler directly). They are post-hook handlers, not agents.

After kill-agents:
- Beckman terminal hook for a task with `posthook_kind: "grade"` runs the 3-step wrapper inline (fetch + call `grade_task` + shape result). `grade_task` itself stays as-is — it already enqueues an overhead sub-task through Beckman.
- Same for `posthook_kind: "summarize_artifact"` → calls `_llm_summarize`.
- `agent_type` rows with values `"grader"` / `"artifact_summarizer"` get rewritten by a one-shot DB migration to a post-hook task shape, OR kept as a thin compat alias that dispatches to the post-hook handler. Pick whichever is cleaner once Phase 4 callsite migration runs.

## Duplicated work to dedupe during port

1. **Step-config refresh** runs twice: `execute()` 2017-2128 AND `_build_model_requirements` 3562-3601. Same workflow JSON re-read. After the port: `react.py` setup phase calls one shared `_refresh_step_from_workflow_json(task)` once before `_build_context` + `_build_requirements`.
2. **Envelope-unwrap** logic exists in 3 places: `_unwrap_final_answer`, `_build_context` retry-block (lines 925-931), and exhaustion-path final-answer extraction (3417-3429). Single helper in `response_parser.py`.
3. **Task-context JSON parse** is repeated everywhere (`isinstance(ctx, str): json.loads(ctx)`). One helper, called once at react-entry.

## Migration order (revised)

Each step shippable, no behavior change until step 6.

1. `profiles.py` + Profile dataclass + 20 entries. AGENT_REGISTRY rewritten as `{name: Profile}`. `BaseAgent` keeps reading `.allowed_tools` etc. from a `_profile` attr it gets from the registry. **No ReAct change.** Tests verify lookup parity.
2. `response_parser.py` extracted; `BaseAgent` delegates. Same for `context_window.py`. Pure code move.
3. `tool_executor.py` extracted (single tool + multi tool + permission + idempotency + schema build). `BaseAgent._execute_react_loop` calls into it. Pure code move.
4. `context_builder.py` extracted (~700 LOC moved from `_build_context` + `_fetch_deps` + format helpers). Pure code move.
5. `react.py` extracted. `BaseAgent.execute` becomes a 5-line shim that calls `react.run(profile, task)`. **Trace replay**: capture 5-10 production task traces (in/out), run through new path, diff. Ship if matched.
6. Dispatcher gains `dispatch(profile_name, task)` entry. Orchestrator switches `get_agent(t).execute(task)` → `dispatcher.dispatch(t, task)`. Keep `BaseAgent` shim as fallback for one cycle.
7. `grader` + `artifact_summarizer` migrate to **`workflows/engine/post_hooks/`** handlers (NOT mr_roboto — they enqueue LLM sub-tasks via Beckman). Existing `agent_type="grader"` rows: either rewrite via one-shot DB migration to `kind="post_hook" + payload.kind="grade"`, or keep `agent_type` as compat alias the orchestrator routes to the post-hook handler. Existing rows are short-lived; drain naturally within hours.
8. Delete `BaseAgent`, `src/agents/*.py` (20 files), `get_agent()`, `AGENT_REGISTRY`. `__init__.py` becomes empty or re-exports profile registry.
9. `_maybe_constrained_emit` → `workflows/engine/post_hooks/constrained_emit.py`, called by workflow_engine after agent completes a step with constrainable schema. Was always workflow-specific; finally lives there.

## Open questions for next session

1. `_prompt_version_override` (DB-loaded prompt) — keep dynamic prompt-versioning? Yes — moves to react.py setup. Profile is the fallback when DB has no active version.
2. Skill-injection mutating `allowed_tools` — confirmed needs per-execution mutable copy. New `react.run` builds `effective_tools = list(profile.allowed_tools)` and mutates that, profile stays frozen.
3. `execute_single_shot` callers — task_classifier, mission planner, others. Audit needed: most can call `dispatcher.request()` directly with no profile. Profiles with `execution_pattern="single_shot"` collapse to `iteration_cap=1` profile arg.
4. `default_tier` / `min_tier` legacy strings — grep callers; if only `_build_model_requirements` reads them, drop after that function moves and difficulty-based requirements take over.
5. Trace-replay harness — does one already exist? Check `tests/agents/` for fixtures. If not, lightweight replay: pickle `(messages, response)` pairs from a `record_model_call` hook, replay via `react.run` with `dispatcher.request` mocked to return canned responses.

## Risk hotspots (don't lose during port)

- **Streaming partial-content recovery on cancel** (`asyncio.CancelledError` block lines 3373-3395 + `self._partial_content` set in `_execute_react_loop`). If react.py loses this, mission cancellation drops partial work.
- **Per-iteration heartbeat bump** (line 2358-2362) — orchestrator's no-progress watchdog keys off it. Must fire from `react.py` at the same cadence.
- **Doğru mu Samet quality check** fires inside checkpoint-recovery (line 2241) and self-reflection (line 3994). Both must remain.
- **Read-file intercept for already-injected artifacts** (line 2856-2880). Workflow-step specific, easy to miss.
- **`tools_hint` override and `_strip_file_tools` / `_strip_web_tools` / `_WRITE_TOOLS` auto-strip** (lines 1972-2012). All run BEFORE react loop starts. Belongs in `react.py` setup phase.
- **`record_model_call` + `record_cost`** fire inside the loop (2547-2562). Dispatcher also calls record paths. Audit: is this double-counting? Likely — dispatcher should own this exclusively, react.py should drop the calls.
- **`audit` + `record_tool_call` + `append_trace`** fire inside tool-exec (2972-3005, 3197-3219). Move to `tool_executor.py` so multi-tool path doesn't drift.

## Verdict

Kill direction sound. Audit complete. Step 1 of handoff plan done.

Next concrete step: write Profile dataclass + harvest 20 entries. That's a 1-PR shippable unit with zero behavior change (BaseAgent reads `.allowed_tools` etc. from `self._profile` instead of class attrs).
