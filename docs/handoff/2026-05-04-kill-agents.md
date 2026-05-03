# Handoff — Kill Agent Classes

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries.
Code/commits/security: write normal.

Anchor: 2026-05-04 (continues from `docs/handoff/2026-05-04-unify-non-beckman-paths.md`)

---

## The Insight

> "Today's react loop should be handled via haLLederiz and/or dispatcher,
> not agents. Maybe we should kill agents altogether (today they are
> just helpful prompts) or have good use for them."
> — user, 2026-05-04

After Phase 1 + Phase 2 + Phase 5 of the singular admission migration shipped
(see `docs/superpowers/specs/2026-05-04-beckman-singular-admission-design.md`),
the question of where ReAct logic should live surfaced. BaseAgent was
designed in an "agents own tasks" world. Today's architecture is task-centric:

- Beckman owns admission + lifecycle
- Dispatcher owns single LLM call execution + retry + mid-attempt swap
- HaLLederiz Kadir owns transport (litellm + streaming + parsing)
- Salako owns mechanical executors
- Orchestrator pump ties them together

In that picture, **BaseAgent doesn't fit anywhere clean.** It's a fourth
executor type ("multi-call iterator") that exists because workflows
didn't exist when BaseAgent was written.

---

## Survey of agent subclasses (Python introspection, 2026-05-04)

20 subclasses. Every one of them is **system_prompt + allowed_tools.
Zero extra methods.**

```
analyst.py                AnalystAgent              69 lines  prompt+tools
architect.py              ArchitectAgent            94 lines  prompt+tools
artifact_summarizer.py    ArtifactSummarizerAgent   61 lines  (no overrides)
assistant.py              AssistantAgent            69 lines  prompt+tools
coder.py                  CoderAgent               118 lines  prompt+tools
deal_analyst.py           DealAnalystAgent         139 lines  prompt+tools
executor.py               ExecutorAgent             75 lines  prompt+tools
fixer.py                  FixerAgent                73 lines  prompt+tools
grader.py                 GraderAgent               90 lines  (no overrides)
implementer.py            ImplementerAgent          82 lines  prompt+tools
planner.py                PlannerAgent              98 lines  prompt+tools
product_researcher.py     ProductResearcherAgent    61 lines  prompt+tools
researcher.py             ResearcherAgent           71 lines  prompt+tools
reviewer.py               ReviewerAgent             82 lines  prompt+tools
shopping_advisor.py       ShoppingAdvisorAgent     163 lines  prompt+tools
shopping_clarifier.py     ShoppingClarifierAgent    73 lines  prompt only
summarizer.py             SummarizerAgent           58 lines  prompt+tools
test_generator.py         TestGeneratorAgent        73 lines  prompt+tools
visual_reviewer.py        VisualReviewerAgent       60 lines  prompt+tools
writer.py                 WriterAgent              153 lines  prompt+tools
```

~1700 LOC of preset config disguised as classes.

---

## What's actually in `src/agents/base.py`

`BaseAgent` is ~5800 LOC. Mix of:

- ReAct iteration loop (the multi-call orchestration)
- Tool execution (calling tool functions, handling results)
- Streaming partial content tracking
- Mid-iteration cost tracking
- exclude_models accumulation on retry
- Response shape parsing (multiple formats)
- Quality checks (degenerate/repetitive output via Doğru mu Samet)
- Output validation against task schemas
- structured_emit / constrained-decoding pass
- self-reflection on failure
- alt-prompt retry
- Cost accounting
- Partial content recovery on cancel
- Memory/context loading helpers
- Tool schema building (allowed_tools → litellm function-call format)
- Workflow envelope unwrapping

Audit needed: not all of this is "ReAct" — some is duplicated with
dispatcher's retry loop, some is per-call behavior that should live
elsewhere, some is dead code.

---

## Proposed Kill Direction

**Agents become a profile registry.**

```python
# src/core/profiles.py
@dataclass
class Profile:
    system_prompt: str
    allowed_tools: list[str]
    iteration_cap: int
    min_tier: str
    prefer_local: bool
    prefer_speed: bool
    prefer_quality: bool
    needs_thinking: bool
    needs_function_calling: bool
    # ... whatever the existing agents expose

PROFILES: dict[str, Profile] = {
    "coder": Profile(
        system_prompt="…",
        allowed_tools=["read_file", "write_file", "shell", …],
        iteration_cap=15,
        min_tier="medium",
        prefer_local=True,
    ),
    # … 20 entries, ported from the deleted subclass files
}
```

**ReAct lives in dispatcher** (or a sibling module dispatcher imports).
HaLLederiz Kadir stays single-call transport. Dispatcher becomes the
multi-call iterator when profile says so:

```python
# src/core/llm_dispatcher.py
async def dispatch(task: BeckmanTask) -> TaskOutcome:
    profile = PROFILES[task.profile]  # was task.agent_type
    if profile.iteration_cap == 1:
        return await _single_call(task, profile)
    return await _react_loop(task, profile)
```

Tool execution moves to `src/core/tool_executor.py` (called from inside
`_react_loop`). Tool registry stays in `src/tools/` unchanged.

**What dies:**
- `src/agents/base.py` (~5800 LOC)
- All 20 `src/agents/*.py` subclass files
- `src/agents/__init__.py:get_agent()` registry
- Orchestrator's `get_agent(agent_type).execute(task)` branch
  (`src/core/orchestrator.py:283`)
- The whole BaseAgent class hierarchy

**What survives** (rehomed, not deleted):
- ReAct iteration logic → `src/core/react.py` (or inlined into dispatcher)
- Tool execution → `src/core/tool_executor.py`
- Tool definitions → `src/tools/` (unchanged)
- `agent_type` column on tasks → renamed `profile` (or kept as
  informational alias for backward compat)

---

## Investigation tasks for next session

Before writing any code, the next session should answer:

1. **What in BaseAgent is genuinely needed vs. accidentally accumulated?**
   Read `src/agents/base.py` in full. Categorize every block as one of:
   - ReAct iteration (move to react.py)
   - Single-call retry / mid-attempt swap (already lives in dispatcher;
     remove the dup)
   - Tool exec (move to tool_executor.py)
   - Pre-call config (model selection knobs, exclude_models, min_ctx) —
     much of this is already in profile metadata; reconcile
   - Post-call accounting (cost, latency, model_stats) — already
     covered by dispatcher / nerd_herd; reconcile
   - Quality checks (Doğru mu Samet pass) — keep, move to react.py
   - Streaming partial content — move to react.py
   - Memory/context helpers — move to react.py or a new context module
   - Workflow envelope unwrapping — move to workflow_engine
   - Dead code — delete

2. **Profile field surface.** Open every subclass and harvest:
   - `name`, `description`
   - `default_tier`, `min_tier`
   - `allowed_tools`
   - `max_iterations` (if set)
   - `get_system_prompt()` body
   - Any other class attribute. Build the canonical Profile dataclass
   from the union.

3. **`grader` and `artifact_summarizer` have no `get_system_prompt`
   override.** Where does their prompt come from? If hardcoded in
   `BaseAgent.get_system_prompt()` defaults, port out. If dynamic
   (computed from task fields), profile gets a callable hook.

4. **Tool execution callsite audit.** Inside BaseAgent there are calls
   that execute tools. Trace one full tool execution end-to-end. What
   does the tool executor need from the agent context (file paths,
   workspace root, vector_store handle, …)? Define a clean interface.

5. **Workflow engine touchpoints.** i2p_v3 and shopping pipeline reference
   `agent_type` strings in their step definitions. Confirm a profile-name
   substitution works without renaming every workflow JSON. Likely
   yes — `agent_type` becomes the profile lookup key.

6. **Test sweep.** `grep -r "from src.agents" tests/` — every importer
   needs migration. Some tests construct agents directly.

7. **Mission planning + classifiers** that decide `agent_type` for new
   tasks (`src/core/task_classifier.py`, mission planner, workflow
   expander) — confirm they emit profile names compatible with the
   registry.

8. **Decide on `_react_loop` location.** New module `src/core/react.py`
   keeps dispatcher thin. Inlined into dispatcher keeps fewer modules.
   Either is defensible; pick one and stick with it.

---

## Migration order (rough)

Each step shippable.

1. Profile registry created in `src/core/profiles.py`. All 20 agents
   ported to entries. Tests verify lookup parity vs current `BaseAgent`
   subclass attribute reads.
2. Tool executor extracted to `src/core/tool_executor.py`. BaseAgent
   delegates to it. No behavior change.
3. ReAct loop extracted to `src/core/react.py`. BaseAgent delegates.
   Verify identical execution traces on a captured workload.
4. Dispatcher gains profile-based dispatch path. Initially behind a
   feature flag. Routes to `react.run()` for profile tasks, to single-
   call for raw_dispatch.
5. Orchestrator branch updated: profile-based tasks bypass `get_agent`,
   go straight to `dispatcher.dispatch`. Side-by-side with old path.
6. Telemetry comparison: run flag-on for a day, verify no regression.
7. Old path deleted. BaseAgent + 20 subclasses removed.
8. `agent_type` column renamed `profile` (optional cosmetic step,
   could stay as legacy name).

---

## Risks / pitfalls

- **5800 LOC of BaseAgent** — easy to miss subtle behavior. Capture
  agent traces from production for a day BEFORE any extraction; replay
  against the new pipeline. Diff.
- **Profile field drift** — every place that reads `agent.something`
  outside BaseAgent breaks. `grep -rn "\.allowed_tools\|\.min_tier\|\.default_tier\|\.max_iterations" src/ packages/` first.
- **Streaming + cancel paths** — partial content recovery on cancel is
  in BaseAgent. Don't lose it during extraction.
- **Doğru mu Samet quality check timing** — currently fires
  per-iteration in BaseAgent. Must fire from react.py or dispatcher,
  same timing.
- **Cost/latency accounting** — dispatcher and BaseAgent both touch
  `record_model_call`. Pick one, delete the other call.
- **Workflow expander** writes `agent_type` into task rows. If profiles
  rename anything, expander needs update.
- **Mission planner** likewise.
- **Backward-compat shims** — for rollback safety, keep `get_agent()`
  as an alias that builds a stub returning `dispatcher.dispatch()` for
  one cycle, then delete.

---

## DON'T

- Don't try to fold ReAct into HaLLederiz Kadir. HaLLederiz is single-
  call transport; iterating + tool exec there breaks its contract.
- Don't skip the audit. BaseAgent is too large to refactor blindly.
- Don't break Phase 1+2+5 of singular admission migration. They shipped
  on 2026-05-04 (commits `89ea7a8` through `e2c0f4c`); kill-agents
  builds on top of them.
- Don't restart agents from scratch as worker classes. The whole point
  is they're config, not behavior.
- Don't introduce a "second profile registry" alongside an existing
  config — find what's already there (`src/agents/__init__.py:get_agent`,
  task classifier's allowed-types list, workflow step definitions) and
  unify.
- Don't `pytest` without `timeout` prefix.

---

## Singular admission migration: where things stand

Shipped 2026-05-04 (commits in order):

| SHA | Phase | What |
|-----|-------|------|
| `89ea7a8` | 1.1 | feat(db): tasks.kind column |
| `82a0e3c` | 1.2 | feat(beckman): enqueue widened (kind/parent_id/await_inline/on_complete/next_task_spec) |
| `190d9f5` | 1.3 | feat(beckman): on_complete + next_task_spec + inline-waiter terminal hook |
| `89445b8` | 2.4 | refactor(dispatcher): request() becomes alias over beckman.enqueue(await_inline=True) |
| `3d73992` | 5.14 | refactor(monitoring): cron-seeded monitoring_check, alerts via notify_user sub-tasks |
| `7030841` | 5.15 | refactor(memory): cron-seeded vector_maint via salako, fixes event-loop wedge |
| `34e6d61` | 2.5 | refactor(beckman): admission owns pool_pressure + fatih_hoca.select + KDV.pre_call + in_flight |
| `f72013e` | 2.5-fixup | test(migration): use setattr DB_PATH not setenv for isolation |
| `e2c0f4c` | 2.5-fixup | fix(in_flight): est_tokens propagation through type-conversion boundaries |
| `d84cd06` | docs | docs(beckman): singular admission spec + plan |

**Phase 3 was originally "agent checkpointable state."** This handoff
**replaces it.** After investigating, agent suspend/resume isn't the
right architectural move. Killing agents is.

**Phase 4 (callsite migrations) and Phase 6 (residue cleanup)** from the
original plan still stand — they're orthogonal to the kill direction.
After agents die, Phase 4 sites (telegram chat, classifier, hooks,
shopping pipeline, vision tool, grader, summarizer, structured_emit,
self-reflection, alt-prompt retry) get re-enumerated against the new
profile-based pipeline. Most of them are non-agent calls that survive
the kill unchanged.

---

## Open questions (next session decides)

1. Profile registry as Python dict (in-process) or YAML (hot-reloadable)?
   Existing `models.yaml` precedent suggests YAML, but type safety
   gets harder. Dict in `src/core/profiles.py` is simpler.
2. `_react_loop` in dispatcher vs new `src/core/react.py`? Probably
   new module — keeps dispatcher's surface tight.
3. Should `raw_dispatch` (single-call, no profile) and profile-based
   (multi-call ReAct) share entry point or be different functions?
   Cleaner: dispatcher.dispatch() always takes a profile arg; profiles
   with `iteration_cap=1` are "raw dispatch" naturally.
4. Workflow JSON references — keep `agent_type` field name or rename
   to `profile`? Cosmetic. Backward-compat suggests keep.
5. Tool executor — module-level functions or a class with config?
   Module-level functions match salako's executor pattern.
6. Migration safety net: dual-write (old agent path + new dispatcher
   path) for one cycle to compare outputs? Worth the complexity?

---

## Files involved

- `src/agents/base.py` — the elephant. ~5800 LOC. Audit first.
- `src/agents/__init__.py` — get_agent registry. Dies.
- `src/agents/*.py` (20 files) — all die after porting.
- `src/core/llm_dispatcher.py` — gains react/profile dispatch path.
- `src/core/orchestrator.py:283` — `get_agent(agent_type).execute(task)`
  branch. Re-routes to dispatcher.
- `src/core/profiles.py` — new, registry.
- `src/core/react.py` — new, ReAct loop.
- `src/core/tool_executor.py` — new, tool exec.
- `src/tools/` — unchanged.
- `src/core/task_classifier.py` — emits profile names.
- `src/workflows/i2p/i2p_v3.json` and shopping workflows — reference
  agent_type strings; verify continued compatibility.
- `tests/agents/` — extensive sweep.
- `packages/fatih_hoca/` — selection consumes `agent_type` field;
  works unchanged if profile name == agent_type.

---

## Saved to memory

- Memory entry `project_kill_agents_20260504.md` to be added by the
  next session as work begins. For now this handoff is the source.
