# Kill-Agents → Data-Driven Profiles Handoff
*2026-06-09. Tree-wide modularization track #6 (`docs/2026-05-31-root-debt-map.md` §D). Continues the kill-agents work (`project_kill_agents_20260504`). Goal: collapse `src/agents/` (29 pure-config classes, 2,792 LOC) into a data registry behind one generic Profile, so adding/editing an agent is a data edit, not a new Python class.*

## Why
`src/agents/` is 30 files / 2,792 LOC. **Every concrete agent is pure config** — class attributes + a static `get_system_prompt` string. Zero custom methods, zero `execute` overrides (verified 2026-06-09: `git grep "    (async )?def " src/agents/*.py` outside base.py returns ONLY `get_system_prompt`). The class machinery buys nothing; it's 29 boilerplate files where a YAML table + one loader would do. This is the "package/abstraction declared, no real polymorphism" smell — same family as the core/models debt P1–P7 just closed.

## Hard constraints (DO NOT violate)
- **KEEP-or-DROP, never MODE-merge.** `feedback_no_agent_modes`: do NOT consolidate agents into one mega-agent toggled by a `mode`/`phase`/`artifact_type` flag. Each profile stays a distinct registry entry — the change is class→data, NOT 29→1. The registry still has 29 keys after.
- **Singleton identity must survive.** `AGENT_REGISTRY` instantiates each agent ONCE at import; `get_agent(type)` returns that cached instance. The runtime mutates per-execution instance attrs (`_original_allowed_tools`, `progress_callback`, `_suppress_clarification`, `allowed_tools`) and restores them in `execute()`'s `finally`. Tests + live code rely on `get_agent(x)` returning the SAME object across calls (e.g. the P5 self-reflection bridge — `tests/core/test_orchestrator_self_reflect_bridge.py` — patches `real_profile.execute` and needs the bridge to resolve that same instance). A data-driven `get_agent` MUST still hand back stable per-type singletons, not fresh objects per call.
- **Prompt source-of-truth is the DB, not the string.** At `execute()` time the runtime loads the live prompt from the `prompt_versions` table via `get_active_prompt(name)` (`coulson/__init__.py::_load_db_prompt_override`); the in-code `get_system_prompt` is a FROZEN REFERENCE / seed only (see the big block comment in `base.py:100-125`). So moving prompts to YAML is mostly a seed-source change — but the seeding path (`/prompt seed <agent>`, `save_prompt_version`) and the prompt-quality test both read the code string today and must be re-pointed at the YAML.

## Current state (verified 2026-06-09)
- 29 agents registered in `src/agents/__init__.py::AGENT_REGISTRY`; `get_agent()` falls back to `executor`.
- `base.py` (187 LOC) is the Profile interface: attribute defaults (`name`, `description`, `default_tier`, `min_tier`, `allowed_tools`, `max_iterations`, `can_create_subtasks`, `execution_pattern`, `enable_self_reflection`, `min_confidence`, `confidence_gate`) + `get_system_prompt` + `_build_context` (delegates to `src/runtime/context`) + `execute` (delegates to `src/runtime`/coulson). The 4092-LOC original was gutted by the runtime extraction (Phase A); base.py docstring says `_build_model_requirements`/`_maybe_constrained_emit` still pending move (A.12/A.13) — **this track intersects that; coordinate.**
- **No external module imports a concrete `*Agent` class** (verified: `git grep "import .*Agent" src/ packages/` outside `src/agents/*` + tests = empty). Everything routes through `get_agent`. Tests DO import concrete classes — they'll need updating.

### The 2 carve-outs (NOT pure static prompts)
| Agent | Dynamic bit | Handling |
|-------|-------------|----------|
| `oncall_agent.py` | `domain = (task.get("context") or {}).get("domain","ops")` interpolated into prompt | either keep as a code subclass, OR move the domain lookup into the runtime context layer and template the YAML |
| `writer.py` | branches prompt on `_detect_markdown_schema(task)` (module-level helper) | same — keep as code subclass OR push the schema-detect into runtime + two YAML variants |

Recommendation: **hybrid** — convert the 27 static agents to data; leave `oncall_agent` + `writer` as thin code subclasses of the generic Profile that override `get_system_prompt`. Don't build a templating engine into YAML for 2 cases (over-engineering; violates the "simple" rule).

## Coupled surfaces that must stay in sync
1. **3 prompt-quality invariants** — `tests/agents/test_prompt_quality.py` asserts each prompt: first line `You are …`, body has must/always + don't/never, body has `final_answer` + fenced ` ```json ` schema. The YAML prompts must still satisfy this; **retarget the test to load from the data registry.**
2. **Per-agent self-reflection blocks** — `packages/coulson/src/coulson/reflection.py::REFLECTION_BLOCKS`. Separate from the agent class; keyed by agent name. The `enable_self_reflection` bool (agent attr) gates them via the P5 bridge (`src/core/dispatch_prep.py`). Preserve both the flags (in YAML) and the coulson blocks (untouched).
3. **Message classifier ↔ registry keys** — `src/agents/signal_classifier.py` + the i2p workflow (`src/workflows/i2p/i2p_v3.json`) reference `agent_type` strings. They must match registry keys exactly. A typo in YAML keys silently routes to the `executor` fallback (the get_agent default) — add a test asserting classifier/workflow agent_types ⊆ registry keys.

## Proposed design
- `src/agents/profiles.yaml` (or `profiles/<name>.yaml` per agent — one file each reads better for diffs and matches the per-step i2p style). Fields = the Profile attribute surface + `system_prompt`.
- One generic `Profile` class (rename/keep `BaseAgent`) that reads a dict and exposes the duck-typed surface the runtime already consumes (`name`, `allowed_tools`, `max_iterations`, `execution_pattern`, `get_system_prompt(task)`, `enable_self_reflection`, …). `get_system_prompt` returns `self._system_prompt` (static) for the 27.
- `get_agent` builds the registry once from YAML at import into per-type singletons (same lifecycle as today). Keep `AGENT_REGISTRY` name + `get_agent` signature for back-compat.
- `oncall_agent` + `writer`: thin subclasses overriding `get_system_prompt`, registered the same way.

## Phased plan (low risk → high)
1. **Scaffold loader + 1 agent.** Add `Profile` (data ctor) + `profiles/` dir; migrate ONE static agent (e.g. `summarizer`) to YAML; wire `get_agent` to serve it from data while the other 28 stay classes. Prove singleton identity + execute path + 3-invariant test on the migrated one. Land to main (canonical-first).
2. **Bulk-migrate the 27 static agents** to YAML. Delete their `.py` files. Keep `oncall_agent` + `writer` as code subclasses. Update `__init__.py` to build the registry from YAML + the 2 subclasses.
3. **Retarget tests:** `test_prompt_quality.py` reads YAML; replace any `from src.agents.X import XAgent` test imports with `get_agent("x")`; add the classifier/workflow-keys-⊆-registry test.
4. **Re-point the prompt seed path** (`/prompt seed`, `save_prompt_version` seeding) at the YAML string instead of the code string. Confirm DB-override at runtime still wins (unchanged).
5. **Guardrail:** extend `tests/test_root_stays_thin.py` — assert `src/agents/*.py` count stays small (only base/Profile + the 2 carve-outs + `__init__`), so new agents land as data, not classes.

## Verification
- `.venv/Scripts/python -c "from src.agents import get_agent; print(get_agent('coder').name, get_agent('coder') is get_agent('coder'))"` → `coder True` (singleton identity).
- `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q` (prompt-quality + any agent tests).
- Live: 1 multi-step mission touching coder/reviewer/researcher; confirm prompts resolve + self-reflection still fires for coder.
- **DO NOT** mix `tests/` and `packages/` in one pytest call (conftest plugin collision). Use `.venv/Scripts/python`; always `timeout`.

## Open decisions for founder
- **One file or per-agent files?** Per-agent (`profiles/coder.yaml`) = cleaner diffs, matches i2p; single `profiles.yaml` = one-glance overview. (Lean per-agent.)
- **YAML vs Python data dict?** YAML keeps it non-code (the goal). But the 2 carve-outs already need Python — a `profiles.py` dict + 2 subclasses is also defensible and avoids a YAML parse dep at import. Pick one.
- **Fold this into finishing `base.py` → coulson (A.12/A.13)?** The generic Profile is the natural home to also retire base.py's residual method bodies. Could be one combined effort or sequential.

## Risk notes
- Hot-path: `get_agent` is called on every dispatch. A YAML parse at import is fine (once); never parse per-call.
- Live-bot `git add -A` storm on `main` — do this in a **git worktree**, merge in a quiet window (see `project_modularization_p5_p6_p7_20260608`).
- This is medium-risk: touches every dispatch's profile resolution + the prompt seed path + 3 coupled test/classifier surfaces. Canonical-first (Phase 1 lands before bulk) is mandatory.
