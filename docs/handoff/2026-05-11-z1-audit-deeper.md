# Z1 Audit — Deeper Pass — 2026-05-11

Goes beyond `2026-05-11-z1-audit.md`. That audit was surface-level. This one traces every Z1 post-hook through the materialization → dispatch chain. Net finding: **all 8 mechanical post-hooks shipped across Tier 4-7 are dead in runtime** — silently filtered at registry gate.

## Root cause

Post-hook execution chain:
1. Step JSON's `post_hooks: [<kind>, ...]` → expander stores in `context["post_hooks"]` (`src/workflows/engine/expander.py:271-273`)
2. Beckman's `determine_posthooks(task, ctx)` reads that list (`packages/general_beckman/src/general_beckman/posthooks.py:276`)
3. **Each kind is filtered against `POST_HOOK_REGISTRY`** (`posthooks.py:281`): `if k in POST_HOOK_REGISTRY and k not in kinds: kinds.append(k)`
4. Surviving kinds spawn post-hook tasks via `_posthook_agent_and_payload(a, source, ctx)` (`apply.py:984`)

The registry only contains 10 kinds: `verify_artifacts`, `code_review`, `grounding`, `imports_check`, `test_run`, `pattern_lint`, `design_system_check`, `openapi_sync`, `typescript_sync`, `migration_apply`.

**None of the Z1-shipped post-hook kinds are in the registry.** They're silently dropped at step 3.

## Dead post-hooks shipped this session

| Tier | Kind | Declared at (JSON) | Status |
|---|---|---|---|
| T5A | `compliance_template_present` | step 1.11a | **DEAD** — registry miss |
| T5A | `compliance_blocker_check` | step 6.6 | **DEAD** — registry miss |
| T5B | `verify_premortem_shape` | step 6.5z | **DEAD** — registry miss |
| T4B | `annotate_html_oids` | (declared as own step 5.30c, not as post-hook — fine) | OK |
| T6A | `find_similar_missions` | step 0.1 | **DEAD** — registry miss |
| T6A | `surface_prior_mission_hints` | step 0.5 | **DEAD** — registry miss |
| T6A | `index_idea_fingerprint` | step 0.1 (added this session) | **DEAD** — registry miss + no caller |
| T6B | `prior_art_min_coverage` | step 1.0 | **DEAD** — registry miss |

Items shipped as standalone mechanical STEPS (executor=mechanical at the step level, not as post-hooks) DO work:
- `regen_artifact`, `regen_bundle` (T4A)
- `propagate_asset_change`, `propose_spec_patch_from_html_diff`, `annotate_html_oids` (T4B)
- `emit_preview_url`, `kill_preview_url` (T4C, step 5.40)
- `attention_check`, `attention_debit` (T5A — wired into `clarify.py`, not step-based)
- `spec_consistency_check` (T5B steps 7.0z..12.0z)
- `index_mission_artifacts` (T6A step 6.7z)
- `init_mission_github_repo` (T6C step 6.7)
- `verify_against_paraflow_goldens` (T7B — `/paraflow_check` only)
- `run_bash_audit` (T7A — Beckman cron, translation verified OK at `apply.py:34`)

And inline post-hooks INSIDE handlers DO work:
- `critic_gate` runs inline inside `git_commit` and `notify_user` handlers (`mr_roboto/__init__.py`)

## Fix shape

Two parts:

### Part A — register Z1 mechanical post-hook kinds

Add to `POST_HOOK_REGISTRY` (`posthooks.py:91`):

```python
"compliance_template_present": PostHookSpec(
    kind="compliance_template_present",
    verb="compliance_template_present",
    default_severity="blocker",
    auto_wire_triggers=[],
    description="Z1 T5A — assert referenced compliance templates exist on disk."),
"compliance_blocker_check": PostHookSpec(
    kind="compliance_blocker_check",
    verb="compliance_blocker_check",
    default_severity="blocker",
    auto_wire_triggers=[],
    description="Z1 T5A — phase-boundary check that required compliance docs were rendered."),
"verify_premortem_shape": PostHookSpec(
    kind="verify_premortem_shape",
    verb="verify_premortem_shape",
    default_severity="blocker",
    auto_wire_triggers=[],
    description="Z1 T5B — assert premortem.md has 3+ scenarios with plausibility + kind."),
"find_similar_missions": PostHookSpec(
    kind="find_similar_missions",
    verb="find_similar_missions",
    default_severity="warning",  # needs_review surfaces to founder; not a blocker
    auto_wire_triggers=[],
    description="Z1 T6A — cross-mission idea dedup. needs_review surfaces to founder."),
"surface_prior_mission_hints": PostHookSpec(
    kind="surface_prior_mission_hints",
    verb="surface_prior_mission_hints",
    default_severity="warning",  # advisory only
    auto_wire_triggers=[],
    description="Z1 T6A — advisory hints from prior missions. Always completes."),
"index_idea_fingerprint": PostHookSpec(
    kind="index_idea_fingerprint",
    verb="index_idea_fingerprint",
    default_severity="warning",
    auto_wire_triggers=[],
    description="Z1 T6A — embed idea_brief into mission_ideas ChromaDB collection after charter lock."),
"prior_art_min_coverage": PostHookSpec(
    kind="prior_art_min_coverage",
    verb="prior_art_min_coverage",
    default_severity="blocker",
    auto_wire_triggers=[],
    description="Z1 T6B — prior_art_report has attempted_solutions, key_lessons, resolvable URLs."),
```

### Part B — extend `_posthook_agent_and_payload`

In `packages/general_beckman/src/general_beckman/apply.py:984`, add a generic branch BEFORE the final `raise`/return:

```python
# Z1 mechanical post-hooks — kind == action name. Source ctx flows through
# as payload; mr_roboto handlers read what they need (mission_id, produces,
# etc.) from the standard fields.
Z1_MECHANICAL_KINDS = {
    "compliance_template_present", "compliance_blocker_check",
    "verify_premortem_shape", "find_similar_missions",
    "surface_prior_mission_hints", "index_idea_fingerprint",
    "prior_art_min_coverage",
}
if a.kind in Z1_MECHANICAL_KINDS:
    produces = list(source_ctx.get("produces") or [])
    return ("mechanical", {
        "source_task_id": a.source_task_id,
        "posthook_kind": a.kind,
        "executor": "mechanical",
        "payload": {
            "action": a.kind,
            "mission_id": source.get("mission_id"),
            "workspace_path": source_ctx.get("workspace_path"),
            "produces": produces,
            # Kind-specific extras (idempotent — each handler picks what it needs):
            "report_path": produces[0] if produces else None,
            "idea_summary": source_ctx.get("idea_summary"),
            "title": source_ctx.get("title"),
        },
    })
```

The mr_roboto handlers in `mr_roboto/__init__.py` already accept these payload shapes (verified — they read `payload.get("mission_id")`, `payload.get("workspace_path")`, etc. with defaults). Idempotent: each handler ignores fields it doesn't need.

### Part C — tests

Per-kind unit test that:
1. Spawns a fake source task with the kind in `post_hooks`
2. Runs `determine_posthooks(task, ctx)`
3. Asserts the kind survives the registry filter
4. Runs `_posthook_agent_and_payload(a, source, ctx)`
5. Asserts the returned payload has the right `action` name and `mission_id` propagated

## Why audit missed this

Audit grep'd `i2p_v3.json` for the action name + grep'd `mr_roboto/__init__.py` for the dispatch branch, then declared the wiring real. It did NOT trace the materialization chain. The phrase "Verified-real" in the prior audit needs that asterisk: "the code exists and could be reached from `/<command>` but the JSON `post_hooks` mechanism does not actually fire it."

The user warning `feedback_zone_deep_dive.md` ("every zone needs v2+ passes; v1 surface 'isolate + recommend' is never enough") applies here. This was a v1 audit.

## Recommended next session

1. Apply Part A + B + C (single subagent dispatch with concrete file:line targets — should be 1-2 hours since the pattern is mechanical)
2. Re-run full pytest including `tests/i2p/` + `packages/general_beckman/tests/`
3. After merge, do a v2 audit of Tier 3 (`check_against_non_goals`, `verify_charter_shape`, etc.) — those may have the same gap since they were shipped under the same assumption that JSON post_hooks "just work"
4. Add a structural test: `test_every_step_post_hook_kind_is_registered.py` that asserts every `post_hooks` entry in `i2p_v3.json` exists in `POST_HOOK_REGISTRY`. This would have caught the entire class of bug
5. Verify Tier 3 mechanical post-hooks (`check_against_non_goals`, etc.) actually fire — same scrutiny

## Other items confirmed dead (from first audit, still relevant)

- `_edit_html_upload` document handler — never wired (T4B)
- `streaming_guard_log` writer — table created but zero INSERT (T5C)
- Root `conftest.py` line 19 sets `KUTAI_CRITIC_GATE=off` setdefault — risk if non-pytest path imports it (low probability but real)
- `legacy_pre_design_tokens` + `legacy_pre_user_flow` columns — no skip_when consumers

## Items confirmed false-alarm

- `run_bash_audit` cron `_executor` key — Beckman's `apply._mechanical_context` translates to `action` at `apply.py:34`. T7A cron WILL fire.

## What was attempted this session

- Added `index_idea_fingerprint` to step 0.1 post_hooks + `legacy_pre_idea_dedup` skip_when (this session). Both DEAD without Part A+B+C above.

## Honest status

Z1's mechanical post-hook layer is structurally broken across 7 of 8 declared kinds. The features WORK if you invoke their mechanical action directly (via Telegram command or via Beckman `enqueue` with the right payload), but the workflow-step-declared post_hooks do not fire. Reviewer/grader hooks (Tier 0-3 baseline `grade`, `verify_artifacts`, `grounding`) DO fire because they're in the registry. Only the Z1-added kinds are dead.

The user's prompt — "no unwired fragments, no shallow scaffolds" — exposed this. The prior audit + the shipping process both failed to catch a structural registry gate. The fix is mechanical (~50 LOC) but needs to land before any of these post-hooks can be considered "wired".
