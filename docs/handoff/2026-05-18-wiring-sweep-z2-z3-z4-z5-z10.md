# Handoff — Wiring sweep of Z2 / Z3 / Z4 / Z5 / Z10

**Date:** 2026-05-18
**Sibling of:** `docs/handoff/2026-05-17-wiring-audit-z0-handoff.md` (which covered Z7 +
yalayut + Z8/Z9 known orphans). This sweep covers the five zones never audited.
**Method:** five parallel read-only agents, one per zone. Each enumerated every shipped
surface (post-hook kind, mr_roboto verb/executor, cron, recipe, tool, Telegram command,
i2p step) and traced for a real production trigger. See the 2026-05-17 handoff §1 for the
method definition.

**Result:** dead wiring is worse than the audit estimated. In every one of the five
zones a *headline feature* is non-functional in production despite green unit tests.
Five are one-to-two-line fixes. The pattern is consistent: `auto_wire_triggers=[]` with
no i2p step, a registered verb with no caller, a founder dial that never reaches the
expander, or an attribute read against the wrong field name.

---

## Priority table — all five zones

| Pri | Zone | Feature dead | Root cause | Fix size |
|---|---|---|---|---|
| **P0** | Z10 | `/mission_cost` always returns a stub | duplicate `cmd_mission_cost` def at `telegram_bot.py:9699` shadows the working one (3159); imports non-existent `src/app/mission_cost.py` | delete dup |
| **P1** | Z4 | Entire founder visual-review loop (album, approve buttons, calibration) never reaches Telegram | `apply.py:3751` `getattr(a,"result",None)` — `PostHookVerdict` has no `result` field, only `raw` → payload always `{}` | 1 line |
| **P1** | Z5 | Entire mobile submit chain (all 9 `14.8.*` steps) excluded for every mission | `conditions.py:69` `platforms_include()` reads `data.get("platforms",[])`; `platform_requirements` artifact has `target_platform`, no `platforms` field → group always False | 3 group conditions in i2p_v3.json |
| **P1** | Z10 | Phase-G trust-calibration loop is a no-op (`/calibration` always empty, prompt injection never fires) | `tasks.confidence_categorical/numeric` never written by any production code → `record_confidence_claim` always returns None → `confidence_outcomes` stays empty | add one write |
| **P1** | Z2 | Cross-mission learning — the READ side — never runs | `inject_lessons` is wired by the expander but absent from `POST_HOOK_REGISTRY`; `determine_posthooks()` filters out any kind not in the registry | 2 lines |
| **P2** | Z3 | Multi-file expansion + `integration_review` + `integration_replay` all dead | `expand_steps_with_multifile` is never called — `runner.py:445` / `hooks.py:2167` call `expand_steps_to_tasks` instead | wire + cascade |
| **P2** | Z3 | `/density` founder dials ignored | `_auto_wire_posthooks` called with no `dial_ctx` (`expander.py:391`) → always `qa_dial="standard"` | thread dial_ctx |
| **P2** | Z5 | `mobile_smoke` / Maestro always soft-skips | step 14.8 `produces` has no `*.flow.yaml`; core recipes ship no smoke flows | recipe + context |
| **P2** | Z10 | `require_confirmation` never auto-armed for irreversible verbs | flag is opt-in; only `/rollback_mission` sets it | expander auto-arm |
| **P3** | Z3 | Layer-aware tooling latent | `build_reflection_prompt` never passed `layer=`; no agent has `inspect_layer`; `run_semgrep_layer_filtered` has no trigger | wire 3 links |
| **P3** | Z2/Z4/Z5/Z10 | misc (below) | — | small |

Cross-zone bonus bug (surfaced by the Z5 agent): **Z8 ops recipes do not load.**
`list_recipes()` expects a two-level `<name>/<version>/recipe.yaml` layout. The Z8
recipes (`backup_verify_sqlite_v1/recipe.yaml`, `cve_scan_*`, `cost_monitor_*`, etc.) sit
one level deep → `list_recipes()` skips them → never discoverable. All 8 mobile recipes
use the correct two-level layout. Add to the Z8 follow-up.

---

## Z2 — build foundation

**P1 — `inject_lessons` orphaned (cross-mission learning READ side dead).**
The expander correctly prepends `inject_lessons` to the first phase-0 task's `post_hooks`
(`expander.py:713`), and the coulson consumer renders a "Watch out for" block from
`lessons_top_n`. But `inject_lessons` is **not in `POST_HOOK_REGISTRY`**
(`posthooks.py`), and `determine_posthooks()` drops any kind not in the registry. The
verb never fires. The DLQ/posthook-fail *writers* work — only the read-back is dead, so
every mission starts with zero recalled lessons.
Fix: (1) add an `inject_lessons` `PostHookSpec` (`auto_wire_triggers=[]`); (2) add the
`apply.py::_posthook_agent_and_payload` dispatch branch.

**P3 — `_apply_hint_from_targets` no-ops on fresh missions.** It returns early when the
workspace dir doesn't exist, and on a fresh mission the workspace isn't created until a
step runs — but expansion happens at task-creation time. So the `write_file`-strip pass
never fires on a new mission; only on re-expansion. Either move it to per-step dispatch
or document it as re-expansion-only.

Correctly wired (do not re-flag): `test_run`, `imports_check`, `pattern_lint`,
`migration_apply`, `openapi_sync`, `typescript_sync`, `design_system_check`,
`emit_dlq_lessons` cron, `mission_lessons` posthook-fail populator, `recipe_picks`
consumer, i2p steps `8.0a`/`8.0b`, `STACK_BLOCKS`, recipe library (instantiable).

---

## Z3 — build review density

**P2 — multi-file expansion is dead, and it cascades.** `expand_steps_with_multifile`
(`expander.py:1024`) has no production caller — `runner.py:445` and `hooks.py:2167` call
`expand_steps_to_tasks` instead. Consequences:
- `integration_review` post-hook (`auto_wire_triggers=[]`, injected only by the multifile
  expander) never fires; the `integration_reviewer` agent is never invoked.
- `integration_replay` + `integration_bisect` likewise never fire.
- The dead path also has a latent `TypeError`: `expander.py:1050` calls
  `to_mission_dial_context(mission_id, raw_dials)` (2 args) but the function takes 1
  (`review_density.py:172`).
- Separately: `integration_reviewer.allowed_tools` lists `ast_signatures`, which is
  **not in `TOOL_REGISTRY`** — any call fails "unknown tool". The real extractor is
  `mr_roboto.extract_signatures`.
Fix: call `expand_steps_with_multifile` from `runner.py`/`hooks.py`; fix the
`to_mission_dial_context` arity; register `ast_signatures` (or correct `allowed_tools`).

**P2 — `/density` dials never reach the expander.** `_auto_wire_posthooks` is called
from `expand_steps_to_tasks:391` with no `dial_ctx`, so it always falls back to
`qa_dial="standard"`. `security_review` / `adr_drift_check` auto-wire on steps
7.4/7.6/7.10/7.11 regardless of the founder's `/density` setting; a founder who set
`qa_dial=quick` still gets them. `accessibility_review` is correctly off-by-default but
turning it *on* via `/density` also has no effect. `contract_review` triggers on
`**/routes/*.py` — no i2p step produces such a path, so it never fires at all.
Fix: thread the resolved `MissionDialContext` into `_auto_wire_posthooks`.

**P3 — layer-aware tooling latent (two broken links).** `inspect_layer` is in
`TOOL_REGISTRY` but no agent has it in `allowed_tools`. `LAYER_BLOCKS` is defined but
`react.py:980` calls `build_reflection_prompt(...)` without the `layer=` argument.
`run_semgrep_layer_filtered` + `forbidden_in_domain.yml` have no trigger path.

Correctly wired: `/density` command itself, `self_critique` sub-iter guard,
`performance_review` (correctly opt-in). `MULTI_FILE_RULES` covers only one stack
(`fastapi+nextjs`); the LLM fallback for other stacks is a documented TODO.

---

## Z4 — visual review

**P1 — the founder visual-review loop never reaches Telegram.** `apply.py:3751` does
`_vr_result = getattr(a, "result", None) or {}`. `a` is a `PostHookVerdict`
(`result_router.py:87`) whose fields are `source_task_id / kind / passed / raw` — there
is **no `result` field**. So `_vr_result` is always `{}`, `captured_paths` is always
`[]`, and `enqueue_visual_review_notice` hits its `if not captured_paths: return` guard
every time. The WebP album, per-breakpoint approve buttons, and calibration buttons are
never sent. The `visrev:` Telegram callback handler is correct but unreachable.
Fix: `apply.py:3751` → `_vr_result = a.raw or {}`.

**P3 — `test_visual_review_in_simple_blocker_verdict_kinds` fails on a brittle verifier.**
The test scans 6 lines backward from each `_apply_simple_blocker_verdict` call for
`visual_review`; the real routing tuple is 18 lines away (`apply.py:3708` vs `3730`).
Routing is correct; widen the window or grep the tuple directly.

**P3 — per-component selector crop scaffolded but unreachable via posthook.**
`capture_screenshots` accepts `components` and crops, but `_posthook_agent_and_payload`
for `visual_review` never propagates `produces.components` into the verb payload. Works
only when `capture_screenshots` is an explicit step.

Correctly wired: `visual_review` posthook auto-wire on `*.tsx/.jsx/.vue/.svelte`;
`capture_screenshots` self-capture inside `visual_review`; `tokens_changed` (warning-only
by design); `capture_mode="device"` (explicit-step by design). Note: `capture_screenshots`
silently soft-skips when host Playwright is missing — no pre-mission warning.

---

## Z5 — mobile track

**P1 — the entire mobile submit chain is excluded for every mission.** The
`mobile_app_submission` / `ios_submission` / `android_submission` conditional groups use
`platforms_include('ios') OR platforms_include('android')`. `conditions.py:69`
`platforms_include()` reads `data.get("platforms", [])`, but the `platform_requirements`
artifact (step 3.6) has no `platforms` field — it has `target_platform`. The condition
is always False, so all nine `14.8.*` steps (gen_ci, export_web, preview, screenshots,
submit, submit_play, review_status×2) are skipped for *every* mission including real
mobile ones.
Fix: change the three group conditions to `target_platform in ('mobile','both')`, or add
a derived `platforms` field to the artifact schema + step 3.6 instruction.

**P2 — `mobile_smoke` / Maestro always soft-skips.** Step 14.8 has
`post_hooks:["mobile_smoke"]`, but its `produces` contains no `*.flow.yaml` and
`context.maestro_flows` is unset → `flow_paths=[]` → `maestro_run` returns
"no flow paths supplied" (soft pass) every time. The 3 core recipes ship no smoke flows;
`mobile_offline_sync` doesn't either (only `mobile_push` / `mobile_deep_links` do).
Fix: ship smoke flows in the core recipes and thread them into step 14.8 context.

**P3 — `mobile_release_rejection` recipe fails its own posthook.** Its `recipe.yaml`
declares `post_hooks:[imports_check, test_run]` but ships no `tests/` dir → after
instantiation `test_run` runs pytest with zero tests → exit 5 + `total==0` → fail.
Fix: drop `test_run` (it is a markdown playbook recipe) — `mobile_ci` correctly omits it.

**P4 — `no_mobile_app` skip_when never auto-populated.** Nine steps declare
`skip_when:["no_mobile_app"]` but nothing reads `platform_requirements.target_platform`
to add `no_mobile_app` to `active_conditions` (`runner.py:411`). Passive gate. Lower
priority once P1's conditional groups work, but matters for non-i2p callers.

Intentional non-wiring (confirmed against the doc, do NOT "fix"): `android_build`,
`eas_build`, `eas_submit` verbs — the free-first / GitHub-Actions founder decision means
Android/EAS builds run inside the generated CI YAML, not as orchestrator-host verbs.
Correctly wired: `expo_cli`, `gen_mobile_ci`, `fastlane`, recipe discovery +
RECIPE_PARAM substitution, `frontend_platform` group, `target_platform` (step 3.6),
`capture_mode="device"` (gated only by the P1 bug).

---

## Z10 — cross-cutting

**P0 — `/mission_cost` always returns a stub.** `cmd_mission_cost` is defined twice in
`telegram_bot.py` (3159 and 9699); Python keeps the last. The 9699 version imports
`src.app.mission_cost`, which does not exist → silent failure → returns
`"...T2A (cost accounting) not merged yet. Stub response."` The working formatter
(`src.infra.cost_wiring.format_mission_cost`) is used by the shadowed 3159 definition.
Fix: delete the 9699 definition + its duplicate `add_handler` (1973).

**P1 — Phase-G trust-calibration loop is a no-op.** `_record_and_resolve_confidence`
(`apply.py:48`) is correctly called from six grade/grounding/reviewer sites, but it reads
`tasks.confidence_categorical` / `confidence_numeric` — columns **no production code ever
writes**. So `record_confidence_claim` always returns `None`, `confidence_outcomes` stays
empty, `recompute_reliability_scores()` yields zero rows, `/calibration` always shows
"no data", and the prompt-builder calibration injection never fires. The `min_confidence`
gate logic in `react.py:991` is real and does block — but only on the in-response
`confidence` field; it never persists it.
Fix: persist `confidence` to the `tasks` row (in `react.py` after the gate, or in
`apply.py` at task finalize).

**P2 — `require_confirmation` never auto-armed for irreversible verbs.** The
confirmation flow (request → `mission_event_drain` posts to Telegram → reaction →
`resolve_confirmation`) is wired correctly, but no workflow JSON sets
`require_confirmation:true`; only `/rollback_mission` does. Irreversible i2p verbs
(`mission_deliverable_bundle`, `notify_user`, `emit_preview_url`,
`init_mission_github_repo`) all run unconfirmed.
Fix: auto-arm `require_confirmation` for `irreversible`-tagged verbs in the expander/
dispatch when an idle-confirm policy is set.

**P3 — confirmation poll loop is a skeleton.** `_await_confirmation` busy-polls 0.5s ×
120 (60s) and holds the task slot; a slow founder makes the task `fail`. Meant to be
replaced with event-driven unblocking.

**P3 — egress allowlist is caller-opt-in, not enforced.** `shell.py:1034` only checks
when `request_egress_to` is passed; arbitrary shell commands bypass it. No iptables-level
enforcement (acknowledged in-code as v1 best-effort).

**P3 — per-mission Chroma namespace migration is dry-run by default.** Old flat-namespace
data needs `scripts/migrate_chroma_to_per_mission.py --apply`; not automatic.

Correctly wired: `mission_pacing_check` / `confidence_calibration_recompute` /
`mission_event_drain` crons (seeded + dispatched); `/calibration`; `/mission` pacing
block; per-mission Docker container lifecycle (`add_mission` → `_maybe_complete_mission`);
i2p steps `15.10b/15.10c/15.14b` (demo gate blocks `15.14`). The Z9 `model_pick_log`
`call_category='reinforce'` write path IS wired (`record_verdict.py`) — it is a separate
loop from Z10's `confidence_outcomes`; see the 2026-05-17 handoff §3c on unifying them.

---

## Suggested order

1. **The five one-liners first** — P0 Z10 `/mission_cost`, P1 Z4 `a.raw`, P1 Z2
   `inject_lessons` registry, P1 Z5 conditional-group condition, P1 Z10 confidence
   write. Each unblocks a headline feature; each is hours not days. Add a host-path test
   per fix (the unit tests passed *with* these bugs — that is the whole problem).
2. **Z3 multi-file expansion** — wire `expand_steps_with_multifile`, which cascades to
   `integration_review` + `integration_replay`; fix the `to_mission_dial_context` arity
   and `ast_signatures` registration in the same pass.
3. **Dial threading** (Z3 `/density`) and **`require_confirmation` auto-arm** (Z10).
4. **P2/P3 remainder** + the Z8 recipe-layout bug.
5. Only then a real prototype mission run — it will still surface what static grep
   missed, but with far less noise.

Every fix here is "connect existing correct code", not new features. The scaffolding is
built; it was just never plugged in.
