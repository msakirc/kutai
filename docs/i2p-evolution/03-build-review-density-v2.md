# Z3 v2 — Build review density (corrected audit + alt shapes + wiring + migration)

**Supersedes** v1 surface ([03-build-review-density.md](03-build-review-density.md)) per
the "every zone needs v2+" feedback. Inherits frame + open questions;
corrects stale claims; expands each recommendation with alt shapes,
wiring details, severity policy, and phased migration; converts the
seven unbatched areas into a 5-tier plan modelled on Z1/Z2 completion
pattern.

## Frame (unchanged)

Replace single-pass build with real iteration. Replace single-file
feature emission with per-file decomposition + cross-file consistency
review. Add domain-specific QA modalities (a11y, security, perf,
contract) on the same `blockers: {field, levels}` framework Z2 shipped.
Architectural drift gate + layer-aware tooling + pre-merge integration
replay close the long tail.

Visual review remains split to [04-build-visual-review.md](04-build-visual-review.md).

## Stale claims in v1 (fix here)

| v1 claim | Reality | Source |
|---|---|---|
| Doc header reads `# Z2 — Build review density` | Misnumbered. This is Z3 — Z2 is `02-build-foundation.md`. | filename `03-` |
| "code_review post-hook runs once per build step" implies it auto-wires | Auto-wire DISABLED: `auto_wire_triggers=[]`. Must opt-in per step. Currently wired only on 7.9/7.10/7.11 (frontend components). | `packages/general_beckman/src/general_beckman/posthooks.py:105-115`, `src/workflows/i2p/i2p_v3.json` |
| "Existing severity-blockers gate (10.1 owasp_audit, 10.5 encryption_logging_review) is the template" — true but partial | Gate structure is `blockers: {field, levels}` with `levels: ["critical","high"]`. Apply-side evaluation in `apply.py` writes `blocker_count` into context; verdict handling per-kind. | `i2p_v3.json` steps 10.1, 10.5; `general_beckman/apply.py:~1743` |
| "Sub-iter guards (Z1 G grounding) — relevant?" | YES, already structured for this. `MAX_SUB_CORRECTIONS=3` in `coulson/guards.py:46`. Grounding + format + hallucination guards dispatch via `check_sub_iter_guards`. Self-critique is a 4th guard slot, not a new loop. | `packages/coulson/src/coulson/guards.py:46`, `react.py` |
| "feat.4 backend_service today: one call emits the model + the service" | Confirmed. Single-file/single-call. No sub-task expansion exists. | `i2p_v3.json` feat.* steps |
| Implicit "code_reviewer agent reviewed by other graders" | NO — `_NO_POSTHOOKS_AGENT_TYPES` excludes code_reviewer from grading. Its verdict IS the gate. New review agents must follow the same exemption rule. | `general_beckman/posthooks.py:37-38` |
| Z3 lists 4 QA modalities (a11y/security/perf/contract) as parallel work | Three of them (a11y, security, contract) collapse onto the Z2 T3C `run_semgrep` shape + per-language CLI wrappers. Only perf needs a genuinely new sandbox (browser or k6). | Z2 v2 §5 design_system_check |
| "ADRs needed for drift gate" treated as TBD | ADRs already produced + schema-verified. Step 4.4 emits `mission_{mission_id}/.adr/database_schema_decision.json` + `register.md`; `mr_roboto/verify_adr_shape.py` enforces 7-field schema. | `i2p_v3.json:4.4`, `mr_roboto/verify_adr_shape.py:50+` |
| "Inbound deps from 02 hard pre-req" | Yes — but specifically: `POST_HOOK_REGISTRY` (10 kinds), shared `run_semgrep`, `regen_and_diff`, severity-blockers eval, expander auto-wire. All shipped. | `posthooks.py:91-233` |

**Net consequence**: foundation is broader than v1 read. The post-hook
registry + auto-wire + shared semgrep + severity-blockers + ADR schema
all exist. Z3 is **(a)** new review *kinds* using existing engines,
**(b)** the multi-file expander surgery, **(c)** a self-critique
sub-iter guard (not a new outer loop), **(d)** one new agent type
(`integration_reviewer`), **(e)** two genuinely new sandboxes
(browser-driven a11y, perf harness).

## Seven recommendations — alt shapes + pick + wire

Pattern: a) name b) trigger c) alt shapes d) **PICK** e) wiring f) severity.

### 1. Self-critique sub-iter guard (NOT a new outer loop)

- **Trigger:** any step with `agent ∈ {coder, implementer, fixer, test_generator}` and `produces` non-empty.
- **Alt shapes:**
  - **A.** New outer loop (build → review → fix → re-review) in expander. Adds workflow nodes; doubles JSON size; founders see retry storms in Telegram.
  - **B.** New sub-iter guard inside coulson `check_sub_iter_guards` — agent gets `build_self_critique_message()` after a successful `final_answer` but before commit. One extra LLM call inside same iteration. Reuses `MAX_SUB_CORRECTIONS` budget.
  - **C.** Post-hook `self_critique` running between `verify_artifacts` and `code_review` — would invoke the **same** agent type out-of-process; loses in-memory tool context.
- **PICK:** **B**. Already-paved code-path. The grounding guard shape from Z1 is exactly the template — `coulson/grounding.py` is a pure-function module; mirror as `coulson/self_critique.py`. Bonus: B preserves the LLM's working context (tools+history) which post-hook reload cannot.
- **Wire:**
  - New module `packages/coulson/src/coulson/self_critique.py` with `build_self_critique_message(diff_summary, produces)` + `check_self_critique_sub_iter(state) -> SubIterVerdict`.
  - Register in `guards.py::check_sub_iter_guards` dispatch list, between grounding and format guards.
  - Add `SELF_CRITIQUE_OPT_OUT_AGENT_TYPES = frozenset({"code_reviewer", "researcher", "decider"})` — non-emitting roles skip.
  - Budget knob: `MAX_SELF_CRITIQUE_PASSES: int = 1` (separate from `MAX_SUB_CORRECTIONS`; "review your own diff once" is the value, repeated re-prompting just inflates cost without finding new bugs — verified pattern from anthropic/superpowers commentary).
- **Severity:** critic verdict shape: `{verdict: "clean"|"issues", findings: [{severity, file, why}]}`. `severity ∈ {blocker, warning}`. Blocker findings → sub-iter retry with critique appended to next prompt. Warning findings → logged to `mission_lessons` (T4 from Z2), do not block.

### 2. Multi-file feature expansion

- **Trigger:** step `template_id ∈ {backend_service, frontend_component, full_stack_feature}` AND mission has `multi_file_expansion: true` in spec (founder dial; off by default v1).
- **Alt shapes:**
  - **A.** LLM-inferred per-feature: agent reads openapi paths + model + acceptance crits, emits sub-task list. Highest fidelity; non-deterministic; founder loses preview unless we add a confirmation gate.
  - **B.** Static per-stack/per-template inference rules in expander: `backend_service` → `{model, schema, service, repository, error_mapper, fixtures, tests}`; `frontend_component` → `{component, hook, story, test}`. Deterministic; misses stack-specific shapes (Django: model+admin+serializer+viewset+urls).
  - **C.** Hybrid: static rules per `(template_id, tech_stack_detected)`; LLM fallback when stack+template combo missing from table. Founder sees inferred list before dispatch (existing `clarify` mechanical kind from mr_roboto is the confirmation rail).
- **PICK:** **C**. Resolves v1 open question ("static vs LLM") with the same dial we used for stack picking. Static table covers 90%; LLM fallback covers tail. Founder preview via `clarify` is free — that verb already exists.
- **Wire:**
  - New module `src/workflows/engine/multifile.py` with `expand_template(template_id, stack, artifacts) -> list[SubTaskSpec]`.
  - Static table `MULTI_FILE_RULES: dict[tuple[str, str], list[str]]` keyed on (template_id, stack_slug).
  - Expander hook in `src/workflows/engine/expander.py` runs **before** post-hook auto-wire: parent step replaced with parent + N child steps; parent retained as orchestration marker that holds the integration_review post-hook.
  - Each sub-task gets its own `produces` slot (single file) + inherits parent's `code_review` post-hook + auto-wired guardrails per file pattern.
  - Founder confirmation: new mechanical step `propose_subtask_list` inserted before expansion when `mission.review_density_dial == "strict"`; emits the list via `clarify`; on accept, expander proceeds. On `quick` dial, no confirmation.
- **Severity:** sub-task failure isolates to that sub-task's retry budget (existing Beckman logic). Parent step waits on all children + final `integration_review`.

### 3. `integration_review` post-hook (new kind + new agent type)

- **Trigger:** end of any multi-file-expanded parent step (auto-wired by the expander hook above).
- **Alt shapes:**
  - **A.** Use existing `code_reviewer` over concatenated diff — context window blows up on real features; can't render cross-file relationships cleanly.
  - **B.** New agent type `integration_reviewer` with prompt focused on signature alignment, type match, error-branch test coverage, naming consistency, migration↔model alignment. Reads all sub-task `produces` files together.
  - **C.** Pure mechanical: AST-diff signatures across files, fail on mismatch. Cheap but misses semantic drift ("you exposed `get_user_by_email` from the service but the route handler still calls `get_user_by_id`").
- **PICK:** **B + C as belt-and-suspenders**. AST mechanical pre-check (cheap, deterministic) feeds findings as context into the LLM reviewer. Mirror of grounding's mechanical-first → LLM-judge pattern.
- **Wire:**
  - Agent: `src/agents/integration_reviewer.py` — config-only per `feedback_no_agent_modes` (sys_prompt + tools `{read_file, file_tree, ast_signatures}`, zero methods).
  - Add to `_NO_POSTHOOKS_AGENT_TYPES` (its verdict IS the gate).
  - Add to `REFLECTION_BLOCKS` so it gets the reviewer self-reflection.
  - Mechanical pre-check verb: `extract_signatures(files) -> dict[file, list[Signature]]` in mr_roboto; uses tree-sitter for TS, ast for Python. Signature mismatches passed as context into the LLM reviewer prompt.
  - Post-hook registry entry: `"integration_review"` with verb `"integration_reviewer"`, `default_severity="blocker"`, `auto_wire_triggers=[]` (parent-step injection from multifile.py only).
- **Severity:** `blockers: {field: "severity", levels: ["critical", "high"]}`. On fail: failing sub-task IDs returned; retry budget per child resets to N=1 with cross-file context appended.

### 4. QA modalities (collapsed onto existing rails)

Reframe v1's 4-modality table around what's *new* vs *recombine existing*:

| Kind | Engine | New work? | Severity |
|---|---|---|---|
| `security_review` | shared `run_semgrep` + `security.yml` rule pack (OWASP top10, secret patterns); per-stack: `bandit` for Python, `npm audit --json` for JS — all 3 already shell-out shape | NO new engine. New rule pack + 2 thin verb wrappers (`run_bandit`, `run_npm_audit`). | CVSS≥7 OR semgrep `error`-severity → blocker |
| `accessibility_review` | `@axe-core/cli` against built preview URL (tunneled-preview infra from Z1 T4C is the host) | New mechanical verb `run_axe(url)`; depends on tunneled preview being live | violations with `impact ∈ {critical, serious}` → blocker |
| `contract_review` | `schemathesis` against running app (uses openapi.json already produced) | New verb `run_schemathesis(spec_path, base_url)` | any `5xx` or schema mismatch → blocker |
| `performance_review` | `lighthouse` (web) OR `k6` (api) — declared per-feature in spec block | New verb `run_lighthouse(url, thresholds)` OR `run_k6(script, thresholds)` | threshold breach → blocker; soft-perf warning level founder-tunable |

- **PICK rationale:** collapses 4 superficially-distinct items onto **one new pattern** (mechanical shellout verb + JSON parse + severity map) repeated 4×. Each verb is ~40 LOC.
- **Wire:**
  - 4 new kinds in `POST_HOOK_REGISTRY`.
  - Auto-wire triggers:
    - `security_review`: any step with `produces` matching `*.py|*.ts|*.tsx|*.js|requirements.txt|package.json`.
    - `accessibility_review`: any step under `phase_8` (core implementation) emitting `*.tsx` AND mission has `accessibility_dial != "off"`.
    - `contract_review`: any step that produces a route file under `routes/`.
    - `performance_review`: opt-in only via explicit `post_hooks: ["performance_review"]` (cost is real; don't auto-fire).
  - All 4 use the shared `blockers: {field, levels}` rule. Severity field name mapped per-tool in verb output normalization.
- **Severity policy:** defer thresholds to founder dial — keep `mission.qa_dial ∈ {"quick", "standard", "strict"}` knob; strict promotes `serious` → blocker, quick demotes `serious` → warning.

### 5. `adr_drift_check` post-hook

- **Trigger:** any step in phase_8 (core implementation) with `produces` matching code patterns AND mission has any ADRs in `.adr/`.
- **Alt shapes:**
  - **A.** Pure LLM: feed ADR text + diff to `code_reviewer` with drift-focused prompt. Highest recall; noisy false-positives without grounding.
  - **B.** Pure mechanical greps: ADRs declare forbidden imports/patterns in a structured `falsification_signal` field (already in schema!) → grep against produced files. Cheap, false-negative on semantic drift.
  - **C.** Mechanical-first → LLM judge on the gray zone. Greps catch hard violations; LLM ruled in only when grep ambiguous.
- **PICK:** **C**, BUT also: schema enhancement — extend `verify_adr_shape` to require `falsification_signal` follow a structured shape (`{forbidden_imports: [], forbidden_patterns: [regex], required_test_coverage: bool}`) so the mechanical pre-check has data to work with. Schema bump → `_schema_version: "2"`.
- **Wire:**
  - Verb `check_adr_drift(adr_register_path, produced_files) -> {clean|drift, findings}` in mr_roboto.
  - Post-hook kind `"adr_drift_check"`, verb=`check_adr_drift`, default_severity=`blocker`.
  - Auto-wire trigger: `phase_8.*` + `produces` non-empty + `.adr/register.md` exists.
- **Severity:** structured violations → blocker. Gray-zone LLM-flagged → warning (don't block on judgment calls; surface for review).

### 6. Layer-aware tooling

- **Trigger:** read-time enrichment for any code-emitting agent; no new post-hook.
- **Alt shapes:**
  - **A.** Founder declares layer→path mapping per project in spec; tool `inspect_layer(path)` does pure lookup. Founder labor; deterministic.
  - **B.** Heuristic-only (path contains `domain/` → domain, `infra/` → infra, etc.). Zero config; brittle.
  - **C.** Heuristic + spec-override. Heuristic supplies default; spec block overrides per glob.
- **PICK:** **C**. Founder pays for divergence, not for the common case. Spec block lives in pre-code phase artifact (`tech_stack_detected.json` extension or new `layer_map.json` artifact in phase_4).
- **Wire:**
  - New tool `inspect_layer(path) -> Layer` in `src/tools/` (Layer = `domain|adapter|infra|test|ui|unknown`).
  - Mission-context bundler reads `layer_map.json` if present; falls back to heuristic table.
  - Coulson reflection: extend `REFLECTION_BLOCKS` with per-layer blocks. `domain` layer → "no framework imports allowed"; `adapter` layer → "translate between domain types and infra types only"; etc.
  - Apply in `build_reflection_prompt` similar to the stack-blocks plug-point from Z2 T4C.
- **Severity:** N/A (informational). But: `pattern_lint` rule pack gains layer-aware rules — `forbidden_in_domain.yml` enforces no-framework rule via semgrep with layer constraint, runs only on `inspect_layer == "domain"` files.

### 7. Pre-merge integration replay

- **Trigger:** end of any feature parent step (after `integration_review` passes).
- **Alt shapes:**
  - **A.** Rerun full integration suite against current commit + last N feature commits in random shuffle. Catches order-dependent regressions. Cost: N× test runtime.
  - **B.** Mutation-style: rerun **only** tests touched by current feature against last N commits. Cheap but misses cross-feature interactions which is the whole point.
  - **C.** Hybrid: full suite at feature boundary (one-time), spot-check (touched tests) at sub-task boundaries.
- **PICK:** **C** with founder dial. Quick mode: spot-check only. Standard: full at feature boundary. Strict: full + bisect on fail.
- **Wire:**
  - Verb `integration_replay(commits, suite_glob, shuffle_seed, mode) -> ReplayVerdict` in mr_roboto.
  - Post-hook kind `"integration_replay"`, default_severity=`blocker`.
  - Auto-wire: parent feature step (the one holding `integration_review`) gains `integration_replay` as a sibling post-hook running after `integration_review` passes.
  - Shuffle seed: `mission_id` (deterministic per mission; recorded in audit). Closes v1 open question.
  - Bisect on fail: when standard mode reports red, strict mode auto-bisects via git-bisect-style binary search on the N commits; emits a `lesson` row with the offending commit pair.
- **Severity:** any red in replay → blocker. Bisect identifies the breaking pair → retry the **later** sub-task with the **earlier** failing test as feedback (mirror the Beckman retry-with-feedback pattern).

## Cost transparency (resolves cross-link to Z10)

Each new kind declares a `cost_band ∈ {cheap, moderate, heavy}` in
registry metadata. Heavy kinds (perf, integration_replay, multi-file
expansion fan-out) contribute to the **per-feature retry cost meter**
already wired in Z10 cross-cutting. When meter exceeds founder dial,
mission emits Telegram nudge "feature retried Nx; continue?".

## Founder dials

One dial per dial-shaped concern, all in `mission.review_density`:

```yaml
review_density:
  qa_dial: standard       # quick | standard | strict
  accessibility_dial: on  # on | off
  multi_file_expansion: false   # true | false
  integration_replay: standard  # quick | standard | strict
```

Defaults are conservative (no fan-out, no a11y, standard everything
else) — Z3 ships latent and founders opt-in per project.

## Tier batches (Z1-style)

### T1 — Foundation (no new behavior; preps registry + multifile rails)
- **T1A**: `POST_HOOK_REGISTRY` extended with `cost_band` metadata field + dial-aware auto-wire logic (`auto_wire_triggers` becomes a callable that takes mission dials).
- **T1B**: `src/workflows/engine/multifile.py` scaffold + `MULTI_FILE_RULES` table seeded with `backend_service` and `frontend_component` for `fastapi+nextjs` stack only.
- **T1C**: Founder-dial schema in mission config + `mission.review_density` resolver + telegram dial-set command.

### T2 — Self-critique + integration_reviewer (cheap, biggest leverage)
- **T2A**: `coulson/self_critique.py` module + sub-iter guard wired; `MAX_SELF_CRITIQUE_PASSES=1`; opt-out frozenset.
- **T2B**: `integration_reviewer` agent type (config-only) + `_NO_POSTHOOKS_AGENT_TYPES` add + reflection block.
- **T2C**: `extract_signatures` verb in mr_roboto (tree-sitter TS + ast Python); `integration_review` post-hook kind wired; expander wires it as parent's terminal post-hook when multi_file expansion fires.

### T3 — QA modalities (4 new kinds, shared shape)
- **T3A**: `security_review` — semgrep `security.yml` rule pack + `run_bandit` + `run_npm_audit` verbs; kind wired with auto-wire on code produces.
- **T3B**: `accessibility_review` — `run_axe` verb against tunneled preview; kind wired phase_8 + tsx + dial=on.
- **T3C**: `contract_review` + `performance_review` — `run_schemathesis` + `run_lighthouse`/`run_k6` verbs; contract auto-wires on routes/, perf opt-in only.

### T4 — Drift + layer
- **T4A**: ADR schema v2 (structured `falsification_signal`); `verify_adr_shape` updated; ADR-emitting steps (4.4 today; more in 01-pre-code follow-ups) migrate.
- **T4B**: `check_adr_drift` verb + `adr_drift_check` post-hook + auto-wire on phase_8 produces.
- **T4C**: `inspect_layer` tool + `layer_map.json` artifact + REFLECTION_BLOCKS layer extension + `forbidden_in_domain.yml` semgrep pack.

### T5 — Integration replay (biggest cost; ship last)
- **T5A**: `integration_replay` verb (sandbox checkout + suite run + verdict).
- **T5B**: `integration_replay` post-hook kind + auto-wire as sibling-of-integration-review on parent feature steps.
- **T5C**: bisect helper + lesson emission on bisect-identified breaks.

## Migration / back-compat notes

- **i2p_v3.json**: additive. All new post-hooks auto-wire only when triggers + dials match; nothing existing breaks.
- **ADR schema v2**: bumps `_schema_version` to "2"; `verify_adr_shape.py` accepts both v1 and v2 during a 1-mission window; phase_4 steps re-emit on next mission.
- **code_reviewer carve-out**: extend `_NO_POSTHOOKS_AGENT_TYPES` with `integration_reviewer` in T2B. Test: ensure `general_beckman/posthooks.py` test suite covers the new exemption.
- **Cost meter integration**: T1A's `cost_band` field is the contract with Z10 cross-cutting; coordinate with whoever owns the meter so heavy bands feed the right counter.
- **Tunneled preview dependency for T3B**: Z1 T4C shipped emit-only tunneled preview. T3B needs *consumable* preview URL. If the preview pattern is not yet read-back-friendly, T3B blocks on a small Z1 follow-up (write a `last_preview_url.txt` artifact at preview emit).
- **Founder-dial back-compat**: missions started before Z3 ships have no `review_density` block; resolver defaults to conservative settings (multi_file=off, accessibility=off, qa=standard).

## Resolved open questions from v1

- **Multi-file inference rules.** Hybrid: static per `(template_id, stack_slug)` table; LLM fallback for missing combos. Founder preview via `clarify` on strict dial only.
- **Iteration cost ceiling.** Self-critique = 1 pass (not 5). Build-review-fix outer loops *not introduced*; instead sub-iter guards + per-kind retry budgets do the work. Bounded naturally.
- **Pair-programming model selection.** Drop pair-programming as a separate concept — self-critique sub-iter + integration_reviewer (different agent) achieves the diversity goal more cheaply. No two-agent ping-pong.
- **Integration-replay determinism.** Mission_id as shuffle seed; recorded in audit.
- **ADR drift granularity.** Mechanical-first (structured `falsification_signal`) + LLM judge on gray zone.

## New open questions surfaced by v2

- **Self-critique prompt shape.** Generic "review your own diff for omissions/bugs" vs role-conditioned ("you wrote a coder diff — check for off-by-one, error handling, side effects")? Try generic v1, role-conditioned v2 if findings rate underwhelming.
- **`falsification_signal` adoption.** Some ADRs (DB schema choice) have natural structured signals; design choices (REST vs GraphQL) don't. Allow `falsification_signal: null` with mechanical-skip + LLM-only path.
- **Tunneled preview readiness.** T3B blocks until preview URL is consumable. Audit Z1 T4C result before T3B kickoff.
- **Cost-band feedback loop.** Once meter is live, do strict-dial features measurably retry more? If not, the dial is theater. Telemetry from `model_pick_log` + new `posthook_run_log` table.

## Agent task brief

When picking up this doc:
1. Read Z2 v2 ([02-build-foundation-v2.md](02-build-foundation-v2.md)) — guardrail registry + auto-wire patterns are the substrate.
2. Verify all Z2 tiers are merged (T1-T5 per memory `project_z2_t3_shipped`; T4-T5 pending — coordinate sequencing).
3. Pick a tier; partition by sub-tier (A/B/C) for parallel dispatch (Z1 merge pattern applies).
4. Each sub-tier ships independently: new verb + new kind + auto-wire rule + tests + JSON wiring example.
5. Validate AST + JSON + tests before merge.
6. Add `## Updates` entry.

## Cross-references

- **Inbound:** [02-build-foundation-v2.md](02-build-foundation-v2.md) (hard pre-req: registry + semgrep + severity-blockers + auto-wire). [01-pre-code.md](01-pre-code.md) (ADRs produced + schema-verified).
- **Outbound:** [04-build-visual-review.md](04-build-visual-review.md) (another modality on the same shape — visual review = `visual_review` kind + browser sandbox; design from Z3 should aim to absorb cleanly). [09-growth.md](09-growth.md) (analytics hypotheses iterate fast only if review density holds quality).
- **Cross:** [10-cross-cutting-plan.md](10-cross-cutting-plan.md) (cost meter consumes `cost_band` from T1A; Z10 telemetry surfaces "feature retried 8x — continue?" prompts).

## Updates

- 2026-05-08 — v1 initial (Wave 3+5, themes T4+T6+T9).
- 2026-05-11 — v2 written. Re-audit confirms registry + semgrep + severity-blockers + ADR schema all shipped in Z2. Z3 collapses to: self-critique sub-iter guard (not outer loop), multi-file expander surgery, integration_reviewer agent, 4 QA modality kinds (collapsed onto shared shellout+semgrep shape), ADR drift gate (mechanical+LLM), layer-aware tooling, integration replay. Pair-programming dropped in favor of self-critique + integration_reviewer. 5 tier batches.
- 2026-05-12 — **T1 SHIPPED**. T1A (cost_band field on all 18 registry kinds + MissionDialContext dataclass + polymorphic auto_wire_triggers callable form + resolve_triggers helper + expander dial_ctx threading), T1B (`src/workflows/engine/multifile.py` scaffold: SubTaskSpec dataclass + MULTI_FILE_RULES seeded for `(backend_service, fastapi+nextjs)` and `(frontend_component, fastapi+nextjs)` + FILE_ROLE_TO_PATH + pure `expand_template` function), T1C (`missions.review_density_json` column migration + `src/workflows/review_density.py` resolver with ReviewDensityDials dataclass + get_dials/set_dial + to_mission_dial_context bridge + `/density` Telegram command). 3 parallel agents on isolated worktrees, sequential merge per Z1 pattern. One merge conflict on `posthooks.py` (both A and C added MissionDialContext; kept C's plan-aligned vocab, dropped A's). 58 new tests; 336 total passing (3 skipped, 0 failed). Commits 76c0761 (T1A), b334b87 (T1B), 553dfc3 (T1C), 04e7059 (dedup fixup).
- 2026-05-12 — **T2 SHIPPED**. T2A (`packages/coulson/src/coulson/self_critique.py` — pure module with `MAX_SELF_CRITIQUE_PASSES=1`, opt-out frozenset for non-emitting/reviewer roles, `build_self_critique_message()` + `check_self_critique_sub_iter()`; wired into `guards.py` between grounding and format guards; uses dedicated counter, does NOT consume `MAX_SUB_CORRECTIONS` pool), T2B (`src/agents/integration_reviewer.py` config-only agent with cross-file consistency prompt + `read_file`/`file_tree`/`ast_signatures` tools; `_NO_POSTHOOKS_AGENT_TYPES` carve-out; reflection block; `task_classifier.py` REVIEW cluster entry), T2C (`packages/mr_roboto/src/mr_roboto/extract_signatures.py` AST signature extractor with Python + soft-skip TS/JS; `integration_review` post-hook kind in registry; `apply.py` `_enrich_integration_review_payload` async helper injects signatures into reviewer step-context; expander `_maybe_expand_multifile` helper integrates with T1B `expand_template` + T1C dial). 3 parallel agents; merge required canonical T1B/T1C overrides (T2C duplicated prerequisites with incompatible signatures). 121 new tests (25 T2A + 17 T2B + 17 T2C + others); 457 total passing (3 skipped, 0 failed). Commits b334b87→adc6c53 (T2A), 84737bb (T2B), 701a9b4 (T2C), 753e281 (merge), d15a274 (canonical alignment).
- 2026-05-12 — **T3 SHIPPED**. T3A (security.yml rule pack + `run_bandit` + `run_npm_audit` + `security_review` composite verb), T3B (`run_axe` verb + T3B follow-up: emit_preview_url writes `.preview/last_preview_url.txt` so a11y/contract/perf can consume), T3C (`run_schemathesis` + `run_lighthouse` + `run_k6` + `performance_review` composite). POST_HOOK_REGISTRY grew to 23 kinds. New `_dial_get(ctx, key, default)` helper accepts MissionDialContext OR dict OR None — lets callable triggers work from both canonical expander path + ad-hoc test paths. `_apply_simple_blocker_verdict` generic helper in apply.py handles all 4 T3 kinds with shared pass/fail/retry/DLQ semantics. Three parallel agents built incompatible duplicates of T1A primitives — manual integration via `git show worktree-...` + file copy + registry/apply wiring kept canonical T1A. 100 new tests (35 T3A + 32 T3B + 33 T3C); 10 skip-marked tests assert agent-specific non-canonical paths and are kept for intent-tracking. 547 total Z2+Z3 passing (13 skipped, 0 failed). Commit consolidated: integration done locally without per-tier worktree merges (worktree branches diverged too far from main after T2C merge).
- 2026-05-12 — **T4 SHIPPED**. T4A ADR schema v2 (`verify_adr_shape.py` accepts v1 string + v2 object {forbidden_imports, forbidden_patterns, required_test_coverage}; i2p_v3.json step 4.4 bumped; 22 new tests). T4B `adr_drift_check` post-hook (mechanical violation gate; `check_adr_drift.py` parses .adr/ + greps; cost_band=cheap; auto-wires all code globs when qa_dial!=off; 24 new tests). T4C layer-aware tooling (`src/tools/inspect_layer.py` returns domain|adapter|infra|test|ui|unknown via spec-override .spec/layer_map.json + heuristic; LAYER_BLOCKS in coulson reflection; forbidden_in_domain.yml semgrep pack; `run_semgrep_layer_filtered` verb; step 4.3 produces layer_map.json; 49 new tests). POST_HOOK_REGISTRY now 24 kinds. 3 parallel agents; manual integration kept canonical T1A primitives. _dial_get hardened: None values map to default. 684 total Z2+Z3 passing (13 skipped, 0 failed). Commit: feat(z3,t4) at HEAD.
