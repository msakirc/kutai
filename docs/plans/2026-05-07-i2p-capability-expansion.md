# i2p Capability Expansion — Plan

**Date:** 2026-05-07
**Frame:** i2p stays ambitious. Goal is *denser feedback loops* + *more
real capabilities*, not narrower scope. Brainstorm output grouped into
themes, themes ordered into shippable waves with dependencies.

## Premises

- i2p's shape is right; what's missing is feedback density and
  breadth of capability surface. Single-pass build steps + thin
  reviews + zero cross-mission memory are the bottleneck.
- G grounding (shipped 2026-05-06) gives us "did the agent write the
  files" — necessary but not sufficient.
- Most quality wins come from **mechanical** checks (no LLM judgment),
  closing **drift gaps** (spec ↔ code), and **iteration** (multi-pass
  review instead of one-shot).
- Visual review needs real framing/state-setup infra; treat as its
  own subproject, do not block other waves on it.
- Mobile track is large scope; ships last.

## Themes (12 buckets)

### T1 — Mechanical guardrails (no LLM)
Cheap post-hooks, big drift prevention. Each is a mr_roboto verb +
posthook kind.
- Imports vs declared deps (parse imports → diff against
  requirements.txt / package.json → fail or auto-add)
- Design-system enforcement (scan emitted JSX/TSX for raw color hex,
  inline styles, non-design-system imports)
- Forbidden-pattern lint (custom rules: no `console.log`, no
  `time.sleep` in tests, no `assert True`)
- Migration-applies-clean gate (sandbox DB → `alembic upgrade head` →
  assert no errors → teardown)
- Test-running gate (`mr_roboto.run_pytest` / `run_jest` after every
  test-emitting step)
- Cross-type sync check (regenerate frontend types from openapi,
  diff against committed types)

### T2 — Specs as source of truth (closes spec→code drift)
Mechanical contract enforcement.
- OpenAPI generated from FastAPI route decorators (native), not a
  separate artifact that drifts
- Frontend types from `openapi-typescript` (generated, not authored)
- Schema-typed migrations (alembic autogenerate / prisma migrate
  diff against model state)
- Contract fuzz (schemathesis against live staging)

### T3 — Tooling discipline
Force the right tool for the job; dampen rewrite-the-world.
- Diff-shaped edits: route to `patch_file` / `edit_file` /
  `apply_diff` for steps targeting existing files; reserve
  `write_file` for true creation
- Tool-routing layer: per-step `tools_hint` becomes binding, not
  advisory
- Stack-aware system prompts (FastAPI 0.115 idioms vs Django vs
  NestJS), loaded at dispatch based on tech_stack_decision

### T4 — Multi-pass review loops
Turn one-shot into iteration.
- Self-critique pass: same agent reviews own diff with critic prompt
  (1 extra LLM call per step, cheap)
- Pair-programming pattern: drafter + critic, back-and-forth until
  critic clean. Bounded by issue-severity-clean, not iteration count.
- Iterate-until-clean inner loop replaces single-pass build steps

### T5 — Cross-mission memory + recipe library
Biggest multiplier on everything else.
- Per-stack playbook (FastAPI gotchas, NextJS RSC gotchas) injected
  into stack-aware prompts
- Per-domain failure library: "auth-flow recent failures: forgot
  CORS preflight (4/7), session vs token confusion (3/7)"
- Recent-DLQ digest at mission start
- Recipe library: vetted templates for auth, file upload, search,
  pagination, payments, notifications, image resize, soft delete,
  audit log. Each recipe ships with own test suite.
- Planner picks recipe + customizes; doesn't reinvent

### T6 — Multi-file feature expansion
Replaces single-call feat.4-9 with real per-file decomposition.
- feat.4 (backend_service) expands into sub-tasks per inferred file:
  types, validators, service, repository, error mapper, tests,
  fixtures
- Each sub-task gets own produces + grounding + code_review
- Cross-file integration reviewer at end (signature alignment, type
  match, test coverage of error branches)

### T7 — Mid-mission control
Recoverable mistakes + async user collaboration.
- `request_review` action (non-blocking, distinct from
  `needs_clarification`); posts to Telegram with LGTM / inline-comment
  / discuss buttons; comments later become `revision` tasks
- Real revision_policy: agent flags "phase 4 decision wrong because
  X" → revision_policy actually re-runs phase 4 deltas without full
  restart
- State snapshot + `restart_from <phase> --tweak <field>=<value>`
- Bisect-on-break: integration test red → bisect to breaking commit
  → retry that feature with failing test as feedback
- Live mission Telegram thread (persistent thread per mission;
  `[milestone]` / `[blocker]` / `[asking]` posts)

### T8 — Observability
Mid-mission visibility instead of post-mortem-only.
- Live mission dashboard (web): artifact tree, current step, recent
  model picks, retry counts, failed gates
- Decision audit log (each architectural decision +
  reasoning + rejected alternatives + triggering artifact)
- Per-mission cost ceiling (declared at start; approaching →
  surface tradeoff conversation)
- Quick-vs-thorough mode (severity gate dial, not workflow change)

### T9 — QA modality post-hooks
Beyond code_review — domain-specific reviewers.
- Accessibility review (axe-core, severity-blocker gate)
- Security review (bandit / npm audit / semgrep)
- Performance review (lighthouse score threshold)
- Contract review (schemathesis fuzz, see also T2)
- Visual review (LARGE — own subproject, see T11)

### T10 — Sandboxing + reset-to-green
Safety + recovery primitives.
- Per-mission container (Docker / firecracker) with mounted
  workspace; lets us drop safety rails on `shell` without risk
- Reset-to-green: every commit-after-green is a known-good restore
  point; "roll back to last green" becomes a primitive
- End-of-mission deliverable video (final playwright `--video on` →
  30s clip of the app doing the core flow)

### T11 — Visual review (subproject)
Needs real framing/state-setup infra before it's meaningful.
- Browser state primer (login, navigate to feature path, set
  viewport to design breakpoint, wait for content)
- Screenshot at multiple breakpoints (mobile / tablet / desktop)
- Vision-model diff against design_tokens + screen_specifications +
  wireframes
- Returns structured diffs (color drift, layout shift, missing
  component, breakpoint break)
- Blocked on: state-priming infra, vision-model selection, design-
  system reference image library

### T12 — Mobile track
Largest scope. Splits frontend phases into platform-aware variants.
- New conditional group at phase 4: target_platform ∈ {web,
  mobile_native, mobile_cross_platform, both}
- New tech stack options (Expo / RN, Flutter, native iOS+Android)
- Mobile tooling adapters in mr_roboto: expo_cli, ios_simulator,
  android_emulator
- Device-screenshot visual review (depends on T11)
- Mobile e2e: detox / maestro instead of playwright
- Per-platform test/build/sign/distribute steps

## Waves (shippable order with dependencies)

### Wave 1 — Guardrails + Specs-as-SoT (4 weeks)
**Themes: T1 + T2.** Cheap mechanical wins; compound on every later
wave because they catch drift before it propagates.

Ships:
- 6 new mechanical post-hook kinds (imports_check, design_system_check,
  pattern_lint, migration_apply, test_run, type_sync)
- OpenAPI-from-FastAPI as default (i2p_v3 phase 4 selection nudges
  toward frameworks with native openapi support)
- `openapi-typescript` codegen step replacing manual frontend types
- alembic-autogenerate as default migration flow
- Each post-hook auto-wired in expander when relevant produces
  pattern matches (e.g. `**/migrations/*.py` triggers migration_apply)

Dependencies: G grounding (shipped). No others.

Estimate: 6 new mr_roboto verbs + 6 new beckman posthook kinds + i2p_v3
edits + tests. Mechanical in shape — same pattern as verify_artifacts
× 6.

### Wave 2 — Tooling discipline + Memory layer (3 weeks)
**Themes: T3 + T5.** Memory is the biggest multiplier on everything;
ship it early so later waves benefit.

Ships:
- Tool-routing layer: dispatcher inspects step's `tools_hint` +
  whether target file exists; routes to `patch_file` over
  `write_file` when patching existing
- Stack-aware system prompts: prompt fragments per stack, loaded
  based on tech_stack_decision; injected into agent profile at
  dispatch
- Cross-mission memory schema: `mission_lessons` table (mission_id,
  stack, domain, failure_pattern, fix_pattern, severity)
- Recent-failure injector: at mission start, query mission_lessons
  for current stack/domain, inject top-N as "watch out for"
  context
- Recipe library v1: 5 vetted recipes (auth, file_upload, search,
  pagination, audit_log) with test suites; planner gains a
  `pick_recipe` step that names the recipe to use; expander
  instantiates the recipe template

Dependencies: Wave 1 (so recipe templates can declare the new
mechanical post-hooks).

Estimate: ~12 new files (recipe templates), dispatcher edits,
fatih_hoca prompt-fragment loader, db migration for mission_lessons.

### Wave 3 — Multi-pass reviews + Multi-file expansion (4 weeks)
**Themes: T4 + T6.** Restructures the heavy parts of the feature
template. Big quality jump per feature.

Ships:
- Self-critique inner loop in coulson: after final_answer, agent
  re-prompted with critic-shaped message; critic clean → proceed,
  else fix
- Pair-programming pattern: optional second-agent dispatch for
  hard difficulty steps; drafter+critic exchange bounded by
  issue-severity-clean
- Multi-file expansion: feature template's build steps gain a
  `multi_file: true` flag; expander spawns sub-tasks per inferred
  file based on the spec artifacts (openapi paths, model fields,
  story acceptance criteria)
- Integration reviewer: new agent type + post-hook kind that runs
  after all sub-tasks land; reads all files together; emits
  cross-file findings

Dependencies: Wave 1 (cross-type sync), Wave 2 (recipes inform
which files to spawn).

Estimate: largest of the early waves. ~8 new files in coulson +
beckman + new integration_reviewer agent + i2p_v3 template restructure.

### Wave 4 — Mid-mission control + Observability (3 weeks)
**Themes: T7 + T8.** UX + recoverability layer. After Wave 3 missions
will run longer with more iteration; observability + course-correct
become essential.

Ships:
- `request_review` action (non-blocking) + Telegram inline buttons +
  comment-to-revision-task wiring
- Persistent Telegram mission thread (one thread per mission_id;
  posts go through it)
- Real revision_policy implementation (artifact deltas re-applied
  without full restart)
- Bisect-on-break runner (mechanical; runs `git bisect` against
  the integration test)
- Restart-from-phase command (`/restart_mission <id> --from <step>
  --tweak <field>=<value>`)
- Live mission web dashboard (artifact tree + current state + decision
  log + cost gauge)
- Per-mission cost ceiling (declared at start, surfaces tradeoff
  prompt at thresholds)

Dependencies: Waves 1-3 stable enough to be worth observing.

Estimate: dashboard is biggest piece; rest are wiring against
existing Telegram + DB.

### Wave 5 — QA modalities (3 weeks)
**Theme: T9** (excluding visual). Each post-hook is a mr_roboto verb +
beckman kind, severity-aware via existing blockers rule.

Ships:
- accessibility_review (axe-core)
- security_review (bandit + npm audit + semgrep)
- performance_review (lighthouse threshold)
- contract_review (schemathesis fuzz against staging)
- All four use the existing `blockers: {field: severity, levels:
  [critical, high]}` pattern shipped 2026-05-05

Dependencies: Wave 1 mechanical post-hook framework.

Estimate: smaller than Wave 1 because the framework now exists; each
modality is a wrapper around an existing tool.

### Wave 6 — Sandboxing + Demo deliverable (2 weeks)
**Theme: T10.** Safety + recovery + final deliverable.

Ships:
- Per-mission container runtime (Docker compose template; mission
  workspace mounted; mr_roboto shell verbs route through it)
- Reset-to-green primitive (`/rollback_mission <id>` to last green
  commit)
- End-of-mission video step (playwright `--video on`, attached as
  mission deliverable)

Dependencies: Wave 1+ stable. Container infra is mostly ops work.

Estimate: container scaffolding is the bulk; rollback uses git
already; video is a mr_roboto verb wrapping playwright.

### Wave 7 — Visual review subproject (4-6 weeks)
**Theme: T11.** Standalone subproject; can run parallel to other
waves once kicked off.

Ships:
- Browser state primer module (login flows + path navigation +
  viewport setup + content-ready waits)
- Multi-breakpoint screenshot harness
- Design-token reference library (per-mission baseline images
  from sprint 0)
- Vision-model diff (Claude Vision or specialised UI diff model)
- Structured visual-diff output (color/layout/missing/breakpoint)
- visual_review post-hook kind wired into beckman pipeline

Dependencies: Wave 1 (post-hook framework). Vision-model selection +
state-priming infra are open questions to resolve in week 1.

Estimate: dominated by state-priming infra; the diff and post-hook
plumbing are routine.

### Wave 8 — Mobile track (6-8 weeks)
**Theme: T12.** Biggest scope; ships last because depends on visual
review (T11) and observability (T8).

Ships:
- target_platform conditional group at phase 4
- Expo / RN / Flutter recipes added to T5 recipe library
- Mobile tooling adapters in mr_roboto (expo_cli, ios_simulator,
  android_emulator)
- Mobile e2e: detox / maestro adapters
- Mobile visual review: device-screenshot mode
- Per-platform build/test/sign/distribute step variants

Dependencies: Waves 1, 5, 7. Recipes (Wave 2) inform mobile recipe
shape.

Estimate: biggest. Probably split into 8a (cross-platform via Expo)
and 8b (native iOS+Android) sub-waves.

## Cross-cutting

### Memory schema (introduced Wave 2)
```
mission_lessons
  id          INT PK
  mission_id  INT
  stack       TEXT     -- "fastapi+postgres+nextjs", "django+sqlite"
  domain      TEXT     -- "auth", "file_upload", "search", "deploy"
  pattern     TEXT     -- "forgot CORS preflight"
  fix         TEXT     -- "add CORSMiddleware before routes"
  severity    TEXT     -- "blocker", "warning", "info"
  created_at  TIMESTAMP
  occurrences INT      -- denormalized counter
```

Auto-populated from DLQ entries + post-hook fail verdicts.
Queryable at mission start by stack + domain.

### Recipe format (introduced Wave 2)
```
recipes/
  auth/
    recipe.yaml             # metadata, tech_stack compatibility
    spec.md                 # what this recipe is + when to pick
    prompts/                # stack-aware prompt fragments
    backend.template.py     # parametrized scaffold
    frontend.template.tsx
    tests/                  # ships with own test suite
    migrations/
    lessons.md              # known gotchas
```

Planner gains `pick_recipe` action; expander instantiates the
recipe with feature parameters.

### Integration reviewer agent (introduced Wave 3)
- Agent type: `integration_reviewer`
- Post-hook kind: `integration_review`
- Reads ALL files emitted for a feature together
- Prompt focuses on cross-file consistency: signature alignment,
  type match, test coverage of error branches, naming consistency,
  migration-vs-model alignment

## Open questions

- **Recipe versioning:** how do recipes evolve without breaking
  in-flight missions? (Pin recipe version at mission start; allow
  upgrade only at phase boundaries.)
- **Memory pruning:** mission_lessons grows unbounded. Decay policy?
  (TTL + occurrences-weighted retention.)
- **Vision model choice for T11:** Claude Vision vs dedicated UI-diff
  model vs in-house? (Bench in week 1 of Wave 7.)
- **Container runtime for T10:** Docker (familiar) vs firecracker
  (faster boot, more setup) vs nsjail (lightweight)? (Default
  Docker; revisit after performance data.)
- **Multi-file inference (T6):** how does the expander know which
  files to spawn? (Drive from openapi paths + model fields + acceptance
  criteria; ship with conservative inference + human override.)
- **Cost ceiling unit (T8):** dollars vs tokens vs hours? (Hours
  is most user-meaningful; convert internally.)

## Sequencing summary

```
Wave 1 ── Wave 2 ── Wave 3 ── Wave 4
   │         │         │
   └── Wave 5 (parallel after W1)
   │
   └── Wave 6 (after W1 stable)
   │
   └── Wave 7 (after W1, parallel)
                         │
                         └── Wave 8 (after W2, W5, W7)
```

Total estimate: ~6 months calendar with overlap. Critical path:
W1 → W2 → W3 → W8.

## What this is NOT

- NOT a scope reduction. i2p stays end-to-end.
- NOT a workflow-engine replacement (goal-loop runtime brainstormed
  earlier — discarded as wrong framing).
- NOT a kill list of phases. Phases 9-14 stay; their NEEDS-REAL-TOOLS
  steps unblock as adapters land.
- NOT a tier system (concept/design/scaffold flavors brainstormed
  earlier — discarded as rearranging chairs).

The shape is right. We're making it dense.
