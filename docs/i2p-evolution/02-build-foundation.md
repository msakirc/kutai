# Z2 — Build foundation (mechanical guardrails + specs-as-SoT + memory + recipes + tooling discipline)

## Frame

Make code-zone reliable. G grounding (shipped 2026-05-06) confirms files
were written. This doc covers what comes after: imports actually resolve,
migrations actually apply, types actually align, agent uses the right
tool for the job, doesn't re-invent the wheel each mission, and learns
from past failures.

This is the foundational build doc. Higher-density review loops live in
[03-build-review-density.md](03-build-review-density.md) and depend on
this doc shipping first.

## Current state

- G grounding: L1 sub-iter guard + L2 mechanical post-hook against produces declaration (commits e17a176 → 39e27fd, 2026-05-06).
- Mechanical post-hook framework: verify_artifacts (file existence + min_bytes + parse) + code_review (LLM review with severity gate) + grounding (declarative produces vs tool_calls).
- Schema dialect: `equals`, `pattern`, `unique_by`, `min_items_from`, `blockers: {field, levels}` for severity gates (shipped 2026-05-05).
- Salako verbs: workspace_snapshot, git_commit, verify_artifacts, run_cmd, check_grounding, clarify, notify_user.
- Coulson runtime owns multi-call orchestration; tool_calls audit log captured per execution.
- Recipe library, mission-lessons memory, stack-aware prompts: not yet implemented.
- Diff-shaped tools (patch_file, edit_file, apply_diff) exist but agent defaults to write_file (rewrites whole files even for surgical edits).

## Gaps

### Fixable by automation

**A. Mechanical guardrails (no LLM, run code or contracts)**
- **Imports vs declared deps.** Agent emits `from app.services.user import UserService`; no module exists; no requirements/package.json entry. Caught only at integration test time, after 4 features have piled on the broken one.
- **Migration-applies-clean.** Migration files written, never run. Syntax errors / FK ordering / missing extensions caught at deploy.
- **Test-running gate.** Tests written; never executed. `salako.run_pytest` / `run_jest` exists in plan but not wired as posthook.
- **Cross-type sync.** Frontend types diverge from backend openapi. Mechanical: regenerate via `openapi-typescript`, diff against committed.
- **Design-system enforcement.** Components reinvent styling instead of using primitives. Scan emitted JSX/TSX for raw color hex, inline styles, non-design-system imports.
- **Forbidden-pattern lint.** No `console.log` in committed, no `time.sleep` in tests, no `assert True`. Custom-rules engine over ESLint/ruff/semgrep.

**B. Specs-as-source-of-truth (closes spec→code drift mechanically)**
- **OpenAPI from FastAPI route decorators**, not authored separately. Eliminates one drift class entirely.
- **Frontend types via `openapi-typescript`**, generated, not hand-written.
- **Migrations via alembic autogenerate** (model diff), not hand-written. Catches "model changed but migration didn't" mechanically.

**C. Tooling discipline**
- **Diff-shaped edits.** Agent default `write_file` rewrites whole files even for surgical edits. Burns tokens, drops local context, sometimes silently deletes hand-tweaked work. Force `patch_file` / `edit_file` / `apply_diff` for steps targeting existing files.
- **Tool-routing layer.** Per-step `tools_hint` is advisory today; should be binding when target file exists.
- **Stack-aware system prompts.** Currently one coder agent for Python+TS+SQL. Each language has different idioms. Stack-aware prompt fragments load at dispatch based on tech_stack_decision.

**D. Cross-mission memory + recipe library**
- **No cross-mission memory.** Each mission starts blank; mission 57 lessons aren't fed into mission 58. "Forgot CORS preflight (4/7 auth missions)" should pre-load as context.
- **No recipe library.** Every mission redoes auth/file-upload/search/pagination from scratch. Real engineers reuse vetted patterns.

### Founder territory
- Stack and pattern *taste* (via picking from ranked alternatives in Z1 ADRs) — but the framework choice gets enforced mechanically here.
- Tradeoff calls when a recipe doesn't fit (founder approves customization scope).

## Proposed direction

### Mechanical guardrails — six new post-hook kinds

Each is a salako verb + beckman posthook kind + auto-wire in expander.
Same pattern as verify_artifacts:

| Kind | Verb | Triggers when | Verdict |
|---|---|---|---|
| `imports_check` | parse-and-diff against deps file | code-emitting step | fail if missing import or unused import |
| `migration_apply` | spin sandbox DB → alembic upgrade head | step produces migration | fail on apply error |
| `test_run` | invoke pytest/jest/vitest | step produces tests | fail if red |
| `type_sync` | regenerate types from openapi → diff | step produces openapi or types | fail on drift |
| `design_system_check` | scan JSX/TSX for forbidden patterns | step produces frontend code | fail on hit |
| `pattern_lint` | custom rules via ESLint/ruff/semgrep | every code-emitting step | fail by severity |

Each carries severity-aware gating via existing `blockers` rule.

### Specs-as-SoT shifts

- **Phase 4 stack selection** prefers frameworks with native openapi support; recipes (below) ship with this assumption.
- **Phase 7 backend_scaffold** adopts FastAPI's openapi gen; openapi.json becomes a generated artifact, not authored.
- **Phase 7 frontend_state** uses `openapi-typescript` codegen; no hand-written API types.
- **Phase 8 feature template database_migration** uses alembic autogenerate; agent runs `alembic revision --autogenerate` instead of hand-writing.

### Tooling discipline

- **Tool-routing layer in dispatcher.** Inspect target file existence + step intent. Force `patch_file` over `write_file` for existing files; force `apply_diff` for multi-line surgical edits. Reserved `write_file` for true creation only.
- **Stack-aware prompt fragments.** New directory `src/agents/prompts/stacks/`: fastapi.md, nextjs.md, expo.md, django.md, etc. Loaded at dispatch by fatih_hoca based on `tech_stack_decision` artifact. Injected into system prompt.

### Cross-mission memory

```
mission_lessons (DB table)
  id          INT PK
  mission_id  INT
  stack       TEXT     "fastapi+postgres+nextjs"
  domain      TEXT     "auth", "file_upload", "search"
  pattern     TEXT     "forgot CORS preflight"
  fix         TEXT     "add CORSMiddleware before routes"
  severity    TEXT     "blocker" | "warning" | "info"
  occurrences INT      denormalized counter
  created_at  TIMESTAMP
```

- Auto-populated from DLQ entries + post-hook fail verdicts.
- Queried at mission start by stack × domain.
- Top-N lessons injected into context as "watch out for."
- Decay: TTL + occurrences-weighted retention.

### Recipe library v1

Five vetted recipes:
- `auth/` — email-password, JWT/session, password reset, email verify
- `file_upload/` — multipart upload, S3-compat storage, virus scan stub, file metadata
- `search/` — full-text via Postgres tsvector OR Meilisearch adapter
- `pagination/` — cursor + limit/offset variants, total-count cost-aware
- `audit_log/` — append-only event log, queryable, retention policy

Per-recipe shape:
```
recipes/<name>/
  recipe.yaml       metadata, tech-stack compatibility
  spec.md           what + when-to-pick + when-NOT-to-pick
  prompts/          stack-aware prompt fragments
  backend.template.py   parametrized scaffold
  frontend.template.tsx
  tests/            recipe ships with own test suite
  migrations/       alembic-style template
  lessons.md        known gotchas
```

Planner gains `pick_recipe` action; expander instantiates recipe + per-feature parameters.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Recipe pick | proposes recipes that match feature shape | accepts or rejects | full pre-instantiation |
| Mechanical-gate fail | retries with feedback | escalated only after retry budget | full |
| Drift detected (type sync, design system) | retries source step | explicit unblock if rule too strict | full |
| New recipe proposal | drafts based on observed pattern in 3+ missions | reviews + adds to library | n/a |

## Dependencies

- **Inbound:** [01-pre-code.md](01-pre-code.md) — recipe selection depends on spec quality + ADRs.
- **Outbound:** [03-build-review-density.md](03-build-review-density.md) — multi-pass reviews compose on top of these mechanical gates. [04-build-visual-review.md](04-build-visual-review.md) — visual review is another mechanical post-hook in the same framework.
- **Hard pre-req for:** all later Z2 docs.
- **Cost-transparency hooks:** integrate with [10-cross-cutting.md](10-cross-cutting.md).

## Open questions

- **Recipe versioning.** How do recipes evolve without breaking in-flight missions? (Pin recipe version at mission start; allow upgrade only at phase boundaries.)
- **Memory pruning policy.** TTL + occurrences-weighted retention; specifics TBD per stack/domain.
- **Stack-aware prompt fragments.** How dynamic? (V1: file per stack, loaded by name. V2 if needed: parametric.)
- **Tool-routing layer.** Heuristic-only or LLM-arbitrated when ambiguous? (Heuristic first; LLM second only if needed.)
- **Mechanical-gate runtime cost.** Each gate ~30-60s; running them per-feature compounds. Parallel where safe; serialize where dependencies require. Profile before W1 ships.
- **Custom-rules engine for `pattern_lint`.** ESLint plugin + ruff config? semgrep? (semgrep — multi-language, single config.)

## Agent task brief

When picking up this doc:
1. Read 00-README.md + 01-pre-code.md + this doc.
2. Audit existing salako verbs + beckman post-hook plumbing; quote file refs.
3. For each of the six mechanical guardrails: write a phased plan (verb scaffold → posthook kind → auto-wire trigger → tests). Convert open questions to decisions.
4. Recipe library v1: pick 5 recipes (validate or revise the auth/file_upload/search/pagination/audit_log set), draft the per-recipe directory structure, write one recipe end-to-end as the template for the rest.
5. Memory layer: add the DB migration, write the auto-populate hooks (DLQ + post-hook fail), write the at-mission-start query + injection.
6. Cross-reference outbound to [03-build-review-density.md](03-build-review-density.md).
7. Add `## Updates` log entry with commit refs.

## Updates

- 2026-05-08 — initial doc; absorbs Wave 1 + Wave 2 content from `docs/plans/2026-05-07-i2p-capability-expansion.md`.
