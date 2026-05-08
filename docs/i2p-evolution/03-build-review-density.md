# Z2 — Build review density (multi-pass review + multi-file expansion + QA modalities)

## Frame

Once code-zone foundations from [02-build-foundation.md](02-build-foundation.md)
are in, this is the next compound-yielding investment: replace single-pass
build steps with real iteration; replace single-file feature builds with
per-file decomposition + cross-file consistency review; add domain-specific
QA modalities (a11y, security, performance, contract).

Visual review is split out to [04-build-visual-review.md](04-build-visual-review.md)
because the framing/state-priming infra makes it its own subproject.

## Current state

- Single LLM call per feature template build step (feat.4 backend_service emits one model + one service in one call).
- code_review post-hook runs once per build step; one-shot pass/fail.
- No cross-file consistency check after a feature's files all land.
- No accessibility/security/performance/contract review modalities.
- Existing severity-blockers gate (10.1 owasp_audit, 10.5 encryption_logging_review, shipped 2026-05-05) is the template for new modality post-hooks.
- `code_reviewer` agent type registered (2026-05-06).

## Gaps

### Fixable by automation

**A. Multi-pass review loops (turn one-shot into iteration)**
- **Self-critique pass.** After each build step's final_answer, same agent re-prompted with critic-shaped message: "review your own diff for omissions/bugs/inconsistencies." 1 extra LLM call; cheap; catches obvious issues before posthook chain.
- **Pair-programming pattern.** For hard-difficulty steps: drafter + critic agents, back-and-forth until critic clean. Bounded by issue-severity-clean (no medium+ findings), not iteration count.
- **Iterate-until-clean inner loop.** Replace "build → one review → done" with "build → review → fix → re-review → ..." bounded by quality gate.

**B. Multi-file feature expansion**
- **feat.4 backend_service** today: one call emits "the model + the service." Real shape: agent infers files needed from openapi paths + model fields + acceptance criteria; spawns sub-tasks per file (types, validators, service, repository, error mapper, fixtures, tests).
- Each sub-task: own produces + grounding + code_review.
- No cross-file consistency review at the end. Service signature drifts from route handler; frontend type drifts from openapi schema; test fixture misses error branches. Single-file code_review can't catch.

**C. QA modalities (severity-aware, mechanical posthook framework from 02)**
- Accessibility: axe-core findings, severity-blocker gate.
- Security: bandit (Python) + npm audit (JS) + semgrep (cross-lang) + dependabot.
- Performance: lighthouse score threshold OR k6/locust load profile.
- Contract fuzz: schemathesis against staging openapi.

**D. Architectural drift**
- Mid-implementation, agent ignores phase 4 ADRs. New post-hook: scan emitted code for new dependencies + new patterns + deviations from chosen stack against ADR; fail with citation. Depends on ADRs from [01-pre-code.md](01-pre-code.md).

**E. Layer-aware tooling**
- Codebase introspection tool tags each file/dir with layer (domain/adapter/infra/test). Code-review prompts gain layer context: "this file is core domain — no framework imports allowed."
- Catches "agent leaked Stripe types into the domain model" class.

**F. Pre-merge integration replay**
- Before claiming a feature done: replay test suite against last N committed features in random order; confirm no regressions. Catches "works in isolation" bugs cheaply.

### Founder territory
- Severity threshold tuning per project ("medium findings acceptable for prototype, blocker for prod").
- Quick-vs-thorough mode toggle (severity gate dial).

## Proposed direction

### Multi-pass review machinery

- **Self-critique loop in coulson.** New sub-iter step: after `final_answer`, before commit, agent gets critic-prompted with own diff. If critic clean → proceed; else fix.
- **Pair agent dispatch.** Optional second-agent dispatch flag on hard-difficulty steps. Drafter → critic → drafter → critic until critic-passed.
- **Iterate-until-clean** as the default for code_review post-hook on build steps; bounded by issue-severity-clean rather than max_iterations.

### Multi-file expansion

- **Feature template build step gains `multi_file: true` flag.** Expander reads spec artifacts (openapi paths, model fields, story acceptance criteria); generates per-file sub-tasks.
- **File inference rules** (per-stack, recipe-aware): backend_service expands to {model, schema, service, repository, error_mapper, fixtures, tests}; frontend_components expands to {component, hook, story, test}; etc.
- **Conservative inference + human override.** Founder sees inferred file list before sub-tasks dispatch; can edit.

### Integration reviewer (new agent type)

- Agent type: `integration_reviewer`.
- Post-hook kind: `integration_review`.
- Reads ALL files emitted for a feature together.
- Prompt focuses on cross-file consistency: signature alignment, type match, test coverage of error branches, naming consistency, migration-vs-model alignment.
- Verdict: pass / fail with structured findings; fail retries one or more sub-tasks with cross-file context.

### QA modality post-hooks

| Kind | Tool | Verdict gate |
|---|---|---|
| `accessibility_review` | axe-core via playwright | severity = critical/serious blocks |
| `security_review` | bandit + npm audit + semgrep | CVSS ≥ 7 OR semgrep `error`-rule blocks |
| `performance_review` | lighthouse OR k6 | threshold violations declared in spec block |
| `contract_review` | schemathesis | any 5xx OR contract mismatch blocks |

All use the existing `blockers: {field, levels}` rule.

### Architectural drift gate

- New post-hook `adr_drift_check`. Runs after build steps that introduce dependencies or patterns. Reads relevant ADR; LLM-judges + mechanical-greps for known violations (new framework imports, deviation from chosen state-management lib, etc.).
- Cited fail messages reference ADR by id.

### Layer-aware tooling

- New tool: `inspect_layer(path)` returns layer tag (domain/adapter/infra/test) based on path heuristics + project conventions declared in spec.
- Code-review prompts gain layer context per file.

### Pre-merge integration replay

- Before claiming a feature done, mechanical step: rerun integration test suite against the feature's commit + N prior feature commits in random shuffle. Failure → bisect to which combination breaks → retry the breaking sub-task with that test as feedback.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Multi-file inference | proposes sub-task list | edits before dispatch | full |
| Iterate-until-clean | retries until quality gate clean | overrides "this finding is acceptable" | full |
| Severity threshold | per-project default | dials quick-vs-thorough | full |
| Integration replay fail | bisects + auto-retries | reviews bisect report on persistent fail | full |

## Dependencies

- **Inbound:** [02-build-foundation.md](02-build-foundation.md) hard pre-req — mechanical gates + memory + recipes must exist before review density compounds. [01-pre-code.md](01-pre-code.md) — ADRs needed for drift gate.
- **Outbound:** [04-build-visual-review.md](04-build-visual-review.md) — visual is another modality on the same posthook framework. [09-growth.md](09-growth.md) — analytics-driven hypothesis tests need fast feature iteration, depends on this density.
- **Cross:** [10-cross-cutting.md](10-cross-cutting.md) — cost transparency on the iterate-loops to surface "this feature has retried 8x; OK to continue?".

## Open questions

- **Multi-file inference rules.** Static per-stack vs LLM-inferred per-feature? (Static heuristic first, LLM fallback only when shape doesn't match recipe.)
- **Iteration cost ceiling.** How many iterate-until-clean rounds before escalation? (Default 5; configurable per quality dial.)
- **Pair-programming model selection.** Drafter + critic same model? Different models for diversity? (Different models — different perspectives reduce groupthink.)
- **Integration-replay determinism.** How to seed shuffles for reproducibility? (Mission-deterministic seed; record in audit.)
- **ADR drift granularity.** Greps + LLM judgment, or formal architecture-as-code (eg. arch-unit)? (Greps + LLM v1; formal v2 if drift remains an issue.)

## Agent task brief

When picking up this doc:
1. Read 00-README, 01-pre-code, 02-build-foundation, this doc.
2. Verify dependencies in 02 are sequenced ahead.
3. Convert each section's recommendation to a phased plan: tool/verb scaffolds, posthook wiring, expander changes, agent prompt fragments, tests.
4. Resolve open questions or escalate.
5. Cross-reference outbound to [04-build-visual-review.md](04-build-visual-review.md), [09-growth.md](09-growth.md), [10-cross-cutting.md](10-cross-cutting.md).
6. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; absorbs Wave 3 + Wave 5 + theme T4 + T6 + T9 from `docs/plans/2026-05-07-i2p-capability-expansion.md`.
