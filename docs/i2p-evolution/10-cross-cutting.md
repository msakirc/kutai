# Cross-cutting concerns + sandboxing + demo deliverable

## Frame

Concerns that appear in every zone but are nobody's primary
responsibility. Trust calibration, time awareness, reversibility framing,
cost transparency, provenance. Plus two infra pieces that don't fit a
single zone: per-mission sandboxing/containerization, and the end-of-
mission demo deliverable.

This doc is the "shared spine" — every other zone doc must conform to
the patterns established here. Independent agent picking this up should
expect to drive design decisions that ripple across the whole folder.

## Concerns

### A. Trust calibration

**Gap.** Agent doesn't know what it doesn't know. No "I'm 30% confident
in this architectural choice; here's why." Confidence is implicit /
absent; founders can't tell which decisions warrant scrutiny.

**Solution.**
- Every artifact-producing step gains a `confidence` field (low / medium / high) + `reasoning`.
- Every architectural decision (ADR) gains explicit `confidence` + `reversal_cost`.
- UI surfaces low-confidence + irreversible decisions for explicit human sign-off.
- Confidence calibration training: feed past confidence vs actual-correctness back into prompts ("models that say 'high confidence' here are wrong 40% of the time — be more conservative").
- Per-domain reliability score: which areas of decision-making does this agent actually do well in? Track over time.

### B. Time awareness

**Gap.** Mission has no temporal pacing. "We've been at this 3 weeks" not surfaced. No deadline pressure, no scope-cut judgment when launch date approaches.

**Solution.**
- Mission declares: target launch date, total time budget, per-phase budget.
- Real-time pacing dashboard: where are we vs plan? burndown.
- Approaching deadlines surface tradeoff conversations: "phase 8 backlog has 12 features; at current pace you'll hit launch with 8. Cut which 4?"
- Long-running concerns scheduled (backups, key rotation, dependency updates) cross-ref [08-operations.md](08-operations.md).
- Calendar integration: founder availability, demos scheduled (cross-ref [07-humanish-layers.md](07-humanish-layers.md)).

### C. Reversibility framing

**Gap.** "I'm about to migrate the database schema" should feel different from "I'm about to add a button." Today they look the same in the agent's confirmation flow.

**Solution.**
- Every action gets a reversibility annotation:
  - `full` — undo by file revert / git revert
  - `partial` — undo possible but data may be lost (e.g. dropped column with backfill)
  - `irreversible` — once done, it's done (publish to app store; send email to all users; spin up paid resource)
- High-stakes actions (`partial` / `irreversible`) require explicit founder confirmation in Telegram thread.
- Audit log captures reversibility tag per action.
- "Roll back to last green" primitive (cross-ref [10 sandboxing](#h-sandboxing--reset-to-green) below).

### D. Cost transparency

**Gap.** Founder doesn't know mission cost until invoice. Tokens (LLM
calls), infra (vendor adapters), human-time (own + counsel), opportunity
(months of work) — all opaque.

**Solution.**
- Per-mission real-time cost gauge: tokens × model-rate, vendor API costs, projected-cost-to-completion.
- "This feature will cost ~$X to build" estimates upfront.
- Cost surfaces at decision points: "this multi-pass review iteration will cost ~$2; continue?"
- Per-mission budget ceiling; threshold alerts at 50% / 75% / 90%.
- Quick-vs-thorough mode dial trades cost for quality; dial visible in mission setup.
- Long-tail costs surface (subscription that auto-renews; storage that grows monthly).

### E. Provenance

**Gap.** "Where did this code come from? Which model? Which iteration?
Why?" — currently opaque. Audit trail exists in DB but isn't queryable
from the artifact end.

**Solution.**
- Every artifact tagged with the chain that produced it:
  - Source step ID
  - Model + iteration count + retry count
  - Decisions referenced (ADRs, lessons applied)
  - Reviewers + verdicts
  - Founder approvals (if any)
- Queryable: "show me the provenance of `backend/services/billing.py`" returns the full chain.
- Useful for incident response ("this bug shipped from mission 47, model X, after 3 retries — model X has been having issues; rollback").

### F. Cross-mission memory (re-emphasized — its home is [02-build-foundation.md](02-build-foundation.md) but it's cross-cutting)

- Memory schema (`mission_lessons`) maintained in 02.
- Cross-cutting use: every other zone reads from + writes to it.
- Founder profile + product profile persist across missions; same founder's second mission inherits voice / brand / stack preferences.
- Memory pruning policy: TTL + occurrences-weighted retention (cross-ref 02 open questions).

### G. Per-mission Telegram thread

- Persistent thread per mission_id (cross-ref [07-humanish-layers.md](07-humanish-layers.md) for support patterns).
- Posts: `[milestone]`, `[blocker]`, `[asking]`, `[confirmation_required]`, `[cost_alert]`.
- Founder reactions become typed events (`approve`, `reject`, `comment`).
- Comments → revision tasks against the relevant artifact.
- request_review action surfaces here.

### H. Sandboxing + reset-to-green

- **Per-mission container.** Mission writes go through Docker container with mounted workspace. Lets us drop safety rails on `shell` without risk to host. Default Docker; firecracker / nsjail later if perf needs.
- **Reset-to-green primitive.** Every commit-after-green is known-good restore point. `/rollback_mission <id>` returns to last green commit + state.
- **Resource limits per mission.** CPU / memory / disk caps prevent runaway mission consuming the whole host.
- **Network policy.** Egress limited to whitelisted vendor APIs by default; broader requires founder approval.

### I. End-of-mission demo deliverable

- Final playwright run with `--video on` capturing the core flow.
- 30s MP4 attached to mission deliverable.
- Forces "running demo" to be a real exit criterion, not a checkbox.
- Useful for founder review + investor updates + marketing.
- Mobile equivalent (cross-ref [05-build-mobile-track.md](05-build-mobile-track.md)) records device-screen video.

## Founder territory

- Trust calibration: founder ultimately decides which agent decisions to trust how much. Confidence-and-reasoning helps, but final call is taste.
- Cost cap setting: founder sets ceiling; agent stays under.
- Reversibility judgment: agent can tag, but founder makes the irreversible call.
- Time pressure: founder owns the launch date.

## Proposed direction

### Phase A — Reversibility + provenance (foundational; ship early)
- Add `reversibility` + `confidence` fields to action / artifact schemas.
- Audit log extension: capture reversibility tag.
- Provenance query API: artifact → chain.
- UI: confirmation flow honors reversibility tag.

### Phase B — Cost transparency
- Real-time cost gauge: tokens via existing accounting + vendor cost via adapter (cross-ref [06-real-world-bridge.md](06-real-world-bridge.md)).
- Per-mission budget + threshold alerts.
- Cost-at-decision surfacing.
- Quick-vs-thorough dial.

### Phase C — Time awareness
- Mission gains `target_launch` + `time_budget` fields.
- Pacing dashboard.
- Tradeoff prompts at deadline thresholds.

### Phase D — Telegram thread
- One thread per mission_id (Telegram supports threaded chats / topics).
- Typed event posting (milestone / blocker / asking / etc.).
- Reaction → typed event mapping.
- request_review wires through.

### Phase E — Sandboxing
- Docker template per mission.
- Salako shell verbs route through container.
- Resource limits + network policy.
- Reset-to-green primitive: git-based + state snapshot.

### Phase F — Demo deliverable
- Salako verb: `record_demo(scenario)` — playwright `--video on` + ffmpeg trim.
- Mission template: final phase produces demo as named artifact.

### Phase G — Trust calibration loop
- Track per-decision confidence vs actual-correctness over time.
- Per-domain reliability score (e.g. "agent's confidence in stack decisions correlates 0.7 with downstream success; but confidence in pricing decisions correlates 0.2 — discount accordingly").
- Feed back into prompt-builder.

## Human-in-loop pattern

| Concern | Agent surfaces | Founder decides | Reversibility tag |
|---|---|---|---|
| Confidence | low-confidence decisions for review | trust / override | full |
| Time | budget burn, scope-cut suggestions | which features to cut | full |
| Cost | per-decision estimate, ceiling alerts | quick-vs-thorough, budget cap | full |
| Reversibility | irreversible-action confirmation | sign-off | by definition |
| Provenance | full chain on request | n/a (read-only) | n/a |
| Sandbox | resource alerts | container reconfig | full |
| Demo | drafts video | re-records / approves | full pre-publish |

## Dependencies

- **Inbound:** every other zone's actions need to honor these patterns.
- **Outbound:** every other zone reads these patterns at design time.
- **Especially tight coupling with:** [02-build-foundation.md](02-build-foundation.md) (memory + cost), [06-real-world-bridge.md](06-real-world-bridge.md) (cost + reversibility on real-world actions), [08-operations.md](08-operations.md) (audit + reversibility on on-call actions).

## Open questions

- **Confidence-field schema.** Numeric (0-1) or categorical (low/med/high)? (Categorical for explainability; numeric for computation. Both — categorical surface, numeric storage.)
- **Reversibility taxonomy.** 3 buckets (full/partial/irreversible) or finer? (3 v1; finer if needed.)
- **Cost surfacing frequency.** Per-action vs per-step vs per-phase? (Per-step for normal actions; per-action for >$1 actions.)
- **Telegram thread vs separate chat.** Topics within one chat or per-mission separate? (Topics — keeps founder in one place.)
- **Container runtime.** Docker (familiar) vs firecracker (faster boot) vs nsjail (lightweight)? (Docker v1.)
- **Provenance storage.** Inline in artifact metadata vs separate table? (Separate table; artifact references provenance_id.)
- **Demo recording length.** Fixed 30s vs variable? (Variable, capped 90s.)
- **Trust-calibration loop scope.** Per-mission vs cross-mission? (Cross-mission — sample size matters.)

## Agent task brief

When picking up this doc:
1. Read 00-README + every other zone doc (skim — this doc affects all).
2. Phase A: confidence + reversibility schema + audit-log extension + provenance query.
3. Phase B: cost gauge + budget cap + quick-vs-thorough dial.
4. Phase C: time-awareness fields + pacing dashboard.
5. Phase D: Telegram thread + typed event flow.
6. Phase E: sandboxing template + reset-to-green primitive.
7. Phase F: demo recording verb + mission integration.
8. Phase G: trust-calibration loop scaffolding.
9. Surface any contradictions with other zone docs back into 00-README.
10. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; absorbs cross-cutting concerns from 2026-05-08 round + Wave 6 (sandboxing + demo) from `docs/plans/2026-05-07-i2p-capability-expansion.md`.
