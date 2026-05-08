# i2p Evolution — Index

**Date opened:** 2026-05-08
**Goal:** map the gulf between today's i2p (concept → compiling code) and a
real, human-in-the-loop, idea-to-product system. One doc per zone (or
paired zones); each doc tackles by an independent agent + session.

## What "real product" means here

Something a stranger pays for, uses, complains about, returns to. Not
"code that compiles + tests pass."

The agent does ~90% of the lift; the founder does the irreplaceable 10%
(strategy, brand, relationships, legal personhood, taste arbiter). This
folder maps where each piece of work lives.

## Zone map

| Zone | Doc | Frame |
|---|---|---|
| Z0 — Mission preflight | [z0-mission-preflight.md](z0-mission-preflight.md) | founder + system readiness contract before phase 0 |
| Z1 — Pre-code | [01-pre-code.md](01-pre-code.md) | idea → executable spec |
| Z2 — Build foundation | [02-build-foundation.md](02-build-foundation.md) | mechanical guardrails + specs-as-SoT + memory + recipes + tooling discipline |
| Z2 — Build review density | [03-build-review-density.md](03-build-review-density.md) | multi-pass review + multi-file expansion + QA modalities |
| Z2 — Visual review | [04-build-visual-review.md](04-build-visual-review.md) | screenshot framing + state priming + diff (subproject) |
| Z2 — Mobile track | [05-build-mobile-track.md](05-build-mobile-track.md) | platform-aware frontend + mobile tooling |
| Z3 + Z7 — Real-world bridge | [06-real-world-bridge.md](06-real-world-bridge.md) | accounts, legal, vendor adapters, payments |
| Z4 + Z8 — Humanish layers | [07-humanish-layers.md](07-humanish-layers.md) | marketing, story, comms helper, relationships |
| Z5 — Operations | [08-operations.md](08-operations.md) | monitoring, on-call, incident, support, backups |
| Z6 — Growth | [09-growth.md](09-growth.md) | analytics, hypotheses, A/B, prioritization |
| Cross-cutting | [10-cross-cutting.md](10-cross-cutting.md) | trust, time, reversibility, cost, provenance, sandboxing, demo |

## Sequencing tiers (cross-zone)

```
Tier 0 — Establish founder/system contract (BEFORE everything)
  └ z0-mission-preflight    (every other zone reads its outputs)

Tier 1 — Make code zone reliable
  ├ 02-build-foundation     (most leverage; everything else compounds on this)
  ├ 03-build-review-density
  └ 10-cross-cutting        (trust + provenance + cost — light slice first)

Tier 2 — Bridge to real-world ops
  ├ 06-real-world-bridge    (accounts, deploy adapters)
  └ 08-operations           (monitoring, on-call)

Tier 3 — Close iteration loop
  ├ 09-growth               (analytics + hypothesis + A/B)
  └ 03-build-review-density (deeper passes once data flows back)

Tier 4 — Humanish layers
  ├ 07-humanish-layers      (marketing, story, comms helper)
  └ 04-build-visual-review  (taste-adjacent gating)

Tier 5 — Scope expansion
  └ 05-build-mobile-track   (biggest scope, last)
```

## Cross-zone dependency graph

```
z0-mission-preflight ─ feeds every zone ─────────────────────────────────┐
                                                                         │
01-pre-code ──┐ (reads z0 founder profile + ambition + compliance)       │
              ├──> 02-build-foundation ──┬──> 03-build-review-density ──┐│
              │                          │                              ││
              │                          ├──> 04-build-visual-review ───┤│
              │                          │                              ││
              │                          └──> 05-build-mobile-track ────┤│
              │                                                         ││
              ├──> 06-real-world-bridge ────> 08-operations ────────────┤│
              │   (reads z0 vault + vendor list)                        ││
              │                                                         ││
              └──> 10-cross-cutting ─ all zones honor patterns ─────────┘│
                  (reads z0 cost ceiling + idle policy + reversibility)  │
                                                                         │
07-humanish-layers ── orthogonal, no hard prerequisite ──────────────────┤
                                                                         │
09-growth ─ depends on operations + review density (reads z0 north-star)─┘
```

## Division-of-labor framing

Every zone doc must describe **the parallel founder track** alongside the
agent track. The current i2p only describes agent work. The system stays
honest only if every zone explicitly states:

- What the agent does autonomously
- What the agent prepares but the founder approves
- What the founder does (agent supports but doesn't drive)
- Handoff contract (what agent gives founder, what founder gives back)
- Reversibility (can the agent unwind, or is human commit final?)

## Brutal truths

These are gaps automation cannot close in principle, only support:

- Founder's lived insight into the problem
- Brand voice and taste
- Long-term relationships (customers, investors, journalists, partners)
- Strategic pivots based on gut + signal mix
- Crisis response requiring legal/PR judgment
- Hiring decisions
- Anything requiring legal personhood (signing contracts, KYC, holding accounts)

Per-zone docs must distinguish "agent gap to close" from "founder responsibility forever."

## Per-doc structure (contract for the agents picking these up)

Each zone doc follows the same skeleton:

1. **Frame** — what this zone is, why it matters for "real product"
2. **Current state** — what we have today; what works; what's wired
3. **Gaps** — sharp, prioritized; distinguish "fixable by automation" vs "founder territory"
4. **Proposed direction** — recommendations from the 2026-05-07 / 2026-05-08 brainstorms; cite prior plans where applicable
5. **Human-in-loop pattern** — explicit founder-track for this zone
6. **Dependencies** — what other zone docs must mature before this one
7. **Open questions** — to resolve at kick-off
8. **Agent task brief** — what to do with this doc when picking it up

## How to use this folder

- **Pick a zone doc.** Read this README + that doc + the dependency list.
- **Refine the analysis.** Add concrete, citable evidence from the codebase.
- **Propose specific implementation plans.** Convert recommendations to phased work, with effort + acceptance criteria.
- **Surface contradictions.** When two zones imply different shapes, flag here.
- **Update.** Each doc has an `## Updates` log at the bottom; add findings + dates + commit refs.

## Cross-references — prior planning artifacts

- `docs/plans/2026-05-07-i2p-capability-expansion.md` — 12 themes / 8 waves, seeds 02-build-foundation, 03-build-review-density, 04-build-visual-review, 05-build-mobile-track, 10-cross-cutting
- `docs/handoff/2026-05-05-i2p-grounding-gates.md` — G grounding session
- `docs/handoff/2026-04-27-session-handoff.md` — retry pipeline / reviewer / constrained emit
- Existing `docs/architecture/` — current architecture docs

## Dispatch groups (parallel-safe)

```
Group 0 — z0-mission-preflight (must land first; outputs feed all)
Group A — 01-pre-code + 10-cross-cutting (no other inbound; cross-cutting
          findings may force minor Z1 rev — accept or sequence 10 first)
Group B — 02-build-foundation + 06-real-world-bridge
Group C — 03-build-review-density + 08-operations
Group D — 04-build-visual-review + 09-growth + 07-humanish-layers
Group E — 05-build-mobile-track
```

## Updates

- 2026-05-08 — folder created; README + 11 zone docs scaffolded.
- 2026-05-08 — added z0-mission-preflight as new Tier-0 zone; updated
  zone map + sequencing tiers + dependency graph + dispatch groups.
