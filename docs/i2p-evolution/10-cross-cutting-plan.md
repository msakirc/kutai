# Z10 Cross-Cutting — Master Plan

**Opened:** 2026-05-10. **Reads:** `10-cross-cutting.md` (zone doc).
**Pattern:** z1 tier-by-tier, parallel parts within tier, sequential merge.

## Decisions locked

| Q | Choice |
|---|---|
| Sequencing | Z0+Z1 done — tackle entire zone in one campaign |
| Scope | Full Phases A→G (all 7) |
| `min_confidence` gate | Fail-closed default; per-agent `confidence_gate=warn` opt-out flag |
| `SANDBOX_MODE=local` | Per-mission opt-in via founder confirmation; default `docker` |
| Audit log shape | Extend `registry_events` with `scope='action'`; split if query patterns diverge |
| Reversibility taxonomy | 3 buckets `full|partial|irreversible`; per-verb registry; shell verbs accept caller intent override |
| Telegram | Forum topics, one per `mission_id` |
| Container runtime | Docker v1 |
| Provenance storage | Separate `artifact_provenance` table |
| Reset-to-green | Workspace git tag + DB snapshot + Chroma snapshot, paired |
| `quality_mode` | `quick|balanced|thorough`, default `balanced` |

## Tier map

```
Tier 1 — Foundation (Phase A: reversibility + provenance + plug min_confidence)
  ├ T1A — plug machinery (min_confidence gate, file_locks sweeper, atomic git_commit↔push)
  ├ T1B — verb taxonomy as code (registry, Action.reversibility, shell intent override)
  └ T1C — schema + provenance API (artifact_provenance, schema_migrations ledger, audit ext, confirmation flow)

Tier 2 — Surface gauges (Phases B + D; independent)
  ├ T2A — cost: mission scope wire + iteration_n aggregation + ceiling alerts + quality_mode dial
  └ T2B — telegram: forum topics per mission + typed events + reaction handler + request_review

Tier 3 — Concurrency + time (Phases C + E; E touches DB, sequence carefully)
  ├ T3A — time awareness (mission target_launch + time_budget + pacing dashboard + tradeoff prompt)
  ├ T3B — sandbox per mission (container per mission_id + per-mission opt-in for local + tightened BLOCKED_PATTERNS + egress whitelist)
  └ T3C — cross-mission state safety (shard _tx_lock + Chroma per-mission namespace + reset-to-green primitive)

Tier 4 — Closing the loop (Phases F + G)
  ├ T4A — demo deliverable (record_demo verb + i2p final-phase wire + bundle to mission thread)
  └ T4B — trust calibration (confidence_outcomes table + nightly job + prompt-builder feedback + /calibration command)
```

## Parts × scope × effort

| Part | Phase | Effort | Depends on |
|---|---|---|---|
| T1A | A | 1.5d | — |
| T1B | A | 1.5d | — |
| T1C | A | 2d | T1A + T1B (schema reads from registry; gate hook reads from sweeper) |
| T2A | B | 2d | T1C (mission_id scope on cost_budgets needs schema) |
| T2B | D | 3d | T1C (typed events post via confirmation flow plumbing) |
| T3A | C | 2d | T1C |
| T3B | E | 3d | T1B (verb registry includes shell mode override) |
| T3C | E | 3d | T3B (per-mission container precedes per-mission DB shard) |
| T4A | F | 1.5d | T3B (record_demo runs in container) |
| T4B | G | 2.5d | T1C + T2A (confidence outcomes need provenance + cost telemetry) |

Total ≈ 22d sequential; ~14d wall with parallel parts within tier.

## Dispatch strategy (z1 pattern)

For each tier:
1. Dispatch parts as parallel agents in isolated worktrees (`Agent` w/ `isolation=worktree`).
2. Each part has self-contained brief + acceptance test + commit policy.
3. After all parts return, founder merges into `main` sequentially. Conflict expected on shared files (`db.py`, `mr_roboto/__init__.py`, `actions.py`) — z1 pattern: take-both for db.py, reset-and-inject for mr_roboto inits.
4. Validate: `pytest` targeted run + AST/JSON validate on edited workflow files.
5. Tag handoff doc per tier: `docs/handoff/2026-05-XX-z10-tierN-shipped.md`.
6. Update memory `MEMORY.md` after each tier.

## Acceptance for whole zone

- Every artifact has provenance row (`artifact_provenance`) joinable to model + retry + reviewer.
- Every action verb has `reversibility` tag from registry; `partial`/`irreversible` blocks on founder reaction.
- `min_confidence` empirically gates a low-confidence Coder output in test (was cosmetic; now real).
- `/mission_cost <id>` returns first-pass + retry split, vendor split, total — all populated from real data.
- Two missions run concurrently w/o `database is locked` errors and w/o `chroma_data` interleave.
- `/rollback_mission <id>` restores workspace + mission DB rows + Chroma collection.
- Toy mission produces `demo.mp4` attached to its Telegram forum topic.
- `/calibration` returns per-(model, domain) reliability matrix after ≥30 outcomes.
- Phase A integration test: synthetic mission with one irreversible action stalls until 👍 reaction.

## Risks / gotchas

- `db.py` god-file conflicts every tier. Plan: take-both merge for additive schema; serialize merges.
- `mr_roboto/__init__.py` re-export collisions (z1 hit this). Use reset-and-inject merge pattern.
- Telegram forum topics require chat upgrade to supergroup; one-time founder action.
- Per-mission Chroma collections need migration of existing `chroma_data` (one-shot script in T3C).
- Reset-to-green DB snapshot scope needs care — restore must not clobber other concurrent missions' rows.
- `_tx_lock` shard: identify mission-scoped vs global tables; cross-mission tables (e.g. `models`, `cost_budgets` global rows) keep global lock.

## Updates

- 2026-05-10 — plan opened; tier map fixed; locked decisions from kickoff Q&A.
