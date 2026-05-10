# Handoff — Z1 Tier 0/1/2/3/4/5/6 shipped, Tier 7 dispatched

**Date:** 2026-05-10
**Session focus:** Z1 Tier 6 (cross-mission + ecosystem) via 3 parallel worktree subagents.
**Status:** Tier 0+1+2+3+4+5+6 merged + pushed. 609/612 tests pass (2 pre-existing failures unchanged + T6C visibility test fixed post-merge per founder lock).

---

## What shipped this session (Tier 6)

| Subagent | Items | Mechanical actions |
|---|---|---|
| T6A | P9 + A7 | `find_similar_missions` + `index_idea_fingerprint` + `surface_prior_mission_hints` + `index_mission_artifacts` (phase-6-tail step `6.7z`) + `mission_ideas` ChromaDB collection + `mission_artifacts_index` table + `legacy_pre_idea_dedup` + `legacy_pre_inheritance` |
| T6B | P5 | `packages/vecihi/src/vecihi/prior_art.py` (4 sources: HN, Wikipedia, Wayback, PH RSS) + `find_prior_art` LLM tool + step `1.0 prior_art_search` (researcher agent reused) + `prior_art_min_coverage` post-hook + `prior_art_cache` table + augmented `1.14`/`2.1` consumes |
| T6C | C18 | `init_mission_github_repo` mechanical action (gh-cli, fail-soft) + step `6.7 init_github_repo` + `/github` Telegram cmd (view/init/visibility) + `missions.github_repo_url` col + `legacy_pre_github_init` |

**Founder locks applied at merge:**
- T6C visibility default flipped `private` → `public` per founder Q (post-merge fix in `38e64b7`)
- T6C owner default = current `gh` user (msakirc) — already correct

**Skipped** per Q2 lock: B11 publish-spec-as-MCP (drop entirely).

**~90 new tests** — all green except the visibility-default test that was fixed post-merge.

---

## What's next — Tier 7 (auto-dispatched)

Per master synthesis §5:
- **T7A — B12** quarterly "what if just bash" audit framework
- **T7B — C21** bundle-quality regression fixtures vs Paraflow goldens

**Founder F1-F4 deferred to Z2 per founder lock.** Tier 7 unblocked.

---

## Tier 6 known follow-ups (verify-on-mission)

### T6A
- Telegram surface for `needs_review` decision (Continue / Branch from #N / Abort) not yet wired into `_pending_action[chat_id]` — mechanical action returns `needs_review` correctly but Telegram callback handler missing
- Reuse / diverge button responses for `prior_mission_hints.md` not persisted — file write is the surface
- `domain_keywords` extraction = rule-based regex (bold-bullet brand keywords + heading-word fallback). spaCy/LLM rejected for cold-path speed
- `founder_id` source = `'default'` literal until founder profiles land
- `index_idea_fingerprint` is SEPARATE from `find_similar_missions` — caller must explicitly dispatch after Continue/new-mission confirm to avoid self-match

### T6B
- `find_prior_art` is async; tool wrapper at `src/tools/prior_art.py` is also async
- All 4 sources: HN Algolia + Wikipedia REST + Wayback Availability/CDX + Product Hunt RSS (no public REST)
- Cache TTL 168h (7 days); configurable via `find_prior_art(ttl_hours=...)`
- Telemetry plan (yazbunu `z1_prior_art` event) not yet wired
- Reviewer-prompt criteria additions to `1.13 research_quality_review` not yet added

### T6C
- Visibility default = `public` (founder lock); override via `KUTAI_GITHUB_DEFAULT_VISIBILITY` or payload
- Owner default = current `gh` user; override via `KUTAI_GITHUB_ORG`
- Repo name: `kutai-mission-<id>-<charter_slug>` (kebab, ≤30 chars from charter title; falls back to `unnamed`)
- Separate `mission_<id>/.git_export/` repo (not phase-7's working repo) — write-once snapshot
- Artifact whitelist for initial commit: charter, PRD, ADRs, design_tokens, screen_plans, premortem, compliance_overlay, prior_art_report, visual_brief, intake_todo + recursive `adr/`, `.style/`, `.web/`, `screen_plans/`, etc.
- Visibility flip uses `--accept-visibility-change-consequences` — gh CLI may prompt on some versions; surface in fail-soft path
- DB persistence of `github_repo_url` is best-effort try/except — verify on real mission

---

## Pre-existing test failures (unchanged)

- `packages/mr_roboto/tests/test_clarify_variant.py::test_clarify_variant_choice_sends_keyboard`
- `packages/mr_roboto/tests/test_notify_user.py::test_notify_user_sends_message`
- `tests/test_i2p_v3.py` step-coverage (older missions)
- `tests/infra/test_pick_log.py` (5 unrelated)

---

## Quick orientation check for next session

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
rtk git log --oneline -10
rtk timeout 200 pytest tests/i2p/ tests/mechanical/ tests/telemetry/ packages/mr_roboto/tests/ packages/vecihi/tests/ tests/test_streaming_guards.py -q
ls .claude/worktrees/  # only z0-mission-preflight should remain (+ T7 if still running)
```

Expected: 609 pass / 2 pre-existing fail, on `main`, clean tree.
