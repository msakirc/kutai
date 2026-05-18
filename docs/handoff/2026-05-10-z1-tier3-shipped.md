# Handoff — Z1 Tier 0/1/2/3 shipped, Tier 4-7 pending

**Date:** 2026-05-10
**Session focus:** Z1 (i2p phases 0-6) implementation via parallel worktree subagents.
**Status:** Tier 0+1+2+3 merged to `main`. 261 tests green. 56 commits ahead of origin (not pushed). Founder said "wait" before Tier 4 dispatch — honor that.

---

## Read FIRST

1. **Memory locks (binding):**
   - `memory/project_z1_strategic_locks_20260509.md` — 5 strategic decisions: real-product buyer / skip-MCP / Telegram+web-preview / configs+more-mechanicals / `gorsel_ustasi` image provider abstraction
   - `memory/project_z1_merge_pattern_20260510.md` — worktree conflict resolution (take-both for db.py, reset-and-inject for mr_roboto/__init__.py, validate AST+JSON before commit, partition tier-3 agents by step IDs)
   - `memory/feedback_zone_deep_dive.md` — every zone needs v2+ planning passes
   - `memory/feedback_no_agent_modes.md` — never consolidate via mode/phase flags

2. **Source-of-truth docs:**
   - `docs/i2p-evolution/01-pre-code-master-synthesis.md` — 40-item roadmap across 8 tiers; **§6.5 strategic locks**; Z1/Z2 boundary
   - `docs/i2p-evolution/01-pre-code-paraflow-and-competitors.md` — Paraflow ground truth + 30-tool competitor matrix
   - `docs/i2p-evolution/01-pre-code-plan-v3.md` — concrete schemas + reviewer prompt deltas
   - `docs/i2p-evolution/01-pre-code-additions-claude.md` — A1-A8 (reverse-pitch / non-goals / boring-tech / interview-script / attention-budget / premortem / idea-dedup / cost-curve)
   - `docs/i2p-evolution/competitor-research/_landscape-roundup.md` — competitor matrix

3. **Paraflow ground truth:** `C:\Users\sakir\Dropbox\Workspaces\Bilinc\main\paraflow\` (TruthRate output: charter / personas / PRD / user_flow / 24 screen plans / 24 HTMLs / light+dark style guides). This is the artifact-quality benchmark.

---

## What shipped (commits on `main`, not pushed)

| Tier | Items | Step IDs added |
|---|---|---|
| 0 | P7 schema versioning + B10 rework metric | `verify_schema_version` action; `phase_7_rework_loops` column; `/rework` Telegram cmd |
| 1 | C1+A9+C6+A14 charter consolidation + B1 todo gate + A1 reverse-pitch + B7+C16 visual ingest | `0.0z`, `0.0a`, `0.1` (paraflow-shape charter), `ingest_visual` |
| 2 | P3+C7+A8 ADRs + P4+A2 falsification+non-goals + C2+A4 PRD competitor + interview-script | `4.1-4.10` ADR shape, `4.2a`, `0.0c`, `0.6a`, `1.4a`, falsification triple on phase-3 |
| 3 | C5+C8+A13+A16 design tokens + C20 taste + C12 surfaces + C4+A12 user_flow + C18 shell + C3+A10 per-screen + C14 states + C9+A11 HTML | `5.0`, `5.0a`, `5.0b`, `5.0c`, `5.0d`, `5.20a/b`, `5.30a/b` (chunked) |

Mechanical actions added (~20+): `verify_charter_shape`, `verify_reverse_pitch_shape`, `generate_intake_todo`, `verify_adr_shape`, `verify_adr_register`, `verify_cost_curve_present`, `verify_falsification_present`, `verify_non_goals_shape`, `check_against_non_goals`, `verify_competitive_positioning_shape`, `verify_interview_script_shape`, `request_interview_data`, `verify_taste_emphasis_shape`, `verify_design_tokens_shape`, `derive_token_tag_signature`, `verify_surfaces_shape`, `verify_user_flow_shape`, `verify_screen_inventory_shape`, `verify_shared_shell_shape`, `verify_screen_plan_shape`, `verify_html_prototype_shape`, `verify_screen_consistency`.

Legacy gates: `legacy_pre_p7`, `legacy_pre_charter`, `legacy_pre_adr`, `legacy_pre_falsification`, `legacy_pre_non_goals`, `legacy_pre_competitive_positioning`, `interview_skip_reason`, `legacy_pre_design_tokens`, `legacy_pre_user_flow`, `legacy_pre_per_screen_plans`. All idempotent ALTER + backfill to 1 for existing missions.

Reviewer 5.10 instruction now contains: design checks + T3A design-tokens + T2B non-goals + T3C per-screen+HTML+consistency (one coherent string).

---

## What's next — Tier 4-7

Per master synthesis §5:

- **Tier 4 — iteration loop:** C11+A15 `regen_with` per artifact / C19 bundle-level regen / B2 bidirectional asset↔spec propagation (Paraflow's actual differentiator) / C10 tunneled preview EMIT-ONLY (host = Z2) / C17+A20 two-way HTML edit reflection (Onlook `data-oid` pattern)
- **Tier 5 — compliance + memory + critic:** P6 compliance fingerprint / A5 founder attention budget / A6 premortem / B5 spec-stays-alive (Augment Intent) / B4 Critic gate (Devin pattern) / B3 streaming post-processor guards (v0 pattern)
- **Tier 6 — cross-mission + ecosystem:** P9+A7 cross-mission inheritance + idea dedup (vector-search past `idea_brief`) / P5 web-grounded prior art / C18-github init at end of phase 6 (deferred per Q2 — skip B11/B9 MCP)
- **Tier 7 — standing:** B12 quarterly "what if just bash" audit / C21 bundle-quality regression vs Paraflow goldens

**Founder F1-F4 still open** (web preview host strategy / `gorsel_ustasi` MVP providers / preview viewer scope / preview auth). Not blocking Tier 4-7 dispatch.

---

## Dispatch pattern (proven 7 times)

3 parallel general-purpose subagents per Tier, isolation: `worktree`, phase-scoped step-ID partitions to minimize JSON conflicts. Memory `project_z1_merge_pattern_20260510.md` has the conflict resolution recipe.

Each agent prompt must include:
- Required reads (master synthesis + relevant addition docs + paraflow files when shape-relevant)
- "WORK IN PROVIDED WORKTREE — verify pwd before Write" (P7 worker once landed in main repo)
- "Mechanical actions only (Q4 lock); no new agent configs"
- "`produces` paths `mission_{mission_id}/...` relative to WORKSPACE_DIR (no `workspace/` prefix)"
- "Conventional commits, no `--no-verify`, no remote push"
- Specific step-ID partition vs sibling agents
- "DO NOT touch Z2 packages (gorsel_ustasi, web preview, deterministic compile)"

After all worktrees report done:
1. `cd` to main repo (always verify `git branch --show-current` shows `main`)
2. `git merge --no-ff <worktree-branch>` sequentially
3. db.py: take both with python regex → check AST → manually fix try/except missing `pass`
4. mr_roboto/__init__.py: reset HEAD then `git show <branch>:packages/...` + extract handlers via regex (lookahead must include both `unknown` and `Unknown` mechanical-action fallback strings) + inject before `'    if action == "generate_intake_todo":'` anchor
5. i2p_v3.json: conflicts usually on reviewer instruction text or `_schema_version` lines. Take HEAD when only schema-version differs; manually merge reviewer instruction strings to keep all extension paragraphs. If sibling steps from worktree's flat `steps` array missing on main, inject via JSON walk.
6. test_reviewer_regression.py: extend `skip_stems` for new fixture stems that don't follow `{schema_version, payload}` envelope contract
7. Validate: `rtk python -c "import ast; ast.parse(...)"` for db.py + mr_roboto; `rtk python -c "import json; json.load(...)"` for i2p_v3.json; `rtk timeout 90 pytest tests/i2p/ tests/mechanical/ tests/telemetry/ packages/mr_roboto/tests/test_run.py -q`
8. Commit conflict resolution as merge commit body

---

## Pre-existing test failures (ignore — not introduced by Z1)

- `packages/mr_roboto/tests/test_clarify_variant.py::test_clarify_variant_choice_sends_keyboard` — Telegram mock bug
- `packages/mr_roboto/tests/test_notify_user.py::test_notify_user_sends_message` — same
- `tests/test_i2p_v3.py::test_v3_all_steps_have_difficulty / tools_hint / difficulty_distribution` — step `9.4a` missing fields
- `tests/infra/test_pick_log.py` — 5 failures unrelated

---

## Known gaps / follow-ups

- **Vision capability for B7+C16:** `Cap.VISION` wired but no GGUF in `models.yaml`. Add Qwen2-VL/LLaVA/Moondream OR confirm cloud discovery picks vision-capable models, else `ingest_visual` fail-fast with `vision_capability_unavailable`.
- **P4 schema decision pending:** master synthesis didn't pin which P4 shape (simple triple shipped vs richer `failure_mode` object from plan-v3 with `hypothesis / kill_threshold / we_dont_know_yet` + separate `falsification_register` artifact at `3.11a`). Founder hasn't chosen.
- **`check_against_non_goals` calibration:** stopword aggressiveness tuned on synthetic fixtures only. Re-tune after first real mission.
- **Step-ID drift in T2A spec vs reality:** `4.6 api_contracts` actually `4.6 auth_system_design` (api_contracts at `4.5b`); `4.8 service_topology` folded into `4.3 system_architecture_design`. Documented in `scripts/z1_tier2_patch_i2p_v3.py`.
- **Legacy 5.1-5.11b preserved via skip_when** in T3C (avoids cascade into 11+ depends_on rewrites). Sweep auto-rescue handles for new missions.
- **`_z1_tier3_patched: true` marker** at i2p_v3.json top level — diagnostic, harmless.

---

## Holding instruction

Founder said "wait a minute before dispatching next when these are done." Do NOT dispatch Tier 4 without explicit go.

---

## Quick orientation check for next session

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
rtk git log --oneline -10
rtk git status
rtk timeout 90 pytest tests/i2p/ tests/mechanical/ tests/telemetry/ packages/mr_roboto/tests/test_run.py -q
ls docs/i2p-evolution/
ls .claude/worktrees/  # 7 stale worktrees can be pruned: T0(B10)+T1(charter+visual)+T2(ADR+P4+C2)+T3(tokens+flow+screen)
```

Expected: 261 tests pass, on `main`, ~56 commits ahead origin/main, clean tree.
