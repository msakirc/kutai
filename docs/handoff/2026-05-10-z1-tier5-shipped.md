# Handoff — Z1 Tier 0/1/2/3/4/5 shipped, Tier 6-7 pending

**Date:** 2026-05-10
**Session focus:** Z1 Tier 5 (compliance + memory + critic) via 3 parallel worktree subagents.
**Status:** Tier 0+1+2+3+4+5 merged to `main`. 498/500 tests pass (2 pre-existing failures unchanged).

---

## What shipped this session (Tier 5)

| Subagent | Items | Mechanical actions |
|---|---|---|
| T5A | P6 + A5 | `compliance_fingerprint_collection`, `compliance_template_present`, `compliance_blocker_check`, `attention_check`, `attention_debit` + `compliance_template_render` LLM tool + `/budget` Telegram cmd + `0.4a`, `1.11a` steps + `legacy_pre_compliance` + `founder_attention_budget_minutes` col + `founder_attention_log` table |
| T5B | A6 + B5 | `verify_premortem_shape`, `spec_consistency_check` + `6.5z failure_premortem` step + 6 wave-start steps `7.0z`-`12.0z` + `legacy_pre_premortem` + `legacy_pre_spec_alive` |
| T5C | B4 + B3 | `critic_gate` (post-hook on `git_commit` + `notify_user`) + `coulson/streaming_guards.py` pipeline (4 guards) + `critic_log` + `streaming_guard_log` tables + `legacy_pre_critic_gate` |

**Merge commits on main:** `f9c70a4` (T5A — see git log), `9e71ad8` (T5B), `0f315c4` (T5C). Final commit before push: `0f315c4`.

**~109 new tests:** 35 (T5A) + 34 (T5B) + 40 (T5C). All green.

---

## What's next — Tier 6-7

Per master synthesis §5:

- **Tier 6 — cross-mission + ecosystem:** P9+A7 cross-mission inheritance + idea dedup (vector-search past `idea_brief`) / P5 web-grounded prior art / C18 github init at end of phase 6 (skip B11/B9 MCP per Q2 lock)
- **Tier 7 — standing:** B12 quarterly "what if just bash" audit / C21 bundle-quality regression vs Paraflow goldens

**Founder F1-F4 still open** (web preview host strategy / `gorsel_ustasi` MVP providers / preview viewer scope / preview auth). Not blocking Tier 6-7 dispatch.

---

## Tier 5 known follow-ups (verify-on-mission)

### T5A
- Founder-signoff gating in `compliance_blocker_check` deferred (current rule = "rendered file exists"). `founder_signoffs` table is a follow-up.
- `attention_debit` is wired and dispatch-callable but no auto-debit hook on Telegram reply receipt — caller must dispatch a mechanical `attention_debit` task.
- Pre-hook expander surface deferred — gate currently lives inside `mr_roboto.clarify.clarify()`; centralized expander pre-hook would be cleaner but T5A kept surgical.
- Templates beyond `default/en/privacy_policy.md.j2` are stubs — hand-curate per spec.
- z0 wiring: no runtime z0 module exists yet; spec at `docs/i2p-evolution/z0-mission-preflight.md` §E2 documents budget declaration; future z0 implementation writes to `missions.founder_attention_budget_minutes`.

### T5B
- Wave-start naming convention: `<N>.0z` (e.g. `7.0z spec_consistency_check`). Mirrors T3-T4 mechanical-sibling marker `z`.
- Drift detection: rule-based (4 rules R1-R4: stack_drift, token_drift, surface_drift, non_goal_drift). LLM reviewer at 6.6 owns judgment; B5 only surfaces.
- R5 (charter brand drift) deferred — mentioned in module docstring; lands when first false-positive on real mission tells us what brand-keyword exclusion looks like.
- Compliance-overlay rule deferred — fifth drift rule once T5A's `compliance_blocker_check` telemetry lands.

### T5C
- Streaming integration site: `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py` `_stream_with_accumulator` (after `if delta.content:`).
- `streaming_guard_log` table created but no DB writer wired yet — outcomes surface only via `logger.warning`. Add writer as small follow-up.
- Critic gate uses `LLMDispatcher.request(category=OVERHEAD)` directly (per spec "second LLM, smaller/cheaper"); could be swapped to full Beckman-mediated calls.
- Conftest collision worked around by `os.environ.setdefault("KUTAI_CRITIC_GATE", "off")` in worktree-root `conftest.py`. Production must keep the var unset (or set `on`).
- Step-level workflow integration with standalone `critic_gate` action deferred — append `payload.action == "critic_gate"` siblings to existing irreversible-action steps as needed.

---

## Pre-existing test failures (unchanged from prior handoff)

- `packages/mr_roboto/tests/test_clarify_variant.py::test_clarify_variant_choice_sends_keyboard` — Telegram mock bug
- `packages/mr_roboto/tests/test_notify_user.py::test_notify_user_sends_message` — same
- `tests/test_i2p_v3.py` — 3 step-coverage tests for older missions (`9.4a` missing fields)
- `tests/infra/test_pick_log.py` — 5 unrelated

---

## Quick orientation check for next session

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
rtk git log --oneline -10
rtk git status
rtk timeout 150 pytest tests/i2p/ tests/mechanical/ tests/telemetry/ packages/mr_roboto/tests/ tests/test_streaming_guards.py -q
ls .claude/worktrees/  # only z0-mission-preflight should remain
```

Expected: 498 pass / 2 pre-existing fail, on `main`, clean tree.
