# Handoff — Z1 Tier 0/1/2/3/4 shipped, Tier 5-7 pending

**Date:** 2026-05-10
**Session focus:** Z1 Tier 4 (iteration loop) via 3 parallel worktree subagents.
**Status:** Tier 0+1+2+3+4 merged to `main` and pushed. 389/391 tests green (2 pre-existing failures unchanged).

---

## What shipped this session (Tier 4)

| Subagent | Items | Mechanical actions added |
|---|---|---|
| T4A | C11+A15 + C19 | `regen_artifact`, `regen_bundle` + `regen_log` table + `/regen` Telegram cmd + `regen:` callback |
| T4B | B2 + C17+A20 | `propagate_asset_change`, `propose_spec_patch_from_html_diff`, `annotate_html_oids` + `/edit_html`, `/propagate` cmds + step `5.30c` + `legacy_pre_html_oids` gate |
| T4C | C10+A19 | `emit_preview_url`, `kill_preview_url` + `preview_log` table + `preview_url`/`preview_started_at` columns + `/preview`, `/preview_off` cmds + step `5.40` + `legacy_pre_preview_url` gate |

**Merge commits on main:** `5d8ee0a` (T4A), `a22bba1` (T4B), `3afe05e` (T4C).

**33 new tests:** 10 (T4A) + 15 (T4B) + 8 (T4C). All green.

---

## What's next — Tier 5-7

Per master synthesis §5:

- **Tier 5 — compliance + memory + critic:** P6 compliance fingerprint / A5 attention budget / A6 premortem / B5 spec-stays-alive (Augment Intent) / B4 Critic gate (Devin) / B3 streaming post-processor guards (v0)
- **Tier 6 — cross-mission + ecosystem:** P9+A7 cross-mission inheritance + idea dedup / P5 web-grounded prior art / C18 github init at end of phase 6 (skip B11/B9 MCP per Q2 lock)
- **Tier 7 — standing:** B12 quarterly "what if just bash" audit / C21 bundle-quality regression vs Paraflow goldens

**Founder F1-F4 still open** (web preview host strategy / `gorsel_ustasi` MVP providers / preview viewer scope / preview auth). Not blocking Tier 5-7 dispatch.

---

## Tier 4 known follow-ups (verify-on-mission)

- **Regen versioning scheme picked: versioned siblings, canonical path holds latest.** `mission_42/charter.md` (canonical) + `mission_42/charter.v2.md` (snapshot). Validate on first founder regen.
- **Bundle axis registry seeded with `tone`, `density`, `scope`.** Add more axes via `_KNOWN_AXES` in `packages/mr_roboto/src/mr_roboto/regen.py`.
- **`_invoke_emitter` is a private shim** with deterministic fallback when coulson isn't importable. Production path delegates to `coulson.execute("overhead", task)`. Re-verify on first live regen.
- **`data-oid` annotation parser: BeautifulSoup4** (transitive dep, no new requirement). Format: `"<artifact_slug>:<section>"`. Idempotent.
- **Document-upload glue for `/edit_html` deferred.** `cmd_edit_html` stashes the pending action; the actual document handler that pairs the upload with `propose_spec_patch_from_html_diff` lives outside T4B's scope (notification path is `general_beckman`-owned). Add a follow-up task to wire the upload → proposer flow.
- **Inline "🎯 propagate change" button on artifact-emit notifications deferred** for the same reason — notification points are not centralized yet.
- **`emit_preview_url` is fail-soft.** When `KUTAI_PREVIEW_PROVIDER` env var unset OR `cloudflared` binary missing, writes a `pending: hosting deferred to Z2` placeholder. Real hosting = Z2.
- **Step `5.40 emit_preview_url` depends_on `5.30c`** (not `5.30b.verify_shape`) — picked the later predecessor at merge time so propagation runs after annotation.

---

## Pre-existing test failures (unchanged from prior handoff)

- `packages/mr_roboto/tests/test_clarify_variant.py::test_clarify_variant_choice_sends_keyboard` — Telegram mock bug
- `packages/mr_roboto/tests/test_notify_user.py::test_notify_user_sends_message` — same
- `tests/test_i2p_v3.py::test_v3_all_steps_have_difficulty / tools_hint / difficulty_distribution` — step `9.4a` missing fields (older missions)
- `tests/infra/test_pick_log.py` — 5 failures unrelated

---

## Holding instruction

None. Tier 5 dispatch ready when founder gives the go.

---

## Quick orientation check for next session

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
rtk git log --oneline -10
rtk git status
rtk timeout 90 pytest tests/i2p/ tests/mechanical/ tests/telemetry/ packages/mr_roboto/tests/ -q
ls docs/i2p-evolution/
ls .claude/worktrees/  # only z0-mission-preflight should remain
```

Expected: 389 pass / 2 pre-existing fail, on `main`, clean tree.
