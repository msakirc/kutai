# Z4 — Visual review (v2)

**Supersedes** [04-build-visual-review.md](04-build-visual-review.md) (v1, 2026-05-08). v1 wrote
the spec before Z1 (tunneled preview, ingest_visual, vision capability) and Z3 (run_axe /
run_lighthouse / preview_url emit-kill / cost_band / dial-aware auto_wire) shipped. v2 re-audits
end-to-end, fixes 8 stale claims and 6 schema-level errors, picks the posthook shape against the
actual `PostHookSpec` / `MissionDialContext` API, and lays out a batched tier plan that piggybacks
on existing infra.

## What changed since v1

Hard-audit of the tree on 2026-05-15 (`grep -rn` over `packages/` and `src/`; read of
`expander.py`, `posthooks.py`, `run_axe.py`, `preview_url.py`, `emit_preview_url.py`,
`ingest_visual.py`, `vision.py`, `visual_reviewer.py`, `review_density.py`, `inject_lessons.py`,
`capabilities.py`).

### v1 claims to drop

| v1 claim | Actual state |
|---|---|
| "No screenshot harness wired into i2p_v3" | **Partly true.** No `capture_screenshots` verb, but `emit_preview_url` / `kill_preview_url` mechanicals are live (Z1 T4C `packages/mr_roboto/src/mr_roboto/emit_preview_url.py`). They write `.preview/last_preview_url.txt` so downstream consumers can resolve the URL without re-spawning. |
| "Playwright runner is on the W1 real-tools workstream; needed here" | **Shipped.** `packages/mr_roboto/src/mr_roboto/record_demo.py:319` shells `npx playwright test` inside the workspace container; `run_axe` drives axe-core; `run_lighthouse` drives Lighthouse — both against the preview URL. Playwright is in dev deps. |
| "No vision model selection for diff judgment" | **Already solved.** `ingest_visual.py` calls `src/tools/vision.py::analyze_image`, which submits a spec with `agent_type="visual_reviewer"` + `needs_vision=True` through `general_beckman.enqueue(...)` (singular admission). Fatih Hoca's capabilities.py:270 maps `visual_reviewer` to `Cap.VISION: 1.0` and ranking.py:535 hard-requires VISION-capable models. Failure mode `vision_capability_unavailable` is already a structured signal. **No second bench needed.** |
| "Storybook stories per primitive/composite act as canonical reference" | **Dead.** [Z1 strategic locks](../../../memory/project_z1_strategic_locks_20260509.md) commit to Telegram + web preview only. Baselines must come from elsewhere: design tokens, the previous mission's captured-and-approved frame, or founder-uploaded reference (`ingest_visual` output at `mission_{id}/.intake/visual_brief.md`). |
| "visual_reviewer planned to be killed per kill-agents work; survives as a profile" | **Still a class.** `src/agents/visual_reviewer.py::VisualReviewerAgent`, 3 iters, tools `[read_file, file_tree, web_search]` — no actual screenshot capability. Picks: 0 in i2p (`agents_required` declares it at `i2p_v3.json:18` but no step routes to it). v2: keep class for ad-hoc `/task` use; route i2p visual review through a **mechanical** post-hook, not the agent. |
| "Severity gating via blockers rule" | **Pattern ready but vocab different.** Z3 `accessibility_review` (posthooks.py:502) doesn't use a `blockers: {field, levels}` dict — that's not a real schema. The PostHookSpec carries `default_severity` (`blocker`/`warning`); the verb returns `verdict: pass|fail` after threshold-checking its own findings (run_axe.py:208: `has_blocker = any(f["severity"]=="blocker" for f in findings)`). Findings carry `severity` ∈ `{blocker, warning, info}` (run_axe.py:45). v2 lock: match this exact shape — not the v1 4-tier `critical/high/medium/low`. |
| "Per-mission baseline images captured from sprint 0" | **Half-true.** No sprint-0 capture exists. `ingest_visual` produces `visual_brief.md` per mission — that's structural reference, not pixel baseline. First captured frame in a mission becomes the implicit baseline once founder approves; subsequent runs diff against it. |
| "Vision-model diff — bench Claude Vision vs GPT-4o vs ResNet" | **Don't bench again.** Reuse `analyze_image` + Fatih Hoca selector. Cost ceiling via existing `cost_band="heavy"` + dial-aware auto_wire (callable). |

### v2 schema corrections (these had wrong details in the first v2 draft)

These are mistakes in the prior v2 surface I'd written, caught by the deeper audit:

1. **`cost_band` literal is `cheap | moderate | heavy`** (posthooks.py:155 `Literal`). v2 draft wrote `"high"` — would have failed type check.
2. **`PostHookSpec` has no `blockers`, `requires_inputs`, `skip_when_pending_inputs`, or `auto_wire_on_produces` fields.** Real fields: `kind`, `verb`, `default_severity`, `cost_band`, `auto_wire_triggers`, `description`. Pending-URL handling is the verb's responsibility (run_axe.py:134 `if not _is_real_url(preview_url): ... skipped=True`); not a registry field.
3. **`auto_wire_triggers` callable form** receives `MissionDialContext | dict | None`; resolved via `_dial_get(ctx, key, default)` helper (posthooks.py:195). The dial-off case returns `[]` to suppress wiring entirely.
4. **Severity vocab inside findings**: `blocker | warning | info` — matches run_axe. Driver-level severities (critical/serious/moderate/minor in axe) are mapped via `_impact_to_severity`. visual_review needs an equivalent: map model's 4-tier verdict → 3-tier finding severity.
5. **Dial registration**: `visual_dial` doesn't exist. Add to `_ALLOWED` (`src/workflows/review_density.py:30`) + `ReviewDensityDials` dataclass (line 42) + `MissionDialContext` (posthooks.py:18). `accessibility_dial` is the precedent.
6. **`analyze_image` signature** takes ONE filepath (vision.py:7). For (captured, baseline) pair diff, either extend to accept `list[str]` (build multi-image messages array) or write a sibling `compare_images`. Extending is cleaner — single tool serves single-image audit + pairwise diff. visual_brief.md ingest already passes a single image, so backward-compat via positional kwarg.

## What stays from v1

- Output shape ≈ structured findings list (component, kind, severity, description, expected, observed) — but recoded to match run_axe finding dict shape: `{severity, file, url, impact, why, source}` (run_axe.py:86). visual_review adds: `kind`, `component`, `expected`, `observed`, `breakpoint`, `route`.
- Breakpoint set (375 / 768 / 1280 / 1920) is right.
- "No pixel-perfect compare; semantic compare via vision model" is right.
- Founder loop (👍 / wrong-this-is-fine / broken → mission_lessons) is right; now wireable via `upsert_mission_lesson(stack, domain, pattern, ...)` (`src/infra/mission_lessons.py`) with `dedup_key` schema. STACK_BLOCKS / `inject_lessons` (mr_roboto) read back top-N.
- "Phase 1 emulated, phase 2 native-device" right; native-device handoff to [05-build-mobile-track.md](05-build-mobile-track.md).

## Current adjacent infra (load-bearing)

- `packages/mr_roboto/src/mr_roboto/preview_url.py::is_real_url` — pending-URL filter (Z3 R2 consolidated; commit ecff23c7 extended to schemathesis + lighthouse)
- `packages/mr_roboto/src/mr_roboto/emit_preview_url.py` / `kill_preview_url.py` — Z1 T4C cloudflared tunnel; writes `mission_{id}/preview_url.txt` AND `mission_{id}/.preview/last_preview_url.txt`. Idempotent via `_kill_prior_tunnel`. Persists `preview_log` + `missions.preview_url` rows.
- `packages/mr_roboto/src/mr_roboto/run_axe.py` — **canonical template for `visual_review` verb shape**: pending-URL soft-skip → tool-availability soft-skip → subprocess (or vision call) with timeout → parse output → severity-mapped findings → `{verdict, findings, skipped, reason}` envelope
- `packages/mr_roboto/src/mr_roboto/run_lighthouse.py` — same template; second precedent
- `packages/mr_roboto/src/mr_roboto/record_demo.py` — already drives `npx playwright test` inside workspace container; webm output. Validates container-playwright pattern works. (v2 chooses **host playwright** instead since cloudflared URL is public — simpler + faster + no container assumption.)
- `packages/mr_roboto/src/mr_roboto/ingest_visual.py` — vision-capable model call (`_call_vision` → `analyze_image`); structured failure modes (`vision_capability_unavailable`, `no_images_readable`, per-image error); `SCHEMA_VERSION = "1"` frontmatter pattern
- `src/tools/vision.py::analyze_image` — single-image vision wrapper; goes through `general_beckman.enqueue(raw_dispatch=True, await_inline=True)`; degenerate-output check via `dogru_mu_samet` wired
- `packages/general_beckman/src/general_beckman/posthooks.py::POST_HOOK_REGISTRY` — registry pattern; 6 review kinds; insertion order is dispatch order (expander prepends matches)
- `packages/general_beckman/src/general_beckman/posthooks.py::_NO_POSTHOOKS_AGENT_TYPES` — judge-of-judge exclusion list; mechanical is already excluded (good — visual_review-as-mechanical doesn't get graded)
- `src/workflows/engine/expander.py::_auto_wire_posthooks` (line 89) — iterates registry, resolves triggers (static or callable) with mission's `MissionDialContext`, fnmatch-checks against flattened `produces`, prepends matches
- `src/workflows/review_density.py` — `_ALLOWED` dial vocab + `get_dials(mission_id)` + `set_dial(mission_id, key, value)`. Used by Telegram `/density` and by expander wire-up at expand-time. (`migrations: 2026-05-12-missions-review-density` adds `review_density_json` column.)
- `src/infra/mission_lessons.py::upsert_mission_lesson` (key on `_dedup_key(stack, domain, pattern)`; supports `suppressed=1` founder-mute); `top_mission_lessons(stack, domain, limit)` for read-back
- `packages/mr_roboto/src/mr_roboto/inject_lessons.py` — STACK_BLOCKS injection at mission start (Z2 T4C); Coulson renders "## Watch out for" block from `context.lessons_top_n`
- `packages/fatih_hoca/src/fatih_hoca/capabilities.py:270` `visual_reviewer` row with `Cap.VISION=1.0`; `ranking.py:535 "vision": {"visual_reviewer"}` — vision sub-call routes correctly via `agent_type="visual_reviewer"` even when the outer task is `agent_type=mechanical`

## Gaps that remain

### Mechanical-side gaps

**G1. No `capture_screenshots` verb.** Required behaviors:
- Resolve preview URL from `mission_{id}/.preview/last_preview_url.txt` (already written by `emit_preview_url`), fall back to `mission_{id}/preview_url.txt`.
- Soft-skip when `not is_real_url(url)` (mirror run_axe.py:134).
- For each route in step's `produces.routes` (new field on step JSON) OR inferred from `produces` globs (`pages/foo.tsx` → `/foo` for Next; convention table per framework), navigate via Playwright.
- Per breakpoint in `{375, 768, 1280, 1920}` (configurable via `.kutay/visual.yaml`), set viewport, wait `networkidle` + optional step-declared `ready_signal` (CSS selector `[data-visual-ready]`).
- Inject deterministic CSS: `* { animation: none !important; transition: none !important; caret-color: transparent !important }` to defuse animation/timing flake. `prefers-reduced-motion: reduce` emulation.
- Capture to `mission_{id}/.visual/captured/{step_id}/{route_slug}_{breakpoint}.png`.
- Return `{ok, captured_paths, skipped, reason}` envelope.

**G2. No `visual_review` mechanical verb.** Mirrors run_axe shape exactly:
- Resolve preview URL + captured frame list + baseline frame list (`mission_{id}/.visual/baseline/`).
- Resolve design_tokens + screen_specifications artifacts from store (pattern: `social_preview_check.py:42 _load_artifact`).
- For each (captured, baseline) pair: ONE vision call via `analyze_image(filepaths=[captured, baseline], question=DIFF_PROMPT)` returning structured JSON.
- For frames without baseline (first frame in mission): single-image audit `analyze_image(filepaths=[captured], question=AUDIT_PROMPT)`. Same call site, different prompt.
- Parse JSON → finding dicts shaped `{severity, file, url, impact, why, source, kind, component, breakpoint, route, expected, observed}`.
- Apply local severity rules over model-emitted hints: color delta-E > 8 → blocker; layout shift > 4px → blocker; named-component-missing → blocker; brand-token violation → blocker; typography off-scale → warning; shadow/micro-spacing → info.
- `has_blocker = any(f["severity"]=="blocker" for f in findings)`; `verdict = "fail" if has_blocker else "pass"`.
- Soft-skip path for `vision_capability_unavailable` (reuse `_is_vision_capability_unavailable` from `ingest_visual.py:155` — import, don't copy).
- Soft-skip path for no captured frames (G1 already skipped).
- Per-call timeout (default 60s) and global per-mission ceiling.

**G3. `analyze_image` extension to accept multi-image.** Current signature `analyze_image(filepath: str, question: str)`. Extend to `analyze_image(filepaths: list[str] | str, question: str)` — when list, assemble `content` array with multiple `image_url` entries. Backward-compatible (single-string still works). Falls under Z4 since visual_review is its first multi-image consumer; ingest_visual stays single-image.

**G4. No baseline store layout.** v2 lock:
- **Per-mission baselines** live at `mission_{id}/.visual/baseline/{route_slug}_{breakpoint}.png`. Created by founder action only (no auto-promotion of captured → baseline; first-frame audit is single-image against tokens until founder approves).
- **Cross-mission baselines** live at repo-root `.visual_baseline/{component}_{breakpoint}.png`, version-controlled, refreshed when design tokens change. Trigger: hash `design_tokens.json` at baseline-extract time; mismatch → regenerate.
- No Storybook (Z1 lock).
- Disk cleanup: `mission_{id}/.visual/captured/{step_id}/` retained per mission for replay; `kill_preview_url` does NOT touch `.visual/`.

**G5. No `visual_review` PostHookSpec entry.** Concrete row to add to `posthooks.py::POST_HOOK_REGISTRY`:

```python
# Z4 T3A — visual_review against tunneled preview.
"visual_review": PostHookSpec(
    kind="visual_review",
    verb="visual_review",
    default_severity="blocker",
    cost_band="heavy",
    # Callable: when visual_dial=on → frontend globs; else empty.
    # Accepts MissionDialContext OR dict OR None.
    auto_wire_triggers=lambda ctx: (
        ["*.tsx", "*.jsx", "*.vue", "*.svelte"]
        if _dial_get(ctx, "visual_dial", "off") == "on"
        else []
    ),
    description=(
        "vision-model diff against tunneled preview URL. "
        "blocker findings (color/layout/missing-component) → fail; "
        "warning (typography) / info (shadow) do not block."
    ),
),
```

**G6. No `visual_dial` in `MissionDialContext` or `ReviewDensityDials`.** Required edits:
- `src/workflows/review_density.py:30` `_ALLOWED["visual_dial"] = {"on", "off"}`
- `ReviewDensityDials` dataclass: add `visual_dial: str = "off"` (line 42)
- `get_dials` loop at line 86: add `"visual_dial"` to the iteration key tuple
- `packages/general_beckman/src/general_beckman/posthooks.py:18` `MissionDialContext` dataclass: add `visual_dial: str = "off"`
- `to_mission_dial_context` mapper (line 182): include `visual_dial=dials.visual_dial`
- Migration: existing rows return default `"off"`; no schema migration needed (JSON column).

**G7. No expander auto-wire for multi-file expansion.** When `multi_file_expansion=on`, the expander breaks template steps into per-file sub-tasks (Z3 T2). visual_review wired naively would fire per sub-file — N vision calls when 1 suffices. Mitigation: `integration_review` precedent — `auto_wire_triggers=[]` on registry, but expander injects on parent integration step. For visual_review: keep the auto_wire callable for non-multifile path; suppress wiring inside per-file sub-tasks; expander injects visual_review as sibling on the parent integration step. Add suppression flag to `_auto_wire_posthooks` or check `context.parent_step_id`.

**G8. No posthook-fail retry context injection.** On visual_review fail, source frontend step retries — but without the visual diff, agent re-emits same broken code. Apply.py's retry path already feeds posthook output into source step's failure context (Z2 T6 pattern). visual_review must emit findings in the standard envelope; apply.py picks it up. Verify: trace `_apply_posthook_verdict` → ensure findings appear in retry context as `visual_diff` (named) so agent prompt builder can render them.

**G9. No founder action for baseline approval.** Z2 T6 founder_actions pattern (z6_admission) tracks `founder_actions_emitted` but per-action registry not yet enumerated. Net-add Telegram command `/approve_baseline {mission_id} {step_id}` that copies `mission_{id}/.visual/captured/{step_id}/*.png` → `mission_{id}/.visual/baseline/`. Best place: extend `src/app/telegram_bot.py` command handlers OR add to z6 founder_actions registry (preferred if registry exists; check before T4).

**G10. No Telegram thread surface for visual diffs.** Each visual_review run emits a Telegram message with thumbnails + structured summary. Bandwidth concern: 4 breakpoints × thumbnail ≈ 200KB/step. Compress to ≤80KB each (PNG → WebP, max dim 600px) before send. Place: `src/app/telegram_bot.py` posthook notification path (search for existing `accessibility_review` surface — likely the precedent).

### Founder-territory gaps (unchanged from v1)

- Sprint-0 design baseline approval — first-frame manual approval via /approve_baseline.
- Severity calibration ("this color is fine") — founder reaction in Telegram → `upsert_mission_lesson(stack="frontend", domain="visual", pattern=f"{component}:{kind}", lesson="...", source_kind="founder_reaction")`. dedup_key collapses repeated lessons.
- Final taste call on ambiguous diffs — escalation via existing posthook needs_review surface (no new path).

## Posthook shape (locked, against real schema)

```python
# inserted into POST_HOOK_REGISTRY in posthooks.py (after Z3 T3 block):
"visual_review": PostHookSpec(
    kind="visual_review",
    verb="visual_review",
    default_severity="blocker",
    cost_band="heavy",
    auto_wire_triggers=lambda ctx: (
        ["*.tsx", "*.jsx", "*.vue", "*.svelte"]
        if _dial_get(ctx, "visual_dial", "off") == "on"
        else []
    ),
    description=(
        "vision-model diff against tunneled preview URL. "
        "Color/layout/missing-component blockers fail; typography "
        "warnings + shadow info do not block."
    ),
),
```

**No "judge of judge."** visual_review is mechanical (verb runs vision call internally + parses + severity-maps). `mechanical` is already in `_NO_POSTHOOKS_AGENT_TYPES` (posthooks.py:74), so no grader spawns on its output. dogru_mu_samet quality-check still runs on the vision response inside `analyze_image`.

## Alt shapes considered and rejected

1. **Visual review as the existing LLM agent (`visual_reviewer` class) wired into i2p directly.**
   Rejected: 3-iter ReAct loop returning free-form markdown, no structured findings, no severity gate. Tools `[read_file, file_tree, web_search]` don't include browser or screenshot. Severity downstream of unstructured prose is impossible to gate cleanly. Keep class for ad-hoc `/task visual_reviewer ...` (single-image audits, ad-hoc diagram review).

2. **Pixel-diff via OpenCV / ImageMagick / `pixelmatch-py`.**
   Rejected: alpha noise + antialiasing + Windows font hinting + cloudflared compression noise. Z1 render target is Telegram preview / cloudflared tunnel — we don't control render fidelity tightly enough for pixel compare to mean anything. Even with `pixelmatch-py`'s threshold knob, false-positive rate would dominate.

3. **Visual review as a sub-iteration guard (Z3 T2A self_critique shape) instead of a post-hook.**
   Rejected: visual review is heavy (browser + vision call ≈ 8–15s/breakpoint); sub-iter would multiply by retry count. Post-hook fires once per step; retries via apply.py path consume one cycle each. Z3 T2A's frozenset opt-out exists for exactly this kind of carve-out — visual_review opts OUT of sub-iter and IN to posthook.

4. **Per-component (Storybook) baselines.**
   Rejected: Z1 lock on Telegram + web preview only. No Storybook means no per-component canonical render. Per-route capture + optional CSS-selector crop (`page.locator('[data-visual-id=login-form]').screenshot()`) deferred to T5 if found needed.

5. **Fold visual_review into `accessibility_review`.**
   Rejected: different cost band (axe is fast subprocess; vision is LLM call), different blocker thresholds (a11y blocks on `serious`; visual blocks on `high` brand-cost vs legal-cost), different fixer (a11y → code; visual → code + token). Single verb would muddle severity rules. Sister posthooks, not merged.

6. **Capture inside workspace container (record_demo pattern).**
   Rejected: cloudflared URL is public; host playwright reaches it fine. Container adds: build dep on workspace dev server being inside the container (true today but fragile), extra spawn cost, and complicates founder-local debugging. record_demo containers because it needs the dev-server-in-container for E2E replay; visual_review only needs the public URL.

7. **Multi-file expansion: fire visual_review per sub-file.**
   Rejected: integration_review already established the "fire on parent integration step, not per sub-task" pattern (posthooks.py:469, `auto_wire_triggers=[]` + expander-injected). visual_review follows same; auto-wire path fires only when not in multi-file sub-task.

8. **Pin the vision model.**
   Rejected: capabilities.py already requires `Cap.VISION=1.0` for `visual_reviewer` agent_type. Fatih Hoca's selector picks the best vision-capable model available per cost. Pinning would defeat per-mission cost dial. Cost ceiling enforcement: ~4 routes × 4 breakpoints × $0.005 ≈ $0.08/step × 20 frontend steps ≈ $1.60/mission. Heavy cost_band keeps it dial-gated by default.

9. **CSS-selector-cropped per-component capture in T1.**
   Rejected for T1; useful later. Default = full-page-per-breakpoint. Per-component (selector-based) extension lands in T5 if mission types demand it.

## Wiring (end-to-end, post-shipping)

```
step "build login form" (frontend code emit, produces ["src/pages/login.tsx"])
  │
  │ expander auto-wires (visual_dial=on):
  │   post_hooks = ["grounding", "visual_review", ...]
  │
  ├── mr_roboto: capture_screenshots (sibling mechanical, injected after step)
  │    reads: mission_{id}/.preview/last_preview_url.txt
  │    inputs: produces.routes (declared) OR inferred ("/login" from path)
  │    output: mission_{id}/.visual/captured/{step_id}/login_{375,768,1280,1920}.png
  │    soft-skip: not is_real_url(url) → {ok: true, skipped: true, reason: "no preview"}
  │
  ├── posthook visual_review (cost-gated, dial-gated)
  │    reads: captured paths + baseline paths (if any) + design_tokens artifact
  │    one vision call per (captured, baseline) pair via analyze_image(list, DIFF_PROMPT)
  │    first-mission frames: single-image audit against tokens
  │    output: {verdict, findings, skipped, reason}
  │    on fail: source step retries with apply.py-injected `visual_diff` context
  │              (rendered in agent prompt: "## Visual findings:\n- /login@375 ...")
  │
  └── on first-pass success: founder Telegram thread
       message: thumbnails + structured summary
       reactions: 👍 / ❌ this color is fine / 🔧 broken
       reactions → mission_lessons.upsert(stack="frontend", domain="visual",
                                          pattern=f"{component}:{kind}",
                                          dedup_key=auto)
       founder /approve_baseline {mission_id} {step_id}:
         copy mission_{id}/.visual/captured/{step_id}/*.png
           → mission_{id}/.visual/baseline/
```

## Determinism / flake mitigations (T1+T2 must include)

- **Animations/transitions**: CSS injection (`* { animation: none !important; ... }`) + `page.emulate_media({prefers_reduced_motion: 'reduce'})` before capture.
- **Async data loading**: `await page.wait_for_load_state('networkidle')` + step-declared `ready_signal` CSS selector wait-for.
- **Time-dependent UI** (clocks, "ago" timestamps): inject `Date.now` stub via Playwright `add_init_script` with fixed epoch. Configurable per mission; default fixed-epoch enabled.
- **Random data** (UUID-shaped IDs, randomized order): pass deterministic seed via `add_init_script({Math.random})` stub. Configurable; default seed enabled.
- **Font rendering jitter** (sub-pixel hinting): accepted noise; severity rules don't trigger on micro-diffs.
- **Locale/timezone**: `page.context.set_default_timeout` + `Accept-Language: en-US`; document as known constraint.
- **Light/dark mode**: capture light by default. If step declares `produces.color_modes: ["light", "dark"]`, capture both per breakpoint (2× cost).
- **Auth-gated routes**: if step declares `produces.requires_auth: true`, run scenario from `tests/e2e/*.spec.ts` (record_demo precedent at `_resolve_scenario_path`) to authenticate, then navigate. Otherwise default = unauthenticated.

## Privacy / security

- **PII in screenshots**: test data may show realistic-looking emails/names. Mitigation: redact via Playwright `evaluate` overlay before capture (replace `[data-pii]` content with `█`); document as known risk for T2 unless step opts out.
- **Secrets in URLs**: cloudflared URL contains opaque random subdomain; no token in path. preview_url.txt may contain pending placeholder strings — sanitized by `is_real_url` filter before logging.
- **Vision API upload**: captured PNGs go to the vision provider (could be cloud — Claude Vision / GPT-4o per Fatih Hoca selection). Cloud uploads gated by `visual_dial=on` + `cost_band=heavy` + Telegram preview cost ack (z6).

## Migration / order of operations

Net-add only. No deprecation.

- `visual_reviewer` agent class stays callable via `/task visual_reviewer ...` for ad-hoc single-image audits and diagram review.
- `analyze_image` extended to accept multi-image; backward-compatible (single-string filepath still works → wrap to list internally).
- New posthook `visual_review` is opt-in via `visual_dial=on` (default off); existing missions see zero behavior change.
- New step field `produces.routes` is additive; expander falls back to glob-based inference when absent.

## Tier plan (batched, no pauses)

Founder does irreplaceable 10% (sprint-0 baseline approval, severity calibration); agent does 90%.

**T1 — Capture rail (parallel-safe with T2).**
- T1A. `capture_screenshots` mr_roboto verb skeleton + pending-URL soft-skip + breakpoint loop + route iteration + host playwright invocation + determinism injection (CSS / motion / Date / random). Per-step output dir.
- T1B. `produces.routes` schema extension to i2p step format + framework inference table (Next, Remix, SvelteKit). Document inference rules in step schema doc.
- T1C. `mission_{id}/.visual/captured/` workspace layout + retention policy + envelope shape tests.
- Tests: pending-URL soft-skip, missing route inference, breakpoint set override via `.kutay/visual.yaml`.

**T2 — Diff rail (parallel-safe with T1).**
- T2A. `analyze_image` extension: accept `filepaths: list[str] | str`; assemble multi-image messages content array. Backward-compat unit test on single-string. ingest_visual unchanged.
- T2B. `visual_review` mr_roboto verb: reads captured + baseline + tokens + specs; one vision call per pair (or single-image audit when no baseline); structured JSON parse; severity rules; finding envelope shape; `verdict` decision; vision-capability soft-skip (import `_is_vision_capability_unavailable` from ingest_visual). Per-call + per-mission timeout.
- T2C. Severity rule table in `.kutay/visual.yaml` (founder-overridable): delta-E threshold, layout-shift px threshold, named-component-missing → blocker, etc. Defaults match v1 intent.
- T2D. Diff prompt + audit prompt; structured JSON schema. Compatible with constrained-decoding pass (Z2 work).
- Tests: vision capability missing → skipped; first-frame audit (no baseline); diff happy path; degenerate vision output via dogru_mu_samet; structured finding parse.

**T3 — Registry + auto-wire (CANONICAL-FIRST: T1+T2 must land to main before T3 dispatches per [canonical-first-for-tier3plus](../../../memory/feedback_canonical_first_for_tier3plus.md)).**
- T3A. `visual_review` PostHookSpec entry in `posthooks.py::POST_HOOK_REGISTRY`.
- T3B. `visual_dial` added to `_ALLOWED` (review_density.py), `ReviewDensityDials` dataclass, `MissionDialContext`, `to_mission_dial_context`. `/density` Telegram command surfaces it.
- T3C. Expander auto-wire test against frontend globs with `visual_dial=on` → wires; with `off` → no wire.
- T3D. Multi-file expansion suppression — wire visual_review on parent integration step only (not per sub-file). Reuses `integration_review` injection point.
- T3E. apply.py retry-context injection: ensure visual_review findings flow into source step retry as `visual_diff` named context. (Likely already works via generic posthook-fail path; T3E = verification + test, not new code.)
- Tests: registry kind count +1 (= 26 total); expander idempotency; sub-file suppression; retry context contains findings.

**T4 — Founder loop.**
- T4A. Telegram thread surface (Z10 cross-cutting): visual_review run emits message with WebP thumbnails (≤80KB each) + structured summary + reaction buttons.
- T4B. `/approve_baseline {mission_id} {step_id}` Telegram command + handler: copy `mission_{id}/.visual/captured/{step_id}/*.png` → `mission_{id}/.visual/baseline/`. Idempotent.
- T4C. Reaction handler → `upsert_mission_lesson(stack="frontend", domain="visual", pattern=f"{component}:{kind}", ...)`. Wire into existing reaction-callback infrastructure (search telegram_bot.py for `accessibility_review` reaction precedent).
- Tests: thumbnail size cap, /approve_baseline idempotency, lesson dedup_key collision.

**T5 — Cross-mission baseline + Z5 handoff.**
- T5A. `.visual_baseline/` repo-root store + version control + `tokens_changed` detection (hash `design_tokens.json` at extract time).
- T5B. Per-component capture extension (CSS selector crop) — opt-in via `produces.components: [{name, selector}]`. Default = full-page.
- T5C. Z5 handoff: native-device capture (`xcrun simctl io` / `adb exec-out screencap`) replaces playwright viewport when [05-build-mobile-track.md](05-build-mobile-track.md) lands.
- Tests: cross-mission baseline rebuild on token-hash mismatch, selector-crop happy path, Z5 hook present (no impl).

## Dependencies

- **Inbound (all live):**
  - Z1 T4C tunneled preview ✅ `emit_preview_url` / `kill_preview_url`
  - Z1 B7 ingest_visual + vision capability ✅ `analyze_image` + `_is_vision_capability_unavailable`
  - Z2 T4 mission_lessons + inject_lessons ✅ `upsert_mission_lesson` + STACK_BLOCKS
  - Z2 T6 founder_actions (z6_admission) + posthook-fail recipes ✅ (recipe library at `recipes/`, but our retry context injection is via apply.py generic path, not the recipe bundle)
  - Z3 T1A cost_band + dial-aware auto_wire ✅ `_dial_get` helper + callable triggers
  - Z3 T3B accessibility_review template (preview-URL consumer, severity mapping) ✅
- **Outbound:**
  - [05-build-mobile-track.md](05-build-mobile-track.md) — native-device capture phase 2 (T5C handoff)
  - [10-cross-cutting.md](10-cross-cutting.md) — Telegram thread surface (T4A)

## Open questions (narrow, no benches required)

- **Per-route inference per framework.** Next.js, Remix, SvelteKit each have different routing conventions. T1B will need a small table. Punt: hardcode Next-style (`pages/foo.tsx` → `/foo`, `app/foo/page.tsx` → `/foo`) in T1; extend in T5 if needed.
- **Container vs host playwright.** Locked to host (public cloudflared URL). Founder local debugging can run host-direct without container assumption.
- **Baseline drift after token change.** Hash `design_tokens.json` at extract time; mismatch → regenerate cross-mission baselines (T5A). Per-mission baselines stay (founder-approved per mission).
- **Cost ceiling per mission.** ~4 routes × 4 breakpoints × ~$0.005 ≈ $0.08/step × ~20 frontend steps ≈ $1.60/mission. `cost_band=heavy` + `visual_dial=off` default keeps it gated. Per-mission ceiling enforced via existing z6_admission cost-ack on `heavy` posthooks (memory `[Z6 COMPLETE]`).
- **CSS-selector crop vs full-page.** Default full-page in T1; selector crop opt-in via `produces.components` in T5.
- **Cross-browser.** Only Chromium in T1 (matches record_demo). Firefox/WebKit capture deferred.
- **Auth scenarios.** Reuse `tests/e2e/*.spec.ts` per record_demo's `_resolve_scenario_path`. If absent and step needs auth, soft-skip with `requires_auth: scenario_missing` reason.
- **PII redaction policy.** Conservative default in T2: redact `[data-pii]` overlay if attribute present. Opt-out via `produces.redact_pii: false`. Stronger redaction (face blur, ML PII detect) deferred.

## Agent task brief (for whoever picks T1 up)

1. Read 00-README + this doc + Z3 T3B accessibility_review (`packages/mr_roboto/src/mr_roboto/run_axe.py` as the canonical template) + Z1 T4C `emit_preview_url.py` + Z2 T6 founder_actions overview + Z2 T4 mission_lessons + `ingest_visual.py` + `src/tools/vision.py` + `src/workflows/engine/expander.py::_auto_wire_posthooks` + `packages/general_beckman/src/general_beckman/posthooks.py`.
2. Land T1 + T2 in parallel worktrees; merge canonically to main BEFORE dispatching T3 (per `feedback_canonical_first_for_tier3plus`).
3. T3 sequential on main. T4 + T5 sequential after T3.
4. No tier pauses. Don't ask for greenlights between tiers (per `feedback_no_tier_pauses`).
5. Update `## Updates` at the bottom after each tier ships.
6. Tag tier ships: `z4-t1-shipped`, `z4-t2-shipped`, `z4-t3-shipped`, `z4-t4-shipped`, `z4-t5-shipped`. Final `z4-complete-YYYY-MM-DD`.
7. After Z4 close, write `project_z4_closed_{date}.md` memory entry per the zone-close pattern (see Z1/Z2/Z3 closed entries).

## Updates

- 2026-05-15 — v2 written. Re-audit corrected 8 stale v1 claims + 6 schema-level errors in the prior v2 draft (cost_band literal, PostHookSpec field names, severity vocab, dial registration, analyze_image signature, multi-file expansion injection). Concrete file paths + line numbers throughout. Tier plan replaces v1 Phase A–D. Adjacent infra (preview_url, ingest_visual, run_axe template, mission_lessons, cost_band, dial-aware auto_wire) all shipped; remaining work is two mechanicals + one `analyze_image` extension + one PostHookSpec + new `visual_dial` field + auto-wire suppression for multi-file + founder loop. Cost math: ~$1.60/mission worst case at heavy cost_band, dial-off by default.
