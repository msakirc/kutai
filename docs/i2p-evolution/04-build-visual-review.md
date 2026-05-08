# Z2 — Visual review (subproject)

## Frame

Visual reviewer agent exists in the agent registry but is disconnected
from feedback loops. Real visual review needs nontrivial infra: place
the subject in a known state, frame the screenshot, capture across
breakpoints, diff against a reference, return structured issues. None
of that infra exists. Treat as its own subproject because it does not
fit the "wrap a CLI tool in a mr_roboto verb" pattern of other QA modalities.

## Current state

- `visual_reviewer` agent registered in agent_zoo (planned to be killed
  per kill-agents work; for visual review specifically the role survives
  as a profile, not a class).
- No screenshot harness wired into i2p_v3.
- Playwright runner is on the W1 real-tools workstream; needed here.
- No design-token reference image library; no per-mission baseline
  screenshots from sprint 0.
- No vision model selection for diff judgment.

## Gaps

### Fixable by automation (with significant infra investment)

**A. State priming**
- Login flow → navigate to feature path → set viewport → wait for content → assert ready state. Without this, screenshot captures a half-loaded page.
- Per-feature priming script that the visual reviewer can replay.
- Test fixture data for predictable visual state (e.g. specific user with known data; not random seed).

**B. Multi-breakpoint capture**
- Mobile (375px), tablet (768px), desktop (1280px), wide (1920px) at minimum.
- Per-breakpoint capture; per-breakpoint diff vs reference.

**C. Reference baseline**
- Per-mission baseline images captured from sprint-0 design system; updated whenever design tokens change.
- Story-driven baselines (Storybook stories per primitive/composite) act as canonical reference.

**D. Vision-model diff**
- Compare captured screenshot against reference + design_tokens + screen_specifications.
- Returns structured diffs: color drift, layout shift (px), missing component (named), breakpoint break (which one).
- Not pixel-perfect compare (alpha noise / antialiasing kills that). Semantic compare.

**E. Severity gating**
- Critical: layout broken at any breakpoint, missing component named in spec.
- High: color drift > N delta-E from token, padding off > 4px from spec.
- Medium: typography sized off scale, button label inconsistent.
- Low: shadow/elevation mismatch, micro-spacing.
- Critical/high block via existing `blockers` rule.

**F. Connection to feedback chain**
- visual_review post-hook kind, mechanical executor that runs the harness, vision-model judgment.
- Failure → retry source frontend step with structured visual diff overlay.
- Pass → proceed.

### Founder territory
- Approving the design baseline (sprint 0 sign-off).
- Calibrating severity thresholds ("this color is fine"; agent learns).
- Final taste judgment on ambiguous cases (escalation path from agent).

## Proposed direction

### Phase A — Infrastructure (most work)
- Mr. Roboto verbs:
  - `prime_state(scenario)` — log in, navigate, seed test data, wait for ready signal
  - `capture_screenshots(path, breakpoints)` — playwright-based, per-breakpoint
  - `extract_baseline(component_name)` — pull from Storybook story or design-token export
- Storage: `mission_workspace/visual/baseline/` for references; `mission_workspace/visual/captured/` for current run; `mission_workspace/visual/diffs/` for diff artifacts.
- Test-fixture seeder: deterministic test data per scenario; idempotent setup/teardown.

### Phase B — Vision diff
- Vision model selection (open question; bench in week 1).
- Diff prompt: "Compare these two images. List structural differences (missing component, layout shift, color drift, breakpoint break) at severity (critical/high/medium/low). Return JSON."
- Output schema: structured findings array (component, kind, severity, description, expected, observed).

### Phase C — Posthook integration
- New beckman posthook kind `visual_review` (mechanical, since the LLM call is wrapped inside the mr_roboto verb that returns structured findings — no "judge of judge").
- Severity-gated via `blockers: {field: severity, levels: [critical, high]}`.
- Auto-wire in expander on steps that emit frontend code.

### Phase D — Founder loop
- Visual diffs surface in mission Telegram thread (image attachments + structured summary).
- Founder reactions (👍 / "wrong, this color is fine" / "this is broken") feed into mission_lessons memory.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Baseline capture | extracts from Storybook + tokens | approves baseline as sprint-0 sign-off | full pre-mission |
| Per-feature visual review | captures + diffs against baseline + reports | reviews diffs in Telegram thread; calibrates severity | full |
| Ambiguous severity | flags as needs_clarification | makes call | full |
| Baseline update | proposes after design-token change | approves new baseline | full |

## Dependencies

- **Inbound:** [02-build-foundation.md](02-build-foundation.md) — needs the posthook framework + recipes (visual recipes for common UI patterns). [03-build-review-density.md](03-build-review-density.md) — fits naturally into the modality posthook framework once that's in.
- **Real-tools workstream** — playwright runner is shared with `test_run` posthook in [02-build-foundation.md](02-build-foundation.md).
- **Outbound:** [05-build-mobile-track.md](05-build-mobile-track.md) — mobile visual review uses device-screenshot mode, depends on this doc's harness shape.

## Open questions

- **Vision model.** Claude Vision (good general; cost real)? GPT-4o (similar)? Dedicated UI-diff model (more accurate, more setup)? In-house ResNet/CLIP-based diff (cheap, less semantic)? **Bench in week 1.**
- **Storage for baselines.** Repo-committed (auditable, version-controlled, but bloats repo) vs object storage (clean repo, but external dep). (Repo-committed under `.visual/baseline/` with LFS if size grows.)
- **State priming reusability.** Per-feature scripts vs scenario library? (Scenario library; recipes ship with named scenarios.)
- **Flake handling.** Antialiasing / animation timing / async data introduces noise. Tolerance threshold? (Per-finding kind; layout shift > 4px matters; pixel noise < 5% ignore.)
- **Test fixture data.** Random seed vs canonical test users? (Canonical test users; recipes ship them.)
- **Mobile visual review.** Native device screenshots (xcrun + adb) vs viewport-emulated playwright? (Phase 1 emulated; phase 2 device when [05-build-mobile-track.md](05-build-mobile-track.md) lands.)

## Agent task brief

When picking up this doc:
1. Read 00-README + 02-build-foundation + 03-build-review-density + this doc.
2. Pick the vision model — bench Claude Vision vs GPT-4o vs ResNet baseline on 5 known-good and 5 known-bad UI samples.
3. Build Phase A infra (state prime, capture, baseline extract); skeleton + tests.
4. Build Phase B vision diff with chosen model; structured output.
5. Wire Phase C posthook + auto-expander; tests.
6. Plan Phase D founder loop integration with [10-cross-cutting.md](10-cross-cutting.md) Telegram thread design.
7. Resolve open questions or escalate.
8. Cross-reference outbound to [05-build-mobile-track.md](05-build-mobile-track.md).
9. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; absorbs Wave 7 + theme T11 from `docs/plans/2026-05-07-i2p-capability-expansion.md`. Acknowledged limit: state-priming infra is the hard part; "wrap a vision model" is the easy part.
