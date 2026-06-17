# Surface as a single source of truth (i2p) — design

**Date:** 2026-06-17
**Status:** Stage 1 implemented; Stages 2–3 specified, gated.
**Origin:** Founder hit the 5.0b "📱 Bu ürün hangi platformları hedefliyor?" Telegram
pause and asked "why pause — I already said *app* in the mission description, and
how could everything before this run without knowing the surface?"

## The defect

The "surface" of a product (mobile / web / desktop / admin) is one product fact.
The i2p pipeline currently captures it **three times, independently, in three
shapes, none reconciled:**

1. **Phase 0 intake — `0.0a.draft` Slot 12** ("which surfaces ship first?").
   Captured, but `generate_intake_todo.py` renders it as a **freeform markdown
   checkbox** (`_CANONICAL_TOPICS`), never a structured field. It evaporates:
   `product_charter` (0.1), `idea_brief` (0.6), `prd_final` (2.11b) have **no
   surface/platform field**.

2. **Phase 3 — `3.6 platform_and_accessibility_requirements`** emits
   `platform_requirements.target_platform ∈ {web, mobile, both}` (schema v2),
   **re-derived from PRD prose** (not from the intake answer). This is the
   **load-bearing** signal — it drives the actual build:
   - `4.2 tech_stack` (Expo/RN vs web framework)
   - `frontend_platform` conditional group → `7.5` (web) vs `7.5m` (Expo)
   - `expander.py::select_platform_variants` → feature template `feat.7–10` vs `feat.7m–10m`
   - `mobile_app_submission` / `ios_submission` / `android_submission` groups (14.8.*)
   - `9.7 cross_platform_testing`
   - Test rail: `tests/workflows/test_z5_platform_branching.py`, `tests/test_workflow_conditions.py`

3. **Phase 5 — `5.0b surfaces_lock`** asks the founder *again* via Telegram and
   writes `.charter/surfaces.json` (`surfaces ∈ {mobile, web, desktop, admin}`,
   schema v1). **Design-lane only**: feeds `5.0c user_flow`, `5.0d screen_inventory`,
   `5.20a/b per-screen plans`, `5.30a/b html prototypes`.

### Why this is broken

- **Double-booking.** `target_platform` (3.6) and `surfaces` (5.0b) are two
  decisions for one fact, with no reconciliation. Founder taps "mobile + web" at
  5.0b after 3.6 derived "web" → stack already built web-only at 4.2 → screens
  planned for mobile too → silent mismatch, no error.
- **Inverted order.** The build commits to the surface at 3.6/4.2 (phase 3–4).
  5.0b "locks" it at phase 5 — *after* the stack is chosen. The lock is theater
  for the web/mobile axis.
- **Noise.** For the axis that matters (web vs mobile), 5.0b adds nothing 3.6
  didn't already decide. Its only unique tokens are `desktop` (which *should*
  affect the build — deciding it at phase 5 is the same inversion) and `admin`
  (the one genuinely-late-safe axis — admin is usually `/admin` routes on the
  existing stack).
- **Wasted human input.** The founder already answered at intake (Slot 12). It
  was thrown away, so 3.6 re-derives and 5.0b re-asks.

## Target architecture

**One surface fact. `target_platform` is canonical. `surfaces` is a derived
projection. No independent human re-ask.**

```
intake Slot 12 (structured)  ──►  3.6 target_platform (canonical, build signal)
                                        │
                          derive ◄──────┘
                                        ▼
                              surfaces.json (projection, design lane)
                                        │
                                        ▼
                            5.0c / 5.0d / 5.20 / 5.30
```

`target_platform → surfaces` map:
- `web`    → `["web"]`           (primary `web`)
- `mobile` → `["mobile"]`        (primary `mobile`)
- `both`   → `["mobile", "web"]` (primary `mobile`)

The human gate, when surface is genuinely ambiguous, lives at **3.6** (which
already has `may_need_clarification` + "If multiple platforms equally important,
trigger needs_clarification"). It is asked **once, early, before the stack**.

## Staged delivery

### Stage 1 — derive surfaces from `target_platform`, no pause  *(implemented 2026-06-17)*

- `5.0b` gains `depends_on: 3.6` (guarantees `platform_requirements` exists; phase
  order already implies it — this just makes it explicit).
- `5.0b` clarify branch: load `platform_requirements.target_platform`, derive the
  surface set, write `surfaces.json` (`source="derived"`), **complete — no
  keyboard, no pause**.
- Fallback chain (only when `platform_requirements` is missing/garbage): the
  existing text-inference smart gate (`surface_infer`, high/medium/low) →
  blocking keyboard on low. Never silently skips.
- Blast radius: `packages/mr_roboto/` + one `depends_on` edge in `i2p_v3.json`.
  **Does not touch** the Z5 build rail (`expander.py`, `conditions.py`,
  `target_platform` enum, their tests).
- Net: web/mobile surface is now consistent-by-construction with the build; the
  founder is never paused for what 3.6 already settled. Supersedes the
  2026-06-16 text-inference smart gate as the *primary* path (kept as fallback).
- Trade-off accepted: `desktop`/`admin` are no longer offered at 5.0b. `desktop`
  was an inversion bug there anyway (see Stage 2); `admin` deferred to Stage 3.

### Stage 2a — desktop/admin in the design lane, deterministic, no pause  *(implemented 2026-06-17)*

A lower-risk decomposition discovered during implementation: the surface fork is
not "build vs nothing" but **design lane** (desktop/admin → more screens at
5.0c/5.0d, safe at phase 5) **vs build lane** (stack/scaffold/variants, the
rail-touching part). 2a delivers the safe design-lane half:

- `5.0b` derive reconstructs the full surface set:
  `merge_surfaces(target_platform, surface_signal.surfaces)` — web/mobile from
  the canonical build signal (`target_platform`), desktop/admin layered on from
  the deterministic `surface_signal` (3.5z). `primary_surface` stays a build
  surface so it aligns with what the stack was built for.
- `clarify._load_surface_signal_surfaces` reads `.charter/surface_signal.json`.
- Net: desktop/admin design coverage restored (Stage 1 had dropped them), now
  deterministic and pause-free, web/mobile still consistent with the build.
- **Does not touch** `target_platform`'s enum or the Z5 build rail. 63 surface
  tests green.

### Stage 2b — desktop as a first-class BUILD platform  *(gated — large, touches Z5 rail)*

Make desktop influence the actual build, not just the design: stack picks a
desktop shell (Tauri/Electron), a desktop scaffold step (mirroring `7.5m`),
feature-template desktop variants (mirroring `feat.*m`), an `expander.py` desktop
branch, a `frontend_platform` desktop arm, schema bump. This mirrors the entire
Z5 mobile track — a multi-session effort on a well-tested rail. **Gated on
explicit go-ahead + confirmation that desktop is a real near-term target** — not
worth destabilizing the rail speculatively.

### Stage 3 — structured surface capture at intake

Make `generate_intake_todo` persist Slot 12 as a structured field (or a dedicated
early `surfaces.json` with `source="intake"`), and make 3.6 **read** it instead of
re-deriving from PRD prose (falling back to derivation only when intake is silent).
Closes the "founder answered, we threw it away" loop. Lower risk than Stage 2 but
depends on Stage 1's projection being the single sink.

## Consumers verified (audit 2026-06-17)

- `target_platform` producers/consumers/branch-points/tests: see audit — 3.6
  (producer); 4.2, 9.7, 14.8, feature template `feat.*m`, conditional groups
  `frontend_platform`/`*_submission`/`seo_implementation` (consumers);
  `conditions.py`, `expander.py` (evaluators); `test_z5_platform_branching.py`,
  `test_workflow_conditions.py` (tests). Stage 1 changes **none** of these.
- `surfaces` producers/consumers: 5.0b (producer); 5.0c/5.0d/5.20a/5.20b/5.30a/5.30b
  (consumers via `surfaces_config`/`surfaces`/`user_flow` frontmatter);
  `verify_surfaces_shape`, telegram `send_surface_keyboard`/`_handle_surface_choice`.
  Stage 1 keeps `surfaces.json` produced (just derived, earlier-consistent) so all
  consumers are unaffected.
