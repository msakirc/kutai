# Z1 — Paraflow + competitor landscape, mapped to KutAI

**Date:** 2026-05-09
**Author:** Claude (own investigation, ground truth from `Bilinc/main/paraflow/` user output)
**Scope:** Bigger frame than the original Z1 doc — looking at the entire AI-design-tool category as competitors to what i2p phases 0-6 should produce. Then filtered through KutAI's specific constraints (Telegram-only UI, async/AFK founder, local-first inference, long-running autonomous missions).

**⚠️ Read §11.5 first.** First-pass framing in §1 treated Paraflow output
as user-authored methodology. Corrected: Paraflow is an AI tool; the
TruthRate folder is its generated output. Everything else here is correct
but reads in light of the §11.5 reframe — Paraflow is competitor benchmark,
not pattern-to-imitate.

---

## 1. Paraflow ground truth (extracted from real output)

You produced a real product spec for **TruthRate** using Paraflow. Folder shape:

```
paraflow/
├── Global Context/
│   ├── product_charter.md        70 lines  — positioning + brand keywords + JTBD + mission + 6 solutions
│   ├── persona_business_owner.md 52 lines  — Carlos Rodriguez (named, single page)
│   └── persona_reviewer.md       — Maya Chen
├── Feature Plan/
│   ├── prd.md                    76 lines  — background + objectives + user stories + key features + key flows + competitive analysis
│   ├── user_flow.md              113 lines — 3 mermaid diagrams (mobile reviewer, web business owner, admin)
│   └── <screen>_screen_plan.md   24 files  — one per screen, ~25 lines each
├── Screen & Prototype/
│   └── <screen>.html             24 files  — mobile 390×844, real designed
└── Style Guide/
    └── mobile_fact_primary_*.md  light + dark variants
```

### Paraflow's distinct moves (concrete, not speculative)

1. **Charter is one fat doc, not 7 micro-artifacts.** Sections: Positioning / Brand Keywords (5) / Core Problem & JTBD / Goals & Mission (mission + 6 desired-outcomes) / Solutions We Own (6 solutions, each: what-it-solves, typical-path, outcome, **boundaries**, **guiding-principles**). The boundaries field is anti-goal at solution granularity.

2. **PRD has explicit competitive-analysis section** at the spec level (not just a research note). Five sub-sections: Landscape / Value Thesis / Strengths-Weaknesses / Our Differentiators / Switching Costs & Risks / Notes. Forces the founder to commit to a positioning *relative to specific named competitors* (Yelp, TripAdvisor, Google, Amazon, Trustpilot for TruthRate).

3. **Non-goals declared in PRD section 2** as "Boundaries" — explicit list (advanced auto fact-verification, AI-content detection, social networking, e-commerce, i18n).

4. **User stories are persona-tagged + lettered** (A-J). Story-level granularity, not feature-level.

5. **Key flows are narrative, not just diagrams.** Six worked examples per flow (Trigger / Path / Result). Reads like acceptance criteria scenarios.

6. **User flow doc holds Mermaid for three audience tracks** (mobile reviewer, web owner, admin) — same product, three diagrams. Makes multi-platform/multi-role architecture visible at spec time.

7. **Per-screen plan is short** (~25 lines) and section-tagged: `## Search Bar`, `## Featured Content`, `## Quick Actions`, `## Personalized Recommendations`, `## TabBar`. Sections become DOM regions in the HTML.

8. **Style guide has audience descriptor in filename**: `mobile_fact_primary_light.style-guide.md`. The "fact_primary" tag means the style was designed *around* a content-type emphasis (facts vs reviews). Style guide is product-bound, not generic.

9. **Style guide → HTML is tight.** Forest green `#1B4D3E` from style guide appears verbatim in HTML (`background-color: rgba(27, 77, 62, 1)`). Spacing scale, type scale, shadow recipes — all from guide. Real lockstep.

10. **HTML uses real (AI-generated?) images**, not placeholders. Per-image alt text describes the scene. Static hosted at paraflowcontent CDN.

11. **HTML uses Tailwind + Iconify (Lucide icons)** — production-realistic stack, not lorem-ipsum. Founder + downstream agents can lift HTML directly.

12. **HTML viewport hard-coded to mobile 390×844** (iPhone 14 Pro). Plus separate web flow for business dashboard. Multi-form-factor at spec time.

### What Paraflow lacks (relative to v1-v3 of Z1 plan)

- Zero ADRs, zero alternatives-considered, zero falsification fields.
- Personas are concise but unsourced (no evidence, no interview count).
- No regulatory / compliance fingerprint at intake.
- No prior-art "graveyard" — competitive analysis is from training data, not searched.
- No reversal-cost reasoning.
- No cost-curve modeling.
- One-shot output assumed; no iterative refinement loop captured in folder.

This matches A1-A14: Paraflow nails artifact shape; v1-v3 + A1-A8 add audit/grounding rigor that Paraflow doesn't.

---

## 2. Competitor landscape (own knowledge, labeled by confidence)

Confidence levels: **H** (high — used or recently inspected), **M** (training data + general awareness), **L** (heard about, may be stale).

### Charter+spec generators (closest to paraflow's frame)

| Tool | Confidence | Strongest move | Note |
|---|---|---|---|
| **Paraflow** | H (your output) | Charter + PRD + per-screen + style + HTML bundle | Spec-first, then HTML |
| **Stitch (Google)** | M | Text + image → multi-screen Figma export | Replaced Galileo AI; Figma-output focus |
| **Uizard** | M | Sketch / screenshot → wireframe → polished | Theme generator; multi-screen consistency lock |
| **Visily** | L | Sketch → wireframe; templates-heavy | Mid-fidelity bias |

### Code-output generators (skip artifacts, output real code)

| Tool | Confidence | Strongest move | Note |
|---|---|---|---|
| **v0 (Vercel)** | H | Prompt → React + shadcn/ui + Tailwind components, chat-iterate | Component-grade output; project-grade slowly catching up |
| **Lovable (lovable.dev)** | M | Full-stack from prompt; Supabase integrated; GitHub sync | Ships deployed app; visual edit + chat edit |
| **Bolt.new (StackBlitz)** | M | In-browser WebContainer; real npm + terminal + filetree | "Watch it build" experience |
| **Replit Agent** | M | Full app from prompt; runs in their cloud + DB | Tightly hosted |
| **Cursor Agent / Windsurf Cascade** | M | IDE-mode multi-file generation | Code-first not design-first |

### Designer-grade generators (taste-leaning)

| Tool | Confidence | Strongest move | Note |
|---|---|---|---|
| **Subframe** | M | Token-system first; production CSS/component shipping | Designer fluency, not just throwaway mocks |
| **Magic Patterns** | M | Component-library-aware generation | Re-uses existing design system |
| **Tempo Labs** | M | Storybook visual editor + React two-way sync | Sits in between design and code |
| **Plasmic** | M | Visual builder + AI; component library bridge | Headless CMS + visual model |
| **Onlook** | L | Open-source; direct DOM-edit of running React app | Two-way design↔code |
| **Framer / Webflow + AI** | M | No-code site builder + AI assist | Marketing-site bias |

### Adjacent: full-app cloud agents

| Tool | Confidence | Strongest move | Note |
|---|---|---|---|
| **Devin (Cognition)** | M | Long-running autonomous agent; full project lifecycle | Closest in autonomy to KutAI |
| **OpenDevin / SWE-agent** | M | OSS autonomous coding agents | Code-task focused |
| **Magic.dev / Augment** | L | "Software engineer" agents | Limited public detail |

### Patterns observed across the category

| # | Pattern | Tools that exemplify |
|---|---|---|
| P1 | Chat-first iteration loop (never one-shot) | v0, Lovable, Bolt, Cursor |
| P2 | Live preview with hot reload | Bolt, Lovable, Tempo, Onlook, Replit |
| P3 | Component-library awareness (shadcn/MUI/etc.) | v0, Magic Patterns, Subframe |
| P4 | Multi-screen shared shell (nav/header consistency) | Stitch, Uizard, Subframe |
| P5 | Token-system first-class style guide | Subframe, Magic Patterns, Plasmic |
| P6 | AI-generated assets baked in | Paraflow, v0 (placeholders), Stitch |
| P7 | Two-way design↔code sync | Onlook, Tempo, Plasmic |
| P8 | One-click deploy to host | v0→Vercel, Lovable→Netlify, Bolt→Netlify, Replit→Replit |
| P9 | GitHub integration | Lovable, v0, Bolt |
| P10 | Templates / starter kits | Visily, Uizard, Lovable, Replit |
| P11 | Charter / positioning artifact | Paraflow uniquely (others jump straight to UI) |
| P12 | Image / sketch input | Uizard, Visily, Stitch, v0 |
| P13 | Empty/loading/error states explicit | Subframe, Magic Patterns |
| P14 | Mobile + Web split UI in same project | Paraflow (mobile reviewer + web owner) |
| P15 | Multiple style-guide variants (light/dark/density) | Paraflow, Subframe |
| P16 | Streaming generation (UI updates live) | v0, Lovable, Bolt |
| P17 | Long-running async agent | Devin, KutAI |

---

## 3. KutAI's specific ambitions and constraints

To filter what to copy, restate what KutAI actually is and isn't:

**Is:**
- Telegram-native interface (text + file uploads, voice memos, inline keyboards)
- Async / AFK founder (long mission gaps acceptable)
- Long-running multi-step missions (hours to days)
- Local-first inference + cloud burst (cost-aware; GPU is real)
- Modular package architecture (mr_roboto, fatih_hoca, beckman, coulson, vecihi…)
- Cross-mission memory (Chroma) + skill library
- Real product output ambition (not just code that compiles)
- Turkish-market vertical strength (shopping pipeline)

**Is not:**
- Not an in-browser IDE / not a visual editor
- Not session-bound (founder doesn't sit at the screen)
- Not a hosted SaaS (runs on user's box)
- Not deploying anything today (no built-in deploy infra)
- No web UI surface beyond yazbunu logs
- No real-time collaboration

This list determines fit per pattern.

---

## 4. Pattern × KutAI fit matrix

For each pattern, score:
- **Fit:** ✅ strong fit / 🟡 needs adaptation / ❌ fights the constraints
- **Effort:** S / M / L
- **Leverage:** how much it moves the bar

| # | Pattern | Fit | Effort | Leverage | Adaptation note |
|---|---|---|---|---|---|
| P1 | Chat iteration loop | ✅ | M | High | KutAI-native via Telegram clarify; needs `regen_with: "<change>"` action on every artifact |
| P2 | Live preview | 🟡 | M | High | No browser. Adapt: tunneled URL (cloudflared/ngrok) for HTML prototype; share link in Telegram |
| P3 | Component-library awareness | ✅ | M | High | New `4.X component_library_decision` ADR; phase 5 generation reads the ADR, uses real component names |
| P4 | Shared shell consistency | ✅ | S | Medium | `screen_inventory.md` declares shared shell; per-screen plan inherits + overrides; reviewer rejects re-invented shells |
| P5 | Token-system style guide | ✅ | M | High | `design_tokens.json` (not markdown). Phase 7 codegen consumes directly |
| P6 | AI-generated assets | ✅ | M | Medium | KutAI has GPU. Use local SDXL or burst to cloud (Replicate/fal.ai). Mechanical action `generate_image(alt, dimensions, style_seed)` |
| P7 | Two-way design↔code sync | 🟡 | L | Medium | Async-friendly version: founder edits HTML; post-hook `reflect_html_changes` on next mission interaction proposes spec patch for confirm |
| P8 | One-click deploy | 🟡 | M | High | Phase 7+ scope, but Z1 must produce artifacts deploy expects (env spec, build command). Mechanical `publish_static_preview(workspace) -> url` for Z1 prototypes |
| P9 | GitHub integration | ✅ | S | Medium | KutAI can push via gh-cli already; mission-private repo at phase 6.7+. Z1 outputs become first commit |
| P10 | Templates / starter kits | ✅ | M | High | Recipe library (overlaps Z2). Per-product-type templates: SaaS-CRUD / mobile-content / dashboard / e-commerce. Pre-fills charter sections |
| P11 | Charter artifact | ✅ | S | Very High | A9 — replaces 7 phase-0 micro-artifacts with one fat charter |
| P12 | Image / sketch input | ✅ | S | Medium | Telegram already accepts photos. Phase 0 founder sends moodboard ref / competitor screenshot; vision agent extracts |
| P13 | Empty/loading/error states | ✅ | S | Medium | Per-screen plan declares states (paraflow does this implicitly; make explicit) |
| P14 | Mobile + Web split | ✅ | M | Medium | Mission-config field `surfaces: [mobile, web, desktop]`; per-surface flow + screen plans + style guide |
| P15 | Multi-variant style guide | ✅ | S | Medium | A13 + paraflow precedent: light + dark + density variants as token sets |
| P16 | Streaming generation | ✅ | already done | — | KutAI streams via hallederiz_kadir. Surface streaming progress in Telegram (typing indicator + checkpoint messages) |
| P17 | Long-running async agent | ✅ | already done | — | KutAI's existing strength. Devin is the closest competitor on this axis |

---

## 5. KutAI advantages no competitor has

1. **Persistent multi-day missions.** v0/Lovable/Bolt sessions die. KutAI missions live in DB until founder kills them. Spec can mature for weeks.
2. **Telegram = always-with-you.** Founder pitches from a coffee shop, reviews from bed. No tool requires laptop + browser.
3. **Cross-mission learning.** Skill library + Chroma. Mission #19 knows mission #12 was killed for reason X (A7 idea-dedup proposal).
4. **Local GPU inference for asset gen.** Image generation cost ≈ electricity, not $/image. Can iterate moodboards freely.
5. **Local-first cost ceiling.** Founder can run dozens of speculative missions without burning cloud budget. Competitors are all SaaS-priced per generation.
6. **Voice + photo as first-class founder input.** Whisper for voice memos, vision for screenshots. Most competitors are text-only or text+image.
7. **Multi-agent specialization.** Researcher does prior-art, architect does ADRs, designer does mockups. Competitors are generally one prompt → one model.

---

## 6. What to copy, ranked by leverage × fit

### Tier 1 — copy directly, high leverage, low effort

- **C1. Charter as one fat doc** (paraflow shape). Replaces phase-0 atomization. **A9 confirmed.**
- **C2. PRD with named-competitor analysis section** (paraflow's section 6). Forces evidence-grounded positioning. New step `1.4 competitive_positioning_lock` consuming `1.3 competitive_landscape` outputs.
- **C3. Per-screen plan as short markdown unit** (paraflow shape, 20-30 lines, sectioned). New phase 5 sub-loop. **A10 confirmed.**
- **C4. User flow doc with Mermaid per audience track** (paraflow shape). New `5.0 user_flow_lock` step. **A12 confirmed.**
- **C5. Style guide as deliverable** (paraflow precedent, all competitors). **A13 confirmed.**
- **C6. Solutions-with-boundaries-and-guiding-principles** (paraflow charter pattern). Per-solution non-goals at finer grain than mission-wide. **A14 confirmed.**

### Tier 2 — copy with KutAI adaptation, high leverage, medium effort

- **C7. Component-library ADR at phase 4** (v0 / Magic Patterns / Subframe). New step `4.X component_library_decision`. **A17 confirmed.**
- **C8. design_tokens.json instead of markdown style guide** (Subframe / Plasmic). Codegen consumes directly. **A16 confirmed.**
- **C9. HTML prototype per screen via mechanical generator** (paraflow). Mobile-first, Tailwind, Iconify. New step `5.X html_prototype_render` per screen plan. **A11 confirmed.**
- **C10. Tunneled preview URL per mission** (Lovable / Bolt adapted). cloudflared tunnel for `workspace/mission_<id>/.prototype/`. Founder shares with friends. **A19 confirmed.**
- **C11. Iterative refinement: `regen_with: "<change>"` on every artifact** (v0 / Lovable). Founder doesn't restart. Loop until accept. **A15 confirmed.**
- **C12. Multi-surface support** (paraflow mobile+web). Mission config `surfaces: [mobile|web|desktop]`. Drives flow + screen + style branching.
- **C13. AI-generated images via local SDXL** (paraflow has CDN, KutAI has GPU). Mechanical action `generate_image`; placed in `workspace/mission_<id>/.assets/`.
- **C14. Empty/loading/error states explicit per screen** (Subframe). Per-screen plan must declare; reviewer rejects screens without all four states.

### Tier 3 — copy as later-phase additions

- **C15. Templates / starter kits keyed by product-type** (Visily / Uizard / Lovable). Recipe library overlap with Z2. Charter pre-filled for "SaaS-CRUD" / "mobile-content" / "dashboard" / "e-commerce" / "AI-tool".
- **C16. Sketch / screenshot input** (Uizard / Visily / Stitch). Telegram-native; vision agent extracts wireframe-as-text.
- **C17. Two-way reflection: founder edits HTML, system proposes spec patch** (Onlook async-adapted). **A20 confirmed.**
- **C18. GitHub repo init at end of phase 6** (Lovable / v0). Z1 outputs become first commit in mission-private repo.

### Tier 4 — adapt or skip; doesn't fit KutAI's shape

- **In-browser visual editor** (Onlook / Tempo / Plasmic). Skip. KutAI has no browser surface; Telegram is the UI.
- **Live hot-reload preview** (Bolt WebContainer). Skip; replaced by tunneled URL (C10).
- **Real-time collaboration** (Plasmic / Subframe). Skip. KutAI is single-founder.
- **Marketing-site builder bias** (Framer / Webflow). Skip. KutAI's missions are products, not sites.

---

## 7. Specific things Paraflow does that none of the code-gen tools do

Worth amplifying because they're under-used in the broader category:

1. **Charter as separate artifact from PRD.** Most tools collapse to one. Paraflow's split lets the charter be reused across multiple PRDs (multiple product iterations of the same vision). Maps cleanly to KutAI's cross-mission inheritance (A7 + P9).
2. **"Solutions We Own" with boundaries inline.** Forces "what this solution refuses to do" at solution-level. Code-gen tools don't have a "solution" abstraction at all.
3. **Persona files as separate artifacts** with named individuals (Maya Chen / Carlos Rodriguez), not "the user." Easier to write user stories against.
4. **PRD competitive analysis as a required section.** Code-gen tools assume the founder already knows their position. Paraflow re-asks at PRD time.
5. **Style guide variant tagged by content emphasis** (`mobile_fact_primary_*`). The style isn't generic; it's optimized for a specific content-type primacy. KutAI can tag `mobile_<primary_content_type>_<theme>.style-guide.md`.
6. **HTML uses semantic IDs** (`71:1` etc.) with `orderindex` and `relativetransform`. Looks like Paraflow stores positional info for a visual editor. KutAI doesn't need this — but the pattern of "HTML carries enough metadata to be re-edited by the agent" is correct.

---

## 8. What Paraflow gets wrong (or doesn't address) that KutAI shouldn't copy

- **Self-hosted CDN dependency** (`static.paraflowcontent.com`). KutAI must serve from `workspace/mission_<id>/.assets/` or push to mission's own static host.
- **Single style guide variant per mission.** Paraflow has light + dark but doesn't address density/locale/accessibility variants.
- **Personas not evidence-backed.** KutAI's P1 evidence intake is strictly stronger; don't downgrade.
- **No iteration loop captured in artifacts.** Folder is a snapshot; no "regen_with" history. KutAI should preserve the chain (founder request → diff → new version) for debugging spec drift.
- **HTML viewport hard-coded.** Paraflow ships single mobile size. KutAI should generate at target sizes per `surfaces` config.
- **No accessibility audit.** Alt-text is present; ARIA roles, contrast checks, keyboard nav not visible. KutAI should add reviewer criterion.

---

## 9. Sequencing recommendation

Combine with v3's locked sequence (P7 → P4 → P3 → P6 → P5 → P1+P8 → P2 → P9):

| Order | Lands together | Reason |
|---|---|---|
| 0 | P7 (spec versioning + reviewer regression) | Already locked first |
| 1 | C1 (charter consolidation) + C6 (solutions+boundaries) | Reshapes phase 0; everything else inherits the charter |
| 2 | P4 (failure-mode column) | Already locked |
| 3 | P3 (ADRs) + C7 (component-library ADR) + C8 (design_tokens.json) | All ADR-shape; same merge; tokens informs phase 5 immediately |
| 4 | P6 (compliance) + C2 (named-competitor PRD section) | Spec-locking phases need their finals |
| 5 | C3 (per-screen plans) + C4 (user flow) + C14 (states explicit) + C12 (multi-surface) | Phase 5 reshape, all together |
| 6 | C5 (style guide deliverable) + C13 (AI images) | Style locked before HTML |
| 7 | C9 (HTML prototype gen) + C10 (tunneled preview URL) + C11 (regen_with) | Phase 5/6 build artifacts; preview makes it real |
| 8 | P5 (prior-art) parallel to above | Doesn't block |
| 9 | P1+P8 (intake + conflict surface) | After charter shape stabilizes |
| 10 | P2 (taste delegation) + C15 (templates) | Brand work needs charter |
| 11 | P9 (cross-mission inherit) + A7 (idea dedup) + C18 (github init) | After enough missions exist |
| 12 | C17 (two-way reflection) + C16 (sketch input) | Polish, async-adapted |

---

## 10. Call-outs for cross-zone coordination

- **Z2 (foundation):** C15 templates / starter kits live in recipe library (Z2 owns).
- **Z2 (review density):** C14 explicit states feed Z2 reviewer prompt criteria.
- **Z3+Z7 (real-world bridge):** C18 GitHub repo init + C10 tunneled preview need vendor-account contract from Z3.
- **Z4+Z8 (humanish):** C13 AI image gen + C12 multi-surface mobile output overlap with Z4 (mobile track) and Z8 (taste).
- **Z6 (growth):** preview URL (C10) becomes data point for "did founder share with N people, did N click?" — early signal Z6 should capture.
- **Z10 (cross-cutting):** AI image generation (C13) needs sandboxing pattern (Z10 owns); tunneled preview (C10) needs reversibility / cost-tracking (Z10 owns).

---

## 11. Open questions for you

1. **Confirm KutAI ambition for HTML output.** Production-grade Tailwind/shadcn (heavier infra), or paraflow-style design-quality but not necessarily ship-ready?
2. **Local SDXL vs cloud image API for C13.** GPU thrash risk — image gen during mission may compete with llama-server for VRAM. Burst to fal.ai instead?
3. **Tunneled preview (C10) — cloudflared (free, requires CF account) vs ngrok (auth) vs self-hosted reverse proxy on a tiny VPS?**
4. **Multi-surface (C12) — phase-0 founder declares `surfaces` upfront, or agent infers from product-type?**
5. **Templates library (C15) — curated by you or LLM-generated from the first 20 missions?**

---

## 11.5. CRITICAL REFRAME (added 2026-05-09 after user correction)

Earlier sections incorrectly framed Paraflow output as "methodology user
followed." It is not. **Paraflow is an AI tool. The TruthRate folder is
Paraflow's generated output from a founder prompt.** Implications:

### R1. Paraflow is a direct competitor to i2p phases 0-6, not a pattern to copy

Same problem space (idea → executable spec + design + clickable prototype),
same input-channel (founder prompt + few clarifications), comparable output
shape. The TruthRate folder is what i2p phases 0-6 *should* produce for a
similar input. Today i2p produces less coherent, less designed, less
clickable output for more founder effort.

### R2. The input → output ratio is the real benchmark

Paraflow probably asks ~5-10 founder questions and fans out into:
- charter (70 lines, coherent)
- 2 named personas
- PRD (76 lines with named competitor analysis)
- user_flow.md (113 lines, 3 mermaid diagrams)
- 24 per-screen plans
- 24 mobile HTML prototypes (forest green, real images, Tailwind+Iconify)
- light + dark style guides

i2p today asks 50+ founder decisions across phases 0-6 and produces:
- 7+ scattered phase-0 micro-artifacts (idea_brief, problem_statement, etc.)
- text-only design phase (no HTML)
- no per-screen plans
- no user flow
- no style guide
- no images

**The gap is not "i2p needs more steps." The gap is "i2p needs to generate
more per founder-question answered."** Today i2p atomizes; Paraflow
consolidates+expands.

### R3. Taste-from-product-essence is a missing capability

Paraflow style guide is named `mobile_fact_primary_light.style-guide.md`.
The `fact_primary` tag means the AI inferred from TruthRate's content-type
emphasis (facts vs reviews) that fact-cards should be the visual primary,
then designed the palette around that primacy (forest green for trust +
amber/slate/coral for content-type categorization).

i2p has no step that derives style emphasis from product essence. P2 (taste
delegation) lets agent generate 3-5 mood boards but doesn't tie them to
*what the product is centrally about*. New capability needed:
**taste-extraction-from-charter** as input to style generation.

### R4. The iteration loop is at the bundle level, not the component level

Founder doesn't say "regen this card." Founder says "make it more clinical"
or "skip the warm tones" or "darker theme primary." Paraflow regenerates
the affected slice across all 24 HTMLs + style guide consistently.

i2p's chat-iteration plan (C11/A15) was framed per-artifact. Should be
*per-axis*: founder gives a directional change, agent identifies all
artifacts touched, regenerates coherently. Different shape than v1/v2/v3
captured.

### R5. Quality benchmark is now concrete, not abstract

For any future i2p phase 0-6 mission targeting a similar product
(consumer-facing mobile content app), generate the bundle and **diff
against Paraflow's TruthRate output.** Specific deltas:
- Charter coherence (does it read like one mind?)
- Style-guide fitness (do colors mean something product-specific?)
- HTML quality (renderable, designed, real-looking content?)
- Per-screen completeness (all states declared, all interactions named?)
- Time-to-bundle (how many founder turns?)

Reviewer regression suite (P7) gains: "compare bundle quality against
Paraflow-output golden samples." Measurable.

### R6. Re-prioritize the C1-C18 list

Some items were misranked because I thought paraflow was a folder pattern.
Knowing it's an AI bundle generator:

- **C1 charter consolidation:** still high. But add: charter generation
  must happen in *one LLM pass* (~5k tokens), not multiple cascading steps.
  The cascading is what makes i2p slow + incoherent.
- **C8 design_tokens.json:** higher leverage than I scored. Style as JSON
  consumed by every downstream artifact (HTML, screen plans, future
  codegen). Single source of truth.
- **C9 HTML prototype gen:** higher. Should generate all screens in one
  fan-out from the locked spec, not screen-by-screen. Coherence matters
  more than per-screen tunability.
- **NEW C19 — bundle-level regeneration on directional change.** Founder
  says "darker / more clinical / less warm" → agent identifies affected
  artifacts (style guide + every HTML + maybe screen plans for color refs)
  → regenerates all coherently. Mechanical action `regen_bundle(axis,
  direction)`. This is the v0/Lovable iteration pattern but at multi-
  artifact level.
- **NEW C20 — taste-from-essence step.** Phase 5 step
  `5.X taste_extraction_from_charter`: reads charter solutions + brand
  keywords, derives `taste_emphasis` field (e.g., `fact_primary`,
  `community_primary`, `discovery_primary`). Style guide generation
  conditioned on `taste_emphasis`.
- **NEW C21 — bundle-quality regression against Paraflow goldens.** Reviewer
  regression fixtures (P7) include Paraflow output for a few archetypal
  product types; new i2p output diff'd for coverage + coherence + design
  fitness.

### R7. What KutAI can do that Paraflow can't (re-stated, sharper)

Paraflow is one-shot SaaS bundle gen. KutAI can:
- **Iterate over weeks.** Paraflow session has implicit time-pressure;
  KutAI mission lives in DB. Founder can think for 3 days, come back, ask
  one change.
- **Persistent revisions across regenerations.** Every Paraflow regen
  starts somewhere; KutAI can keep founder's accumulated edits as
  "preserve this" annotations across regens.
- **Cross-mission inheritance** (P9 + A7). Mission #19 inherits
  Mission #12's style choices for the same founder. Paraflow doesn't
  remember.
- **Real engineering rigor on top.** Paraflow has zero ADRs / falsification
  / compliance. v1-v3 + A1-A8 + (now) Paraflow-shape bundle gen = unique
  combo nobody offers.
- **Telegram + voice + photo input.** Paraflow is web prompt only.

So competitive frame: **i2p phases 0-6 should hit Paraflow's bundle quality
+ add KutAI's persistence, evidence rigor, and async iteration.** The bar
is no longer "be better than today's i2p" — it's "match Paraflow's
generated quality, then exceed via rigor + persistence + Telegram-native."

---

## 12. Bottom line

Paraflow is a goldmine because it's a real production proof that the artifact-first (not code-first) frame works. Most code-gen competitors skip the spec layer entirely and pay later in re-work. Paraflow's frame plus v1-v3+A1-A8's audit rigor plus competitor patterns adapted to async/Telegram is uniquely possible for KutAI — no competitor has KutAI's combination of long-running missions, cross-mission memory, local cost ceiling, and Telegram interface.

Specific items worth landing first (Tier 1 + Tier 2): **C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11**. That's 11 items, ~3-5 weeks of work, transforms i2p phases 0-6 from "concept → text spec" to "concept → clickable mobile prototype your friends can use." Real product surface area before phase 7 even starts.
