# Cluster Summary — Mixed Angles (Plasmic / Onlook / Replit Agent)

**Date:** 2026-05-09
**Three different angles on idea-to-product:**
- **Plasmic** = visual builder + headless CMS, AI is auxiliary.
- **Onlook** = OSS, two-way DOM↔code sync, design-led.
- **Replit Agent** = autonomous chat agent, hosted runtime, full backend bundled.

## Side-by-side

| Dimension | Plasmic | Onlook | Replit Agent |
|---|---|---|---|
| **Entry** | Drag-drop / Figma | Prompt + visual + import | Prompt + integrations |
| **Output** | React in your repo / headless API | Next.js+Tailwind on disk | Live app in Replit cloud |
| **Iteration** | Visual canvas | Visual ↔ chat ↔ code (fused) | Chat + self-test loop |
| **Two-way sync** | Limited (codegen) | Best-in-class (`data-oid`) | None |
| **Backend** | Connectors (Supabase/Shopify/HTTP) + own CMS | Thin (user's choice) | Built-in DB+Auth+Secrets |
| **Spec/PRD** | None | None | Implicit task list |
| **Multi-route** | Yes | Yes | Yes |
| **Deploy** | User's infra or hosted | Self-host or Freestyle | Replit cloud (default) |
| **OSS** | Hybrid (MIT + AGPL) | Apache 2.0 fully | None |
| **Pricing** | $10/user/mo → Enterprise | OSS free / hosted opaque | $0 → $25 → $100 + effort |
| **Model** | Undisclosed, AI auxiliary | OpenRouter + Morph/Relace | Claude Sonnet 4.x + routing |
| **Lock-in risk** | Medium (headless mode) | Low (your code, your disk) | High (runtime + DB) |

## What each teaches i2p

### From Plasmic — **Composability and Roles**
- **Component registration**: let users bring existing React components into the i2p workflow as drag-and-drop primitives. Agent composes rather than always re-generating.
- **Branching + review** for non-dev review gates — maps to KutAI's Telegram approval pattern.
- **Headless vs codegen split** — same artifact, two delivery modes. Don't force one.
- **Marketers/content as first-class users** — Plasmic monetizes per-seat by serving non-devs. KutAI's todo/shopping flows already serve non-dev users; lean into that.

### From Onlook — **The Source-of-Truth Discipline**
- **`data-oid` instrumentation** is the unlock for any visual editing on top of generated code. Build it in from day one if a canvas is on the roadmap.
- **Split planning model from apply model** (frontier LLM plans, Morph/Relace applies). Maps directly to a coulson optimization: tiny fast model for diff application, big model only for plan.
- **Source-of-truth = files on disk** — never store the canonical artifact only in the platform DB.
- **Three loops fused** (visual/chat/code) on same source — KutAI today is chat-only; even a read-only preview linking back to source would be a leap.
- **OSS as positioning** — Onlook's pitch "your code is yours" resonates with developers burned by lock-in. KutAI's user-repo-first stance is similar; market it.

### From Replit Agent — **The Autonomy Frontier**
- **Self-test reflection loop is table stakes** — but real test execution beats LLM self-grading. Coulson should invoke the actual test runner.
- **Parallel sub-agents** (auth/db/backend/frontend concurrent) — Beckman queue could dispatch independent steps in parallel, not serial. Throughput multiplier.
- **Effort-based pricing** maps user intuition better than tokens or steps. Right model for cloud lane.
- **Hosted DB+Auth removes decisions** — KutAI could ship opinionated defaults (e.g. "your i2p project ships with embedded SQLite + auth scaffold by default; override if you want").
- **"Live URL in 60s" UX** — anything that requires user-side install is at a disadvantage. Consider a hosted preview lane.
- **Black-box routing is a weakness** — Fatih Hoca's `model_pick_log` transparency is a real differentiator. Surface it in user-facing telemetry.
- **Web→mobile in same project** — artifact-agnostic mission spec with per-target adapters is a future lane.

## Cross-cutting observations

1. **None of the three has a real PRD/spec artifact.** All three treat the design or chat history as the spec. KutAI's z0-mission-preflight + charter discipline is a genuine differentiator if executed — competitors are weak here.

2. **Two-way sync is rare and hard.** Only Onlook does it well. If KutAI wants visual editing, copy `data-oid` exactly; don't reinvent.

3. **The "app exists right now" axis splits the field.** Replit owns it (hosted runtime). Plasmic/Onlook ship code, requiring a user-side run step. Decide which side KutAI plays on per-mission — i2p could offer both via a "preview lane" + "export lane."

4. **Pricing models are converging on credits/effort, not seats.** Replit's effort-based pricing is the trend. Plasmic's per-seat model looks dated.

5. **Backend bundling is the moat.** Replit's DB+Auth+Secrets bundle is genuinely sticky. Plasmic's CMS is similar. Onlook is weakest here. KutAI's mission state + skill memory + shopping intelligence are a similar bundle — frame them as the moat.

6. **Open-source positioning** earns trust but doesn't auto-win. Onlook is OSS but pre-1.0; Replit is closed but production-trusted. KutAI's existing local-first stance is a credibility asset; OSS would amplify it but isn't required.

## Top 3 surprises (for the report)

1. **Plasmic underplays its AI story** — given the 2026 market, this is conspicuous. Either they have a strategic reason (enterprise customers don't want LLM-only tools) or they're getting outflanked by Onlook/Replit. Likely both.

2. **Onlook splits planning model from apply model** (OpenRouter for plan, Morph/Relace for apply). This is a sophisticated cost optimization most agent systems don't bother with — direct lesson for coulson runtime.

3. **Replit Agent 4 is bolting a visual canvas onto a chat-first agent** — convergence with Plasmic/Onlook's visual-first approach. The whole field is collapsing toward "chat + visual + code, all on the same source." KutAI being chat-only (Telegram) is a positioning question to confront.

## Confidence

- Plasmic: Medium-High on product/integration; Medium on AI features (under-marketed).
- Onlook: High on architecture (extensively OSS); Medium on hosted pricing (gated).
- Replit Agent: High on pricing/pillars; Medium on exact model routing (obscured).
