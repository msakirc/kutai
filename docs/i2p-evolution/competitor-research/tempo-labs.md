# Tempo Labs — Designer-Grade AI Tool Deep-Dive

**Domain:** tempo.new (verified — they migrated from tempolabs.ai branding)
**Founded:** 2023, YC S23, by Kevin Michael and Peter Gokhshteyn (Perpetua alums); based Toronto
**One-liner:** Visual editor for React, powered by AI — "feels like a design tool, functions like an IDE."
**Research date:** 2026-05-09
**Confidence:** Medium-high on features/pricing; model partially disclosed (GPT-4 + Claude 3.5 Sonnet exposed via credits).

---

## 1. Input flow

- **Text prompts** for whole views or components.
- **Image upload** for brand/style extraction.
- **Figma plugin import** (V2 feature) — designs in Figma → React directly.
- **Drag-and-drop visual editing** in their browser-based IDE.
- **Existing repo import** — point Tempo at any React project.
- **Storybook import** for component libraries.

## 2. Output type

- **React** (primary), **React Native via Expo** (V2, mobile).
- **Full applications**, not just components — pre-built SaaS templates for Stripe, Polar, Supabase, Convex+Clerk indicate a "ship a product, not a prototype" stance.
- Code lives in **your** GitHub repo, your hosting.

## 3. Iteration loop

- **Three-way:** prompt + visual edit + raw code.
- **Bidirectional with VSCode + GitHub:** edit in Tempo, push to GitHub, pull into VSCode for refinement, push back, reimport.
- "Back and forth between prompt-editing UI elements and custom code adjustments" (per reviewers) is the explicit workflow.
- The strongest "round-trip" story in the cluster.

## 4. Token system

- Tokens managed via the visual editor across "components, layouts, and styles."
- Less prescriptive than Subframe's Tailwind-config-as-source-of-truth — Tempo bends to whatever your codebase already uses.
- Brand-style extraction from uploaded images can seed tokens.

## 5. Multi-screen consistency

- Maintained via the imported/generated component library; visual editor enforces consistency by editing the actual components, not screen-local copies.
- Multi-page apps are first-class (SaaS templates ship full app shells).

## 6. Bring-your-own component library

- **Yes.** Two paths: import Storybook OR generate a custom library. Existing React projects are imported as-is.
- This is closer to Tempo's headline value than Magic Patterns' library-import.

## 7. Charter / spec / PRD generation

- **None advertised.** Visual + prompt input only. No PRD layer.

## 8. Two-way sync

- **Strongest in cluster.** Full GitHub round-trip:
  - Tempo → push to GitHub → pull in VSCode → edit code locally → push back → Tempo reflects changes.
- This solves the "drift after dev customization" problem that hits Subframe.

## 9. Export

- Code into **your GitHub repo**, deployable on **your** infra (Vercel/Netlify/anywhere).
- **Figma plugin works in both directions:** Figma → Tempo (import) is the documented direction; round-trip back to Figma less clear.
- React Native bundles via Expo for mobile apps.

## 10. Pricing

| Plan | Cost | Notes |
|---|---|---|
| Free | $0 | Limited credits, 50 GPT-4 / Claude 3.5 Sonnet prompts |
| Pro | $30/mo | 150 credits/month |
| Pro top-up | +$50 | 250 bonus credits, non-expiring |
| Agent+ | $4,500/mo | **Human-assisted** feature dev: 1–3 features/day, 48–72hr turnaround |

The Agent+ tier is unusual — it's a managed "AI + humans" service, not just metered API.

## 11. Underlying model

- **Partially disclosed.** Free tier exposes "up to 50 GPT-4 or Claude 3.5 Sonnet prompts."
- V2 also added "AI reasoning + Gemini Search" — multi-model stack, not single-vendor.

## 12. Recent updates (2025-2026)

- **V2 (Feb 2026):** Figma Plugin, AI reasoning + Gemini Search, SaaS templates (Stripe/Polar/Supabase/Convex+Clerk), **Expo support for React Native**.
- Launched on YC's "Launch YC" (Feb 2026).
- Mobile/Expo angle is a clear 2026 wedge — neither Subframe nor Magic Patterns competes here.

## 13. Differentiators claimed (vs v0/Lovable/Subframe/MP)

- **IDE-grade round-trip** with VSCode + GitHub (vs Subframe's one-way CLI sync).
- **Mobile via Expo** — only one in the cluster.
- **SaaS templates with real backends wired** (Stripe/Supabase/Clerk) — closer to "ship a product" than "ship a UI."
- **Agent+ managed-service tier** — humans-in-the-loop for hard features, an unusual hybrid offering.
- **Multi-model stack** (GPT-4 + Claude + Gemini), not locked to one vendor.

## 14. Limitations

- **React/React-Native only** — no Vue, no Svelte.
- **Pro credits cap real usage** — 150/mo is tight for active projects; top-ups are pricey.
- **Agent+ at $4,500/mo** is enterprise-only; the Pro→Agent+ price gap is huge.
- **Visual editor performance** unverified at large-app scale.
- **No PRD layer.**
- **Two-way Figma sync direction back to Figma** unclear from public docs.

---

## What KutAI i2p should learn

- **Round-trip via GitHub + VSCode** is the right answer to the drift-after-customization problem. KutAI should treat the repo as the source of truth and the canvas as a view, not the inverse.
- **Multi-model stack** (GPT-4 + Claude + Gemini) is operationally smart — different prompts hit different models. Mirrors Fatih Hoca's existing routing thesis.
- **SaaS template wiring** (Stripe + Supabase + auth pre-baked) is what turns "UI prototype" into "shipped product" — the integration layer is where i2p can win.
- **Mobile via Expo** is an under-served wedge — every other competitor is web-only. KutAI's i2p plan already has a mobile track (`05-build-mobile-track.md`); Tempo is the closest direct comparable.
- **Agent+ ($4,500/mo human-assisted)** is the price-anchor for premium AI dev services — useful upper bound for KutAI tier strategy.

## Sources

- [Tempo homepage (tempo.new)](https://www.tempo.new/)
- [Y Combinator company page](https://www.ycombinator.com/companies/tempo-2)
- [Fondo: Tempo Labs launch story (founders, technical approach)](https://fondo.com/blog/tempolabs-launches)
- [AIChief Tempo Labs review 2026](https://aichief.com/ai-design-tools/tempo-labs/)
- [Expo blog: Tempo + Expo for mobile](https://expo.dev/blog/how-tempo-and-expo-redefine-mobile-app-development)
- [Tempo Figma plugin](https://www.figma.com/community/plugin/1463689183126672406/tempo-ai-powered-figma-to-code-react)
- [Tempo Figma plugin docs](https://tempolabsinc.mintlify.app/FigmaPlugin)
- [Product Hunt: Tempo Labs](https://www.producthunt.com/products/tempo-labs)
