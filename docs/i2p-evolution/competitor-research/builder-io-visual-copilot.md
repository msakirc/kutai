# Builder.io — Visual Copilot

**Date:** 2026-05-09
**Confidence:** M
**Category:** Design-to-code conversion (Figma → framework code), with CMS ancestry

---

## 1. What it is

Builder.io is a long-running visual CMS / page builder; its 2026 AI surface is **Visual Copilot**, a Figma plugin + CLI that converts Figma designs into React / Vue / Svelte / Angular / Qwik / Solid / HTML in real time, using a custom model trained on 2M+ data points. The differentiator vs. raw "screenshot → code" is **component mapping**: Visual Copilot reads the target codebase and rewrites the generated output to use the team's existing components instead of one-off divs.

## 2. Input flow

- Figma file (primary) — designer hands off frames.
- Optional repo connection so the AI can map design elements to existing component library.
- CLI flow: `visual-copilot` command pulls Figma assets directly into the codebase.

## 3. Output type

Code in the user's chosen framework + styling library (Tailwind, plain CSS, Emotion, Styled Components, CSS Modules). Optionally Next.js / Nuxt / SvelteKit meta-framework variants.

## 4. Iteration loop

- AI iterates the code for preferred CSS library / framework.
- Cursor + Visual Copilot integration: generate first cut, then keep editing in Cursor with full codebase context.
- Windsurf integration analogous.
- No "regen this screen with X change" UX comparable to v0; iteration happens in the developer's IDE, not in Builder.

## 5. Spec / charter

None. Visual Copilot starts from finished Figma frames; the spec/charter problem is upstream. Builder.io has separate CMS-style content models, but they're not a product spec.

## 6. Multi-screen / multi-page

Yes via Figma frames; each frame becomes a component or page. No automatic flow / routing scaffolding — that's a developer concern.

## 7. Style / theming

Honors the Figma design (colors, spacing, type). Maps to chosen styling lib. With component mapping enabled, the codebase's tokens win over Figma's raw values.

## 8. Deploy / export

Code lands in the developer's repo. Builder.io's CMS side has its own hosting, but Visual Copilot is purely a dev-side workflow tool.

## 9. Pricing

Builder.io free tier covers the Figma plugin for individuals. Paid tiers (Growth, Business, Enterprise) for team features, higher conversion volume, CMS usage. Specific Visual Copilot quotas not consistently listed.

## 10. Underlying model

Custom-trained model (2M+ data points), not a public LLM. Likely a layered system with a vision/structure model + LLM rewriter.

## 11. Recent updates

- **Visual Copilot CLI** — direct repo integration, granular per-frame conversion.
- **Cursor / Windsurf integrations** — tight handoff to agentic IDEs.
- **Component mapping** — read-the-codebase, reuse-existing-components flow.

## 12. Notable strengths / limitations

- **Strength:** Component mapping is the right insight — generic AI codegen produces unmaintainable code; mapping to an existing system is what makes it ship-able.
- **Limitation:** Requires a Figma source of truth. Doesn't help an i2p flow that wants to *generate* designs from spec.
- **Lesson for KutAI:** the "map to existing components" pattern matters when KutAI grows into multi-mission territory — recipe library / pattern reuse should hook into a similar mapper so phase-5 codegen reuses the founder's accumulated components instead of generating new ones each mission.

## 13. Sources

- [Introducing Visual Copilot — Builder.io blog](https://www.builder.io/blog/figma-to-code-visual-copilot)
- [Visual Copilot CLI announcement](https://www.builder.io/blog/visual-copilot-cli)
- [Figma to Code with Cursor and Visual Copilot](https://www.builder.io/blog/figma-to-cursor)
- [AI-Powered Figma to Code](https://www.builder.io/figma-to-code)
