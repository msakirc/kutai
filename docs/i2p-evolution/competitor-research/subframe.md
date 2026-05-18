# Subframe — Designer-Grade AI Tool Deep-Dive

**Domain:** subframe.com (verified)
**Founded:** ~2023 (early-stage, YC alum)
**One-liner:** "The AI design tool built for code." Pixel-perfect React+Tailwind, no AI slop.
**Research date:** 2026-05-09
**Confidence:** Medium-high. Pricing and tokens corroborated across 3 sources; underlying LLM not disclosed.

---

## 1. Input flow

- **Visual canvas first.** Drag-and-drop on auto-layout components — not a free-canvas Figma clone. Everything you place is already a real React component with props/variants.
- **AI assistance "anywhere":** "Ask AI" generates designs from a prompt OR an image, but it generates **using your design system** (tokens + components), not blank-slate.
- **Inline AI** for direct edits on a selected element.
- **Image clone** path supported (paste a screenshot, get a Subframe-rendered approximation).
- No spec/PRD upload path advertised — input is design-shaped, not requirement-shaped.

## 2. Output type

- **React + TypeScript + Tailwind + Radix UI primitives.** Synced to your repo via `npx @subframe/cli sync`.
- Code is **not LLM-generated at export time** — the canvas IS the source of truth, exporter is deterministic. Reviewer (DesignWhine) calls it "a skin over an actual component library."
- Output granularity: components, pages, full multi-page apps. Not a runnable backend.

## 3. Iteration loop

- **Both chat AND visual.** Designers iterate visually; AI variants are generated in batches (multiple distinct directions, not a single re-roll), no per-generation credit gating on Pro.
- **Real-time multiplayer + comments on layers** (Figma-style).
- **Engineer loop:** `subframe sync` pulls updated React into the local repo; design changes propagate as code diffs in the codebase.

## 4. Token system

- **First-class.** Tokens for colors, typography, shadows, borders held centrally as the project's "Theme."
- **Sync target = Tailwind config.** Editing `Brand 600` re-themes every button/badge using that token across all pages, both in the canvas AND in the exported `tailwind.config`.
- Atomic-design-style cascade: tokens → primitives → components → pages.

## 5. Multi-screen consistency

- Strong. Components are shared instances with props/slots/variants — change a component once, propagates everywhere.
- Responsive: built-in auto-layout + mobile breakpoints.
- **Caveat:** canvas capped at 1280px width — anything wider is handled in code, not visually.

## 6. Bring-your-own component library

- Yes — the model is "your existing design system lives in Subframe." 47 built-in components ship as Radix-based starters (their open-source repo `SubframeApp/subframe` confirms this), and you can fork/extend or replace.
- **Limitation:** React-only. No Vue/Svelte support.

## 7. Charter / spec / PRD generation

- **None.** Subframe is design-side-of-handoff. No PRD, no acceptance-criteria capture, no requirements doc generation. This is where it leaves room for an i2p layer above it.

## 8. Two-way sync

- **One-way (design → code) primarily.** CLI sync pushes Subframe → repo.
- The MCP server lets coding agents (Claude Code, Cursor, Codex) **read** the design system back, but custom logic written in code does NOT round-trip back into the canvas.
- Reviewer (DesignWhine) flags this as "potential drift" once devs extend components.

## 9. Export

- React/TS/Tailwind/Radix into your repo (local files, you own them).
- No Figma export advertised (Subframe positions itself as the Figma alternative, not exporter to it).
- No Vue, no Svelte, no plain-CSS-only path.

## 10. Pricing

| Plan | Cost | Key limits |
|---|---|---|
| Free | $0 | 1 project, 5–10 pages, 1 AI prototype, 24h history |
| Pro | $29/editor/month | unlimited projects/pages/AI, custom fonts, 7d history |
| Custom | Enterprise | contact sales |

Sources differ slightly on free-tier page count (5 vs 10) — likely changed during 2025-2026.

## 11. Underlying model

- **Not disclosed publicly.** Their MCP integration name-drops Claude/Cursor/Codex as **clients**, but the model behind "Ask AI" is unstated. Likely a frontier model wrapped with heavy system-prompting around the design-system context.

## 12. Recent updates (2025-2026)

- **MCP server** for AI coding tools (Claude Code, Cursor, Codex, Replit) — major 2025-2026 push.
- **Agent Skills** (mentioned alongside MCP).
- Open-source `SubframeApp/subframe` repo (CLI + Radix components + docs).
- Continued positioning shift from "design tool" toward "design-system-of-record for coding agents."

## 13. Differentiators claimed (vs v0/Lovable)

- **Real component library, not vibe-codegen.** Output matches what you already use in production rather than an LLM hallucinating Tailwind classes per call.
- **Designer-grade UX** (visual canvas, multiplayer, comments) instead of chat-only.
- **No "AI slop":** export step is deterministic, not LLM-mediated.
- **Designed for coding agents to consume** (MCP-first stance is unusual).
- vs Figma: production code is the source of truth, not a side-export.

## 14. Limitations

- **No blank-canvas / brand exploration.** Auto-layout-only; reviewers call it "genuinely limiting" for early-stage visual brainstorming.
- **One-way sync.** Devs extending components → drift risk.
- **React-only.**
- **1280px canvas cap** awkward for marketing pages or wide dashboards.
- **Performance degrades** on complex pages (in-browser rendering bottleneck).
- **No PRD/spec layer** — must be supplied externally.
- **Underlying LLM opacity** — hard to predict Ask AI quality/cost trajectory.

---

## What KutAI i2p should learn

- **Tokens-as-Tailwind-config** is the cleanest design-system-to-code primitive seen in this cluster. The canvas owns the abstract tokens; the codegen target is concrete and version-controllable.
- **Component-library-as-source-of-truth + deterministic export** kills a huge class of LLM hallucination. AI assists composition, not pixel-pushing.
- **MCP-first** means Subframe is positioning itself as a dependency for coding agents — KutAI could either consume Subframe (cheap) or provide the same shape itself (expensive but owned).
- The missing PRD/spec layer is the **gap an i2p tier above this should occupy**.

## Sources

- [Subframe homepage](https://www.subframe.com/)
- [Subframe Concepts docs](https://docs.subframe.com/learn/concepts)
- [DesignWhine review (4.5/5, deep production-readiness teardown)](https://www.designwhine.com/subframe-review-a-design-tool-production-ready/)
- [Banani.co Subframe + MCP review](https://www.banani.co/blog/subframe-ai-review)
- [SubframeApp/subframe GitHub (CLI, Radix components)](https://github.com/SubframeApp/subframe)
- [Subframe — Design Systems page](https://www.subframe.com/design-systems)
