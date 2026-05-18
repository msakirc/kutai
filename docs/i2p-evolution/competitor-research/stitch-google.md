# Stitch (Google) — Competitor Research

**Date:** 2026-05-09
**Cluster:** AI design tools
**Confidence:** Medium-High (official Google blog + multiple reviews; some specifics on internals are inferred)

## TL;DR

Stitch is Google Labs' AI UI generator, launched as the rebrand/successor of Galileo AI after Google's mid-2025 acquisition. Free during Labs phase. The March 2026 "Stitch 2.0" update introduced a multi-screen "vibe design" canvas (up to 5 connected screens), voice commands, MCP server for IDE round-trip, and Figma + HTML/CSS export. Powered by Gemini 3.0 / 2.5 Pro/Flash with model selection.

## 1. Input Flow

- Primary: **natural-language prompt** describing business objective, user feeling, or inspiration. ("vibe design" — describe intent, not specs.)
- Image upload (screenshots, inspiration boards).
- `DESIGN.md` upload — agent-friendly markdown design-system file.
- Voice commands on the canvas (real-time critique + edit).
- Code snippets allowed as canvas context.
- Iteration: chat in canvas; multiple rounds against generated screens.

## 2. Output Type

- High-fidelity UI screens (not just wireframes), but reviewers consistently call them "colored wireframes" — flat, generic visuals.
- Up to **5 interconnected screens** per generation with logical navigation.
- Interactive prototype via "Play" button (auto-wired without manual hotspot setup).
- Exports: **Figma** file, **HTML/CSS** code, `.zip` bundle.

## 3. Iteration Loop

- AI chat editing inside the canvas.
- Per-element regeneration supported.
- Voice-driven edits ("real-time design critiques").
- Design agent "reasons across the entire project's evolution" — implies project-level memory, not per-message.
- Limit: 15 redesign credits/day in standard tier; reviewers report difficulty maintaining cohesion past 3 related screens.

## 4. Style + Theming

- Theme generated from prompt and (optionally) `DESIGN.md`.
- No first-class palette upload UI surfaced in reviews; theming primarily prompt-driven.
- Visual consistency across screens reportedly automatic.

## 5. Multi-Screen Consistency

- Yes — up to 5 screens, automatic user-journey mapping between them.
- Single design agent maintains cross-screen design system.
- Limitation: cohesion degrades past ~3 screens for complex flows.

## 6. Charter / Spec / PRD Generation

- No PRD artifact. Closest: `DESIGN.md` is a portable design-system spec, not a product spec.
- Stitch jumps directly from intent → UI; no requirements step.

## 7. Image Generation

- UI-internal images (hero, product visuals, backgrounds) integrate with **Nano Banana / Nano Banana Pro** (Gemini 2.5 Flash Image / Gemini 3 Pro Image) — separate model, not auto-pipelined into Stitch output by default but composable.
- Stitch itself focuses on UI structure; visual assets are a layered workflow.

## 8. Export

- **Figma** export (file or via Figma plugin path).
- **HTML/CSS** code export.
- `.zip` bundles for sharing.
- **MCP server** — two-way sync with IDEs (Cursor, Blackbox, Antigravity, Claude Code-compatible per ecosystem reports).
- AI Studio handoff for further code work.

## 9. Pricing

- **Free** during Google Labs phase (still free as of May 2026).
- Limits (post-2.0 expansion):
  - 350 standard generations/month (Gemini 3.0 Flash / 2.5 Flash).
  - 200 experimental Pro generations/month (Gemini 3.0 Pro / 2.5 Pro).
  - 400 daily design credits + 15 redesign credits.
- No credit top-up. Paid plans expected Q4 2026 (analyst speculation: 30-50% under Figma).

## 10. Underlying Model

- **Gemini 3.0 Pro / Flash** and **Gemini 2.5 Pro / Flash** with user-selectable model.
- Image generation pairs with Nano Banana (Gemini-image family).

## 11. Recent Updates (2025-2026)

- **Mid-2025**: Google acquired Galileo AI.
- **2025**: Relaunched as Stitch (Google Labs).
- **March 18-19, 2026**: Stitch 2.0 — vibe design, AI-native infinite canvas, 5-screen multi-screen, voice commands, MCP server, model picker, expanded free quota, Figma export polished.
- Skills/SDK framework released alongside (2.4k stars per Google blog reference).

## 12. Limitations

- Output "feels like a colored wireframe" — flat buttons, generic cards, lacks polish.
- Figma imports often have cluttered/poorly-structured layers — designers re-do hierarchy.
- Hard cap at ~3 cohesive screens; 5-screen flows lose consistency.
- No credit top-up; daily limits force pacing.
- HTML/CSS export quality acceptable for prototypes, not production.
- Limited freehand annotation / sketch input (text + images dominate).
- No PRD or upstream product-spec artifact — designers/PMs must bring intent.

## What i2p Should Notice

1. **`DESIGN.md`** as a portable, agent-readable design contract is exactly the kind of artifact i2p's pre-code phases should produce — Stitch's bet is that a markdown design-system file is the right unit for handoff to coding agents. Consider a `UI.md` / `THEME.md` artifact in i2p Z1.
2. **MCP two-way sync** — Stitch chose MCP as the IDE bridge. i2p Z6 (real-world bridge) should treat MCP as the export interface, not a one-shot file dump.
3. **5-screen ceiling is a real constraint** — even Google can't keep cohesion past ~3-5 screens. i2p should chunk multi-screen generation rather than ask one prompt for 20 screens.
4. **"Vibe design" framing** (intent-first, not spec-first) is the user-facing prompt pattern — but reviewers flag it produces generic output. i2p should pair intent with concrete UI patterns from the skill library.

## Sources

- [Stitch official site](https://stitch.withgoogle.com/)
- [Google Labs blog: Design UI using AI with Stitch](https://blog.google/innovation-and-ai/models-and-research/google-labs/stitch-ai-ui-design/)
- [Banani — Google Stitch AI Review 2026](https://www.banani.co/blog/google-stitch-ai-review)
- [Banani — Google Stitch Pricing 2026](https://www.banani.co/blog/google-stitch-pricing-and-credits)
- [NxCode — Stitch Pricing 2026](https://www.nxcode.io/resources/news/google-stitch-pricing-plans-complete-guide-2026)
- [Tech-Insider — Vibe Design and 5-Screen Canvas](https://tech-insider.org/google-stitch-ai-design-tool-march-2026-update/)
- [UXPin — Stitch Updates & Alternatives](https://www.uxpin.com/studio/blog/google-stitch-ai-design-tool-updates-ui-ux/)
- [Imagine.art — Stitch + Nano Banana workflow](https://www.imagine.art/blogs/google-stitch-overview)
- [Google DeepMind — Nano Banana Pro](https://deepmind.google/models/gemini-image/pro/)
