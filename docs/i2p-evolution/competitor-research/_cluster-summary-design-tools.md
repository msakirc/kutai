# Cluster Summary — AI Design Tools (Stitch, Uizard, Visily)

**Date:** 2026-05-09
**Cluster:** AI design tools (idea → UI prototype)
**Files:** [stitch-google.md](./stitch-google.md), [uizard.md](./uizard.md), [visily.md](./visily.md)

## Comparison Table

| Dimension | Stitch (Google) | Uizard (Miro) | Visily |
|---|---|---|---|
| Founded / Origin | 2025 (rebrand of Galileo AI, acq. mid-2025) | 2018, Copenhagen, acq. by Miro 2024-05-27 | ~2021, independent |
| Primary input | Prompt + image + DESIGN.md + voice | Prompt + sketch + screenshot | Prompt + screenshot + sketch + URL + template |
| Output fidelity | "Colored wireframe" → mid-fi | Mid-fi mockups, themed | High-fi wireframes, themed |
| Multi-screen | Up to 5, auto-linked | Multi-screen flows, themed | Unlimited (Pro), Smart Components |
| Theming | Prompt + DESIGN.md | Theme generator + brand kits | URL/image extraction + premade |
| Iteration | Chat + voice + per-element | Conversational section edits (AD 2.0) | Chat + drag-drop + collab cursors |
| Figma export | Yes (free) | No (SVG workaround) | Yes (Pro+) |
| Code export | HTML/CSS, MCP to IDE | React + CSS (Pro) | HTML/CSS (Pro) |
| Image gen | Composes with Nano Banana | Library only | Library only |
| Pricing | Free (Labs phase) | Free / $12 / $39 / Custom | Free / $11 / $29 / Custom |
| Model | Gemini 3.0/2.5 Pro/Flash (selectable) | Proprietary (undisclosed) | Undisclosed |
| Spec/PRD artifact | DESIGN.md (design-system only) | None | None |
| Biggest weakness | Generic visual output, ~3-screen cohesion ceiling | No Figma export, AI quality "amateur" | No auto-layout, shallow prototyping |
| Killer feature | MCP IDE round-trip + voice canvas | Conversational section-level edits | URL theme extraction + screenshot-to-editable |

## Cross-Tool Patterns

1. **Input pluralism is table stakes.** All three accept text + image + sketch. Differentiation is at the *edges*: Stitch adds voice + DESIGN.md, Visily adds URL theme extraction, Uizard leans on its pix2code research heritage. None settled on a single canonical input.
2. **No PRD step.** All three jump from intent → UI. None produces a charter, spec, or PRD as a first-class artifact. This is the wide-open lane KutAI's i2p sits in — pre-design phases are not commodified.
3. **Theming is decoupled from screens** in two of three (Visily extracts; Uizard generates). Stitch's DESIGN.md is the same idea in markdown. The artifact schema lesson: theme tokens ≠ component instances ≠ screen layouts; keep them separable.
4. **Multi-screen consistency degrades fast.** Even Google admits ~5-screen ceiling with cohesion loss past 3. Nobody has cracked 20-screen apps from a single prompt — chunking + per-flow regeneration is universal.
5. **Figma is the de-facto handoff target**, not code. Code export exists but is "good enough for prototype, not production" across all three. MCP (Stitch's bet) is the first credible alternative — IDE round-trip beats one-shot file dumps.
6. **Acquisition is the exit pattern.** Galileo → Google (2025), Uizard → Miro (2024). Pure-AI-design as a standalone product has not held independence at scale; it gets absorbed into a collaboration platform or a model lab. Visily is the lone independent.
7. **Pricing converges**: free starter with credit caps, ~$11-12/editor/mo Pro, ~$29-39 Business, custom Enterprise. Credit-based + seat-based hybrid is the norm.
8. **Output quality plateau.** All three plateau at "good enough for stakeholder demo, not production." Reviewer language is identical across reviews — "feels like a colored wireframe," "amateur," "needs designer refinement."

## What i2p Should Learn From This Cluster

1. **Own the pre-design layer.** None of these tools produce a charter, PRD, user-research synthesis, or feature-priority artifact. KutAI's Z0/Z1 phases sit upstream of where this cluster begins. *Don't compete on the design step — feed it.*
2. **Adopt MCP for design-tool handoff.** Rather than build a UI generator, have i2p hand a `DESIGN.md` + spec into Stitch/Visily/Figma via MCP and let the design tool render. i2p's value is the *upstream* spec, not the pixels.
3. **Theme as a separable artifact.** i2p's design artifact schema must split: `theme.json` (tokens), `components.json` (smart components), `screens.json` (instances). Mirrors all three tools and enables cheap theme swap.
4. **Chunk multi-screen generation.** Even Google can't do >5. i2p's mobile-track (Z5) should plan in 3-5 screen flows, not "generate the whole app."
5. **URL theme extraction is the highest-leverage cheap pre-step.** Implement before any LLM design call. Visily proved the pattern.
6. **Plan for the acquisition trap.** This cluster gets absorbed. KutAI's defensibility is the autonomous *agent loop* (Telegram-driven, idea → product, multi-phase) — not any single tool replacement. Stay above the design layer.
7. **Voice + chat-driven iteration is the converging UX.** Stitch added voice; the rest will follow. KutAI's Telegram interface is structurally well-positioned — voice notes already work as input, and conversational iteration is native to chat.
8. **Code export quality is the unsolved problem.** Whoever generates production-quality, framework-native code from design wins the next round. KutAI's Z2-Z5 build phases can leapfrog by treating *code generation* as the primary output and design as a deliverable along the way, inverting the cluster's priority.
