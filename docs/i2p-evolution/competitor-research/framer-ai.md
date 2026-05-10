# Framer AI (incl. Workshop)

**Date:** 2026-05-09
**Confidence:** M
**Category:** Designer-first site builder with AI prompt + AI components

---

## 1. What it is

Framer is a no-code site builder leaning into design taste. Its 2026 AI surface has two distinct features:

1. **Framer AI (site generation)** — text prompt → full multi-page site (layout, copy, images, navigation, responsive breakpoints) inside the Framer canvas.
2. **Workshop (⌘K)** — generate custom components from a prompt; output is editable, production-ready, with built-in property controls and automatic style matching to canvas. Workshop is the bigger 2026 leap.

## 2. Input flow

- Site generation: paragraph-style prompt ("a SaaS landing page for a project management tool with dark theme").
- Workshop: prompt for a specific component (cookie banner, visual effect, tabs, accordion).
- Wireframer (separate tool): build site page structure with AI before styling.

## 3. Output type

- Hosted Framer site.
- Editable canvas components (not just rendered HTML).
- Workshop components have property controls — designer can configure without code.

## 4. Iteration loop

Visual canvas iteration is primary. Workshop components regenerate on prompt revision; site-level regen less granular than v0's per-screen. Framer's strength is post-AI hand-editing.

## 5. Spec / charter

None as artifact. Wireframer step gives a structural pre-design but no PRD / persona / charter doc.

## 6. Multi-screen / multi-page

Yes — Framer's primary unit.

## 7. Style / theming

Canvas-bound styles (fonts, colors). Workshop matches generated components to canvas style automatically — closes the visual-coherence loop most AI-generated pieces lose.

## 8. Deploy / export

Hosted on Framer (primary). Code export limited.

## 9. Pricing

Free tier; Mini ($5/mo), Basic (~$15/mo), Pro (~$30/mo), Business / Enterprise. AI features included on paid tiers with usage limits.

## 10. Underlying model

Not specified publicly per feature.

## 11. Recent updates

- **Workshop** is the headline 2026 capability — moves Framer from "AI-assisted templates" toward "AI-generated bespoke components."
- Wireframer for structural pre-design.
- Tight integration with content translation, image gen.

## 12. Notable strengths / limitations

- **Strength:** Designer taste is real; output looks better than most prompt-to-code tools by default. Workshop closes the "components match canvas style" loop.
- **Limitation:** Marketing-site bias. Not built for stateful product apps with auth, db, complex flows.
- **Lesson for KutAI:** Workshop's "match canvas style automatically" is the same problem Builder.io's component mapping solves — KutAI's recipe library should propagate style tokens / component registry into every codegen call, not regenerate from scratch.

## 13. Sources

- [Framer AI](https://www.framer.com/ai/)
- [Framer Workshop](https://www.framer.com/workshop/)
- [Framer Wireframer](https://www.framer.com/wireframer/)
- [Framer Review 2026 — VibeCoding](https://vibecoding.app/blog/framer-review)
