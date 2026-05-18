# Onlook — Competitor Research

**Date:** 2026-05-09
**Sources:** onlook.com, github.com/onlook-dev/onlook, github.com/onlook-dev/web, LogRocket/BrightCoding writeups, openalternative.co.
**Confidence:** High on architecture (extensively documented in OSS repo); Medium on hosted pricing (gated behind sales contact).

## 1. Input Flow
Multi-modal entry:
- **Text prompt** (chat-driven generation).
- **Image prompt** (paste a screenshot, get a layout).
- **Figma import**.
- **GitHub repo import** (work on existing project).
- **Templates**.

Then iteration is **visual on the live preview**, not prompt-only.

## 2. Output Type
**Next.js + TailwindCSS source code on the user's disk.** Not a proprietary format. Files update in real-time during visual edits. Deployable via shareable Onlook link or custom domain through Freestyle hosting.

## 3. Iteration Loop
**The headline differentiator.** Three loops fused:
1. Visual: drag/resize/restyle on a live preview iframe.
2. Chat: AI generates React+Tailwind diffs.
3. Code: edit `.tsx`/`.css` directly; HMR reloads.
All three operate on the same source files concurrently — "design in the real product."

## 4. Style + Theming
Brand assets + design tokens managed in editor. Tailwind classes are the styling target. Component detection scans existing code for reusable units.

## 5. Multi-Screen / Multi-Route
Multi-page navigation supported. Responsive breakpoints maintained when dragging elements (Tailwind responsive classes auto-adjust).

## 6. Charter / Spec / PRD
**No explicit spec artifact.** Onlook is design-led; the visual canvas IS the spec. AI chat history is the closest thing to a charter trail.

## 7. Backend Integration
**Weak / nascent.** Onlook focuses on the front-end design loop. Backend wiring is left to the developer (Next.js API routes, user's choice of DB). No first-class Supabase/auth integration matching Plasmic or Replit.

## 8. Two-Way Sync — THE CORE MECHANISM
- **`data-oid` attributes** instrument the rendered DOM, mapping each node back to a source line.
- Visual edit → look up `oid` → patch corresponding JSX → HMR reload.
- AI-generated code edits stream via **diff-match-patch** so both canvas AND file tree update simultaneously.
- Code edits flow back to canvas because the canvas is just a live Next.js dev server in an iframe.

This is the most elegant two-way sync of the three tools surveyed.

## 9. Deploy
- **Self-hosted:** local via Bun runtime (Apache-2.0 OSS).
- **Hosted:** onlook.com cloud, deploys via Freestyle.
- Project containerization via **CodeSandbox SDK + Docker** for hosted previews.

## 10. OSS vs SaaS
**Apache 2.0 OSS, fully self-hostable.** Hosted version exists for convenience. Positioned as open-source alternative to Bolt.new, Lovable, V0, Replit Agent, Figma Make, Webflow.

## 11. Pricing
- Self-hosted: free.
- Hosted: not published — "contact sales / book demo." (Confidence: Medium — pricing was gated as of fetch; may have changed.)

## 12. Underlying Model
**OpenRouter** (provider-agnostic LLM access) plus **Morph Fast Apply** and **Relace** for code-edit application. Morph/Relace specialize in fast diff application — distinct from the planning model. So Onlook splits "what to change" (OpenRouter to a frontier model) from "how to apply it" (Morph/Relace).

## 13. Recent Updates 2025-2026
- v0.2.32 (2025-07-17) latest tagged release at fetch time.
- Active development; project flagged "under development."
- Web version (`onlook-dev/web`) splits desktop and browser flavors.
- Roadmap mentions non-Next.js + non-Tailwind support.

## 14. Limitations
- **Stack-locked** to Next.js + Tailwind today.
- **Backend-thin** — not an idea-to-product tool, more a design-to-product tool.
- **Pre-1.0** — instability expected.
- **Hosted pricing opaque** — friction for evaluation.
- **Image-prompt fidelity** likely limited (true of all such tools).

## What this teaches i2p
1. **`data-oid`-style instrumentation is the unlock for two-way edits.** If KutAI ever exposes a visual review canvas, instrumenting generated code with stable IDs back to source is the right pattern.
2. **Split planning model from apply model.** Onlook uses a frontier LLM for diff generation but Morph/Relace for diff application — fast, cheap, deterministic. KutAI's coder agent could similarly delegate "apply this patch" to a non-LLM or small-model lane.
3. **Source-of-truth = files on disk, not platform DB.** Onlook's "your code is yours" stance maps to KutAI's user-repo-first philosophy. Avoid platform lock-in.
4. **Three iteration loops fused (visual/chat/code) on same source.** KutAI's iteration today is chat-only via Telegram — adding even a read-only visual preview that links back to source files would be a leap.
5. **Provider-agnostic via OpenRouter** — Fatih Hoca already does this conceptually; Onlook validates it as a product strategy.
