# Plasmic — Competitor Research

**Date:** 2026-05-09
**Sources:** plasmic.app, github.com/plasmicapp/plasmic, G2/Capterra/SaaSWorthy 2026 listings.
**Confidence:** Medium-High on product/integration; Medium on AI features (Plasmic's AI story is under-marketed compared to Onlook/Replit).

## 1. Input Flow
Primary input is **drag-and-drop in Plasmic Studio** (visual canvas), not prompt. Secondary inputs:
- **Figma import** via "best-in-class Figma to code plugin" — translates Figma to DOM/CSS.
- **Codebase import** — register existing React components so they appear in the visual builder.
- AI assist exists but is auxiliary, not the entry point. (Confidence: Medium — Plasmic's homepage downplays AI vs competitors.)

## 2. Output Type
**Production React code OR headless API rendering.** Three delivery modes:
1. **Headless / loader API** — fetch designs at runtime (CMS-style).
2. **Codegen** — sync as actual React components into a repo.
3. **Custom-component registration** — your existing components become drag-and-drop primitives.

Output ships into the user's existing codebase; Plasmic itself doesn't host the app (though hosted preview exists).

## 3. Iteration Loop
**Visual canvas first, code second.** Branching + multiplayer editing + review/approval before publish. Non-developers (marketing/content) iterate without devs in loop. Iteration is WYSIWYG; chat is not the primary loop.

## 4. Style + Theming
Design tokens, design systems built on react-aria for accessibility primitives. Responsive layout system. "Full design freedom" — not template-locked. Theme/brand assets managed in Studio.

## 5. Multi-Screen / Multi-Route
Yes — pages, components, and CMS models are first-class. Internationalization built in. Static site generation supported.

## 6. Charter / Spec / PRD
**No spec artifact.** Plasmic is a builder, not a planner. The "spec" is the visual design itself.

## 7. Backend Integration
- Built-in connectors: **Supabase, Contentful, Shopify**, plus arbitrary HTTP/GraphQL.
- Plasmic ships its **own headless CMS** (flexible content modeling, i18n, versioning, A/B testing, segmentation).
- Auth + roles built in.

## 8. Two-Way Sync
**Limited.** Codegen mode syncs designs → code, but edits in code don't flow back to the visual canvas the same way Onlook does. Custom-component registration is the closest thing to two-way (code defines components, Studio composes them).

## 9. Deploy
User-controlled. Plasmic generates artifacts; users deploy to their own infra (Vercel, on-prem, behind firewall). Plasmic also offers hosted publish via CDN.

## 10. OSS vs SaaS
**Hybrid.** GitHub repo (`plasmicapp/plasmic`) is open: most code MIT, `platform/` directory AGPL. Studio (the visual editor) is the SaaS hook. 6.8k stars, 675 forks (as of fetch).

## 11. Pricing (2026)
- Free tier exists.
- **Pro: $10/user/month** entry point per G2/Capterra.
- Higher plans (~$100/mo+) for teams with branching, advanced CMS, segmentation.
- Enterprise custom.

## 12. Underlying Model
**Not publicly disclosed.** Plasmic does not market a specific LLM partnership the way Replit/Onlook do. AI features (component generation, design assist) appear lightweight and auxiliary.

## 13. Recent Updates 2025-2026
- Active project (commits, releases ongoing).
- Project Jam streams covering cookies/analytics/auth.
- No flagship "Plasmic AI" launch comparable to Agent 3/4 or Onlook AI chat.
- (Confidence: Medium — Plasmic's marketing has not loudly pivoted to AI-first.)

## 14. Limitations
- **AI is not the headline.** Users coming from Cursor/Lovable/Replit may find the AI assist underwhelming.
- **Visual-first means designers/marketers lead** — not idea-to-product from text.
- **Codegen complexity** — keeping Plasmic Studio + repo in sync requires CI discipline.
- **Vendor lock-in risk** in headless mode (designs live in Plasmic's API).
- Pricing per-seat scales painfully for large content teams.

## What this teaches i2p
1. **Component-registration pattern** — i2p workflows could let users register their own existing React components as building blocks; the agent then composes rather than re-generates everything.
2. **Branching + review for non-dev iteration** — KutAI's Telegram review flow could mirror Plasmic's review-before-publish gating.
3. **Headless vs codegen split** — i2p could ship the same artifact in two modes: live preview (hosted) and exported code (user repo). Don't force one path.
4. **CMS-as-companion** — Plasmic's value isn't just the builder; it's the CMS+builder bundle. KutAI's todo/mission DB could be exposed similarly.
