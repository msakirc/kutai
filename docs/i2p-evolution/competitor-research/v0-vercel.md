# v0 (Vercel)

> Research date: 2026-05-09. Domain rebranded `v0.dev` → `v0.app` in Jan 2026; old URLs 301 to the new host.

## What it is

v0 is Vercel's AI app builder. It started in late 2023 as a UI-component generator (React + shadcn/ui + Tailwind) and over 2025 was rewritten into a full-stack "AI agent that helps anyone create real code and full-stack apps and agents" — UI, backend, DB, auth, deploy. Vercel positions it as the "vibe-coding" front door to its hosting platform; deployment is one-click to Vercel and code can be pushed to GitHub.

It is now a product surface (chat + visual design mode + sandbox runtime + Git panel + iOS app) on top of a composite model pipeline. The brand mission per Vercel's blog is that "2026 will be the year of agents" with end-to-end agentic workflows authored in v0.

## Input flow

- **Primary input**: a single natural-language prompt in a chat window. No mandatory clarification rounds — v0 jumps straight to a generation on the first turn.
- **Multi-modal**: accepts screenshots, file uploads, and a Figma import for converting mockups to code. ChatPRD and similar tools have an "Open in v0" handoff that pastes a structured PRD into the chat.
- **GitHub import**: can import an existing repo and pull Vercel env vars into the sandbox.
- **Clarification behavior**: v0 does not run a charter/persona/requirements-elicitation pass. Users (or external PRD tools) front-load context themselves. Best-practice guides explicitly tell users to "break the PRD down by UI sections" because v0 expects component-flavored input, not a discovery dialogue.

## Output type + stack

- **Stack**: Next.js + React + Tailwind + shadcn/ui is effectively locked. Vercel's blog phrasing: "modern tools like Next.js, Tailwind, shadcn/ui, and more." Reviews repeatedly flag this as a constraint ("React-only output, reliance on specific libraries like Tailwind CSS").
- **Output unit**: full apps with routes/pages, server actions, DB integrations (Snowflake, AWS, etc.), auth — runnable inside a Vercel-hosted sandbox runtime. Old "Block" (single-component) output is still supported but no longer the default.
- **Code shape**: you get real source files in the sandbox; "Quick Edit" and "AutoFix" models patch them streamingly. No native option to target Vue/Svelte/SwiftUI/Flutter.

## Iteration loop

- Chat-driven regen: every user turn is a delta on the running project. v0 versions each generation; you can fork or roll back.
- **Visual Design Mode**: live preview where you click into the rendered UI to fine-tune details (colors, copy, spacing) without retyping a prompt.
- **Quick Edit model**: dedicated specialist model for "updating text, fixing syntax errors, or reordering components" so trivial edits don't burn a full base-model call.
- **AutoFix (`vercel-autofixer-01`)**: streaming post-processing model that fixes errors in <250ms during generation, and a "LLM Suspense" layer that swaps non-existent lucide-react icons within 100ms.
- **Known pain**: manual edits get overwritten by subsequent generations — "carefully refined customizations disappearing when you ask v0 to make other changes" is a recurring complaint.

## Multi-screen / multi-route

Yes — since the 2025 full-stack rewrite, v0 emits multi-route Next.js apps (templates ship dashboards, e-commerce, landing pages) rather than single screens. Earlier versions were component-only; that limitation is gone. Routes still tend to be added incrementally inside one chat rather than planned up-front from a sitemap.

## Charter / spec generation (or absence)

**Absent.** v0 has no built-in PRD/persona/charter step. The official docs page titled "PRD design" is guidance for users to *write their own* PRD in component-shaped chunks before prompting. Third-party tools (ChatPRD's "Open in v0", external PRD generators) fill that gap. Custom Project-level instructions and uploaded style guides exist on Team plans, but those are static config, not an interactive elicitation flow.

This is the single biggest functional gap vs. an i2p workflow that owns idea → spec → design → build.

## Image generation

v0 does **not** ship a first-party image-generation step inside the build flow. Generated apps use placeholder images by default. Real image generation is delegated to:
- The Vercel AI Gateway image-generation capability (DALL-E, SDXL via API, last updated March 2026), which an *app you build with v0* can call.
- Manual asset upload / brand-image swap by the user.

So images are an integration concern, not a product feature of v0 itself.

## Style + theming

- **Design Systems** feature: v0 can scaffold a design system (colors, typography, spacing) for a project; the system is then referenced across subsequent generations.
- Project-level custom instructions on Team plans let you pin a style guide.
- Inference: v0 will infer a palette/typography from the prompt by default and surface it in Visual Design Mode for tweaking.
- shadcn/ui tokens are the underlying primitive, so theming is constrained to what shadcn exposes.

## Component library awareness

Tightly opinionated:
- shadcn/ui is the default component vocabulary.
- lucide-react for icons (with the AutoFix layer auto-correcting hallucinated icon names).
- Tailwind utility classes for layout.
- Next.js conventions (App Router, server actions) for backend wiring.
- DB connectors: Snowflake, AWS integrations cited; Vercel Postgres/KV/Blob via the platform.
- Beyond the shadcn/Tailwind/Next stack, awareness of arbitrary npm libraries is best-effort via the base model + RAG, not a curated catalog.

## Deploy

- One-click deploy to Vercel is the headline path.
- **Git panel**: branch creation and PR opening directly from the chat — added in 2025 along with GitHub-repo import and env-var sync.
- Generated apps run inside a Vercel-managed sandbox runtime during iteration, so preview ≠ production deploy is a single button.
- No first-party deploy to non-Vercel hosts; you can pull source via GitHub and deploy elsewhere manually.

## Pricing

(From `v0.app/pricing`, USD.)

| Tier | Price | Credits / limits | Notes |
|---|---|---|---|
| Free | $0 | $5/month credits, **7 messages/day** hard cap | Visual Design Mode, GitHub sync, Vercel deploy |
| Team | $30/user/mo | $30/user credits + $2/day login bonus per user | Shared chats, Projects, centralized billing, SSO |
| Business | $100/user/mo | Same credit pool as Team | Training opt-out by default, priority support |
| Enterprise | Custom | Priority queue, SAML SSO, RBAC, SLAs | "Your data is never used for training" |

**Token-based model pricing** (per 1M tokens, in/out):
- v0 Mini — $1 / $5
- v0 Pro — $3 / $15
- v0 Max — $5 / $25
- v0 Max Fast — $30 / $150 (2.5× faster)

Cache-token discount across all tiers. Credit consumption is variable per generation — a HN post titled "A working app, a happy friend, and a $50 bill" captures the predictability complaint.

## Underlying tech

- **Composite model architecture** (Vercel's term). Base reasoning model is swappable; pipeline stays stable.
- Base models: `v0-1.5-md` and `v0-1.5-lg` use **Claude Sonnet 4**; legacy `v0-1.0-md` used Claude Sonnet 3.7. Not GPT-4.
- **RAG** layer over hand-curated code examples + docs; injected into prompts via embeddings + keyword match. Vercel explicitly chose a read-only example filesystem over web search to keep prompts cacheable.
- **Quick Edit model** for trivial diffs.
- **`vercel-autofixer-01`** fine-tuned model — comparable quality to GPT-4o-mini but "10 to 40 times faster" on fixes.
- **LLM Suspense** streaming layer corrects common errors mid-stream (<100ms) without an extra model call.
- Reported headline metric: `v0-1.5-md` produces error-free code 93.87% of the time vs. 64.71% for raw Claude Sonnet — Vercel's case for the wrapper.
- Public model API: v0 models are exposed via an API, also available as a litellm provider.

## Recent updates 2025-2026

- **Full-stack rewrite (2025)**: from component generator to app builder with sandbox runtime, DB/API/auth support.
- **Git panel**: branch/PR from chat.
- **GitHub repo import + env-var sync** from Vercel projects.
- **Token-based billing** replaced fixed credit counts; v0 Mini/Pro/Max/Max Fast tiers introduced.
- **Composite model family blog** (v0-1.0-md → v0-1.5-md/lg with Sonnet 4 base).
- **Team plans GA**: Projects with custom instructions, shared chats/Blocks, SSO, centralized billing.
- **iOS app** for prompt/iterate from mobile.
- **Domain move**: `v0.dev` → `v0.app` (Jan 2026).
- **Stated 2026 direction**: "year of agents" — author end-to-end agentic workflows (with AI models included) and deploy on Vercel infra.
- **Security incidents**: April 2026 Vercel breach via a Context AI hack exposed limited customer credentials; July 2025 reporting on cybercriminals using v0 to mass-produce phishing login pages — both reputational, not v0-product-feature changes.

## Limitations

Most-cited from reviews and HN/Reddit:
1. **React/Next/Tailwind/shadcn lock-in.** No Vue, Svelte, SolidJS, native mobile, or non-Tailwind styling.
2. **Manual edits clobbered** by subsequent generations.
3. **Export pain**: blank screens, incomplete exports, components that work in v0 preview but break in prod.
4. **Debugging is thin** — server-side exceptions can leave you unable to even preview the project; few first-class debug tools.
5. **Cost unpredictability** under token-based billing for complex full-stack generations.
6. **Quality regressions on long sessions** — users report v0 becomes "buggy to the point of being unusable" deep in a chat; prompts fail to complete.
7. **Misaligned non-developer expectations** — sentiment data (Reddit) shows ~55% positive; the gap is impressive demo vs. production-grade output.
8. **No spec/charter elicitation** — the model jumps straight to UI; you bring the requirements.
9. **No first-party image generation** in the build flow.
10. **Vercel-coupled deploy** — leaving the platform requires manual GitHub-sourced redeploy.

## Sources (URLs)

- https://v0.app/ (homepage; redirected from v0.dev)
- https://v0.app/pricing
- https://v0.app/docs
- https://vercel.com/blog/v0-composite-model-family
- https://vercel.com/blog/how-we-made-v0-an-effective-coding-agent
- https://vercel.com/blog/v0-plans-for-teams
- https://vercel.com/blog/introducing-the-new-v0
- https://vercel.com/blog/category/v0
- https://chat.v0.dev/docs/prd-design
- https://www.chatprd.ai/docs/v0-integration
- https://capacity.so/blog/what-is-v0-dev
- https://www.taskade.com/blog/v0-review
- https://trickle.so/blog/vercel-v0-review
- https://news.ycombinator.com/item?id=45163212 ("$50 bill" weekend)
- https://news.ycombinator.com/item?id=47824463 (April 2026 Vercel security incident)
- https://thehackernews.com/2025/07/vercels-v0-ai-tool-weaponized-by.html
- https://thehackernews.com/2026/04/vercel-breach-tied-to-context-ai-hack.html
- https://www.nxcode.io/resources/news/v0-by-vercel-complete-guide-2026
- https://www.nxcode.io/resources/news/v0-alternative-2025
- https://docs.litellm.ai/docs/providers/v0
- https://aiengineerguide.com/blog/vercel-v0-api/
- https://vercel.com/docs/ai-gateway/capabilities/image-generation

## Implications for KutAI i2p

- **Spec/charter is a wide-open moat.** v0 has no idea→PRD→persona elicitation; users (or ChatPRD-style external tools) front-load the context. KutAI's Phases 0-3 (charter, persona, requirements, executable spec) are the part v0 deliberately skipped. Don't compete on UI codegen quality — compete on owning the discovery + spec arc that v0 leaves to humans.
- **Composite-pipeline pattern validates KutAI's multi-package architecture.** v0's "swap base model freely + keep specialist models (Quick Edit, AutoFix, Suspense) stable" is the same shape as Fatih Hoca + HaLLederiz Kadir + Doğru mu Samet + Mr. Roboto. The interesting lesson is the *streaming* post-processors (<250ms autofix, <100ms icon swap) — KutAI's quality checks today run after generation, not during the stream. A streaming guard layer is a cheap win.
- **Stack lock-in is a feature, not a bug, for codegen reliability.** v0 hits 93.87% error-free by constraining itself to one stack with curated RAG examples. KutAI's coder agents would benefit from a similarly narrow "blessed stack" with hand-curated example library, rather than open-ended any-language codegen.
- **Visual Design Mode parallels matter for Z2.** v0's "click the rendered UI to edit" loop is the iteration UX users now expect after a prototype lands. A Telegram-only iteration loop (text-only deltas) will feel primitive; Z2 should plan for a web preview surface where the user clicks → KutAI patches.
- **Pricing telegraphs an upper bound on value capture.** Free tier is 7 messages/day; $5 of "credits" burns fast on full-stack runs ($50 for one weekend project is a documented HN data point). KutAI's local-LLM-first economics are a real edge for hobbyist/founder users priced out of v0 Max usage — lean into "unlimited iterations on your own GPU" as a positioning angle rather than matching feature breadth.
