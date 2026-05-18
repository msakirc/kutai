# Bolt.new — Competitor Research

**Vendor:** StackBlitz
**URL:** https://bolt.new
**Researched:** 2026-05-09
**Researcher:** subagent (web-only, no hands-on)
**Confidence:** Medium-High on capabilities and pricing (multiple confirming sources). Low-Medium on internal model orchestration and exact 2026 features (vendor blog/X posts move fast and not all reviews agree).

---

## TL;DR

Bolt.new is StackBlitz's "prompt → running full-stack web app in your browser" agent. The differentiator is not the LLM — it's the **WebContainer** runtime: a real Node.js + npm + terminal sandbox that runs entirely client-side in the tab. The agent gets full filesystem, package manager, terminal, and browser-console control, so it can install deps, run dev servers, and iterate on a real app without a backend VM. Output is a deployable React/Vite/Next-style app; one-click deploy to `*.bolt.host` (Netlify under the hood) and GitHub sync are first-class. Pricing is **token-metered**, $25/mo Pro buys ~10M tokens with one-month rollover, and heavy users routinely report burning millions of tokens debugging.

The model is primarily Anthropic Claude (Sonnet historically; reviews mention Opus 4.6 with adjustable reasoning added in 2026). No charter/spec artifact in the KutAI sense — there is an "Enhance Prompt" expander that produces a PRD-ish blob the AI then implements, but the unit of work is "a whole app" and discovery happens in the preview pane, not in a separate spec phase.

Hard ceiling: **WebContainer is Node-only in practice.** Python is experimental and has no `pip`; no Docker, no native binaries, no system services. So Bolt is structurally locked to JS/TS web apps — by design.

---

## 1. Input flow

- **Primary:** chat prompt in the bolt.new entry box.
- **Also accepted:** Figma file import, GitHub repo import, custom design system import (Material, Chakra, Shadcn, Porsche, WaPo design systems are listed). Image input is mentioned by some reviews as accepted; the nxcode comparison explicitly says Bolt does **not** do image-to-code (only v0 does), so treat "image upload" as "attached for context", not "screenshot → React".
- **Enhancer:** an "Enhance Prompt" / "AI Enhancer" button rewrites a one-liner into a longer PRD-style brief before generation.
- **Rounds:** single-shot generation followed by chat-driven iteration; no fixed round count. Vendor explicitly recommends writing 3 alternate opening prompts and picking the best output.
- **System prompts:** as of Jan 2026 users can set per-project and global system prompts (libs/conventions Bolt should always use).

## 2. Output type

- **A full Node project running live in the tab**, served by the dev server inside WebContainer. Stack defaults: Vite + React + Tailwind. Next.js is explicitly supported. Expo (React Native web) is wired in for mobile.
- File tree + Monaco-style editor + terminal + live preview are all visible in one IDE-like UI.
- "If it runs on StackBlitz, it runs on Bolt" — meaning anything the WebContainer Node runtime can host.

## 3. Iteration loop

- Three editing surfaces: **chat** (prompt → diff), **visual preview editor** (click element → tweak), and **direct file edit** in the in-browser IDE.
- "Attempt Fix" button on errors — agent reads logs, patches, retries. This is the highest token burner.
- File-tree edits are persisted in WebContainer FS; HMR via Vite gives sub-second reload.

## 4. Multi-screen / multi-route

- Yes, full client-side routing supported (React Router / Next.js routing inherited from the chosen stack). No special "screen graph" abstraction; routes are whatever the generated framework supports.
- Reviews flag context degradation past ~15-20 components — the agent starts losing track of cross-route state, props, and shared types.

## 5. Charter / spec generation

- **No durable spec artifact.** "Enhance Prompt" generates a PRD-ish text blob in the chat that the agent then implements; it is not stored as a separate doc the user iterates on, and it is not used as a contract across iterations.
- Discovery is implicit: under-specify and Bolt fills gaps with defaults (default login page, default data model, duplicated pages). User either pre-loads the prompt with constraints or discovers the gaps by inspecting the preview.
- This is the **largest delta vs KutAI's i2p**: Bolt is "prompt → running app", not "prompt → charter → plan → build". For KutAI's i2p evolution, the charter step is a real differentiator if KutAI can avoid the "default app appears, now you reverse-engineer your spec" antipattern.

## 6. Image generation

- 2026 Pro tier added **AI image editing in chat** ("edit specific parts of images directly inside the chat interface"). Sources are unclear whether base images are generated, stocked, or user-uploaded.
- For app content (hero images, product images), behavior reported in reviews is mostly **placeholders + Unsplash-style stock URLs** in generated code. No confirmed in-house image generator wired into the page output.

## 7. Style + theming

- Default: **Tailwind**, generated inline in components.
- Design system import lets the agent ground in Material / Chakra / Shadcn / Porsche / WaPo / custom systems — Pro+ tiers can attach per-package prompts to a design system so the agent uses the right primitives.
- Responsive preview switcher (mobile/tablet/desktop viewports).
- No explicit color-token / theme-config artifact — theme is whatever Tailwind config + design-system import implies.

## 8. Backend support inside WebContainer

- **Real:** Node.js servers (Express, Hono, Next API routes), npm install of any pure-JS package, in-process SQLite-via-WASM, fetch to third-party APIs, Cloudflare Workers (Wrangler) deploy targets.
- **Not real inside WebContainer:** Python with pip, native binaries, Docker, real Postgres/MySQL daemons, anything needing a kernel.
- **External backends, wired automatically:** **Supabase** (DB + auth + storage + edge functions) is the canonical backend integration; **Stripe** for payments; **Google SSO**; **Netlify** for hosting; **Expo** for mobile. "Bolt Cloud" (V2, Oct 2025) bundles managed DB / auth / storage / edge functions / analytics / hosting under StackBlitz so users don't have to wire Supabase manually.

## 9. Deploy

- **One-click to `*.bolt.host`** (Bolt-managed, Netlify-backed).
- **Netlify** direct, with editable Netlify URLs (recent feature).
- **GitHub export** → user can deploy anywhere (Vercel, Cloudflare Pages, etc.). The nxcode comparison lists Vercel as a deploy target; that is via GitHub sync, not a native button.
- Custom domains gated behind Pro.

## 10. GitHub sync

- First-class. Push current project to a GitHub repo, pull updates back, both directions. Supports private repos. This is the documented escape hatch for "I want to leave Bolt and keep building locally."

## 11. Pricing (verified from bolt.new/pricing)

| Tier | Price | Tokens | Notable |
|---|---|---|---|
| Free | $0 | 300K/day, 1M/month cap | Public/private projects, Bolt branding on deployed sites, 10MB uploads, hosting, 333K web requests, unlimited DBs |
| Pro | $25/mo | 10M/mo min, no daily cap | No branding, private sharing, 100MB uploads, 1M web requests, **token rollover (1 extra month)**, custom domains, SEO boost, expanded DB, AI image editing |
| Teams | $30/member/mo | Pro per member, no pooling | Centralized billing, RBAC, org sharing, private NPM registries, design-system per-package prompts |
| Enterprise | Custom | Custom | SSO, audit logs, compliance, 24/7, custom SLAs/integrations |

**Token model:** every chat turn, every "Attempt Fix", every regeneration burns tokens. Reports of 1.3M tokens/day burned on a single project; some users have spent $1,000+ debugging a stuck app. The economic pressure to "get it right in fewer rounds" is the user-visible cost of Bolt's "agent does everything" model.

## 12. Underlying model

- Documented: **Anthropic Claude** (README explicitly mentions Claude/Anthropic; for ~all of 2024-2025 it was Claude Sonnet 3.5).
- 2026 reviews mention **Opus 4.6 with adjustable reasoning depth** as an upgrade option, presumably on Pro+. Not officially confirmed on bolt.new directly in any page I could fetch.
- No mention of model choice exposed to end-user beyond reasoning-depth slider.

## 13. Recent updates (2025-2026)

- **Jul 2025:** token rollover (1 extra month).
- **Oct 2025 — Bolt v2:** "Bolt Cloud" launched (managed DB + auth + storage + edge functions + analytics + hosting), autonomous-debugging push (claimed "98% fewer errors"), claim of "1000× larger projects" handleable, native Stripe.
- **Jan 2026:** per-project + global system prompts shipped (X announcement).
- **2026:** Opus 4.6 routing with reasoning slider (per third-party reviews); AI image editing added to Pro; Teams design-system per-package prompts; editable Netlify URLs; team templates; January 2026 internal benchmarks claim 40% build-perf improvement vs 2024.

## 14. Limitations

**WebContainer-imposed (structural):**
- **Node.js only in practice.** Python is experimental, no `pip`, only built-in modules. No Ruby, Go, Rust, Java runtimes.
- **No Docker.** No way to run arbitrary container images.
- **No native deps / binaries.** Packages with C/C++ extensions either don't install or fall back to JS shims.
- **No real long-lived backend services.** Postgres/Mongo/Redis must be external (Supabase / managed).
- **No local filesystem outside the tab** — the WebContainer FS is sandboxed; persistence requires GitHub push or Bolt Cloud.
- Browser tab = process. Close it = state gone unless synced.

**Agent-imposed (model behavior):**
- **Context degradation past ~15-20 components / files.** Agent starts forgetting earlier decisions, rewrites working code.
- **Token burn on debugging.** "Attempt Fix" can spiral; users routinely report multi-million-token loops.
- **Struggles with deep logic, persistent state, multi-layer integrations** (Banani review). Good at scaffolding, weaker at non-trivial business logic.
- **No true spec / charter contract** — drift between user intent and generated app accumulates across long sessions.

**Product-imposed:**
- **Cloud-only.** No local Bolt; if bolt.new is down or you're offline, you're stuck (mitigated by GitHub sync).
- **No image-to-code** (per nxcode comparison; v0 wins this).
- **Limited mobile/desktop output** — Expo present but reviews indicate it's React-Native-web-leaning, not parity with native tooling.
- **Tokens expire** on Free; rollover only on paid and only 1 month.

---

## Comparison axes for KutAI i2p

| Axis | Bolt.new | KutAI i2p (current) | Implication |
|---|---|---|---|
| Surface | Browser IDE | Telegram chat | KutAI wins on ambient/async; Bolt wins on visual feedback |
| Runtime | WebContainer (Node, in-browser) | Local subprocess + git | KutAI is not Node-locked; can ship Python, CLI, services |
| Spec artifact | "Enhance Prompt" blob, ephemeral | Charter (planned), durable | KutAI charter is a real differentiator if it stays in workflow |
| Iteration | Chat + preview + file edit | Chat + step graph | Bolt's preview is its killer feature; KutAI lacks visual feedback |
| Image gen | Editor in 2026, not generation | None confirmed | Both weak; opportunity |
| Backend | Supabase / Bolt Cloud | Local SQLite + filesystem | KutAI more flexible, less polished |
| Deploy | One-click bolt.host / Netlify | None default | Bolt has clear "ship" button; KutAI does not |
| Pricing | Token-metered, $25-30/seat | Self-hosted, GPU-bounded | Different economics; KutAI's local model = no per-token guilt |
| Multi-route | Yes (framework-native) | N/A (no UI output) | Out of scope until KutAI does UI |
| Lock-in escape | GitHub sync | Files on disk already | KutAI structurally less locked-in |

**Top 3 things to steal:**
1. **One-click visible preview** of whatever was just generated. Telegram-only loop is a huge UX gap vs Bolt's "see it running in 30s." Even a screenshot + hosted URL after each i2p phase would close it partially.
2. **"Enhance Prompt" expander** before charter generation. KutAI's charter quality depends entirely on input prompt quality; a cheap LLM pre-pass that turns one-liners into structured PRDs would lift everything downstream.
3. **Per-project system prompt** (Jan 2026 Bolt feature). Persistent project conventions ("always use X library, follow these naming rules, target this audience") — KutAI mission_state could carry this through every dispatcher call.

**Top 3 things NOT to copy:**
1. The token-metered economic model — KutAI's local-first means there's no reason to charge per token, and Bolt's pricing creates user anxiety on every retry.
2. WebContainer-style runtime lock-in — KutAI's strength is shipping anything (Python scripts, CLI tools, services), not just web apps.
3. "Default app appears, now reverse-engineer your spec" antipattern — KutAI's charter step is the right discipline; don't break it for speed.

---

## Top 3 surprises

1. **Bolt has no real spec/charter artifact.** The "Enhance Prompt" expander is one-shot text the agent then ignores; there's no contract the agent honors across edits. KutAI's charter approach is genuinely differentiated, not a duplicate of what Bolt already does.
2. **WebContainer is a much harder constraint than I expected.** Python has no pip. No native deps. No Docker. Bolt is structurally a Node-only web-app builder, full stop — which is why it's so good at React/Vite output and why it has no answer for KutAI's "ship a Python service" use case.
3. **The agent can burn $1,000 on one debug spiral.** Users report 1.3M tokens/day routinely. The "Attempt Fix" loop has no hard budget cap visible to the user, and the token-meter pricing makes every regeneration feel expensive. KutAI's local-GPU "free retries" model is a real comfort-of-use advantage that should be marketed, not hidden.

---

## Sources

- [bolt.new landing](https://bolt.new)
- [bolt.new/pricing](https://bolt.new/pricing)
- [stackblitz/bolt.new GitHub README](https://github.com/stackblitz/bolt.new)
- [Banani — Bolt.new AI Builder: 2026 Review](https://www.banani.co/blog/bolt-new-review)
- [nxcode — V0 vs Bolt.new vs Lovable 2026 comparison](https://www.nxcode.io/resources/news/v0-vs-bolt-vs-lovable-ai-app-builder-comparison-2025)
- [WebContainers landing](https://webcontainers.io/)
- [StackBlitz blog — Native Language Support in WebContainers (WASI)](https://blog.stackblitz.com/posts/announcing-wasi/)
- [bolt.new on X — per-project + global system prompts (Jan 2026)](https://x.com/boltdotnew/status/1883949779572646008)
- [Bolt support — Prompting Effectively](https://support.bolt.new/best-practices/prompting-effectively)
- [Bolt blog — Prompting tips for Bolt](https://bolt.new/blog/prompting-tips-for-bolt)
- [SurePrompts — Bolt.new Prompting Guide 2026](https://sureprompts.com/blog/bolt-new-prompting-guide)
- [devclass — Bolt.new launch coverage](https://devclass.com/2024/10/16/stackblitz-bolt-new-blurs-boundaries-between-web-development-and-skilled-use-of-ai-prompts/)
- [Taskade — Bolt.new $25/mo review 2026](https://www.taskade.com/blog/bolt-review)
- [Capacity — What Is Bolt.new Complete Guide 2026](https://capacity.so/blog/what-is-bolt-new)
