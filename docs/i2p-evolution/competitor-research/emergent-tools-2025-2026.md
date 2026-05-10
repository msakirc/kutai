# Emergent 2025-2026 tools (the ones we hadn't named)

**Date:** 2026-05-09
**Confidence:** M
**Category:** Roundup of newer entrants surfaced during this research that should be on KutAI's radar

This file consolidates short briefs on tools discovered during research that aren't covered in the original cluster list and aren't in the existing `01-pre-code-paraflow-and-competitors.md` doc. Top-5 emergent picks ranked at the end.

---

## 1. Emergent.sh

- **What:** YC-backed, $70M Series B at $300M valuation, $50M ARR in 7 months, 6M+ users in 190 countries. "Vibe-coding + no-code" hybrid: full-stack app generation via multi-agent system that reasons, plans, builds, tests, deploys autonomously. Frontend + backend + databases + integrations + hosting from prompt.
- **Why notable:** the most commercially successful pure-play autonomous app builder of 2025-2026 by raw revenue. Validates the "multi-agent autonomous team" narrative at scale.
- **Closest-to-KutAI:** medium — also multi-agent, also end-to-end. KutAI differentiation: local-first, founder-async, Telegram surface, cross-mission persistence.

## 2. Manus.AI

- **What:** Monica.im (China). Launched March 2025. "General-purpose AI agent" — gives the agent a full virtual computer (browser, terminal, file system) and lets it complete multi-step tasks autonomously, including building apps. As of early 2026 still invite-only, 500K+ waitlist.
- **Why notable:** The "agent has a real computer" framing is the right one. Devin pioneered, Manus popularized. Both prove that long-running autonomous tasks need a stateful workspace, not just a chat.
- **Closest-to-KutAI:** high — KutAI's `workspace/mission_<id>/` model is the same primitive. The contrast is operational: KutAI's workspace is shell-bound and on Yaşar Usta's process tree, not a hosted VM.

## 3. Devin (Cognition AI)

- **What:** Purpose-built autonomous SWE; own dev environment with IDE/browser/terminal; writes, tests, debugs, deploys. More "code-focused" than Manus.
- **Why notable:** the canonical reference for "autonomous SWE in a sandbox." Pricing is enterprise-y; not founder-friendly.
- **Closest-to-KutAI:** medium — same shape but enterprise positioning vs. solo-founder.

## 4. Softgen

- **What:** Conversational input → full-stack Next.js app. 175K+ builders. Supabase / Firebase backends, Stripe, one-click Vercel deploy. Specializes in business tools (dashboards, admin, CRMs). $33/year + AI credits — unusual annual-license pricing.
- **Why notable:** vertical focus (business tools) and unusual pricing model. Niche-down strategy.
- **Closest-to-KutAI:** low — different shape (web-only, no spec phase).

## 5. Create.xyz / Anything

- **What:** Rebranded from Create.xyz to "Anything" in 2025-2026. Text prompt → working code, focused on speed and lightweight projects. Frontend / layout strong, backend logic weaker.
- **Why notable:** demonstrates that "low-stakes prototype" is a distinct buyer category — not all founders want production-ready.
- **Closest-to-KutAI:** low.

## 6. Codev (co.dev)

- **What:** YC-backed; text → production-ready Next.js full-stack apps; 15K+ builders, 17K+ apps. Export clean code, continue in VS Code.
- **Why notable:** "exports clean code, hand off to IDE" is a friendlier dev shape than Lovable's hosted-only flow.
- **Closest-to-KutAI:** medium — code-export discipline matches KutAI's "founder owns the workspace" stance.

## 7. Trickle AI

- **What:** Natural language → working products (apps + websites). Focuses on functional layouts, logic flows, data structures from idea — not drag-and-drop blocks.
- **Why notable:** described as "most talked about" 2026 builder in some review channels. Real momentum.
- **Closest-to-KutAI:** low-medium.

## 8. Databutton

- **What:** AI-native dev platform; describe app + features + sketches; collaborate with AI agent on iteration.
- **Why notable:** sketch-as-input is a real-world founder behavior. Vision-input is undervalued by most builders.
- **Closest-to-KutAI:** medium — sketch input maps to KutAI's Telegram photo handling (already wired). Worth the C16 priority bump.

## 9. Claude Artifacts (Anthropic)

- **What:** First-party Anthropic feature. Claude generates artifacts (web apps, tools, dashboards) that run on Anthropic infra. **Live Artifacts** (April 2026) refresh with current data; **MCP-connected artifacts** read/write Asana, Calendar, Slack. Sharing is free; usage charged to viewer's Claude subscription.
- **Why notable:** This is the *most strategically relevant* emergent tool for KutAI. Anthropic owns model + runtime + distribution + payments-via-subscription. Any AI-app builder dependent on Claude (which is most of them) is now competing with Anthropic's own product. The MCP connection story matters — KutAI's mechanical executors should expose MCP-compatible interfaces so missions can be hosted *anywhere*, not just KutAI's local stack.
- **Closest-to-KutAI:** medium-high in shape (single-prompt → app), but radically different in business model.

## 10. Cline / Cursor (Agent) / Windsurf (Cascade)

- **What:** Three IDE-bound autonomous coding agents converging on similar shape. Cline is OSS (Apache 2.0, 5M+ installs, BYOM no-markup); Cursor is the market leader at 360K paying users; Windsurf was acquired by Google early 2025. All three run multi-step agentic loops in the user's IDE.
- **Why notable:** these are the *foundation layer* for solo-developer founder workflows in 2026. KutAI's coulson runtime overlaps with what these tools own at the IDE level. Differentiation: KutAI is not bound to a developer surface — Telegram-first founder, no IDE required.

## 11. GitHub Copilot

- **What:** ~15M developers; free tier + $10/mo Pro; deepest install base.
- **Why notable:** distribution baseline. Any tool that can't beat Copilot in some axis (taste, autonomy, multi-mission, async) won't get adopted.

## 12. Antigravity (Google) and Kiro (AWS)

- **What:** Mentioned in 2026 coding-agent comparison roundups as new agentic IDEs from cloud incumbents. Less mature than Cursor/Windsurf.
- **Why notable:** the cloud platforms can't let coding agents become a foreign moat; they will subsidize and integrate aggressively. Any KutAI play that depends on cloud-LLM goodwill needs a multi-provider hedge.

---

## Top-5 emergent tools KutAI should specifically track

Ranked by *strategic relevance to KutAI's i2p evolution*, not raw popularity:

1. **Claude Artifacts (Anthropic)** — same model provider, same shape, deeper pockets, MCP-native distribution. The platform-risk fight. Action: design KutAI's mechanical executors as MCP-compatible from the start so missions can deploy anywhere.

2. **Pythagora** — closest *philosophical* competitor to i2p evolution (spec-first, step-by-step, human approval per step). Already in the dedicated doc; flagged here too because the 14-agent architecture is the reference shape. Action: study their agent decomposition and compare against KutAI's existing roster (Yaşar Usta / Fatih Hoca / Mr. Roboto / etc.) — there may be missing roles.

3. **Manus.AI** — proves "agent has a real computer" at scale. Action: harden KutAI's `workspace/mission_<id>/` semantics; add provenance trails for every file mutation so the founder can review what the agent did, exactly like Manus's session-replay UI.

4. **Databutton** — sketch-as-input is the founder behavior most builders ignore. Action: bump C16 (sketch / screenshot input) priority; KutAI's Telegram photo path is 80% of the work already.

5. **Emergent.sh** — most commercially successful pure-play autonomous builder; validates that the buyer for "build me a whole product" is real and has money. Action: study their pricing and conversion funnel as the reference for any KutAI commercial milestone.

## Sources

- [Emergent.sh homepage](https://emergent.sh/) and [Emergent YC profile](https://www.ycombinator.com/companies/emergent)
- [Manus AI Review 2026 — Taskade](https://www.taskade.com/blog/manus-ai-review)
- [Best Manus Alternatives 2026 — Vellum](https://www.vellum.ai/blog/best-manus-alternatives)
- [Softgen homepage](https://softgen.ai/)
- [Anything (Create.xyz)](https://www.create.xyz/)
- [Codev homepage](https://www.co.dev/)
- [Trickle AI review — Vitara](https://vitara.ai/what-is-trickle-ai/)
- [Databutton homepage](https://databutton-landing.webflow.io/)
- [Claude Artifacts Help Center](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)
- [Anthropic Upgrades Claude Artifacts — InfoQ](https://www.infoq.com/news/2025/06/anthropic-artifacts-app/)
- [The 2026 AI Coding Assistant Showdown — DEV](https://dev.to/linou518/the-2026-ai-coding-assistant-showdown-cursor-vs-copilot-vs-windsurf-vs-cline-vs-claude-code-64e)
- [Coding Agents Comparison — Artificial Analysis](https://artificialanalysis.ai/agents/coding)
- [Cline vs Windsurf — Qodo](https://www.qodo.ai/blog/cline-vs-windsurf/)
