# Replit Agent — Competitor Research

**Date:** 2026-05-09
**Sources:** replit.com/products/agent, replit.com/agent4, blog.replit.com/effort-based-pricing, blog.replit.com/agent-v2, latent.space Agent 4 writeup, hackceleration/serenitiesai 2026 reviews.
**Confidence:** High on pricing/pillars; Medium on exact model routing (Replit doesn't fully publish per-tier models; community reports point to Claude Sonnet 4.x + GPT-class fallbacks).

## 1. Input Flow
**Prompt + (optionally) integrations.** Natural-language chat is the primary entry point: "describe your app or website idea." Optional context:
- Linear, Notion, Excel imports for Agent 4.
- Existing Repl import (work on existing code).
- Web search auto-invoked when current info needed.
- "Extended Thinking and High-Power Models" toggle for hard tasks.

## 2. Output Type
**A running, deployed application inside Replit's cloud.** Not source-code-to-disk first — the app exists as a Repl, runnable immediately, with database, auth, secrets, and a public URL provisioned. Code is exportable but Replit's value-prop is "app exists right now."

## 3. Iteration Loop
**Chat-driven, with a self-test reflection loop.** "Agent tests its own work so you don't have to." Browser-simulation testing claimed 3x faster + 10x cheaper than computer-use models. Visible task progress (planner-style steps). Agent 4 adds **parallel task execution** — auth/db/backend/frontend can run as separate concurrent agents.

## 4. Style + Theming
Agent 4 adds **"Design Freely" / infinite canvas** with visual tweaking, multi-select, variant generation. So Agent 4 is bolting a Plasmic/Onlook-style visual layer onto its previously chat-only loop.

## 5. Multi-Screen / Multi-Route
Yes. Web apps with multiple routes are standard output. Agent 4 also converts web apps to mobile apps within the same project.

## 6. Charter / Spec / PRD
**Implicit, not artifact-first.** Agent generates a plan/task list visible to the user before executing. Agent 4's "Build Together" pillar emphasizes the team planning while Agent handles execution — so planning is surfaced but lives in chat, not as a discrete PRD doc.

## 7. Backend Integration
**Strongest of the three.** Built-in:
- **Database** (Postgres-compatible, hosted by Replit).
- **User Authentication** (Replit Auth).
- **Secrets** management.
- Third-party: Stripe, OpenAI, etc., with key vaulting.
- Agent can build other agents and orchestrate workflows (meta-agent capability).

## 8. Two-Way Sync
**No.** Replit Agent is generation-and-test; visual edits don't sync back to code the way Onlook does. The "infinite canvas" in Agent 4 is for design ideation, with Agent translating to code one-way.

## 9. Deploy
**Replit cloud, immediate.** Apps run in Replit-hosted containers with public URLs out of the box. Custom domains supported on paid tiers. Export to GitHub/local possible but not the primary flow.

## 10. OSS vs SaaS
**Pure SaaS.** No open-source version. Lock-in is real — runtime, DB, auth all Replit-managed.

## 11. Pricing (2026)
Four-tier post-Feb-2026 restructure:
- **Starter $0** — explore only, 1 published app, limited Agent intelligence.
- **Core $25/mo** ($17/mo annual) — full Agent, $20–$25/mo credit, unlimited published apps.
- **Pro $100/mo** — replaces Teams; up to 15 builders, pooled credits, 1-month rollover, priority support.
- **Enterprise** — SSO/SCIM, custom.
- **Effort-based pricing** for Agent: was $0.25/checkpoint flat; now bundled per-request priced by **time + compute**. Simple <$0.25, complex tasks more. High-power-model toggle adds cost.

## 12. Underlying Model
**Not officially fully disclosed; community-reported as Claude Sonnet 4.x family for primary work, with GPT/Gemini fallbacks.** "High Power Model" toggle suggests routing to a more capable tier (likely Opus-class or Sonnet-thinking) for hard tasks. Sub-agents may use cheaper models. (Confidence: Medium — Replit obscures exact routing.)

## 13. Recent Updates 2025-2026
- **Agent v2** (early 2025): reflection loop, self-testing.
- **Agent 3**: maturity pass, improved long-running tasks.
- **Agent 4** (2026): four pillars (Design Freely, Build Together, Ship Anything, Move Faster), parallel task execution, infinite canvas for design, web→mobile conversion, multi-artifact (apps/sites/decks/videos) in one workspace, Linear/Notion/Excel integrations.
- **Effort-based pricing** rollout replacing per-checkpoint.
- Pricing tier consolidation (Teams → Pro) Feb 2026.

## 14. Limitations
- **Lock-in** — runtime, DB, auth all Replit-managed; export possible but disruptive.
- **Cost unpredictability** — effort-based means a single hard prompt can cost meaningfully more than a checkpoint.
- **No two-way visual sync** — Onlook beats it on design fidelity.
- **Black-box model routing** — users can't reliably predict which model handles which task.
- **Browser-sim testing** is impressive but not a substitute for real integration tests on user's stack.
- **Long-context drift** in extended autonomous runs (community reports).

## What this teaches i2p
1. **Reflection loop with self-test is table stakes** — KutAI's coulson runtime already retries; matching Replit's "agent tests itself" requires invoking the real test runner inside the agent loop, not just LLM self-grading.
2. **Effort-based pricing is the right user-facing abstraction** — for KutAI's eventual cloud lane, billing by "time + compute" rather than tokens or steps maps better to user intuition.
3. **Parallel sub-agents** (Agent 4's auth/db/backend/frontend in parallel) — KutAI's Beckman task queue could dispatch independent workflow steps as parallel agents instead of serializing. Big throughput win on multi-feature missions.
4. **Hosted DB + Auth removes a class of decisions** — by choosing for the user, Replit eliminates the "which Postgres provider?" rabbit hole. KutAI could ship an opinionated default stack.
5. **Web→mobile conversion in same project** — interesting expansion lane: i2p mission spec could be artifact-agnostic, with adapters per target.
6. **Black-box model routing has cost** — users complained about unpredictable spend. Fatih Hoca's transparency (model_pick_log) is a real differentiator. Lean into it.
7. **Don't underestimate "app exists right now"** — Replit's killer demo is "prompt to live URL in 60s." Anything KutAI ships that requires user-side `npm install` is at a UX disadvantage.
