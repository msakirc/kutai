# Augment Code — IDE Agents, Intent Workspace, Cosmos

Date: 2026-05-09
Confidence: High — Augment's marketing site is unusually concrete about the orchestration pattern; Medium on real-world reliability (mixed third-party reports).

## 1. Input flow
- **IDE agent** in VS Code / JetBrains: chat panel, multi-step task execution.
- **CLI** with the same Context Engine.
- **Intent Workspace** (separate macOS desktop app): you write a **spec**, approve a **plan**, and a fleet of agents executes. This is the closest analog to KutAI's i2p mission flow in the entire cluster.
- **Cosmos** (newest, 2026): "operating system for agentic software" — integrates Augment with Slack, GitHub, Jira etc rather than being its own surface.

## 2. Output type
- IDE: multi-file diffs, terminal use, code review comments.
- Intent: a fully orchestrated execution with parallel implementor agents writing code, a Verifier agent checking against the spec, and a Coordinator routing tasks.
- Code review: inline GitHub PR comments scoped by full-codebase context.

## 3. Iteration loop
- IDE agent: standard accept/reject diff loop, multi-step within one task.
- Intent: **Coordinator → Implementor waves → Verifier**, each wave gated. Spec is the loop invariant — agents check back against it; Verifier flags missing edge cases / contract deviations before reaching human review.
- Specialist roles (six built-in): **Investigate, Implement, Verify, Critique, Debug, Code Review**.

## 4. Charter / spec / PRD generation
- **Yes — spec is first-class.** Intent's whole mental model is spec-driven: "specs stay alive" across the workspace lifecycle.
- The spec acts as the contract between agents; the Coordinator decomposes it, the Verifier compares output against it.
- Closest tool in this cluster to what KutAI's i2p phases 1–6 (concept → charter → design → plan) produce.

## 5. Multi-screen UI generation
- Not a marketed strength. Intent is described in terms of services, contracts, and tasks — not screens or flows.
- Augment positions against backend / production-systems work; mobile UI gen is not in their messaging.

## 6. Style awareness
- **Context Engine** is the marquee feature: live indexed understanding of the entire stack — code, dependencies, architecture, history. Marketed as outperforming peers on SWE-Bench Pro.
- Style adherence comes from this index, not from explicit Rules files.

## 7. Async / background
- **Intent runs Implementor agents in parallel waves** — that is the async story.
- Each workspace is **isolated** ("every workspace is isolated"), echoing Cursor's per-VM model but framed around the spec rather than the branch.
- Less public detail than Cursor on the runtime (cloud VM? container? unspecified in marketing). Confidence here is Medium.

## 8. Deploy
- N/A directly. Cosmos integrates with deploy-adjacent tools but Augment does not host apps.

## 9. Pricing (2026)
- **Indie** $20/mo — 40K credits, 1 user, Context Engine, MCP, SOC 2.
- **Standard** $60/user/mo — 130K credits, up to 20 users.
- **Max** $200/user/mo — 450K credits, includes **Cosmos** (new).
- **Enterprise** custom — unlimited users, SSO/OIDC/SCIM, CMEK, ISO 42001, dedicated support.
- Credits **pooled at team level**; top-ups $15 per 24K credits, expire after 12 months.
- Credit consumption "varies based on task complexity" — same forecasting headache as Cursor.

## 10. Underlying model
- Multi-model — uses frontier models behind the scenes (Claude, GPT, Gemini class) selected per agent role. Augment doesn't ship its own LLM; the moat is the Context Engine + orchestration.

## 11. Recent updates (2026)
- Intent (macOS) launches as the spec-driven orchestration workspace.
- Cosmos launches as the agentic-software OS layer integrating Slack/Jira/GitHub.
- Code review productized as standalone offering ("only AI reviewer that thinks like a senior engineer" — marketing claim).
- Intent Pricing guide published explaining credit semantics.

## 12. Limitations
- **Intent is macOS-only** — hard blocker for many teams.
- Credit-based billing obscures real cost per task.
- Less third-party adoption signal than Cursor/Windsurf.
- No multi-screen UI generation; not a frontend / app-builder play.
- Marketing precision exceeds independent verification — treat the SWE-Bench Pro / "no-slop" claims as vendor-stated.

## Why Augment matters most for KutAI i2p
Of the four tools, **Augment Intent is the only one whose mental model overlaps the i2p mission shape**: spec → plan → parallel implementor waves → verifier → human review. The Coordinator/Implementor/Verifier triad is essentially a productized i2p workflow on top of an IDE. Worth studying as both validation (the pattern works in the market) and as a competitor to differentiate against.

## Sources
- [Augment Code homepage](https://www.augmentcode.com/)
- [Augment Code pricing](https://www.augmentcode.com/pricing)
- [Intent product page](https://www.augmentcode.com/product/intent)
- [Coordinator-Implementor-Verifier pattern](https://www.augmentcode.com/guides/coordinator-implementor-verifier)
- [Spec-driven development guide](https://www.augmentcode.com/guides/spec-driven-development-ai-agents-explained)
- [Intent: a workspace for agent orchestration](https://www.augmentcode.com/blog/intent-a-workspace-for-agent-orchestration)
- [Intent Pricing breakdown](https://www.augmentcode.com/guides/intent-pricing)
- [Augment Code review (Computertech)](https://computertech.co/augment-code-review/)
- [Augment Code review (Awesome Agents)](https://awesomeagents.ai/reviews/review-augment-code-intent/)
- [Augment vs Claude Code](https://www.augmentcode.com/guides/claude-code-vs-augment-code)
