# Sourcegraph — Cody (Enterprise) and Amp (Individual + Teams)

Date: 2026-05-09
Confidence: High on the Cody-deprecation / Amp-pivot story; Medium on Amp internals (newer product, docs less mature than Cursor's).

## Important framing
Sourcegraph **deprecated Cody Free/Pro on July 23, 2025**. Cody now exists only as **Cody Enterprise ($59/user/mo)** — a context-augmented assistant for huge codebases. Their agentic IDE play for individuals is **Amp**, a separate product. Both are relevant to KutAI but answer different questions.

## 1. Input flow
- **Cody Enterprise**: chat panel in VS Code / JetBrains, autocomplete, slash commands ("Generate unit test", "Document"), and **multi-repo context** queries via the Sourcegraph code graph.
- **Amp**: chat panel in VS Code extension **plus a CLI**. CLI was rebuilt from scratch in 2026.

## 2. Output type
- Cody: code suggestions, chat answers, autocompletes, refactors with diff preview. More assistant than agent.
- Amp: full agent — multi-file diffs, tool use (shell, search), and **agentic code review** that pre-scans diffs and posts structured feedback.

## 3. Iteration loop
- Cody: per-suggestion accept/reject; not autonomous.
- Amp: agent loop with **Deep Mode** (extended-reasoning autonomous problem solving). Iteration ends when Amp self-declares done or hits a tool budget.
- Amp's review agent runs in a dedicated panel scoped to a commit range / diff.

## 4. Charter / spec / PRD generation
- No spec/PRD generator on either. Cody's strength is **reading** the existing codebase as the implicit spec via the code graph.
- Amp has a **walkthrough skill** that produces annotated diagrams — the closest "explain the system" output.

## 5. Multi-screen UI generation
- Not a focus area for either product. Both target backend / large-codebase work, not greenfield UI generation.

## 6. Style awareness
- **Cody's defining advantage**: the Sourcegraph code graph indexes the **whole organization's code** — multi-repo, cross-service. Style awareness is structural (real call graphs, real types) rather than vector-similarity.
- Context Filters (enterprise) restrict what context can be sent to which model.
- Amp inherits Sourcegraph code-graph context when used in an enterprise that has it; standalone, it's repo-local.

## 7. Async / background
- Cody: no background agent.
- Amp: agentic execution is interactive in the panel; no Cursor-style cloud-VM async mode. Code review agent is the closest async surface — it runs in the background of a PR.
- Roadmap: terminal/CLI version of the review agent (not yet shipped as of May 2026).

## 8. Deploy
- N/A.

## 9. Pricing (2026)
- **Cody Enterprise**: $59/user/mo. **No individual or Pro tier** any more.
- **Sourcegraph Enterprise** platform (the code-graph backend): from **$16K/year**, scales with team size, includes AI credits.
- **Amp for individuals**: free with starter credits ($10 for old Cody Free users, $40 for old Cody Pro users at migration time); paid usage beyond that.
- Enterprise platform "works with Claude Code, Cursor, Codex, Amp, and more" — Sourcegraph is repositioning the code graph as a context-supply layer, not just its own assistant.

## 10. Underlying model
- Cody: Claude (Sonnet/Opus), GPT, Gemini — selectable per workspace; enterprise admins gate which.
- Amp **Smart Mode** routes to top-tier models including **Claude Opus 4.7**.
- No proprietary Sourcegraph LLM; the moat is the code graph, not the model.

## 11. Recent updates (2025 – May 2026)
- Jun 2025 announcement → Jul 23, 2025: Cody Free/Pro shut down.
- Late 2025 / early 2026: Amp public launch, VS Code extension, CLI rebuild.
- Apr 2026 onward: Amp adds **agentic code review** with dedicated review-focused agent and toolset.
- Sourcegraph platform pivots to "context for any agent."

## 12. Limitations
- Cody has no spec/PRD generation; it answers questions about code, doesn't compose product features end-to-end.
- No multi-screen UI generation.
- Cody Enterprise minimums (platform price + per-seat) gate it to mid/large orgs.
- Amp is younger; less battle-tested than Cursor / Windsurf.
- Amp review agent is **VS Code only** as of May 2026 — CLI version on roadmap but unshipped.

## Sources
- [Sourcegraph pricing](https://sourcegraph.com/pricing)
- [Cody plan changes blog](https://sourcegraph.com/blog/changes-to-cody-free-pro-and-enterprise-starter-plans)
- [Cody VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=sourcegraph.cody-ai)
- [Amp page on Sourcegraph](https://sourcegraph.com/amp)
- [Amp homepage](https://ampcode.com/)
- [Amp on VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=sourcegraph.amp)
- [Cody review 2026 (WeavAI)](https://weavai.app/blog/en/2026/04/30/sourcegraph-cody-review-2026-enterprise-ai-at-59-mo/)
- [Amp adds code review (Tessl)](https://tessl.io/blog/amp-adds-agentic-code-review-to-its-coding-agent-toolkit/)
- [Amp review (Second Talent)](https://www.secondtalent.com/resources/amp-ai-review/)
- [Amp directory entry (authorityaitools)](https://authorityaitools.com/tools/cody)
