# Cursor — Composer, Agent Mode, Background Agents

Date: 2026-05-09
Confidence: High on pricing/features (vendor + multiple recent reviews); Medium on internal Background Agent runtime details (vendor docs are thin, third-party guides fill gaps).

## 1. Input flow
- Three surfaces in the IDE: **Tab** (autocomplete), **Composer** (chat-style multi-file edits, accept/reject diffs), and **Agent Mode** (autonomous tool-using loop on local machine).
- **Background Agents** are launched from the Composer panel ("Send to background") or via the `/multitask` slash command, which spawns parallel async subagents that split a request into chunks.
- **Cursor 3.0** (April 2026) introduced an "agent-first interface" — the agent feed, not the file tree, is the primary surface.

## 2. Output type
- Composer: multi-file diff stream, applied as you watch; per-hunk accept/reject.
- Agent Mode: same diff output plus tool calls (terminal, search, file ops).
- Background Agents: full feature implementations on a separate git branch, ending in a pull request.

## 3. Iteration loop
- Inline diff blocks per file with **Accept all / Reject / Accept selected hunk**. Composer keeps the conversation thread alive across edits.
- Background Agents iterate autonomously (test → lint → fix) until they self-declare done; the PR is the iteration boundary, not per-file.

## 4. Charter / spec / PRD generation
- No first-class spec/PRD generator. **Cursor Rules** files (`.cursor/rules/*.mdc`) act as a lightweight charter — project conventions, style, framework prefs — read on every turn.
- v3.3 (May 2026) added a "Plan Mode" in CLI that produces a plan first, then executes; closest equivalent to a generated spec.

## 5. Multi-screen UI generation
- Yes, but unopinionated. Common pattern documented by users: pair Cursor with **v0.app** (which produces shadcn/ui code) for screens, then have Composer wire navigation.
- For mobile: React Native / Expo workflows produce multi-screen apps from prompts in Composer; quality depends heavily on prompt + Rules.

## 6. Style awareness
- Reads Cursor Rules (project + user level), Skills (recently added), and uses **codebase indexing** (vector + symbol graph) to mirror existing patterns.
- No automatic design-token extraction — style awareness is code-style, not visual-design.

## 7. Async / background — the closest thing to KutAI's mission shape
- **Cloud VM per agent.** Repo cloned into a fresh VM, work happens on a separate branch, environment built from a Dockerfile + setup scripts you commit to the repo.
- **Privacy Mode must be off** (code leaves the developer's machine), and **usage-based billing must be enabled** ($10–$20 minimum funding floor).
- **Snapshots** of the VM let later agents resume from a warm state instead of re-installing deps.
- **Lifecycle**: spawn → clone+setup → autonomous edit/test/lint loop → push branch → open PR → notify in Cursor sidebar. Developer reviews PR like any other.
- **`/multitask`**: in-IDE, splits a single user request into parallel subagent threads (different from Background Agents — these are still your local session, just parallel).
- **v3.3 "Build in Parallel"** (May 7, 2026): one click identifies independent slices of a plan and runs them as async subagents, then a "PR Splitting" quick-action divides the result into dependency-ordered PRs.
- **No public API** to trigger Background Agents programmatically (as of May 2026). Limits: best on small/predictable tasks, not large architectural rewrites.

## 8. Deploy
- N/A. Cursor doesn't host the resulting app; PR + your existing CI is the deploy path.

## 9. Pricing (2026)
- **Hobby** $0 — limited Tab + limited Agent.
- **Pro** $20/mo — unlimited Tab, $20 of API agent credit, Background Agents included.
- **Pro+** $60/mo — 3× model usage.
- **Ultra** $200/mo — 20× model usage, priority on new features.
- **Teams** $40/user/mo — shared rules, SSO, analytics.
- **Enterprise** custom — granular model access, soft spending limits with 50/80/100% alerts (added May 4, 2026).
- Billing shifted from request counts to API-credit dollars in 2026; Background Agents bill against this credit pool.

## 10. Underlying model
- **Composer** (Cursor's own frontier coding model, October 2025) — agentic-tuned, ~4× faster than peers, sub-30s interactive turns. **Composer 2** shipped with Cursor 3.0.
- Plus all major hosted: Claude (Sonnet 4.6, Opus 4.7), GPT-5, Gemini 3.1 Pro, plus "Auto" model router.

## 11. Recent updates (Oct 2025 – May 2026)
- Oct 2025: Composer model launches.
- Feb 2026: Cursor 2.4 — Async Agents, CLI Plan Mode.
- Apr 2026: Cursor 3.0 — agent-first interface, Cloud Agents, Composer 2.
- Apr 30, 2026: Security Review beta (Teams/Enterprise) — automated PR vuln scans + scheduled codebase scans.
- May 1: team plugin marketplace controls (off / on / required).
- May 4: enterprise model-access controls + soft spend limits.
- May 6: per-rule/skill/MCP context-usage analytics.
- May 7 (v3.3): PR Review experience, Build in Parallel, PR Splitting.

## 12. Limitations
- Background Agents need cloud privacy off — blocker for proprietary codebases.
- No spec/PRD generator; Rules are conventions, not requirements artifacts.
- No multi-screen design generator; you bring v0/Figma yourself.
- No public API to trigger Background Agents from external systems.
- Background Agents recommended for "tasks you don't want to babysit," not architectural overhauls — consistent with third-party reports of regressions on large refactors.
- Credit-based billing makes cost forecasting harder than seat pricing.

## Sources
- [Cursor product page](https://cursor.com/product)
- [Cursor pricing](https://cursor.com/pricing)
- [Cursor changelog](https://cursor.com/changelog)
- [Cursor 3 agent-first interface (InfoQ)](https://www.infoq.com/news/2026/04/cursor-3-agent-first-interface/)
- [Cursor 2.4 Async Agents (Agency Journal)](https://theagencyjournal.com/cursors-fresh-2-4-drop-agents-level-up-and-cli-gets-smarter/)
- [Background Agents complete guide (ameany.io)](https://ameany.io/cursor-background-agents/)
- [Steve Kinney course notes on Background Agents](https://stevekinney.com/courses/ai-development/cursor-background-agents)
- [Cursor 2.0 architecture guide (DigitalApplied)](https://www.digitalapplied.com/blog/cursor-2-0-agent-first-architecture-guide)
- [v0 + Cursor 2.0 mobile workflow (WeAreBrain)](https://wearebrain.com/blog/building-an-app-with-ai-v0-cursor-ai/)
- [Cursor pricing 2026 (AI Productivity)](https://aiproductivity.ai/blog/cursor-pricing/)
