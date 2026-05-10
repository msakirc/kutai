# OpenHands (formerly OpenDevin) — Competitor Research

**Researched:** 2026-05-09
**For:** KutAI i2p-evolution
**Confidence:** High — open source, ICLR 2025 paper, SDK paper Nov 2025, active GitHub.

## Snapshot

OpenHands is the leading open-source autonomous coding agent. Born March 2024 as "OpenDevin" (a community response to closed Devin), renamed OpenHands. ICLR 2025 paper. Currently ~65k GitHub stars, 100+ releases, 3,500+ commits. Backed by All-Hands AI (the company spun out to commercialize it). Released a major **Software Agent SDK** in November 2025 (arxiv 2511.03690), refactoring V0's monolithic design into a modular V1 SDK.

OpenHands is the closest open-architecture analog to what KutAI is building: event-sourced, sandboxed, multi-agent-capable, MCP-integrated.

---

## 1. Input flow

- CLI: `openhands` with prompt
- Web GUI: chat interface like Devin's
- REST API (V1 SDK)
- GitHub App: tag `@openhands` on issues/PRs to trigger
- Conversation can attach repos, files, URLs

V1 SDK is **interface-agnostic** — same agent core, multiple front-ends.

## 2. Output type

- Code changes inside the sandboxed workspace (mountable to host or git-pushable)
- PR via GitHub integration
- Conversation transcript + event log (replayable)
- Browser actions, shell output

The V1 SDK separates "agent produced these events" from "those events were applied to a real repo" — outputs are events first, side effects second.

## 3. Spec / charter generation

Not by default — CodeAct agent jumps into ReAct-style execution. But:

- **Microagents** can inject planning / spec discipline as domain prompts
- The codebase ships an evaluation harness with 15+ benchmarks but no built-in "write a charter first" gate
- Users layer this on via prompt or via custom agents

Compared to Devin's Interactive Planning, OpenHands defaults are more "just start hacking", less "draft a spec and approve". Spec discipline is opt-in via microagent or sub-agent.

## 4. Planning approach

ReAct loop is the default. The CodeAct agent at each step either:

1. **Converses** — asks clarification, confirms with the user, or
2. **Acts via code** — emits a CodeAct (bash, python, browser action, file edit)

No mandatory upfront DAG. Multi-agent delegation exists (CodeActAgent can spawn web-browsing or code-editing specialists), but orchestration is emergent rather than templated.

V1 SDK adds **deterministic replay** and **event-sourced state** — you can rewind a session and re-execute deterministically. Closer to time-travel debugging than to a planning system, but it gives you the foundation to build planning on top.

## 5. Tool use

- **Bash shell** (`CmdRunAction`) inside Docker sandbox
- **Python execution** (Jupyter-style stateful kernel)
- **File operations** (`FileWriteAction`, `FileReadAction`, edit-with-linter)
- **Browser** (`BrowseURLAction` — Playwright-driven, with screenshot observations)
- **MCP tools** — V1 SDK adds first-class Model Context Protocol integration; any MCP server becomes a tool
- **Custom tools** — typed tool system, easy to add

Tool boundary matches Devin's three-tool philosophy (shell + editor + browser) with Python kernel as a bonus.

## 6. Memory / persistence

- **EventStream** — every action and observation is an immutable event in an append-only log; this IS the memory
- **ConversationMemory** — processes event history into LLM-consumable messages (with condenser to handle context limits)
- **Condenser** — lets sessions exceed context window indefinitely; full replay still possible from EventLog even after compression
- **Microagents** — Markdown files (with YAML frontmatter) holding domain knowledge / conventions; auto-loaded when triggered by keywords or always-on
- **AGENTS.md** — repo-level conventions file the agent reads on entry (similar to KutAI's CLAUDE.md pattern)
- **agent-memory skill** — extension that persists facts across sessions in AGENTS.md format

Notable gap: no built-in cross-session episodic memory. The agent re-discovers per conversation (a recurring critique).

## 7. Multi-step orchestration

- Single conversation = single agent loop with EventStream as state
- Multi-agent: CodeActAgent can delegate to specialists (browser agent, editor agent)
- V1 SDK exposes the agent as a composable Python class — you can build orchestrators on top
- Docker sandbox is persistent for the session, can be reattached
- Server mode supports concurrent conversations (multi-tenant)

Long-running coherence is solved by EventStream + Condenser, not by an explicit DAG.

## 8. Failure recovery

- Linter feedback in edit actions (catches syntax errors before commit)
- Test runs via shell — agent reads output, retries
- Replay/rewind via EventStream (you can re-execute from any event)
- Human-in-the-loop: `MessageAction` to request clarification
- No formal critic-LLM gate by default (unlike Devin)

Recovery is execution-loop driven. If the agent gets confused, the human intervenes via the chat surface.

## 9. Cost model

- **Free** (open source, MIT license)
- You pay your own LLM API costs (Claude, GPT, local — model-agnostic)
- Optional: All-Hands managed cloud service (commercial, pricing per usage)

OpenHands Index (their internal benchmark, Jan 2026) compares model costs to give users a realistic baseline.

## 10. Benchmark performance

- ICLR 2025: state-of-the-art among open-source agents at publication
- **~77% on SWE-Bench Verified** with Claude Sonnet 4.5
- **72%** earlier with prior Claude
- OpenHands Index (Jan 2026) — their own benchmark, Claude Opus and gpt-5.2-codex are top
- Strong on multi-benchmark eval harness: SWE-bench, HumanEvalFix, ML-Bench, BIRD, Gorilla, GAIA, AgentBench, WebArena, etc.

Among open-source agents, OpenHands is the leader on SWE-bench.

## 11. Recent updates 2025-2026

- **ICLR 2025**: paper accepted, formal academic milestone
- **Mid-2025**: rebrand from OpenDevin → OpenHands
- **November 2025**: Software Agent SDK released (arxiv 2511.03690) — V0 monolith → V1 modular SDK with event sourcing, MCP integration, opt-in sandboxing
- **January 2026**: OpenHands Index benchmark launched
- **March 2026**: V1.5.0 release (3500+ commits since launch)

## 12. Limitations / complaints

- **No project memory across sessions by default** — re-discovers codebase each time (acknowledged publicly: "Coding Agents Without Project Memory Re-Discover Codebases Every Session")
- **Setup friction** — Docker required, microagent system has learning curve
- **No native planner gate** — needs custom microagents or a layered orchestrator for spec-first work
- **Model cost passthrough** — free framework but Claude/GPT bills can match or exceed Devin
- **Browser stability** — Playwright-based, occasionally flaky on complex SPAs
- **Multi-agent orchestration is DIY** — V1 SDK gives primitives, doesn't ship a full multi-agent product

## What KutAI should learn / contrast

- **Steal:** EventStream / event-sourced state pattern. KutAI's task table is row-per-step; an immutable event log per mission would unlock replay, debug, and Coulson tracing.
- **Steal:** Microagents pattern (Markdown + YAML frontmatter, keyword-triggered, domain-specific knowledge injection). Lighter-weight than full agent classes, perfect for KutAI's "knowledge per domain" needs (shopping, scraping, geocoding).
- **Steal:** Condenser concept — KutAI already does context compression in some places, but not as a first-class subsystem with full-event-log replay guarantee.
- **Steal:** AGENTS.md repo convention — KutAI does this with CLAUDE.md, OpenHands validates the pattern as industry-standard.
- **Steal:** Typed tool system + MCP first-class. KutAI tools are loosely typed; formalizing would help.
- **Contrast (KutAI edge):** cross-mission memory + skill library. OpenHands explicitly lacks this; KutAI's auto-skill-capture is a differentiator.
- **Contrast (KutAI edge):** local model orchestration (DaLLaMa swap budget, Fatih Hoca scoring). OpenHands is model-agnostic but assumes API-call models; KutAI's local lane is unique.
- **Contrast (KutAI edge):** Telegram async + cross-mission task queue. OpenHands is conversation-bounded, no built-in long-mission queue.
- **Contrast (KutAI risk):** OpenHands SDK is becoming the open standard. KutAI should consider being SDK-compatible (consume OpenHands microagents, expose KutAI tools as MCP) rather than reinvent.

## Sources

- [OpenHands GitHub](https://github.com/OpenHands/OpenHands)
- [OpenHands Paper (ICLR 2025, arxiv 2407.16741)](https://arxiv.org/abs/2407.16741)
- [OpenHands Software Agent SDK Paper (arxiv 2511.03690)](https://arxiv.org/html/2511.03690v1)
- [OpenHands Runtime Architecture Docs](https://docs.openhands.dev/openhands/usage/architecture/runtime)
- [Microagents Overview](https://docs.openhands.dev/openhands/usage/microagents/microagents-overview)
- [OpenHands Deep Dive (DEV)](https://dev.to/truongpx396/openhands-deep-dive-build-your-own-guide-1al0)
- [OpenHands Index (Jan 2026)](https://www.openhands.dev/blog/openhands-index)
- [Memory Critique (MemU blog)](https://memu.pro/blog/openhands-open-source-coding-agent-memory)
- [Building Secure Runtime with Daytona](https://www.daytona.io/dotfiles/building-a-secure-openhands-runtime-with-daytona-sandboxes)
