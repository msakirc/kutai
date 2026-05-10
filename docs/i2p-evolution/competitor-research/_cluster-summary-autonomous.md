# Cluster Summary — Autonomous SWE Agents

**Researched:** 2026-05-09
**Cluster:** Devin, OpenHands (OpenDevin), SWE-agent + adjacent (Factory.ai Droids, Magic.dev, Augment Code)
**For:** KutAI i2p-evolution competitive positioning
**Confidence:** High on shape of the landscape, medium on Devin proprietary internals.

## The shape of the landscape

Three archetypes have crystallized:

1. **Closed productized agent** (Devin, Factory Droids) — full-stack: VM, planner, critic, UX, billing. Sells autonomy as a product.
2. **Open framework + SDK** (OpenHands, SWE-agent) — sells the substrate, runs on any LM, leaves orchestration to you.
3. **Context-engine IDE plugin** (Augment Code, Magic.dev) — leans on giant context / repo indexing inside the developer's IDE; agent is conversational, not autonomous in the Devin sense.

KutAI doesn't cleanly fit any of these. Closest to (1) Devin in autonomy + persistent VM + async UX, but with a **Telegram-first, mission-queued, local-model-capable, cross-mission-memory** twist. The combination is genuinely unoccupied.

## Orchestration patterns observed

| Pattern | Devin | OpenHands | SWE-agent | Factory | Augment |
|---|---|---|---|---|---|
| Upfront plan / DAG | Yes (Interactive Planning) | No (ReAct) | No (ReAct) | Yes (per-Droid) | No |
| Planner / Critic / Executor split | Yes | No (single CodeAct) | No | Partial | No |
| Multi-agent delegation | Yes (sub-sandboxes) | Yes (V1 SDK) | No | Yes | No |
| Event-sourced state / replay | Internal | **Yes, first class** | Trajectory log | Unknown | No |
| DAG mid-flight re-planning | **Yes** | No | No | Yes | No |

**Convergence point:** event-sourced trajectories + replay are becoming table stakes (OpenHands V1, Devin's replay timeline, SWE-agent's trajectory log). KutAI's task table is the analog but it's row-per-step state, not an immutable event log. Worth upgrading.

**Divergence point:** "plan first vs ReAct first". Devin and Factory plan; OpenHands and SWE-agent ReAct. The right answer depends on the task: Devin's spec-first wins on multi-day missions, ReAct wins on tight bug fixes. **KutAI's i2p workflow is plan-first by construction** — that's the right call for the missions KutAI tackles.

## Persistence patterns observed

| Layer | Devin | OpenHands | SWE-agent | Augment |
|---|---|---|---|---|
| Codebase index | Vector snapshot per session | AGENTS.md + microagents | None | **Persistent live index** |
| Conversation memory | Replay timeline | EventStream + Condenser | History processors | Memories across sessions |
| Cross-session episodic | Wiki + Knowledge (curated) | None default | None | Memories (auto) |
| Cross-mission learning | Implicit (Wiki) | None | None | Memories |
| Repo conventions | Knowledge entries | AGENTS.md | None | Inferred from index |

**KutAI advantage is concentrated here.** Three KutAI capabilities have no equivalent in any of these tools:

1. **Skill library** — auto-captured from successful 2+ iteration tasks, pruned on bad effectiveness, ranked Bayesian. Devin's Wiki and Augment's Memories are human-curated or conversation-scoped; nobody has KutAI's auto-capture-and-prune skill economy.
2. **ChromaDB semantic memory + RAG injection** for agent context — OpenHands' microagents are keyword-triggered, KutAI's is vector-recall. KutAI's is more powerful but heavier.
3. **Cross-mission task queue with quota look-ahead** — Beckman's design is unique. Everyone else assumes one task at a time per user.

## Failure-recovery patterns observed

| Pattern | Devin | OpenHands | SWE-agent | Factory |
|---|---|---|---|---|
| Iterate-until-green | Yes | Yes | Yes | Yes |
| Critic LLM gate | **Yes** | No | No | Partial |
| Linter-feedback-on-edit | Yes | Yes | **Pioneered** | Yes |
| Re-plan on repeated failure | **Yes (DAG re-write)** | Manual | No | Yes |
| Human escalation channel | **Slack ping** | Chat message | None | Slack/Linear |
| Budget circuit breaker | ACU exhaustion | Token budget | Step limit | Token budget |
| DLQ / quarantine | No (session-bound) | No | No | Unknown |

**KutAI advantage:** the **DLQ + cross-mission failure tracking** (`/dlq retry`, failure-aware Fatih Hoca scoring, retry pipeline hardening) is more sophisticated than anything observed. Devin re-plans within a session; KutAI tracks failure across missions and adapts model selection.

**KutAI gap:** no formal **Critic LLM gate** before execution. Devin's separation of Planner → Critic → Executor is a clean pattern KutAI's pipeline doesn't have. Fatih Hoca chooses the model, runtime executes, but no second LLM reads the diff for "logic + security" before commit. Worth considering for high-stakes steps in i2p.

## Adjacent tools — limited public detail

- **Factory.ai Droids** ($50M Series B Sep 2025, $150M Series C 2026): generalist Droids that work from terminal/IDE/Slack/Linear/browser. Customer-claimed metrics: 31x faster feature delivery, 96% on-call MTTR reduction. LM-agnostic, interface-agnostic. Public architectural detail is thin — mostly marketing. Closest to Devin in productization, more Slack/Linear-native.
- **Magic.dev**: research-stage. Their LTM-2-mini model claims **100M token context** (~10M LoC, ~750 novels), with a 1000x cheaper sequence-dim algorithm than Llama 3.1 405B attention. Building Magic-G4 (H100s) and Magic-G5 (GB200) supercomputers. **No shipping product yet**, just model research. If LTM-2 ships, it changes the game on "do you even need a vector index?". Watch but don't depend on.
- **Augment Code**: IDE plugin (VS Code, JetBrains) with persistent **Context Engine** (real-time codebase index including commit history + cross-repo deps + architecture). 200K token agent context, **Memories** that auto-update across conversations, Code Checkpoints (rollback), MCP integration, multi-modal (Figma, screenshots). Strong on large codebases (100k+ files). Launched April 2025. Conversational pair-programmer, not autonomous-mission agent — different problem from KutAI but Memories pattern is worth noting.

## Where KutAI wins (asymmetric advantages)

1. **Async + Telegram + cross-mission memory** combo — nobody else has all three. Devin has async + Slack but no cross-mission memory. OpenHands has memory primitives but no async/queue. Augment has memory but is IDE-bounded.
2. **Local model lane** — DaLLaMa swap orchestration, Fatih Hoca's 15-dim scoring, perf_score by hardware reality. Every other tool assumes API-call models. KutAI can run a long mission overnight at near-zero marginal cost.
3. **Mission-queue with quota look-ahead** — Beckman is uncommon. Most tools are conversation-bound; KutAI runs a job system.
4. **Skill library auto-economy** — auto-capture, auto-prune, Bayesian ranking. No competitor has this.
5. **Telegram-first UX** — meets the founder where they are (phone, away, asynchronously). Devin's Slack approach is the validation; Telegram is freer.
6. **Domain depth** — 15 Turkish e-commerce scrapers, geocoding stack, free-API registry. Nobody else cares about these. This is the real moat: depth in a chosen domain beats breadth in framework features.

## Where KutAI must catch up

1. **Event-sourced trajectory log** with deterministic replay (OpenHands V1 has this; KutAI should). Coulson's runtime is the right place to add it.
2. **Critic LLM gate** for high-stakes steps (Devin pattern). Cheap and high-leverage, especially for `git_commit` and deploy steps.
3. **Microagents-style domain knowledge injection** — Markdown + frontmatter, keyword-triggered. KutAI has skills but not lightweight domain prompts. The two patterns are complementary.
4. **Interactive Planning checkpoint** as a first-class i2p phase — Z0 is moving in this direction; explicitly model it after Devin's approve-the-plan-then-execute UX.
5. **MCP-as-tool-system** — formalize tool surface so KutAI can both consume MCP servers (extensibility) and expose itself as MCP (integration into Claude Code, Augment, etc.).
6. **PR-as-output discipline** — Devin's "deliverable is a reviewable PR" is sharper than KutAI's "task done, see logs". Even non-code missions benefit from a single artifact.

## What to deprioritize

- **Bigger context windows** — Magic.dev's LTM bet may pan out, but for KutAI's mission shape (long horizon, many small steps, persistent memory) condensing + vector recall + skill library is the right architecture. Don't chase context size for its own sake.
- **SWE-bench scores** — KutAI doesn't compete on issue-fix benchmarks. Track SWE-Bench Pro (long-horizon, multi-file, 46% SOTA) as a leading indicator of the underlying model capability KutAI rides on, but don't optimize for it.
- **IDE plugin** — Augment owns this surface. KutAI is mobile/Telegram/async. Different user.

## Strategic verdict

The autonomous SWE agent space is bifurcating: **closed productized** (Devin, Factory) chasing enterprise dollars, **open framework** (OpenHands) chasing developer mindshare. Both assume desktop + synchronous + single-task-at-a-time + cloud-LM economics.

KutAI's bet — **mobile-first async + cross-mission memory + local-model-capable + domain-deep** — is orthogonal to that bifurcation. The patterns from Devin and OpenHands transfer (event-sourced state, critic gates, Interactive Planning, microagents), but the strategic position doesn't overlap. There's no head-on competitor for what KutAI is becoming.

The risk is execution, not positioning.

## Sources

- See per-tool files for citations:
  - `devin.md`
  - `opendevin-openhands.md`
  - `swe-agent.md`
- Adjacent tools:
  - [Factory: Droids unleashed (SiliconANGLE)](https://siliconangle.com/2025/09/25/factory-unleashes-droids-software-agents-50m-fresh-funding/)
  - [Factory.ai Pricing](https://factory.ai/pricing)
  - [Factory $150M Series C 2026](https://tech-insider.org/factory-ai-150-million-series-c-khosla-coding-droids-2026/)
  - [Magic.dev 100M Context Window blog](https://magic.dev/blog/100m-token-context-windows)
  - [Augment Code: Context Engine](https://www.augmentcode.com/context-engine)
  - [Augment Agent launch blog](https://www.augmentcode.com/blog/meet-augment-agent)
  - [WorkOS: Augment Code analysis](https://workos.com/blog/augment-code-context-is-the-new-compiler)
