# Cluster Summary — IDE-Mode AI Agents (Cursor, Windsurf, Cody/Amp, Augment)

Date: 2026-05-09
Scope: What this cluster teaches KutAI's i2p evolution. Special focus on **Cursor Background Agents** (closest async shape to KutAI) and **Augment Intent** (closest spec-driven shape to i2p phases 1–6).

## Where each tool sits

| Tool | Primary surface | Spec/PRD? | Background async? | Multi-screen UI? | Closest to KutAI on... |
|---|---|---|---|---|---|
| Cursor | Composer + Agent + Background | No (Rules only) | **Cloud VM + branch + PR** | Via v0 pairing | Mission async runtime |
| Windsurf Cascade | In-editor chat | No (`.windsurfrules`) | Weak — Hooks only | Browser preview only | Style-aware iteration |
| Sourcegraph Cody/Amp | Chat + CLI + PR review | No | Review agent only | No | Code-graph context depth |
| Augment Intent | macOS workspace + IDE + CLI | **Yes — spec is the contract** | Parallel implementor waves | No | **i2p mission shape** |

## What this cluster teaches i2p

### 1. The spec-driven mission pattern is real and productized
Augment Intent's **Coordinator → Implementor waves → Verifier** loop is a clean productization of essentially what i2p does in phases 1–8. Validation: the pattern is commercially viable. Risk: a well-funded incumbent already owns the framing. KutAI's edge has to be **what the agents do between the bookends**, not the bookends themselves — local model dispatch, Turkish shopping, mission-as-Telegram-conversation, multi-day persistence.

### 2. Cursor Background Agents define the async-runtime baseline
This is the closest existing thing to KutAI's mission shape (long-running, autonomous, returns artifact). Studied closely:
- **Per-task isolated VM** (not just a container — full snapshotted environment).
- **Branch-per-mission** so iteration history is git-native.
- **Setup contract is in-repo**: a Dockerfile + setup script you commit. The repo *is* the agent's environment definition.
- **PR is the completion signal** — review surface is GitHub, not a custom UI.
- **Snapshots warm up subsequent agents** — important for KutAI: cold starts on a Windows GPU box are expensive; warm-state reuse is a known pattern.
- **No public API** to launch them — KutAI's Telegram-as-API is actually a differentiator here.
- **Privacy mode must be off** — proves out the trust gap KutAI sidesteps by being self-hosted.

Implications for i2p:
- Persisting per-mission workspace state (snapshot-equivalent) deserves a first-class slot, not an afterthought.
- A "mission = git branch" mental model would simplify a lot of i2p plumbing and align with how reviewers think.
- Don't reinvent the review surface; lean on PRs / file diffs the human can scan, not custom mission viewers.

### 3. Style awareness is converging on "graph + rules + memory"
- Cursor: indexed codebase + Rules + Skills + Memory
- Windsurf: Codemaps + `.windsurfrules` + auto-generated Memory
- Cody: Sourcegraph code graph (the strongest one)
- Augment: Context Engine
None of them extract **visual** design tokens — that's an i2p differentiator if the multi-screen / mobile track lands.

### 4. Pricing has standardized on credits, against everyone's interest
All four use credit-style billing (Cursor explicitly, Windsurf credits, Augment credits, Amp credits). Reviewers consistently flag this as making cost forecasting hard. **A clean per-mission cost reading is a real product opportunity** for KutAI — the user already wants to know "what did this mission cost me in tokens / GPU-minutes."

### 5. Multi-screen UI generation is an open lane
None of the four does multi-screen UI generation natively. Cursor pairs with v0; the others don't even have that. If i2p's mobile/visual track ships, it competes against v0+Cursor combos, not against any single IDE agent. Smaller competitive surface than expected.

### 6. The "spec stays alive" framing matters
Augment's phrasing — "specs stay alive" — is the right mental model for i2p. The spec/charter must be a first-class artifact that later phases consult, that the Verifier checks against, and that gets revised when reality diverges. KutAI already has this in artifact form; the discipline is enforcing the consultation in every phase, not just emitting it in phase 1.

## Specifically on Cursor Background Agents (deep dive — KutAI's closest mirror)

| Aspect | Cursor BGA | KutAI mission |
|---|---|---|
| Trigger | "Send to background" / `/multitask` in IDE | `/task` or workflow in Telegram |
| Isolation | Per-mission cloud VM | Single shared host, single Beckman queue |
| State persistence | VM snapshots, warm-resume | DB-persisted task state; no warm GPU snapshot |
| Branch strategy | New branch per agent | mr_roboto auto-commit on coder steps; no branch-per-mission |
| Completion signal | PR opened in GitHub | Telegram message + DB row |
| Review surface | GitHub PR diff | Telegram + manual git review |
| API | None public | Telegram-as-API (de facto) |
| Privacy posture | Code leaves the dev's machine | Self-hosted by design |
| Cost model | Credits + max-mode premium | Local GPU = effectively free per-token; cloud only on overflow |

KutAI's wins: privacy, cost, Turkish-domain tools, conversation-shaped interface.
KutAI's gaps to close (or consciously decline): **no branch-per-mission**, **no per-mission isolation**, **no warm-state snapshot for repeated mission shapes**.

## Top 3 surprises

1. **Sourcegraph killed Cody Free/Pro entirely (Jul 2025) and pivoted to Amp.** Cody is enterprise-only at $59/seat. The story isn't "Cody competes with Cursor" — it's "Sourcegraph is repositioning its code graph as a context layer for *all* agents (Cursor, Codex, Amp, Claude Code)." That's a different competitive frame.

2. **Augment Intent's Coordinator/Implementor/Verifier pattern is essentially i2p, productized and shipping.** Not a future risk — a current one. macOS-only is the gap KutAI can exploit, but the conceptual ground is taken.

3. **Cursor 3.3 (May 7, 2026) shipped "Build in Parallel" + "PR Splitting" four days ago.** The async-parallel-execution + dependency-aware PR decomposition story is moving fast; what was a Cursor differentiator over Windsurf in February is being deepened monthly. KutAI's parallel-implementor story (if/when it exists) needs to land soon.

## Confidence

- Cursor: **High**.
- Windsurf: **High** on features/pricing; **Medium** on async (small surface to verify).
- Cody/Amp: **High** on the Cody-deprecation pivot; **Medium** on Amp internals.
- Augment: **High** on marketed orchestration pattern; **Medium** on real-world reliability (vendor-heavy sourcing).
- Cluster-level claims: **High** for the "no spec, no multi-screen, credit-based billing" generalizations; **Medium** for the Cursor-vs-KutAI table where I infer from public docs.

## Files

- `cursor.md`
- `windsurf.md`
- `sourcegraph-cody.md`
- `augment.md`
- `_cluster-summary-ide-agents.md` (this file)
