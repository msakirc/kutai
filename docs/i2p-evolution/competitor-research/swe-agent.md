# SWE-agent (Princeton) — Competitor Research

**Researched:** 2026-05-09
**For:** KutAI i2p-evolution
**Confidence:** High — peer-reviewed paper (NeurIPS 2024), public GitHub, well-documented ACI design.

## Snapshot

SWE-agent is the academic ancestor of the modern autonomous-SWE-agent wave. Released April 2024 by the Princeton NLP group (same team as SWE-bench). NeurIPS 2024 paper. The contribution is conceptual more than productized: it formalized the **Agent-Computer Interface (ACI)** — the idea that LM agent performance depends as much on the tool/UI design as on the underlying model.

SWE-agent is not a product, it is a **research framework + reference implementation**. Mini-SWE-agent (released 2025) is the minimalist 100-line successor that hits 65% on SWE-bench Verified. KutAI should treat SWE-agent as the design-pattern source, not as a competing product.

---

## 1. Input flow

- A **GitHub issue** (URL or local clone + issue text) is the canonical input
- Local file path or repo path
- Custom problem statement (for non-GitHub use cases — competitive coding, CTF/cybersecurity)
- CLI invocation: `sweagent run --config X --instance Y`

Single-shot per issue. No conversational front-end by default.

## 2. Output type

- A **patch file** (`.patch` / `.diff`) that resolves the issue
- Trajectory log (every LM step, every action, every observation) — invaluable for research
- SWE-bench-format submission (for benchmarking)

No PR creation by default — the patch is the unit of output. Wrapping it in a PR is left to the user / harness.

## 3. Spec / charter generation

No. SWE-agent jumps straight into investigation + execution. The issue text IS the spec. There's no planner-then-executor split; ACI design carries the load.

## 4. Planning approach

Pure **ReAct loop** with carefully designed action space:

- Single LM acts as both planner and executor
- Every turn: LM emits a single ACI command (look at file, edit lines, run tests, submit)
- No upfront plan, no DAG, no critic
- History processors compress the context as the trajectory grows
- The thesis: with a good ACI, you don't need elaborate planning — the model can navigate iteratively

Mini-SWE-agent doubles down on this: 100 lines of Python, bash-only, no special tools. Achieves 65% Verified.

## 5. Tool use

The contribution is here. SWE-agent's ACI tools:

- `open <file> [<line>]` — open file, scroll to line, show ~100 lines window
- `goto <line>` — scroll within open file
- `scroll_up` / `scroll_down`
- `edit <start>:<end> <<replacement>>` — edit lines, with **integrated linter feedback**; invalid edits are rejected and a diff is shown
- `find_file <name>`, `search_dir <pattern>`, `search_file <pattern>`
- `submit` — finalize patch
- Bash via a constrained shell

Key innovation: **edit command shows linter errors in-context and discards invalid edits**. This single design choice contributed measurably to SWE-bench performance.

Mini-SWE-agent strips this to bash-only and still hits 65% — suggesting model capability has caught up to where ACI sophistication matters less.

## 6. Memory / persistence

- Per-trajectory only — no cross-session memory
- History processors compress old turns to fit context window
- No vector store, no skill library, no repo wiki

This is by design — SWE-agent is a research scaffold, not a productized agent.

## 7. Multi-step orchestration

- Single LM in a loop, single sandbox, single trajectory
- No multi-agent delegation
- Sandboxed via Docker (or Modal cloud since Jan 2025)
- Coherence comes from history compression + the ACI's stateful cursor (the "open file + line" state is implicit memory)

Multi-agent orchestration is explicitly out of scope for SWE-agent. Forks (e.g. multi-SWE-agent variants in research) extend it, but the canonical version is single-agent.

## 8. Failure recovery

- Edit-with-linter rejects invalid edits and re-prompts
- Failing tests show output, agent retries
- Submit is single-shot — if the patch is wrong, the trajectory ends with a failed submission
- No human-in-the-loop (autonomous benchmark agent)

Recovery is purely "model reads error, model retries". No external critic or human escalation.

## 9. Cost model

- **Free / open-source** (MIT license)
- You pay LM API costs
- Modal cloud option for evaluation (also pay-per-use)

Designed for research budgets, not production economics.

## 10. Benchmark performance

- Original (May 2024, GPT-4 Turbo): **12.47%** on SWE-bench full test (vs 3.8% prior best)
- Mini-SWE-agent (2025): **65%** on SWE-bench Verified with frontier models, in 100 lines
- Modern SOTA on SWE-bench Verified is now ~81% (Claude Opus 4.5) — the gap reflects 2 years of model improvement, not scaffold sophistication
- February 2026 benchmark refresh: scaffolding + environments + token limits upgraded

## 11. Recent updates 2025-2026

- **Jan 2025**: SWE-bench Multimodal integrated; Modal cloud eval support
- **2025**: Mini-SWE-agent released — minimalist ReAct in 100 LoC, demonstrates that scaffolds are converging to "less is more" as models get better
- **Aug 2025**: SWE-bench-Live — 50 new verified issues added monthly
- **Feb 2026**: Major scaffolding/environment/token-limit upgrade
- **Feb 2026**: SWE-bench-Live/Windows + Win-agent (Powershell environment)
- **SWE-Bench Pro**: enterprise-complexity benchmark; best models score only 46% (vs 81% on Verified) — long-horizon multi-file is still hard

## 12. Limitations / complaints

- Not a product — no UX, no persistence, no team features
- Single-issue scope — no multi-feature missions
- Heavy LM cost per trajectory (long ReAct loops)
- ACI design wins are smaller now than at launch (frontier models compensate for weaker tools)
- Patch-as-output is rigid — no exploratory or open-ended work
- Edit command's strictness sometimes traps the agent in loops

## What KutAI should learn / contrast

- **Steal:** ACI design philosophy. Every KutAI tool surface (search, scrape, file edit, bash) should be evaluated through an ACI lens: error messages should teach the LM, invalid actions should fail fast and explain, state should be visible. KutAI's tool design has grown organically; an ACI-first audit is worth the time.
- **Steal:** Linter-feedback-on-edit pattern. KutAI's coder agents would benefit from "edit rejected + diff shown" rather than "edit applied, build fails 3 steps later".
- **Steal:** History processors as a first-class concept (not just truncation). Coulson runtime should treat trajectory compression as a deliberate design surface.
- **Steal:** Mini-SWE-agent's "less is more" — when frontier models reach the codebase, much of KutAI's scaffolding may become noise. Periodically test "what if we just gave Claude bash and let it cook?"
- **Contrast (KutAI edge):** SWE-agent has zero memory, zero multi-mission, zero async. KutAI's whole proposition is the opposite. SWE-agent shows how much you can do without those — but also why it's not a product.
- **Contrast (KutAI edge):** SWE-agent solves "fix this issue", KutAI solves "build this feature, ship it, watch the metrics, iterate". Different problem, but the ACI lessons transfer.
- **Note:** SWE-Bench Pro (46% SOTA vs 81% Verified) tells us the long-horizon multi-file problem is unsolved. This is exactly KutAI's territory. Don't get distracted by Verified scores — Pro is the real bar.

## Sources

- [SWE-agent NeurIPS 2024 Paper (arxiv 2405.15793)](https://arxiv.org/abs/2405.15793)
- [SWE-agent NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf)
- [SWE-agent GitHub](https://github.com/SWE-agent/SWE-agent)
- [ACI Background Doc](https://github.com/SWE-agent/SWE-agent/blob/main/docs/background/aci.md)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [SWE-bench Verified](https://www.swebench.com/verified.html)
- [SWE-Bench Pro (Scale)](https://labs.scale.com/leaderboard/swe_bench_pro_public)
- [SWE-bench-Live](https://swe-bench-live.github.io/)
- [SWE-Bench Explained 2026 (Morph)](https://www.morphllm.com/swe-benchmark)
- [Simon Willison: SWE-bench Feb 2026 update](https://simonwillison.net/2026/Feb/19/swe-bench/)
