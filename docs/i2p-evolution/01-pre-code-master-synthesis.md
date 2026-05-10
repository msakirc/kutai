# Z1 — Master synthesis (post-competitor-research)

**Date:** 2026-05-09
**Inputs:** v1/v2/v3 plans, A1-A21 additions, paraflow ground truth, 10-cluster competitor research (~30 tools surveyed in `competitor-research/`), landscape roundup matrix.
**Goal:** Lock final understanding of KutAI's lane in the i2p phase 0-6 category. Decide what to build, what to skip, what to defer.

This doc supersedes the **prioritization** of v1/v2/v3 + A1-A21 + C1-C21. Individual proposal designs in those docs remain the source of truth for *how* each item is shaped.

---

## 1. The category map (final)

### Tier S — Direct philosophical competitors (study deeply)

| Tool | Bet | KutAI relationship |
|---|---|---|
| **Paraflow** | Spec-bundle artifacts (charter+PRD+screens+style+HTML) from minimal founder input | Same artifact shape; KutAI adds rigor + persistence + Telegram |
| **Pythagora** | Spec-first, multi-agent (14 agents), step-by-step build with explicit tech spec | Same multi-agent + spec-first philosophy; KutAI adds async + cross-mission |

### Tier A — Threats / parallel bets (track closely)

| Tool | Bet | KutAI relationship |
|---|---|---|
| **Lovable** | "Running app IS spec" — no charter, $100M ARR in 8 months at entry tier | Direct opposite bet. Different buyer. Don't compete on speed. |
| **Claude Artifacts** | Same model provider; MCP-native distribution; free-via-subscription | Platform risk. Make KutAI MCP-compatible defensively. |
| **Manus.AI** | "Agent has real computer" with VM | Validates `workspace/mission_<id>/` primitive |
| **Emergent.sh** | $50M ARR in 7 months; multi-agent autonomous builder | Commercial milestone reference |
| **Devin** | Long-running autonomous SWE; Planner/Critic/Executor split | Closest on autonomy; lift Critic-gate pattern |

### Tier B — Pattern sources (lift specific moves, not full philosophy)

| Tool | What to lift |
|---|---|
| **v0** | Streaming post-processor pipeline (autofixer + LLM Suspense + RAG examples) |
| **Subframe** | Deterministic compile from tokens+component-graph (LLM composes, doesn't pixel-push) |
| **Onlook** | Planner-model split from apply-model (Morph Fast Apply / Relace pattern); `data-oid` for two-way DOM↔code |
| **Augment Intent** | Spec-stays-alive (loop invariant); Coordinator → Implementor waves → Verifier |
| **Cursor 3.3** | Build-in-Parallel + dependency-aware PR splitting |
| **Sourcegraph (post-pivot)** | Code-graph as context-supply layer for other agents |

### Tier C — Adjacent / orthogonal (skip or note only)

Stitch, Uizard, Visily, Plasmic, Tempo, Magic Patterns, Builder.io, Webflow, Framer, Wix, Marblism, Aider, Cursor (chat), Windsurf, GPT Engineer, Pythagora-OSS, Mage variants, trymage, Anything, Antigravity, Kiro, Vercel AI SDK, Codev, Trickle, Databutton, Softgen.

Each useful as datapoint; none change KutAI's roadmap.

---

## 2. KutAI's six locked moats (none of 30 competitors has these)

| # | Moat | Why nobody else has it |
|---|---|---|
| M1 | Telegram-only / async-AFK founder | Every competitor assumes founder-at-keyboard, watching the build |
| M2 | Cross-mission inheritance (skill library + ChromaDB) | OpenHands publicly admits this gap; nobody else even mentions it |
| M3 | Local-first inference with cloud burst | Devin $300-500/mo complaints; Bolt $1k debug spirals; KutAI = electricity cost |
| M4 | Mechanical executors as first-class workflow citizens | Mr. Roboto pattern; competitors have hooks but no first-class non-LLM dispatch |
| M5 | Mission-level workspace persistence with provenance | Manus session-replay is closest; lacks longitudinal memory across sessions |
| M6 | Turkish-market + multilingual | Whole category is English-first |

These are not negotiable; every Z1 decision must preserve all six.

---

## 3. KutAI's three category bets (lock these)

### Bet 1 — Spec-first beats spec-skip *for the right buyer*

Lovable's $100M ARR proves spec-skip works for a buyer who wants a working app *now*. KutAI is not chasing that buyer.

**KutAI's buyer:** wants to ship a real product (sells, retains, complies). Will tolerate weeks of spec work because rework cost dwarfs the spec wait. Likely solo founder + small team. The Pythagora buyer.

**Decision:** keep Z1 phases 0-6. Don't collapse to "prompt → app." Justify spec-time via reduced phase 7+ rework, evidenced by P4 (failure-mode column) + R5 (mission-level rework metric in telemetry).

### Bet 2 — Async-AFK-Telegram beats sit-at-laptop *for the right founder*

Cursor 3.3 + Devin are accelerating async story (background agents, parallel builds). They still assume founder reviews via web/IDE.

**KutAI's bet:** founder lives in Telegram. Pitches from coffee shop, reviews from bed. Multi-day missions accumulate spec without founder pressure to "ship by Friday."

**Decision:** every Z1 interaction must be Telegram-shaped. No interaction requires browser. Tunneled preview URLs (C10) are *additive*, not required. Voice memos + photo + chat as primary input.

### Bet 3 — Local-first cost ceiling beats cloud-metered *for cost-sensitive founders*

Bolt $1k debug spirals + Devin $300-500/mo prove cloud-metered creates real founder pain. KutAI runs llama-server on user's box.

**Decision:** preserve local-first fall-through. Cloud burst is opt-in per mission. Image generation defaults to local SDXL (C13) not cloud API. Document expected per-mission electricity cost; reference v0/Lovable token-spend war-stories as positioning.

---

## 4. New propositions surfaced by competitor research (B1-B12)

These are NEW vs C1-C21, sourced directly from competitor patterns.

### B1. Agent-generated todo-list as the only structured gate at intake

**Source:** Paraflow's only structured gate is a generated to-do list the user confirms before generation begins.

**Why land:** Currently i2p phase 0 asks 5+ explicit questions across `0.1`-`0.5`. Paraflow asks zero, generates the todo, founder confirms. Removes interrogation pain.

**Wiring:** new step `0.0a generate_intake_todo` (analyst LLM, ~2k tokens) reading founder pitch + z0 outputs → emits `intake_todo.md` (10-15 items). Founder confirms via Telegram (clarify-shape). On confirm, phase 0.1+ proceeds; on edit, todo regenerates.

**Replaces:** several questions in `0.5 founder_clarification`. Doesn't remove all clarification; consolidates upfront.

### B2. Bidirectional asset↔spec propagation primitive

**Source:** Paraflow's actual differentiator. "Point at asset, describe change, agent updates screen + components + style guide together."

**Why land:** v3 C19 framed bundle-regen but at axis-level only. Paraflow does it at *asset-pointer* level — founder says "this card is too busy" pointing at a specific HTML element, agent ripples back through screen plan + style guide if needed.

**Wiring:** new mechanical action `propagate_asset_change(asset_path, change_description)`. Agent identifies dependents via produces/consumes graph. Cross-artifact regen with founder review per change.

### B3. Streaming post-processor pipeline

**Source:** v0's real moat. `vercel-autofixer-01` (<250ms post-LLM autofix), "LLM Suspense" (<100ms icon-swap), RAG over hand-curated examples. Lifts raw 64.71% → 93.87% error-free.

**Why land:** KutAI's quality checks (Doğru mu Samet) run *after* generation as quality_check. Streaming guard layer would catch errors before they hit the artifact.

**Wiring:** `packages/coulson/src/coulson/streaming_guards.py`. Token-stream interceptors for: malformed JSON, broken markdown fences, common code typos, hallucinated import paths. Cheap rule-based; not LLM. Inline fix or short-circuit.

### B4. Planner / Critic / Executor split with separate model tiers

**Source:** Devin's pattern; Onlook's planner-vs-apply split.

**Why land:** Fatih Hoca picks one model per step. No second LLM gates the diff before commit. For irreversible actions (git commit, deploy, send-to-user), a Critic gate is cheap insurance.

**Wiring:** new mechanical action `critic_gate(action, payload)`. Runs second model (smaller, cheaper) with prompt "would this action break the spec / cause founder fury / leak secrets?" Vetos or passes. Add as post-hook on git_commit, notify_user, publish_prototype actions.

### B5. Spec-stays-alive (spec as loop invariant)

**Source:** Augment Intent. "Specs stay alive" — spec re-validated each implementor wave.

**Why land:** i2p spec drifts across phases 7-12. By phase 8 the design might contradict phase 4. v3 N6 (spec-rot detector) was a partial nod; B5 is the full pattern.

**Wiring:** every phase 7+ wave starts with a `spec_consistency_check` mechanical action: re-reads phase-6-locked spec + phase-N artifacts, identifies drift, surfaces to reviewer. Prevents silent drift; forces explicit spec amendments.

### B6. Deterministic compile from tokens + component graph

**Source:** Subframe's bet. AI composes; deterministic emitter writes React/Tailwind. Output called "production-ready."

**Why land:** Phase 7+ codegen today goes through LLM, hallucinations leak. For UI specifically, given locked design_tokens.json (C8) + component_library ADR (C7), the actual component composition can be generated by a non-LLM emitter. LLM only picks the composition tree.

**Wiring:** Z2 territory mostly. But Z1 must produce inputs the deterministic emitter can consume: tokens + component-graph + screen plan. Don't ship LLM-generated React in Z1.

### B7. Sketch / photo input as first-class founder channel

**Source:** Databutton, Visily, Stitch, Uizard — all accept sketch/screenshot. Telegram already has photo support.

**Why land:** founder shows competitor screenshot, doodled wireframe, pinterest moodboard. Today KutAI doesn't ingest visually.

**Wiring:** mechanical action `ingest_visual(file_path)` → vision-LLM extracts: structural elements (header/cta/grid), inferred intent ("they want X like Y but cleaner"), color/style inferred. Output `visual_brief.md` consumed by phase 0/1/5.

C16 promoted; was Tier 4 → Tier 2.

### B8. Multi-screen consistency ceiling = 3-5 screens

**Source:** Stitch 2.0 reviewer reports — even Google plateaus past 3 cohesive screens. Universal.

**Why land:** KutAI must NOT plan to "generate 24 screens coherently in one pass." Plan around the same ceiling.

**Wiring:** phase 5 generates screens in chunks of 3-5 with explicit shared-shell consistency-check between chunks. C18 (shared-shell + per-screen-inheritance) is the answer; this validates the design.

### B9. MCP-compatible mechanical executors

**Source:** Stitch ships MCP server; Claude Artifacts MCP-native; field converging.

**Why land:** MCP becomes design-to-IDE bridge standard. KutAI's mechanical executors (mr_roboto actions) should expose MCP server interface so external tools can call KutAI artifacts as resources.

**Wiring:** Z6/Z10 territory. Note: keep mr_roboto action signatures MCP-compatible going forward (typed schemas, async, stream-friendly).

### B10. Mission-level rework / regression metric

**Source:** Implicit in every competitor. Lovable's selling point is "no spec needed because rework is cheap." KutAI's selling point is the inverse — "spec rigor reduces rework." This must be measurable.

**Why land:** without telemetry, can't prove KutAI's spec-first bet. Reviewers and founder alike will doubt the wait.

**Wiring:** new schema column `missions.phase_7_rework_loops` (count of rollbacks to phase ≤6). Yazbunu event `phase_rollback` per occurrence. Dashboard metric: missions/week × phase_7_rework_loops average. Compare to founder-reported pre-KutAI baseline.

### B11. "Spec → emit MCP server" output channel

**Source:** Lift from Sourcegraph's pivot (code-graph as context-supply for OTHER agents).

**Why land:** A locked KutAI spec (charter + ADRs + tokens + screen plans) is valuable for OTHER agents (Cursor, v0, Claude Code) to consume. KutAI can be the *spec-supply layer* for the broader agent ecosystem, not just an end-to-end builder.

**Wiring:** mechanical action `publish_spec_as_mcp(mission_id) -> mcp_endpoint`. Hosts the mission's locked artifacts via MCP server. Founder can plug it into any MCP-aware tool. Z6 territory; flag for later.

### B12. Audit existing scaffolding for over-engineering

**Source:** Mini-SWE-agent: 65% SWE-bench Verified in 100 LOC of Python with bash-only. As frontier models improve, ACI sophistication matters less.

**Why land:** KutAI has accumulated layers. Some may be over-engineered against current model capability.

**Wiring:** quarterly audit: "what if we just gave Claude bash and let it cook?" Strip layers in places it's pure overhead. Doesn't apply to anything in Z1 directly; affects Z2 / Z10 reviewers + workflow engine.

---

## 5. Re-prioritization of full proposal stack

Combining v3's P1-P9, A1-A21 (claude additions), C1-C21 (paraflow + competitor patterns), B1-B12 (this synthesis).

### Drop entirely

| Item | Reason |
|---|---|
| Visual editor for KutAI | Field's converging on chat+visual+code; KutAI sticks to chat-only intentionally (M1) |
| Hot-reload preview | Bolt territory; tunneled URL (C10) is async-shape replacement |
| Real-time collaboration | Single-founder; no fit |
| Marketing-site builder pivot | Framer/Webflow lane; not KutAI's product surface |

### Tier 0 — Foundational (block downstream, ship first)

1. **P7 — spec versioning + reviewer regression fixtures** (already locked)
2. **B10 — mission-level rework metric instrumentation** (without telemetry, can't prove the bet)

### Tier 1 — Charter + intake reshape (replaces phase 0 today)

3. **C1 + A9 — charter consolidation** (one fat doc, paraflow shape)
4. **C6 + A14 — solutions+boundaries+guiding-principles** (per-solution non-goals)
5. **B1 — agent-generated todo as only structured gate** (replaces interrogation)
6. **A1 — reverse-pitch / press-release** (founder writes outcome, not just pitch)
7. **B7 + C16 — sketch/photo input first-class** (Telegram photos → vision-extract)

### Tier 2 — Spec rigor (the moat over Paraflow + Lovable)

8. **P4 — failure-mode column** (falsifiability)
9. **A2 — non-goals artifact** (mission-wide, complements per-solution boundaries)
10. **P3 + C7 + A8 — ADRs first-class with component-library + cost-curve fields**
11. **C2 — PRD with named-competitor section** (paraflow pattern)
12. **A4 — interview-script generator** (bridges P1 evidence-absence)

### Tier 3 — Design + prototype (the visible delta vs today's i2p)

13. **C4 + A12 — user-flow doc with Mermaid per audience track**
14. **C3 + A10 — per-screen plan as discrete artifact**
15. **C14 — empty/loading/error states explicit per screen**
16. **C5 + C8 + A13 + A16 — design_tokens.json + multi-variant style guide**
17. **C20 — taste-from-essence step** (paraflow's `fact_primary` tagging)
18. **C13 — local SDXL for image generation** (cost moat)
19. **C9 + A11 — HTML prototype per screen, Tailwind+Iconify** (paraflow shape)
20. **C18 — shared-shell consistency invariants** (with B8's 3-5 screen ceiling)
21. **C12 — multi-surface (mobile/web/desktop) per mission config**

### Tier 4 — Iteration loop

22. **C11 + A15 — `regen_with: "<change>"` on every artifact**
23. **C19 — bundle-level regen on directional change**
24. **B2 — bidirectional asset↔spec propagation** (paraflow's actual differentiator)
25. **C10 + A19 — tunneled preview URL per mission**
26. **C17 + A20 — two-way HTML edit reflection**

### Tier 5 — Compliance + memory + critic

27. **P6 — compliance fingerprint**
28. **A5 — founder attention budget**
29. **A6 — premortem step**
30. **B5 — spec-stays-alive (spec as loop invariant)**
31. **B4 — Critic gate on irreversible actions**
32. **B3 — streaming post-processor guards**

### Tier 6 — Cross-mission + ecosystem

33. **P9 + A7 — cross-mission inheritance + idea dedup**
34. **P5 — web-grounded prior art**
35. **C18 — github repo init at end of phase 6**
36. **B11 — publish-spec-as-MCP**
37. **B9 — keep mr_roboto MCP-compatible going forward**
38. **B6 — deterministic compile from tokens + component graph (Z2 carry)**

### Tier 7 — Audit + standing rules

39. **B12 — quarterly "what if just bash" audit**
40. **C21 — bundle-quality regression vs Paraflow goldens**

### Sequencing DAG

```
Tier 0 ─ Tier 1 ─ Tier 2 ─┬─ Tier 3 ─ Tier 4 ─┐
                          │                    ├─ Tier 5 ─ Tier 6
                          └────────────────────┘
                                                Tier 7 = standing
```

Tier 1 hard-blocks Tier 3 (charter shapes downstream artifacts).
Tier 2 can land parallel to Tier 3 (different surfaces).
Tier 5 needs Tier 4's iteration loop because critic-gate requires regen.
Tier 6 needs Tier 5's compliance shape for cross-mission inherit to be safe.

---

## 6. What this means for v3's locked sequence

v3 locked: P7 → P4 → P3 → P6 → P5 → P1+P8 → P2 → P9.

**Replace with:** [Tier 0] → [Tier 1] → [Tier 2] → [Tier 3 + Tier 4 parallel] → [Tier 5] → [Tier 6] → [Tier 7 standing].

v3's order remains valid *within* tiers. New B1 (todo-gate) lands in Tier 1 alongside C1. New B10 (rework metric) lands Tier 0 with P7. New B5/B4/B3 (spec-alive, critic gate, streaming guards) land Tier 5 after iteration loop is in place.

---

## 6.5. STRATEGIC DECISIONS LOCKED (2026-05-09)

Founder answered the five strategic questions. Decisions now binding on
Z1 roadmap and downstream zones.

| # | Decision | Roadmap implication |
|---|---|---|
| **Q1 — Buyer focus** | **Real product (Pythagora-side).** KutAI ships products that live, sell, retain, comply. Not disposable prototypes. | All Z1 spec rigor justified. **Drop ambition-tier-skips-charter idea.** Charter + ADRs + evidence + compliance mandatory for every mission. No `prototype` ambition opt-out on spec discipline. |
| **Q2 — MCP investment** | **Skip for now.** Re-evaluate Q3 2026. | B9 (mr_roboto MCP-compatible signatures) drops to opportunistic-only. B11 (publish-spec-as-MCP) deferred to 2027 or never. R20 (Anthropic shipping Artifacts) tracked but not pre-mitigated. |
| **Q3 — Surface durability** | **Add slim web preview surface.** Telegram = primary control. Browser = read-only view of mission status / HTML preview / spec inspector. | C10 (tunneled URL) becomes Tier 2 not Tier 4. New zone work: mission dashboard / spec viewer in browser (read-only). M1 reframed: "Telegram-driven, browser-viewable." Adds web-static-host obligation to phase 6.7+. |
| **Q4 — Agent depth** | **Stay current shape (a) + add more mechanical actions (c).** All LLM agents share generic ReAct in coulson — configs only differ. New work goes into more first-class mechanical (non-LLM) actions in mr_roboto. | Drops Pythagora-style deep specialization (b). Confirms KutAI moat M4 (mechanical executors as first-class workflow citizens). Every Tier 5 critic-gate / Tier 4 propagation / Tier 3 verification proposal lands as mr_roboto action, not new LLM agent role. |
| **Q5 — Image generation** | **Provider abstraction with both local + paid.** Same shape as hallederiz_kadir/fatih_hoca. New Turkish-named package (e.g. `gorsel_ustasi`) with selection layer that picks per task: local SDXL vs cloud (fal.ai / Replicate / Together). | C13 expands: not "local SDXL" but "image-gen provider abstraction." New package work, not just one mechanical action. M3 (local-first cost ceiling) preserved via local fallback; cloud burst is opt-in selection. |

### Concrete moves these decisions force

1. **`gorsel_ustasi` package SCOPED TO Z2 (foundation), NOT Z1.** Strategy stays locked here (provider abstraction parallel to hallederiz_kadir + fatih_hoca; local SDXL + cloud burst). Z1 produces HTML with placeholder/stock images; real image generation wires later via Z2 package. Don't pull infra package work into Z1.
2. **Web preview surface SCOPED TO Z2, NOT Z1.** Strategy stays locked (cloudflared/local-port/GitHub-Pages-per-mission, decision F1). Z1 produces `workspace/mission_<id>/.web/` folder content; hosting + viewer ships in Z2.
3. **Drop ambition-tier opt-outs from spec rigor.** Every mission gets charter + ADRs + evidence + compliance. Founder attention budget (A5) handles the "but I don't have time" complaint via deferred-questions log, not by skipping spec.
4. **mr_roboto becomes the dumping ground for new behaviors.** Critic gate (B4), spec-stays-alive checks (B5), streaming guards (B3), asset propagation (B2), all bidirectional propagation primitives (Paraflow's differentiator) — all land as mechanical actions, not new agent configs.
5. **Defensive MCP work shelved.** Don't pre-design mr_roboto signatures around MCP. Refactor later if needed.

### Z1 vs Z2/Z10 scope boundary (2026-05-09)

**Z1 owns (artifact-shape + mr_roboto mechanical actions only):**
- All phase 0-6 artifact shape changes (charter / personas / PRD / ADRs / flow / screen plans / tokens / falsification / compliance fingerprint / evidence index / non-goals / boundaries / premortem / reverse-pitch)
- All mr_roboto mechanical actions wiring those artifacts (todo-gate / verify-schema / critic-gate / asset-propagation / spec-consistency-check / streaming-guards / visual-ingest using existing vision tool / attention-budget-check / record-rollback)
- Reviewer prompt edits + regression fixtures
- Cross-mission inheritance behavior using EXISTING Chroma + skill library
- HTML prototype files using placeholder/stock images

**Z2 owns (heavy infrastructure packages):**
- `gorsel_ustasi` image-gen provider abstraction package (Q5 strategy locked here; package built in Z2)
- Web preview hosting + viewer surface (Q3 web preview locked here; surface built in Z2)
- Deterministic compile from tokens + component graph (B6)
- Recipe / template library
- Component library ADR consumption pipeline

**Z6 / never owns:**
- MCP server (B11) — deferred per Q2
- Publish-spec-as-MCP — deferred

### Open follow-ups from these decisions

- **F1** — Web preview surface: which static-host strategy? cloudflared vs local-port-with-tunnel vs GitHub-Pages-per-mission. Affects M3 (cost) and M5 (provenance).
- **F2** — `gorsel_ustasi` provider list at MVP: local SDXL only, or local + 1 cloud? Cost vs simplicity trade.
- **F3** — Web preview viewer scope: just HTML preview, or also charter / ADR / spec inspector? Bigger surface = bigger build.
- **F4** — How does the web preview surface authenticate? Founder's mission is private; URL-with-token? Login? Telegram-deep-link?

These F1-F4 land as new open questions in Z1 follow-up. Not blocking the
B1+C1+A9 first-merge.

---

## 7. (Original) Five strategic questions for you

These are decisions only you can make; agent research can't answer.

1. **Buyer focus.** Does KutAI target the Lovable buyer ("ship something fast") or the Pythagora buyer ("ship something real, take the time")? The roadmap above assumes Pythagora's buyer. Confirm or pivot.

2. **MCP investment level.** B9/B11 say "make KutAI MCP-compatible." This is real engineering effort. Defensive (against Claude Artifacts) or offensive (KutAI as spec-supply layer for ecosystem)?

3. **Async vs eventual web surface.** M1 locks Telegram-only forever. As field collapses to chat+visual+code (Replit Agent 4 trend), is Telegram-only sustainable, or do we accept a slim web preview surface (C10 tunneled URL = step in that direction)?

4. **Multi-agent depth.** Pythagora has 14 agents. KutAI has ~6 packages (Yaşar Usta / Fatih Hoca / Mr. Roboto / Beckman / Coulson / Vecihi / etc.). Do we go deeper (12-15 distinct agent roles) or hold at current count?

5. **Image generation strategy.** C13 says local SDXL. GPU thrash risk vs llama-server. Cloud burst (fal.ai / Replicate) is cheap but breaks M3. Hybrid: local SDXL for prototype tier, cloud burst for private_beta+ tier?

---

## 8. Risks not previously named

- **R20 — Anthropic ships Claude Artifacts → bundled-app-builder.** Free with Claude subscription. Same model. Removes KutAI's price advantage at scale. Mitigation: M1+M2+M3+M5 still differentiate; Anthropic won't ship local-first or cross-mission inheritance.
- **R21 — Lovable forks downward into spec-first segment.** $100M ARR funds them to expand. Mitigation: deepen Pythagora-side moats; brand KutAI as "the spec-first agent that remembers."
- **R22 — Telegram itself changes / loses founders.** Single platform dependency. Mitigation: abstract interface layer for future Discord/iMessage/etc. shim. Not urgent; Telegram founder share is stable.
- **R23 — Local model quality plateaus while frontier surges.** Cost moat erodes as cloud gets cheaper. Mitigation: Fatih Hoca burst-to-cloud already supports this. Re-evaluate moat M3 quarterly.
- **R24 — Mini-SWE-agent style minimalism becomes industry standard.** KutAI's heavy scaffolding looks dated. Mitigation: B12 quarterly audit; strip what model has caught up to.

---

## 9. Bottom line

Competitor research confirms KutAI's bet is real: spec-first + async-Telegram + local-first + cross-mission memory is uncontested. Two competitors (Paraflow, Pythagora) share the spec-first axis but lack the other three. Lovable + Bolt + v0 dominate the "ship now" bet but charter-empty. Devin/Manus/Augment crowd autonomy but cloud-bound.

The full stack ranks 40 items across 8 tiers. Tier 0-3 (~21 items) is the next 6-12 months of Z1 work. Tier 4-7 stretches to 18 months.

The strategic call: **commit to the Pythagora-side buyer**. Don't drift toward Lovable-side speed metrics; the buyer is different and the moat is different.

The implementation call: **B1 + C1 + A9 ship together as the single biggest UX win** (charter consolidation + agent-generated todo replaces phase 0 interrogation). One merge cycle. Highest visible delta vs today.

The platform call: **answer Q2 (MCP investment) within next 2 weeks**. Claude Artifacts is moving; defensive MCP compatibility is cheap insurance.
