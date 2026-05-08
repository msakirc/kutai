# Z1 — Pre-code (idea → executable spec)

## Frame

Phases 0-6 of i2p_v3: idea → research → requirements → architecture →
design → planning. LLM-friendly because outputs are prose. The zone where
i2p is currently strongest. Real-product gap: spec quality is a fiction
of the founder's pitch instead of a real-world signal; taste decisions
are shallow defaults; failures aren't falsifiable.

## Current state

- 17 phase-0 to phase-6 step groups in `src/workflows/i2p/i2p_v3.json`.
- Output artifacts are markdown / structured JSON consumed by later phases via the artifact blackboard.
- Reviewer agents on key gates (1.13 research_quality_review, 3.11 requirements_review, 4.16 architecture_review, 5.10 design_review, 6.6 project_plan_review) — gated with string `equals` post-emit constraints (shipped 2026-05-05).
- `architecture_pattern_selection` (4.1) and `tech_stack_research_and_selection` (4.2) currently emit a single decision artifact with reasoning; alternatives discarded.
- Phase 0 has `idea_brief`, `problem_statement`, `target_users`, `value_proposition` as separate artifacts.
- No mechanism to inject founder-supplied evidence (interview transcripts, surveys, screenshots).
- No web-grounded prior-art search.

## Gaps

### Fixable by automation
- **Spec is hallucinated, not grounded.** Personas, JTBD, market come from the founder pitch and LLM's training-data echo. No interviews, no surveys, no analytics enter as primary source.
- **Architectural choices are shallow.** Stack picked without modeling cost-curves, team-skill, deployment-constraints, vendor risk. Defaults to NextJS+Postgres regardless of fit.
- **Alternatives are discarded.** ADRs (architecture decision records) absent. Phase 4 emits the chosen pattern; the rejected three are lost. Reviewers can't audit the choice 3 phases later.
- **No falsifiability.** Spec states feature-as-fact ("users want X"). No "we believe X; if false, kill the feature; here's how we'd know."
- **Prior-art blindness.** Agent doesn't know which 4 startups died trying this exact idea. Doesn't search HN, ProductHunt, Crunchbase, Wikipedia, Wayback.
- **Compliance footprint not collected at intake.** Target jurisdictions, user types (consumer/B2B/health/children), data categories — never asked. Hits the team in phase 12+.
- **No structured founder intake.** Founder pastes pitch into Telegram; agent runs. No upload mechanism for interviews, screenshots, voice memos, prior research.

### Founder territory (not gaps to close — gaps to make explicit)
- Brand voice, design language, tone of voice
- Strategic positioning ("are we vs Notion or vs Asana?")
- Pricing intuition
- Whether to build at all

## Proposed direction

### Recommendations from 2026-05-08 round
- **Structured user-research intake.** Founder uploads interview notes, survey results, competitor screenshots, voice memos. Agent extracts structured signal; injects as primary source distinguishable from agent-inferred speculation. Spec artifacts gain `evidence` field that points back to source.
- **Taste delegation.** Agent generates 3-5 brand/design directions with mood boards (real images via web search, not invented). Founder picks. Same for tone-of-voice samples. *Human checkpoint, not gate.*
- **Architecture decision records (ADRs).** Each architectural choice is an artifact: chosen option, alternatives, reasoning, reversal-cost, triggering criteria for revisiting. Reusable across missions; queryable by later phases.
- **Failure-mode column.** Every requirement and architectural choice declares "if this is wrong, what happens, how do we know." Forces falsifiability; populates monitoring rules later.
- **Web-grounded prior art.** New tool `find_prior_art(idea_summary)` — searches HN, ProductHunt, Crunchbase, Wikipedia, Wayback. Returns "graveyard report"; agent reads before locking spec.
- **Compliance fingerprint.** Phase 0 collects target jurisdictions + user types + data categories. Surfaces compliance overlay (privacy policy template, cookie banner, DPA requirements, retention) with explicit gaps for human review. See [06-real-world-bridge.md](06-real-world-bridge.md).

### Recommendations from prior brainstorms (rounds 1-2)
- **Recipe library** for known patterns reduces "design from scratch" surface in phase 4. Shared with [02-build-foundation.md](02-build-foundation.md).
- **Cross-mission memory** so the founder's second mission inherits learnings about their stack, domain, voice. Founder profile + product profile persist across missions.
- **request_review action** as first-class action, distinct from needs_clarification — non-blocking; comments become revision tasks. Wires into the Telegram thread per mission. See [10-cross-cutting.md](10-cross-cutting.md).

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Idea intake | parses pitch, requests structured evidence | dumps unstructured context, answers Q | full |
| Personas | infers from evidence + interviews | corrects misreads | full |
| Architecture pattern | proposes 3 ADRs ranked | picks one OR asks for more options | until phase 8 starts |
| Tech stack | proposes 3 stack ADRs ranked | picks one | high cost after scaffold |
| Brand direction | renders 3-5 mood boards | picks one or rejects all | full pre-launch |
| Compliance fingerprint | drafts overlay from intake | reviews + flags missed jurisdictions | full pre-launch |
| Spec lock | drafts final | sign-off → phase 7 starts | reversible via revision_policy |

## Dependencies

- **Inbound:** none. This is the entry zone.
- **Outbound:** every other zone reads spec artifacts. Spec quality compounds downstream.
- **Hard pre-req for:** [02-build-foundation.md](02-build-foundation.md) (recipes need spec quality), [06-real-world-bridge.md](06-real-world-bridge.md) (compliance fingerprint), [07-humanish-layers.md](07-humanish-layers.md) (brand direction informs marketing).

## Open questions

- **Evidence hierarchy.** When founder evidence contradicts agent inference, who wins? (Default: evidence wins; agent flags contradiction; founder confirms.)
- **Mood-board rendering.** Generate vs scrape from references? (Scrape — generated images don't carry the design context.)
- **ADR storage.** Per-mission vs global library? (Global library; per-mission instances reference + customize.)
- **Compliance template currency.** Who keeps the privacy-policy templates current? (Quarterly cron + founder approval; out of scope for autonomous flow.)
- **Voice memo intake.** Whisper transcription locally? Cloud? (Local Whisper; founder controls the data.)

## Agent task brief

When picking up this doc:
1. Read 00-README.md + this doc.
2. Audit current i2p_v3 phases 0-6 against the gaps above. Quote specific step IDs.
3. Convert each "Proposed direction" item to a phased work plan: what to add to i2p_v3.json, what new artifacts, what new tools, what new post-hooks.
4. Resolve open questions or escalate.
5. Add an `## Updates` entry with findings + commit refs.
6. Cross-reference outbound dependencies (call out what other docs need to know).

## Updates

- 2026-05-08 — initial doc.
