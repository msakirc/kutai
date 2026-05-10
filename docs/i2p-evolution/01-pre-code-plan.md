# Z1 — Pre-code execution plan

**Date:** 2026-05-08
**Source doc:** [01-pre-code.md](01-pre-code.md)
**Reads:** [00-README.md](00-README.md), [z0-mission-preflight.md](z0-mission-preflight.md), `src/workflows/i2p/i2p_v3.json`, `docs/plans/2026-05-07-i2p-capability-expansion.md`

This plan converts the six "Proposed direction" items in 01-pre-code.md
into phased work — citing the i2p_v3 step IDs that exhibit each gap,
and specifying the artifacts, tools, post-hooks, and z0 dependencies
each fix needs. No code; spec only.

## Audit

Every gap in §Gaps→Fixable-by-automation of 01-pre-code.md, mapped to
the specific i2p_v3.json step IDs that exhibit it. Step IDs verbatim
from `src/workflows/i2p/i2p_v3.json` line ranges noted in `Grep`
results above.

| # | Gap (01-pre-code) | Exhibiting step IDs | Evidence (file:line) |
|---|---|---|---|
| G1 | **Spec is hallucinated, not grounded.** Personas / JTBD / market come from founder pitch + LLM training echo; no interviews/surveys/analytics enter as primary source. | `0.1` raw_idea_intake (only intake = founder pitch text); `0.2` problem_statement_extraction (`severity`/`frequency` marked "Needs validation" but never validated); `1.2` target_audience_research (web-only signal, no founder evidence); `2.4` user_personas_and_journeys (input: `audience_research_data_summary` only — no founder evidence channel); `2.9` success_metrics_definition; `3.7` business_rules_extraction. No step exists for evidence intake. | `i2p_v3.json:683-718` (0.1), `:720-751` (0.2), `:937-971` (1.2), `:1632-1681` (2.4) |
| G2 | **Architectural choices are shallow.** Stack picked without modeling cost/team-skill/deploy/vendor-risk; defaults to NextJS+Postgres regardless of fit. | `4.1` architecture_pattern_selection (instruction lists six patterns but emits a single `architecture_pattern_decision` with no cost/team/vendor model); `4.2` tech_stack_research_and_selection (instruction names a fixed candidate list — `React/Next.js, Vue/Nuxt, Svelte/SvelteKit, …`); `4.8` third_party_service_selection. No step models cost-curves, founder team-skill, deploy-constraints, vendor-risk explicitly. | `i2p_v3.json:2544-2581` (4.1), `:2583-2629` (4.2) |
| G3 | **Alternatives are discarded.** ADRs absent — phase 4 emits chosen pattern, the rejected three lost. Reviewers can't audit choice 3 phases later. | `4.1` (output: `architecture_pattern_decision` singular, alternatives stored only inside the chosen-decision blob); `4.2` (output: `tech_stack_decision` singular); `4.14` architecture_decisions DOES emit `adrs` array but is **synthesised post-hoc** from already-locked decisions in 4.1/4.2/4.3/4.4/4.6/4.8 (`depends_on: ["4.1","4.2","4.3","4.4","4.6","4.8"]`) — not an ADR-per-decision artifact during the choice. | `i2p_v3.json:2544-2581` (4.1), `:2583-2629` (4.2), `:3133-3175` (4.14) |
| G4 | **No falsifiability.** Spec states feature-as-fact ("users want X"). No "we believe X; if false, kill the feature; here's how we'd know." | `2.5` feature_brainstorm (item_fields lacks falsification); `2.6` feature_prioritization; `2.7` mvp_scope_definition; `2.9` success_metrics_definition (closest, but generic metrics not per-feature kill criteria); `3.1` functional_requirements_extraction; `3.2` nfr_performance_and_scalability (assumes fixed targets, no "wrong if…"); `3.7` business_rules_extraction; `4.1`/`4.14` ADRs schema has `consequences` but no `falsification_signal` / `revisit_trigger`. `0.3` assumption_identification has `risk_if_wrong` + `validation_method` — closest existing field, but doesn't propagate downstream. | `i2p_v3.json:1683-1723` (2.5), `:1833-…` (2.9), `:2002-…` (3.1), `:753-785` (0.3 — partial precedent) |
| G5 | **Prior-art blindness.** Agent doesn't know which 4 startups died trying this exact idea. Doesn't search HN/ProductHunt/Crunchbase/Wikipedia/Wayback. | `1.3` direct_competitor_identification (collects competitors by name but `status` field only `active/acquired/dead` — no graveyard *narrative*); `1.4` indirect_competitor_identification; `1.7` competitor_review_and_sentiment (focuses on living competitors). No step exists for "who tried this and failed and why." Tools: `web_search`, `smart_search`, `play_store`, `github` exist; no `find_prior_art` / no HN / no Wayback / no Crunchbase tool. | `i2p_v3.json:973-1013` (1.3); `src/tools/` listing — no prior-art tool present |
| G6 | **Compliance footprint not collected at intake.** Target jurisdictions, user types (consumer/B2B/health/children), data categories never asked. Hits team in phase 12+. | `1.11` regulatory_research collects regs **after** market research, not at intake; instruction says "Identify legal, regulatory… for this product" without a structured fingerprint input. `0.4` scope_ambiguity_detection lists `Audience/Scope/Business Model/Constraints/Platform/Differentiation/Priority` categories — does **not** include jurisdictions/data-categories/user-class. No step asks the founder for compliance-relevant facts. | `i2p_v3.json:787-822` (0.4), `:1317-1352` (1.11) |
| G7 | **No structured founder intake.** Founder pastes pitch into Telegram; no upload mechanism for interviews, screenshots, voice memos, prior research. | `0.1` raw_idea_intake instruction: "The user's raw idea is provided in the Context Artifacts section below" — single text channel. No artifact slot for `evidence_uploads`, `interview_transcripts`, `voice_memos`, `competitor_screenshots`, `prior_research_docs`. No mechanical intake step. | `i2p_v3.json:683-718` (0.1), `:701` (instruction text) |

## Phased plans

Each subsection below converts a Proposed-direction item into:
new/modified steps · new artifacts · new tools · post-hooks · effort
· acceptance · z0 dependency · risks.

### Structured user-research intake (closes G1, G7)

**Phase A — Intake mechanism (Telegram + artifact slots)**
- New i2p_v3 steps in `phase_0`:
  - `0.0a evidence_collection_request` (mechanical, agent: `mechanical`,
    executor: `request_evidence_uploads`) — emits a Telegram prompt
    listing accepted evidence types; lands a holding bucket per
    mission. Runs **before** `0.1`. `produces`: `evidence_index`.
  - `0.0b evidence_extraction` (agent: `analyst`) — consumes
    `evidence_index`; produces `extracted_evidence` (structured per-item
    facts with `source_uri`, `source_type`, `extracted_claims`,
    `confidence`). Runs after `0.0a`, before `0.1`.
- Modify `0.1 raw_idea_intake`: add `evidence_index` and
  `extracted_evidence` to `input_artifacts`; instruction gains "if
  evidence contradicts pitch, mark contradictions explicitly under a
  new section *Pitch-vs-Evidence Conflicts*."
- Modify `0.2 problem_statement_extraction`, `2.4 user_personas_and_journeys`,
  `2.9 success_metrics_definition`, `3.1 functional_requirements_extraction`,
  `3.7 business_rules_extraction`: every persona/requirement/rule gains
  an `evidence_refs: [evidence_id, …]` field; if empty, the field is
  set to `["agent_inference"]` so reviewers can audit.
- Modify `2.4` artifact_schema: add `evidence_refs`, `is_inference`
  (bool) to persona item_fields.

**New artifacts**
- `evidence_index` — array of `{evidence_id, source_type ∈ {interview,
  survey, screenshot, voice_memo, prior_research, analytics_export},
  source_uri, captured_at, founder_note, redaction_status}`.
- `extracted_evidence` — array of `{evidence_id, claim, claim_category
  ∈ {pain, current_tool, willingness_to_pay, demographic, jtbd,
  ux_complaint, feature_request}, quote, confidence ∈ {high, medium,
  low}}`.

**New tools (spec only)**
- `mr_roboto.request_evidence_uploads(mission_id, evidence_types: list[str], deadline: datetime?) -> evidence_index`
  — lands in `packages/mr_roboto/`. Sends Telegram message; opens upload window;
  files written to `missions/<id>/evidence/<evidence_id>/`.
- `evidence_extract(evidence_uri: str, source_type: str) -> list[extracted_claim]`
  — lands in `packages/vecihi/` (already does scraping; extends to file
  ingestion). Pipes images → `vision` tool (exists at `src/tools/vision.py`),
  audio → local Whisper (out of scope here, queue as own thing), text → LLM extract.
- `voice_memo_transcribe(audio_path: str) -> transcript` — lands in new
  `packages/whisper_runner/` (local Whisper) — design only, scope to
  follow-on as 01-pre-code §Open questions agreed.

**Post-hook / grounding gate**
- `0.0a` produces an `evidence_index` artifact. The G grounding gate
  verifies the index file exists and is non-empty OR contains the
  literal `{"status": "founder_skipped_evidence"}` (allow skip; mark
  on artifact). No mechanical content gate beyond presence.
- New post-hook `evidence_refs_audit` (mechanical, runs after
  `2.4`/`2.9`/`3.1`/`3.7`): scans the produced array; if >50% of items
  have `evidence_refs == ["agent_inference"]` and ambition tier ≥
  `private_beta`, emit `severity: warning` requesting founder review.
  Wires through the `blockers: {field: severity, levels:[…]}` pattern
  shipped 2026-05-05.

**Effort:** L (intake mechanism + new artifacts ripple through 6 modified steps).
**Acceptance:**
- Mission with founder-supplied interview transcript shows
  `evidence_refs` populated on at least one persona and one
  functional requirement.
- Reviewer (`3.11 requirements_review`) can cite `evidence_id` in its
  finding; gate fails when persona claim has neither
  `evidence_refs` nor `is_inference: true`.

**z0 dependency**
- z0 Phase B (preflight wizard) provisions the per-mission evidence
  bucket path + Telegram thread; Z1 reads `mission_id` + thread_id.
- z0 Phase A (founder profile) tells Z1 whether the founder typically
  has evidence to bring — if `prior_products > 0` and `evidence_culture
  = strong`, intake step *requires* at least 1 piece of evidence;
  otherwise allow skip.

**Risks**
- Voice-memo Whisper local execution is heavy on Windows + shared GPU
  with llama-server. Conflicts with `Never use taskkill on llama-server`
  rule. Defer Whisper to z0 follow-up; v1 supports text + image only.

### Taste delegation (mood boards, tone-of-voice samples)

**Phase A — Brand direction proposals (3-5 mood boards)**
- New i2p_v3 step `5.5a brand_direction_proposals` inserted **before**
  `5.6 brand_and_design_tokens`. Currently `5.6` directly produces
  `brand_identity` + `design_tokens` and only triggers
  `needs_clarification` if "multiple valid directions." That's a weak
  gate — Z1 wants taste *always* delegated.
  - agent: `analyst`
  - tools_hint: `web_search`, `extract_url`, `scrape_image`, `vision`
  - depends_on: `2.1 product_vision_and_positioning`,
    `2.4 user_personas`, `1.8 competitor_ux_evaluation`
  - produces: `brand_direction_options` — array of 3-5 `{option_id,
    name, mood_board: [{image_url, source_url, why_it_fits}],
    tone_samples: [{voice_attribute, sample_copy_blocks: [string,…]}],
    risks, founder_acceptance_signal: null}`.
  - `triggers_clarification: true` (founder picks via Telegram inline
    buttons).
- Modify `5.6 brand_and_design_tokens`:
  - add `brand_direction_options` + `selected_brand_direction_id` to
    `input_artifacts`; remove the inline `needs_clarification` branch
    (delegated upstream now).
  - instruction restated: "Given the founder-selected
    `selected_brand_direction_id`, materialise design tokens consistent
    with that option."

**New artifacts**
- `brand_direction_options` (above schema).
- `selected_brand_direction_id` — `{option_id, founder_note,
  selected_at}` lands via the clarify response handler.

**New tools (spec only)**
- `vecihi.scrape_image(url, dimensions?) -> {local_path, mime, hash, attribution}`
  — lands in `packages/vecihi/`. Scrape-not-generate per §Open question 2.
  Vision tool already exists (`src/tools/vision.py`).
- No new model needed; existing `vision` tool reads scraped boards.

**Post-hook / grounding gate**
- New mechanical post-hook `image_attribution_present` (runs after
  `5.5a`): every `mood_board[].image_url` resolves to a local file +
  has non-empty `source_url`. Block on missing attribution.
- `5.6` gated by presence of `selected_brand_direction_id` in
  blackboard (G grounding catches this for free if the field is named
  in `produces`).

**Effort:** M.
**Acceptance:**
- A mission produces 3-5 mood boards; founder picks one via Telegram;
  `5.6` consumes the chosen option; tokens reflect the selected
  direction (validated by reviewer at `5.10 design_review`).

**z0 dependency**
- z0 Phase A (founder profile) carries `brand_voice_samples` from
  prior products. If non-empty, `5.5a` instruction adds: "anchor at
  least one option to the founder's existing voice; the other 2-4 may
  diverge."
- z0 Phase D (Telegram thread) is required — clarify exchange happens
  in the mission thread.

**Risks**
- "Always 3-5 options" inflates phase 5 cost on prototype-tier
  missions. Gate behind ambition_tier: `prototype` keeps existing `5.6`
  inline path; `private_beta+` activates `5.5a`. This honors
  z0 ambition-tier matrix.

### Architecture Decision Records (ADRs) as first-class

**Phase A — ADR-per-decision instead of post-hoc synthesis**
- Modify `4.1 architecture_pattern_selection`:
  - output_artifacts: `architecture_pattern_decision` → ADR-shaped
    `{adr_id, title, status ∈ {proposed, accepted, superseded},
    context, options_considered: [{name, pros, cons, evaluation_score,
    rejected_because}], decision, consequences, falsification_signal,
    revisit_trigger, reversal_cost ∈ {low, medium, high},
    evidence_refs}`. min_items on `options_considered`: 3.
  - `triggers_clarification: true` always (founder chooses, not LLM
    "trigger if top-2 close").
- Modify `4.2 tech_stack_research_and_selection`: same ADR shape per
  layer (frontend ADR, backend ADR, database ADR, infra ADR — 4 ADRs
  emit, not 1 fused decision).
- Modify `4.6 auth_system_design`, `4.8 third_party_service_selection`:
  same ADR shape.
- **Reframe** `4.14 architecture_decisions`:
  - currently synthesises `adrs` post-hoc; instead, becomes
    `4.14 adr_consolidation` — collects the per-decision ADR artifacts
    into a single browsable register; verifies all-required-ADRs-present;
    emits `adr_register`.
  - `min_items: 3` stays.

**Modified artifacts**
- `architecture_pattern_decision`, `tech_stack_decision`, `auth_design`,
  `third_party_selections` all converge to ADR shape (above).
- New `adr_register` — `{adrs: [adr_id, title, status, decision_summary,
  links_to: [other_adr_id]}, …]}`.

**New tools**
- None. ADR shape is artifact-only.

**Post-hook**
- New mechanical post-hook `adr_completeness_check` (runs at `4.14`):
  asserts every required decision (pattern, frontend, backend,
  database, infra, auth, each vendor in `third_party_selections`)
  has a corresponding ADR with status ∈ {accepted}. Missing → block.
- New mechanical `adr_alternatives_min` (runs after each ADR-emitting
  step): assert `len(options_considered) >= 3`.

**Effort:** M.
**Acceptance:**
- Reviewer at `4.16 architecture_review` can cite which alternatives
  were rejected and why for any decision. The chosen-vs-alternatives
  diff is explicit. Reverse-cost informs phase 8+ change-amplification
  decisions.

**z0 dependency**
- z0 Phase A `founder_profile.technical_comfort` calibrates ADR jargon
  density. Low → ADR `decision` field rendered in plain language with
  jargon defined; high → terse.
- z0 ambition tier sets `adr_alternatives_min` strictness: prototype
  may accept 2 options, revenue_product requires 3+.

**Risks**
- Output token weight goes up (3 alternatives × every decision).
  Cap each `options_considered[].evaluation` to 80 words; cap pros/cons
  to 5 bullets each. Validated against `estimated_output_tokens`.

### Failure-mode column (falsifiability column on every commitment)

**Phase A — Add `failure_mode` field to commitment-shaped artifacts**
- Modify these existing steps' artifact_schema to add a
  `failure_mode` field per item:
  - `2.5 feature_brainstorm` items: add
    `failure_mode: {hypothesis, signal_if_wrong, kill_threshold}`.
  - `2.6 feature_prioritization` retains scoring; adds
    `kill_criteria` field per top-priority feature.
  - `2.7 mvp_scope_definition` items: add `falsification_signal`,
    `success_metric_link` (FK to `2.9` metric).
  - `2.9 success_metrics_definition`: add `kill_threshold` per metric
    (current schema has metric without falsification dial).
  - `3.1 functional_requirements_extraction` items: add
    `if_wrong_consequence` (matches `0.3 assumption_identification`'s
    existing `risk_if_wrong` shape — re-use the precedent).
  - `3.2 nfr_performance_and_scalability`, `3.3 nfr_availability_and_security`:
    add `revisit_trigger`.
  - All ADRs (see ADR phase): `falsification_signal`, `revisit_trigger`
    are part of the ADR shape.

**New artifacts**
- None standalone — fields embedded in existing artifacts.

**New tools**
- None.

**Post-hooks**
- New mechanical `falsification_completeness` (runs after `2.5`, `2.7`,
  `2.9`, `3.1`, `4.1`, `4.2`, `4.6`, `4.8`): every item must have
  non-empty `failure_mode` / `falsification_signal` / `kill_criteria`
  per the artifact's schema. Empty or `"TBD"` → fail.
- The post-hook output feeds directly into the future
  `09-growth.md` monitoring rules ([cross-zone link](#cross-zone-implications)).

**Effort:** M (schema-only ripples but ~7 steps modified).
**Acceptance:**
- A given feature in `mvp_scope_definition` has a kill_threshold that
  references a metric in `success_metrics_definition`.
- Reviewer at `3.11 requirements_review` rejects requirements where
  `if_wrong_consequence == "Needs validation"`.

**z0 dependency**
- z0 Phase F (north-star metric) sets the lens: features whose
  `success_metric_link` doesn't roll up to north-star get flagged at
  `2.6` prioritization. If north-star is "TBD", post-hook is
  warning-only.
- z0 ambition tier: `prototype` skips strict falsification; from
  `private_beta` upward it's a blocker.

**Risks**
- Founders push back on "every feature must declare a kill threshold."
  Make threshold field-optional with explicit `we_dont_know_yet: true`
  flag — that's still falsifiable (it says: TBD, will define before
  beta). The flag is a soft signal not a bypass.

### Web-grounded prior art (closes G5)

**Phase A — `find_prior_art` tool + new step**
- New step `1.0 prior_art_search` in `phase_1`, runs in parallel with
  `1.1 market_existence_research` (both depend on `0.6`).
  - agent: `researcher`
  - tools_hint: `find_prior_art`, `web_search`, `extract_url`
  - produces: `prior_art_report` — `{
      attempted_solutions: [{name, founded_year, status ∈ {alive,
      acquired, dead, dormant}, traction_signal, failure_mode (if dead),
      sources: [url], thesis_summary}],
      adjacent_failures: [{…}],
      key_lessons: [{lesson, evidence_refs}],
      graveyard_count: int
    }`.
- Modify `1.14 go_no_go_assessment`:
  - input_artifacts: add `prior_art_report`.
  - instruction: factor `graveyard_count` and `key_lessons` into the
    score. If ≥3 dead competitors with same thesis → reduce
    `competitive_feasibility` score; surface lessons.
- Modify `2.1 product_vision_and_positioning` instruction: "Reference
  `prior_art_report.key_lessons` and explain how this product avoids
  each failure mode."

**New artifacts**
- `prior_art_report` (above).

**New tools (spec only)**
- `vecihi.find_prior_art(idea_summary: str, domain_keywords: list[str], k: int = 10) -> prior_art_report`
  — lands in `packages/vecihi/` (escalating scraper). Internally calls,
  in parallel:
  - HN Algolia search API (free, no key) — `https://hn.algolia.com/api/v1/search`
  - Product Hunt search (existing scraping path; if blocked, GCSE fallback)
  - Crunchbase via free-tier scrape (no API key; rate limit, polite)
  - Wikipedia category search via MediaWiki API (free)
  - Wayback Machine CDX API (`http://web.archive.org/cdx/search/cdx`)
    for "site existed once, gone now" signal
  Aggregates results, dedupes by name, classifies status via heuristics
  (last commit, domain resolves, SSL valid, last news mention).
  Built on existing `web_search` + `vecihi` infra (no new HTTP stack).

**Post-hook**
- Mechanical `prior_art_min_coverage` (runs after `1.0`):
  `len(attempted_solutions) >= 3` OR explicit
  `{"verdict": "blue_ocean_validated", "evidence": [...]}` payload.
  Empty results → fail (catches "search returned nothing" hallucination).

**Effort:** M (tool ~3-5 days; step + downstream wiring 1-2 days).
**Acceptance:**
- New mission for "AI meal-planner" surfaces ≥3 dead competitors with
  failure_mode populated (e.g. Mealime acquired, Eat This Much status,
  PlateJoy etc.).
- `1.14 go_no_go_decision` cites at least one lesson from
  `prior_art_report.key_lessons` in its reasoning.

**z0 dependency**
- z0 ambition tier `prototype` may skip the heavy prior-art pass
  (cost-bound) — `1.0` becomes a 2-call lite version returning
  graveyard_count only. From `private_beta` upward run full version.
- z0 Phase E (cost ceiling) gates the breadth: tighter ceiling →
  fewer parallel sources.

**Risks**
- Crunchbase scraping is brittle and ToS-grey. Default to
  HN+Wikipedia+Wayback (clearly permitted) as v1; Crunchbase via
  Brave/GCSE fallback only if surfaced naturally.
- Hallucinated competitors are still possible without grounding gate
  on URLs. Add `vecihi.fetch_url(competitor.website_url)` validation
  during the same step — if URL doesn't resolve, mark
  `status: "dead_or_hallucinated"`.

### Compliance fingerprint (closes G6)

**Phase A — Pre-intake compliance fingerprint at phase 0**
- New step `0.4a compliance_fingerprint_collection` (mechanical,
  agent: `mechanical`, executor: `compliance_fingerprint_wizard`)
  inserted between `0.4 scope_ambiguity_detection` and
  `0.5 human_clarification_request`.
  - Triggers a Telegram form: target jurisdictions (multi-select
    EU/US/UK/CA/TR/other), user types (consumer/B2B/health/children/
    financial/government), data categories (PII/payment/health/
    biometric/location/minor's data), data residency requirement,
    age-gate needed?, third-party processors expected.
  - produces: `compliance_fingerprint` artifact.
  - z0 inputs honored: if z0 already collected this at preflight
    (G item in z0), step **skips** with a passthrough — read from
    z0 founder_profile/mission_preflight rather than re-asking.
- Modify `0.6 idea_brief_compilation_and_review`:
  - add `compliance_fingerprint` to `input_artifacts`.
  - new section in idea_brief: `Compliance Footprint`.
- Modify `1.11 regulatory_research`:
  - add `compliance_fingerprint` to `input_artifacts`.
  - instruction: "Use the compliance_fingerprint to scope research:
    only research regulations that the fingerprint flags as
    applicable." — turns 1.11 from open-ended to fingerprint-driven.
- New step `1.11a compliance_overlay` (agent: `analyst`,
  depends_on `1.11`):
  - produces: `compliance_overlay` — `{required_documents:
    [privacy_policy, cookie_banner, dpa, tos, retention_policy,
    age_gate, accessibility_statement], per_document_status:
    {generated_template, founder_review_required, blocker_for_phase},
    monitoring_obligations, data_subject_rights_implementation}`.
  - Each row links to `06-real-world-bridge.md` for execution.

**New artifacts**
- `compliance_fingerprint` — `{jurisdictions: [country_code],
  user_classes: [consumer|b2b|health|children|financial|government],
  data_categories: [pii|payment|health|biometric|location|minor],
  data_residency_required: bool, age_gate_required: bool,
  third_party_processors: [name], collected_at, source ∈ {z0_preflight,
  z1_intake}}`.
- `compliance_overlay` (above).

**New tools (spec only)**
- `mr_roboto.compliance_fingerprint_wizard(mission_id, prefill?: dict) -> compliance_fingerprint`
  — lands in `packages/mr_roboto/`. Multi-step Telegram form; "I don't
  know" allowed per question; agent attempts inference from
  `idea_brief_final` for unanswered fields.
- `compliance_template_render(fingerprint, doc_type) -> draft_md`
  — lands in `packages/mr_roboto/` (or new `packages/legal_lite/`).
  Reads quarterly-refreshed template library from
  `data/compliance_templates/<jurisdiction>/<doc_type>.md.j2`.

**Post-hooks**
- `0.4a` G grounding: artifact non-empty, all required fields present
  (allow `"unknown"` value, but field must exist).
- `1.11a` mechanical `compliance_template_present`: each
  `required_documents[i].generated_template` resolves to a file path on disk.
- New `compliance_blocker_check` runs at every phase boundary ≥ phase 7:
  if any `compliance_overlay.per_document_status[i].blocker_for_phase`
  ≤ current_phase AND `founder_review_required == true` AND no
  founder_signoff → block. (Cross-zone: actual enforcement in
  `06-real-world-bridge.md` and `08-operations.md` pre-launch.)

**Effort:** L (template library + cross-phase enforcement).
**Acceptance:**
- Mission targeting EU + storing PII auto-surfaces GDPR DPA
  requirement before phase 7 starts.
- Mission targeting children's app surfaces COPPA blocker at `0.4a`.
- Reviewer at `3.11 requirements_review` validates that data-handling
  FRs match `compliance_fingerprint.data_categories`.

**z0 dependency**
- z0 §G compliance fingerprint pre-intake — z0 collects high-level;
  Z1's `0.4a` reads z0 output and *only asks the deltas*. If z0 didn't
  run (legacy missions), `0.4a` runs full wizard.
- z0 Phase A founder_profile may carry prior-mission compliance
  posture (e.g. "we're a CA-only consumer SaaS shop") — pre-fill
  defaults.

**Risks**
- Privacy-policy template currency is the bigger risk per §Open Q4.
  Solved by quarterly cron + founder approval (out of scope here);
  meanwhile every generated doc carries a `template_version` +
  `last_reviewed` watermark and a "this is not legal advice" banner.
  Flag clearly in handoff to `06-real-world-bridge.md`.

## Open questions resolved

| # | Question | Tentative (in doc) | Resolution + reasoning |
|---|---|---|---|
| OQ1 | Evidence hierarchy — when founder evidence contradicts agent inference, who wins? | Evidence wins; agent flags; founder confirms. | **Confirm.** Implementation: `0.1 raw_idea_intake` instruction gains a *Pitch-vs-Evidence Conflicts* section that lists every contradiction; `0.5 human_clarification_request` adds those conflicts to its top-5 clarifications when severity ≥ medium. Evidence carries by default (`evidence_refs` overrides `agent_inference`); founder can override evidence with explicit acknowledgement (`override_reason` field on the persona/requirement). Audit trail preserved. |
| OQ2 | Mood-board rendering — generate vs scrape? | Scrape. | **Confirm.** Generated images don't carry market provenance + risk hallucinated brands. Tool: `vecihi.scrape_image` with attribution required. v1 = scrape only; revisit when in-house design tokens library matures (Wave 7 of 2026-05-07 plan). |
| OQ3 | ADR storage — per-mission vs global library? | Global library; per-mission references. | **Confirm with refinement.** Two-layer: (1) global `adr_library` table seeded with canonical templates ("when to pick monolith vs microservices", "when to pick PG vs Mongo"); (2) per-mission ADR instances reference + customize a template. Schema: `mission_adrs.template_id → adr_library.id` (nullable; bespoke ADRs allowed). Cross-mission learning compounds: mission-N can read previous-mission ADRs in same domain via `founder_profile.prior_missions` join. Aligns with z0 §A founder profile + §K mission template selection. |
| OQ4 | Compliance template currency — who keeps templates current? | Quarterly cron + founder approval; out of scope. | **Confirm with hard scope line.** Templates ship with `template_version` + `last_reviewed_date`; any doc older than 180 days emits a warning at `1.11a`. Cron lives in `packages/general_beckman/` scheduled jobs (existing 6h beckman cron pattern from cloud subsystem 2026-04-27). Renewal is a *founder task* in `06-real-world-bridge.md` — Z1 surfaces; doesn't execute. Documented as "irreducible" in 06's founder track. |
| OQ5 | Voice-memo intake — local Whisper vs cloud? | Local Whisper. | **Defer with caveat.** Local Whisper on Windows + shared GPU contests llama-server VRAM (see CLAUDE.md "Never use taskkill on llama-server"). v1 of evidence intake = text + image only; voice-memo handling deferred to a follow-on (`packages/whisper_runner/`) that runs Whisper *on CPU* (slower, won't touch GPU). Z1 ships text+image; voice = post-Z1 enhancement. |

## Cross-zone implications

- **02-build-foundation.md** — Spec quality from Z1 directly drives recipe match in T5. Add note: recipes must declare `compatible_with: {compliance_fingerprint.user_classes, jurisdictions}` so a "children's app + EU" recipe doesn't get picked for a B2B US mission. Recipe library schema gains a compliance compatibility filter.
- **06-real-world-bridge.md** — Compliance overlay (§Compliance fingerprint Phase A above) is the inbound contract; 06 owns execution (vendor adapters for Stripe KYC, DPA signing, cookie banner deployment, age-gate UX). 06 must publish a `compliance_overlay_consumer` interface so Z1's `1.11a` knows what 06 can fulfill vs what's founder-task.
- **07-humanish-layers.md** — `selected_brand_direction_id` from §Taste delegation feeds marketing voice + site copy. 07 must read `brand_direction_options` + selection rather than re-deriving voice. Avoids drift between product brand and marketing brand.
- **10-cross-cutting.md** — Falsification-mode column (§Failure-mode column) is the upstream feeder for 10's monitoring/cost/reversibility patterns. 10 should consume `kill_threshold` per feature into auto-generated alerts; consume `revisit_trigger` per ADR into reversibility tags. Add: `evidence_refs` audit (mechanical post-hook from §Structured user-research intake) needs 10's severity-gate framework to function.

---

## Top 3 cross-zone surprises found

1. **`4.14 architecture_decisions` already emits ADRs — but post-hoc.** It's a synthesiser depending on six already-locked decisions, not a per-decision ADR captured at choice time. The fix isn't "add ADRs"; it's "shift ADR shape upstream into the choosing steps and turn 4.14 into a register." That's a smaller change than the doc framed.
2. **`0.3 assumption_identification` already has `risk_if_wrong` + `validation_method`** — a working precedent for the falsifiability column that could be reused everywhere. The gap isn't "introduce the concept"; it's "propagate this existing field shape to features/NFRs/ADRs that don't have it yet."
3. **z0 already collects compliance fingerprint at preflight (§G).** Z1 must check + extend rather than re-collect, otherwise the founder gets asked the same questions twice. The plan's `0.4a` step is now skip-if-z0-supplied with a delta wizard, not a fresh wizard. Same applies to brand voice (`founder_profile.brand_voice_samples` from z0 §A).
