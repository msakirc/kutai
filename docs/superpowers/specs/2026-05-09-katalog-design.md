# Katalog — Skills, APIs, MCPs Catalog

**Status**: design (brainstorming output, pre-plan)
**Date**: 2026-05-09
**Owner**: KutAI core

## Problem

KutAI's i2p (idea-to-product) workflow burns LLM iterations on tasks that have well-known recipes — auth wiring, scaffolds, CRUD, RAG setup, deploy steps. The existing skill subsystem (`src/memory/skills.py`) auto-grows tiny prompt-injection routing hints from successful tasks but cannot import external recipes. Public skill libraries surveyed (anthropics/skills, obra/superpowers, MetaGPT roles, batteries-included starter kits) carry far more leverage than current internal hints — but no infrastructure exists to fetch, vet, store, and apply them.

Free APIs registry (`src/tools/free_apis.py`) and MCP server discovery face the same shape of problem: external artifacts need discovery, vetting, indexing, exposure to agents, and effectiveness telemetry. Today each is bespoke or absent.

## Goals

- Lazy fetch + apply external recipes per mission (no wholesale registry vendoring)
- Vetted catalog with auditable trust chain
- Pluggable sources (extensible across registries)
- Unified lifecycle for skills, APIs, MCPs (and future artifact types)
- Recipes that can fast-forward whole scaffolding processes — not just hint to LLM

## Non-goals (v1)

- Migrating `src/tools/free_apis.py` onto the framework. Free APIs work likely has bit rot; audit + migrate as separate task. Framework should fit it but not gate skills work on free_apis state.
- Mid-mission skill discovery (breaks reproducibility, defers to v2 if ever)
- Headless-browser scraping for ClawHub / SkillHub (catalogs unscrapeable today; revisit if APIs land)
- Public distribution of KutAI-authored skills (consumer-only for now)

## Design constraints (from brainstorming)

- **No wholesale registry vendoring** — fetch only what's needed
- **Vetted-skill sub-mechanism** — every external skill passes through a gate before agents see it
- **Trusted-source allowlist auto-vet** — anthropics, obra, similar reputable sources auto-approve; everything else requires human review
- **Pre-vetted only at mission time** — agents see only enabled skills; no mid-mission fetch
- **Daily discovery cron** — refresh trusted sources + need-driven search of untrusted catalogs
- **Hybrid preemption** — high-confidence + mechanizable + trusted-source recipes preempt LLM via mr_roboto; lower confidence stays as LLM-callable tool
- **One package** — `packages/katalog/` (single subsystem, internal subdirs; not split read/write)
- **Conceptual unification across artifact types** — skills, APIs, MCPs share lifecycle/discovery/trust/telemetry; per-artifact plugins handle schema/exposure/execution differences

## Artifact types

| Type | Schema shape | Activation cost | Exposure to agent | Execution path |
|---|---|---|---|---|
| skill | manifest + body + assets | none | context inject OR tool register OR role swap | prompt-inject / shell-out / sys_prompt prepend |
| api | endpoint + auth + rate_limit | none | tool register (`call_api`) | HTTP via existing call_api |
| mcp | server cmd + env + tool list | process lifecycle | tool register per MCP tool + server start | MCP protocol |

Skill kinds (sub-types within skill artifact):
- `internal_hint` — auto-grown routing hint (current `skills.py` content); body inline
- `prompt_skill` — markdown SKILL.md (anthropics/superpowers convention); body on disk
- `shell_recipe` — invocation steps (npx wasp, cookiecutter, degit) + post-patch
- `procedure` — ordered tool chain
- `agent_config` — sys_prompt + tool list (MetaGPT-style role)

## Architecture

### Package layout (mutable; firm at module-responsibility level)

```
packages/katalog/
  schema.py          # DB tables, manifest types
  contracts.py       # Plugin protocol (DiscoveryPlugin + AccessPlugin)
  trust.py           # source allowlist, trust scoring
  index.py           # read API for matcher
  matcher.py         # vector + hint match against unified index
  exposure.py        # per-kind exposure to agent context
  executor.py        # kind-dispatch executor (preempt vs LLM-tool path)
  telemetry.py       # injection_count, success, latency, downstream pass/fail
  discovery/
    cron.py          # daily run
    vet_queue.py     # Telegram approval flow
    fetch.py         # staging → vendor/ on approve
    sources/
      github_path.py
      github_topic.py
      awesome_list_md.py
      web_markdown.py
      clawhub_api.py # stub for future
  plugins/
    skill.py         # DiscoveryPlugin + AccessPlugin for skill artifact
    api.py           # later (when migrating free_apis)
    mcp.py           # later (Glama feed → MCP server config)
```

Subdirs are guidance; if implementation finds a different split clearer, that's fine.

### Data model

**Disk** (`vendor/skills/<source>/<name>/v<version>/`):
- `manifest.yaml` — typed metadata (see Manifest below)
- `SKILL.md` — body (for skill artifact)
- `assets/` — optional helper files (scripts, templates)

**DB** (single unified index):
```sql
CREATE TABLE katalog_index (
  id INTEGER PRIMARY KEY,
  artifact_type TEXT NOT NULL,    -- 'skill' | 'api' | 'mcp'
  kind TEXT,                      -- skill sub-type; null for api/mcp
  source TEXT NOT NULL,           -- 'github:anthropics/skills' | 'public-apis-md' | etc.
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  manifest_path TEXT,             -- null for inline (internal_hint)
  body_excerpt TEXT,              -- first ~500 chars for embedding
  embedding BLOB,                 -- vector
  vet_state TEXT,                 -- 'pending' | 'approved' | 'rejected' | 'auto_approved'
  vet_hash TEXT,                  -- content hash at vetting time
  trust_score REAL,               -- 0..1, source_trust × content_review
  mechanizable BOOLEAN,           -- gate for preemption (shell_recipe / procedure only)
  enabled BOOLEAN DEFAULT 1,
  created_at TIMESTAMP,
  vetted_at TIMESTAMP,
  UNIQUE(source, name, version)
);

CREATE TABLE katalog_usage (
  id INTEGER PRIMARY KEY,
  artifact_id INTEGER REFERENCES katalog_index(id),
  task_id TEXT,
  exposed BOOLEAN,                -- shown to agent / preempted
  called BOOLEAN,                 -- agent actually used it
  succeeded BOOLEAN,              -- downstream step passed
  latency_ms INTEGER,
  occurred_at TIMESTAMP
);

CREATE TABLE katalog_sources (
  id INTEGER PRIMARY KEY,
  source_id TEXT UNIQUE NOT NULL, -- 'github:anthropics/skills'
  source_type TEXT,               -- adapter name: 'github_path'
  endpoint TEXT,                  -- repo URL or query
  auth_env TEXT,                  -- env var name for token if needed
  trusted BOOLEAN,                -- auto-vet flag
  enabled BOOLEAN DEFAULT 1,
  last_run_at TIMESTAMP
);
```

### Manifest schema (manifest.yaml)

```yaml
name: wasp-saas-init
version: 1.0.0
artifact_type: skill
kind: shell_recipe
source: github:wasp-lang/open-saas
license: MIT
mechanizable: true
# trust_score is computed from source + content review at vet time, not authored here
intent_keywords: [saas, auth, stripe, full-stack]
inputs_schema: {name: string, db: enum[postgres, sqlite]}
invocation:
  steps:
    - cmd: "npx wasp-cli new -t saas {{name}}"
    - cmd: "cd {{name}} && wasp db migrate-dev --name init"
artifacts: [main.wasp, .env.server]
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
    def expose_to_agent(self, row: IndexRow, task_ctx: TaskContext) -> Exposure: ...
    def execute(self, row: IndexRow, task_ctx: TaskContext, inputs: dict) -> Result: ...
```

Plugins live in `katalog/plugins/<artifact>.py`. Single class can implement both protocols.

### Lifecycle

1. **Discover** (daily cron in beckman scheduled jobs)
   - For each enabled trusted source: source adapter calls `discover()` → list of `SkillRef`
   - Diff against `katalog_index`; new + updated → fetch
   - Need-driven untrusted search: scan recent `katalog_usage` + agent burn telemetry, extract intent keywords from high-burn steps, query untrusted catalogs (github_topic, awesome_list_md), queue results in vet_queue
2. **Fetch** (`discovery/fetch.py`)
   - Source adapter pulls into `vendor/skills/.staging/<source>/<name>/`
   - DiscoveryPlugin parses + validates manifest
   - Compute content hash
3. **Vet**
   - Trusted source → auto-approve (`vet_state=auto_approved`, log entry)
   - Untrusted → enqueue in `katalog_pending` table; `/katalog pending` Telegram command shows manifest + body diff + `[Approve]` `[Reject]` `[Defer]` buttons
4. **Enable**
   - On approve: move staging → `vendor/skills/<source>/<name>/v<version>/`
   - Embed body excerpt → `katalog_index.embedding`
   - `enabled=1`
5. **Update**
   - Re-fetch creates `v<n+1>/`; hash diff highlighted in vet UX
   - Trusted-source minor-version bumps auto-approve; major bumps re-vet (configurable)
6. **Auto-disable** (effectiveness pruning)
   - `katalog_usage.succeeded` rolling rate <30% over N invocations → `enabled=0` + Telegram flag
   - Vetted artifacts never deleted; can be re-enabled after review

### Match + dispatch (hot path)

Existing call site: `coulson/context.py:945` (currently `skills.find_relevant_skills`).

Replaced with `katalog.matcher.match(task)`:
1. Vector similarity over `katalog_index.embedding`
2. Score: `confidence = vector_similarity × source_trust × hint_bonus` (hint_bonus when i2p step carries explicit `recipe_hint`)
3. Return ranked hits
4. Dispatcher branches by kind:
   - `internal_hint`, `prompt_skill` → `exposure.inject_context(rows)` (existing path)
   - `shell_recipe`, `procedure`:
     - `confidence ≥ θ AND mechanizable=true AND source_trust ≥ τ` → preempt: mr_roboto runs `executor.run_recipe` mechanical step
     - else → register as `run_skill_<id>` tool, inject "consider running: …" in context
   - `agent_config` → fatih_hoca prepends sys_prompt at task start

θ and τ are tunable thresholds; defaults set conservatively; per-source overrides allowed.

### Source adapters

Initial set:
- `github_path` — anthropics/skills, obra/superpowers, MetaGPT roles
- `github_topic` — search GitHub by topic for need-driven discovery
- `awesome_list_md` — parse curated lists (awesome-cookiecutter, awesome-mcp-servers)
- `web_markdown` — generic SKILL.md URL fetch
- `clawhub_api` — stub; activates if/when ClawHub exposes API

Adapter contract (defined in `katalog/contracts.py`, implementations in `discovery/sources/`):
```python
class SourceAdapter(Protocol):
    source_type: str
    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]: ...
    async def fetch(self, ref: ArtifactRef) -> Path: ...  # returns staging dir
```

### Telegram UX

- `/katalog` — overview: counts by type/state, vet queue depth
- `/katalog pending` — next pending artifact: manifest + body + `[Approve]` `[Reject]` `[Defer]` buttons
- `/katalog add <url>` — manual import trigger (untrusted by default)
- `/katalog source add <type> <endpoint> [--trusted]` — configure new source
- `/katalog source list`
- `/katalog disable <id>` / `/katalog enable <id>`
- `/katalog stats` — A/B internal vs external vs LLM-only success rates
- `/katalog discover [source]` — force discovery run (skip cron)

### Wiring into existing KutAI

- `coulson/context.py:945` — replace `skills.find_relevant_skills` with `katalog.matcher.match`. Internal hints stay (now flow through unified matcher).
- `mr_roboto` — add `run_recipe` action that delegates to `katalog.executor.run_recipe`
- `general_beckman` — add scheduled job: `katalog.discovery.cron.daily_run()`. Recipe-lane routing: when matcher returns preempting recipe, beckman routes task to mechanical lane with `runner=katalog_recipe`
- `src/memory/skills.py` — keep auto-capture from successful tasks; redirect to write into `katalog_index` with `kind=internal_hint`. Existing `skills` rows migrate via one-shot script.
- `src/app/telegram_bot.py` — add `/katalog ...` command group + callback handlers for vet buttons

### Migration

1. Create `katalog_index`, `katalog_usage`, `katalog_sources` tables
2. Migration script: copy `skills` rows → `katalog_index` with `artifact_type='skill'`, `kind='internal_hint'`, embed `description+strategy_summary`
3. Seed `katalog_sources` with anthropics/skills, obra/superpowers (trusted=1)
4. First discovery cron run populates external skills, auto-vets trusted, queues untrusted
5. Old `skills.py` API kept as thin shim until coulson/grading migrate

## Open issues (defer to plan, not blocking design)

- Trust-score formula calibration (initial heuristic: anthropics/obra = 1.0, GitHub-topic match = 0.5, web_markdown unsigned = 0.3)
- Threshold defaults (θ, τ) — start strict, lower based on telemetry
- Vet UX for multi-file skills (anthropics xlsx skill has helper Python files) — show file tree + per-file approve? Or all-or-nothing?
- Rate limit handling for GitHub API in source adapters
- LLM-assisted vetting later (read manifest + body, flag prompt-injection / unsafe tool calls)
- Mechanizable flag — who sets it? Adapter inference (presence of `invocation.steps` block) plus vetter override

## Testing strategy

- Unit: each source adapter against fixture HTTP responses
- Unit: each plugin's `parse_manifest`, `vet_checks`, `expose_to_agent`, `execute`
- Integration: end-to-end fetch → vet → enable → match → expose with mocked sources
- Telemetry: assert `katalog_usage` rows written on each match
- Migration: assert existing `skills` rows readable via new matcher with byte-identical injection text

## Success criteria

- v1 ships with skill plugin + 4 source adapters + Telegram vet UX
- Anthropics/skills + obra/superpowers fully imported via discovery cron
- At least one shell_recipe (e.g., create-next-app, cookiecutter-django) preempts LLM in i2p mission, end-to-end
- Existing internal-hint matching produces byte-identical context-injection output post-migration
- Telegram `/katalog stats` shows internal-hint vs external-skill A/B
