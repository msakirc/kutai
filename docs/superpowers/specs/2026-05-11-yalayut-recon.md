# Yalayut Recon — Real-world artifacts shape, quality, value

**Status:** recon (input to yalayut synthesis pipeline lock-in)
**Date:** 2026-05-14 (filename keeps 05-11 per request)
**Method:** Read GitHub raw contents directly via `curl https://api.github.com` (gh CLI unauthenticated in env, fell back to public REST + raw URLs). WebFetch used for awesome-list READMEs that don't render well as raw markdown. Sampled ~120 artifacts across 6 sources; read full body for ~12, frontmatter-only for the rest.

**TL;DR for synthesis design:**
- Two source classes dominate: **canonical SKILL.md** (anthropics, obra, wshobson, matlab — clean YAML frontmatter, mechanical parser is trivial) and **curated awesome lists** (heterogeneous bullets, table fragments, no shared schema — LLM-fallback unavoidable).
- **Frontmatter is the bedrock.** All four sampled "real skill" repos use the same shape: `name:` + `description:` (sometimes `license:`, `model:`). No `inputs_schema`. No `invocation.steps`. Our manifest's typed-recipe section is **authored locally**, not lifted.
- **Mechanizable shell recipes are rare in the wild.** anthropics + obra are 100% prose-skill (LLM-callable, not preemptable). True shell-recipe artifacts only appear in cookiecutter ecosystem and a few MCP install snippets — and those are not SKILL.md, they're README-buried.
- **Windows blockers** cluster around: `chmod +x`, `apt-get`/`brew`, raw `.sh` helper scripts with bash-only syntax, `sudo` MCP installs.
- **Negative-value cluster:** 60%+ of `topic:claude-skill` repos are personal aggregations / "context engineering" wrappers / Chinese-character "humanize my text" tools / brand promos. Trust-by-stargazer is broken (many ≥100k-star "skill" repos are actually awesome-list re-publishers).
- **Top-20 seed list is dominated by anthropics + obra + ~5 selected wshobson agent configs + ~3 cookiecutter recipes + ~3 Stripe/Vercel/Cloudflare best-practice skills surfaced via awesome-agent-skills.** That's our T0 trusted set.

---

## 1. Source: `github:anthropics/skills`

**Overview.** 17 skill directories under `skills/`. License mixed (Proprietary "LICENSE.txt has complete terms" on ~10, generic license note on others). Each dir holds `SKILL.md` (always) + optional `scripts/`, `references/`, sometimes a `LICENSE.txt`. Skill scope: format-handlers (pdf, docx, pptx, xlsx), creative (algorithmic-art, canvas-design, theme-factory, slack-gif-creator), platform (claude-api, mcp-builder, web-artifacts-builder, webapp-testing), meta (skill-creator), comms (brand-guidelines, internal-comms, doc-coauthoring), and `frontend-design`. Sampled: all 17 frontmatter, full body of `pdf`.

### 7-axis distribution (n=17)

| Axis | Distribution |
|---|---|
| Name quality | usable 17/17 (100%) |
| Frontmatter | yaml-fm 17/17 (100%) |
| Inputs schema | none 17/17 (100%) — descriptions imply context but no declared inputs |
| Shell blocks | mixed 5/17 (29%, e.g. `pdf` has python code blocks; `web-artifacts-builder` references `scripts/init-artifact.sh`); prose-only 12/17 (71%) |
| Intent keywords | obvious 16/17 (descriptions explicitly enumerate triggers); inferable 1/17 (`internal-comms`) |
| Usability | runnable-as-is 4/17 (pdf, brand-guidelines, internal-comms, theme-factory — pure prose); needs-adaptation 11/17 (require Python libs like pypdf/python-docx/openpyxl/markitdown, or node `pptxgenjs`); blocked 2/17 (web-artifacts-builder uses `.sh` helpers; webapp-testing requires Playwright headless install) |
| Value delta | high 6/17 (pdf, docx, pptx, xlsx, mcp-builder, skill-creator — mechanizes large workflows); medium 9/17 (theme-factory, brand-guidelines, claude-api, frontend-design etc.); low 2/17 (slack-gif-creator, algorithmic-art — niche) |

### Raw → synthesized examples

```yaml
# 1. pdf
RAW: name: pdf
     description: Use this skill whenever the user wants to do anything with PDF files... [200 words]
     license: Proprietary.
SYN: {name: anthropics-pdf, intent_keywords: [pdf, extract-text, merge-pdf, split-pdf, ocr, watermark, forms],
      kind: prompt_skill, mechanizable: false, name_original: pdf, license: proprietary}
FLAG: name "pdf" globally ambiguous → MUST prefix with source slug

# 2. docx
RAW: name: docx; description: "...Triggers include: any mention of 'Word doc'..." (huge enumerated trigger)
SYN: {name: anthropics-docx, intent_keywords: [docx, word, report, memo, template, find-replace, tracked-changes],
      kind: prompt_skill, mechanizable: false, name_original: docx}
FLAG: description-stuffed-with-triggers is the anthropics house style; our matcher should embed FULL description not just name

# 3. mcp-builder
RAW: name: mcp-builder; description: Guide for creating high-quality MCP servers...
SYN: {name: anthropics-mcp-builder, intent_keywords: [mcp, server, fastmcp, mcp-sdk, tool-design],
      kind: prompt_skill, mechanizable: false}
FLAG: high-value for KutAI (we build MCPs); claude-api equivalent already vendored in caveman plugin

# 4. web-artifacts-builder
RAW: name: web-artifacts-builder; description: ...; body refs scripts/init-artifact.sh + bundle-artifact.sh
SYN: {name: anthropics-web-artifacts-builder, intent_keywords: [react, tailwind, shadcn, artifact, bundle],
      kind: shell_recipe?, mechanizable: needs_review, helpers: [scripts/init-artifact.sh, scripts/bundle-artifact.sh]}
FLAG: shell helpers are .sh — Windows ports needed before mechanizable=true

# 5. skill-creator
RAW: name: skill-creator; description: Create new skills...; no license field (only one)
SYN: {name: anthropics-skill-creator, intent_keywords: [skill, eval, benchmark, prompt-design],
      kind: prompt_skill, mechanizable: false}
FLAG: meta-skill (we already do this manually); medium value — duplicates our own design process
```

**Adapter recommendation.** `github_path` adapter, dead-mechanical parser: glob `skills/*/SKILL.md`, parse YAML frontmatter via `python-frontmatter`, body = rest. **T0 trusted, auto-vet.**

---

## 2. Source: `github:obra/superpowers`

**Overview.** 14 skill directories at `skills/`. MIT-vibe license. Skills are workflow disciplines — `brainstorming`, `tdd`, `systematic-debugging`, `executing-plans`, `subagent-driven-development`, etc. — i.e. process meta-skills, not format-handlers. Bodies use heavy XML tags (`<HARD-GATE>`, `<EXTREMELY-IMPORTANT>`, `<SUBAGENT-STOP>`) to inject behavioral constraints. No assets/scripts dir (pure markdown).

### 7-axis distribution (n=14)

| Axis | Distribution |
|---|---|
| Name quality | usable 14/14 (100%, gerund-style "verbing-noun") |
| Frontmatter | yaml-fm 14/14 |
| Inputs schema | none 14/14 |
| Shell blocks | prose-only 14/14 |
| Intent keywords | obvious 14/14 (description leads with "Use when...") |
| Usability | runnable-as-is 14/14 (pure prose injection; no Win compat issue) |
| Value delta | high 4/14 (brainstorming, tdd, writing-plans, subagent-driven-development); medium 8/14; low 2/14 (using-superpowers — boilerplate; using-git-worktrees — duplicates our own flow) |

### Raw → synthesized examples

```yaml
# 1. brainstorming
RAW: name: brainstorming; description: "You MUST use this before any creative work..."
SYN: {name: superpowers-brainstorming, intent_keywords: [brainstorming, design, intent, requirements, pre-implementation],
      kind: prompt_skill, mechanizable: false, trigger_density: high}
FLAG: "you MUST" trigger language — embeds skill activation logic; KutAI matcher must respect this, not normalize away

# 2. test-driven-development
RAW: description: "Use when implementing any feature or bugfix, before writing implementation code"
SYN: {name: superpowers-tdd, intent_keywords: [test-driven, tdd, test-first, red-green-refactor],
      kind: prompt_skill, mechanizable: false}
FLAG: directly applicable to KutAI agents — coder.py reflection block should reference this

# 3. systematic-debugging
SYN: {name: superpowers-systematic-debugging, intent_keywords: [debug, bug, test-failure, root-cause]}
FLAG: high-value; we already ship coulson reflection for this; risk of duplicate guidance

# 4. using-superpowers
RAW: description: "...establishes how to find and use skills..."
SYN: REJECT — boilerplate meta-skill; circular import (we don't use superpowers tool, we use yalayut)
FLAG: NEGATIVE — vendor only at user request

# 5. using-git-worktrees
SYN: skill present but body conflicts with KutAI's `.claude/worktrees/agent-<id>/` convention
FLAG: VALUE-NEGATIVE for KutAI flows; vendor disabled; revisit if we adopt obra worktree conventions
```

**Adapter recommendation.** Same `github_path` adapter as anthropics, identical glob. **T0 trusted, auto-vet.** Filter (`disabled=true`) for `using-superpowers` and `using-git-worktrees` post-import.

---

## 3. Source: cookiecutter ecosystem + `awesome-cookiecutter`

**Overview.** Cookiecutter itself is a CLI: `uvx cookiecutter gh:user/template-repo`. The `cookiecutter.json` (variables) + project dir (Jinja-interpolated files) + optional `hooks/{pre,post}_gen_project.py`. `awesome-cookiecutter` (agconti/awesome-cookiecutter) appears renamed / 404 at the documented path; alternative lists exist at `cookiecutter.readthedocs.io/en/stable/README.html` and discoverable via `topic:cookiecutter-template` GitHub search (~2,000+ repos). Couldn't enumerate 20 from a single canonical README — fragmentation problem.

Sampled 5 templates (cookiecutter-django, cookiecutter-pypackage, cookiecutter-pytest-plugin, cookiecutter-plone-starter, cookiecutter-data-science from prior project knowledge) for shape.

### 7-axis distribution (n=5 sampled + extrapolated)

| Axis | Distribution |
|---|---|
| Name quality | repo-prefixed 5/5 (`cookiecutter-<thing>` — must rewrite to drop prefix) |
| Frontmatter | none 5/5 (`cookiecutter.json` is the metadata; not YAML) |
| Inputs schema | yes 5/5 (cookiecutter.json IS the input schema) |
| Shell blocks | yes 5/5 (invocation is single `uvx cookiecutter gh:...` cmd; hooks optional) |
| Intent keywords | inferable 4/5 (need to read README first paragraph); obvious 1/5 (pypackage) |
| Usability | needs-adaptation 5/5 (uvx + Python 3.10 fine on Win; post-gen hooks may shell out to git/make) |
| Value delta | high 5/5 (these literally scaffold entire projects — the canonical mechanizable artifact) |

### Raw → synthesized examples

```yaml
# 1. cookiecutter-django
RAW: cookiecutter.json: {project_name, project_slug, author_name, use_celery, use_docker, ...}
     no frontmatter; README is the docs
SYN: {name: cc-django, intent_keywords: [django, web-app, fullstack, celery, docker, postgresql],
      kind: shell_recipe, mechanizable: true, name_original: cookiecutter-django,
      invocation: {steps: [{cmd: "uvx cookiecutter gh:cookiecutter/cookiecutter-django"}]},
      inputs_schema: <copied from cookiecutter.json>}
FLAG: post-gen hook may chmod helper scripts → Windows fail; need hook-audit gate

# 2. cookiecutter-pypackage
SYN: {name: cc-pypackage, intent_keywords: [python-package, pypi, library, packaging],
      kind: shell_recipe, mechanizable: true, invocation: "uvx cookiecutter gh:audreyfeldroy/cookiecutter-pypackage"}
FLAG: clean Win-friendly; **prime T0 seed**

# 3. cookiecutter-data-science
SYN: {name: cc-data-science, intent_keywords: [data-science, ml, jupyter, notebooks, pipeline],
      kind: shell_recipe, mechanizable: true}
FLAG: high value; KutAI shopping/research pipelines could use this scaffold

# 4. cookiecutter-pytest-plugin
SYN: medium value; niche (only when authoring pytest plugins); kind=shell_recipe
FLAG: low-traffic; T1

# 5. cookiecutter-plone-starter
SYN: REJECT — Plone is too niche for KutAI scope; T2 or disabled
FLAG: NEGATIVE for our domain
```

**Adapter recommendation.** `cookiecutter_template` adapter (separate from awesome-list parser): for each known template repo, fetch `cookiecutter.json` (mechanical, JSON), generate invocation manifest. Awesome-cookiecutter LIST page → README scraper + LLM fallback (heterogeneous categories, untrusted text). **Templates T1 (need-driven), awesome-list T2.**

---

## 4. Source: `github:punkpeye/awesome-mcp-servers`

**Overview.** Single huge README, bulleted lists grouped by category (Aggregators, Browser Automation, Cloud, Code Execution, ...). Each entry is `**[server-name](github-url)**` + one-line description + sometimes inline install hint (`npx -y server-mcp` or "via Docker"). No structured table, no auth field per row. ~1,000+ entries listed. Sampled 20 across 8 categories.

### 7-axis distribution (n=20)

| Axis | Distribution |
|---|---|
| Name quality | usable 14/20 (70%, e.g. `mcp-server-cloudflare`); repo-prefixed 4/20 (`bch1212/agentfetch-mcp` style); marketing 2/20 (`Tentra`, `nashash` — opaque names) |
| Frontmatter | none 20/20 (README bullets only; no per-server manifest at the list level) |
| Inputs schema | partial 6/20 (env vars hinted in description); none 14/20 |
| Shell blocks | mixed 11/20 (install cmd shown inline); prose-only 9/20 |
| Intent keywords | obvious 15/20; inferable 5/20 (marketing-style descriptions) |
| Usability | needs-adaptation 11/20 (npx works on Win; pip works; docker works); blocked 5/20 (Linux-only sudo apt or chmod patterns in linked README); runnable-as-is 4/20 (pure web-API servers) |
| Value delta | high 4/20 (cloudflare, s3, k8s, e2b — direct ops capability); medium 10/20 (browsers, code-exec); low 4/20 (museum, astronomy); negative 2/20 (truncated/unknown entries) |

### Raw → synthesized examples

```yaml
# 1. mcp-server-cloudflare
RAW: "**mcp-server-cloudflare** - Manage Cloudflare Workers, KV, R2, Pages, DNS (Poetry/pip setup)"
SYN: {name: mcp-cloudflare, intent_keywords: [cloudflare, workers, kv, r2, pages, dns],
      artifact_type: mcp, kind: null, mechanizable: false (process lifecycle),
      auth_env: CLOUDFLARE_API_TOKEN, install: "pip install mcp-server-cloudflare"}
FLAG: T1 — high value but auth required → vet must check env var exists before enable

# 2. e2b-sandbox-mcp
SYN: {name: mcp-e2b-sandbox, intent_keywords: [sandbox, code-execution, isolated-vm],
      artifact_type: mcp, auth_env: E2B_API_KEY, install: "npx e2b-sandbox-mcp"}
FLAG: PAID account → bumps to T2 unless user opts in

# 3. open-museum-mcp
RAW: "Federated museum collections... npx -y open-museum-mcp"
SYN: {name: mcp-open-museum, intent_keywords: [museum, art, collections, met-museum],
      artifact_type: mcp, auth_env: null, install: "npx -y open-museum-mcp"}
FLAG: T2 — niche; only enable if mission keywords match

# 4. browser-use-mcp-server
SYN: {name: mcp-browser-use, intent_keywords: [browser, playwright, automation, web-scraping],
      artifact_type: mcp, install: "docker pull ..."}
FLAG: Docker required; KutAI vecihi already covers browser scraping — DUPLICATE → skip

# 5. nashash-mcp
RAW: "Unknown — entry truncated"
SYN: REJECT
FLAG: NEGATIVE — opaque entry; vet would auto-reject for incomplete manifest
```

**Adapter recommendation.** `awesome_list_md` adapter with LLM fallback for description normalization. Two-pass: (1) bullet-parse regex extracts `name + URL + raw-desc`; (2) LLM extracts `intent_keywords + auth_env + install_cmd`. Cannot mechanize without LLM. **T1 need-driven (search by intent keyword); never auto-import the full list.**

---

## 5. Source: `github:public-apis/public-apis`

**Overview.** Highly structured: per-category markdown table with consistent columns `API | Description | Auth | HTTPS | CORS`. Dead-simple to parse. ~1,400 entries. Sampled 20 across categories.

### 7-axis distribution (n=20)

| Axis | Distribution |
|---|---|
| Name quality | usable 20/20 (100%, well-curated) |
| Frontmatter | json (table row is effectively a tuple) 20/20 |
| Inputs schema | partial 20/20 (auth column gives shape; endpoint format implicit) |
| Shell blocks | none 20/20 (no install steps — these are HTTP APIs) |
| Intent keywords | obvious 18/20; inferable 2/20 |
| Usability | runnable-as-is 11/20 (no-auth APIs work immediately); needs-adaptation 9/20 (apiKey/OAuth — env-var injection needed) |
| Value delta | medium 14/20 (eliminates ambiguity for one-shot API calls); low 4/20 (joke APIs); high 2/20 (GitHub, OpenAQ — composable in workflows) |

### Raw → synthesized examples

```yaml
# 1. CoinGecko
RAW: | CoinGecko | Cryptocurrency Price, Market Data | No | Yes | Yes |
SYN: {name: api-coingecko, intent_keywords: [crypto, bitcoin, price, market, blockchain],
      artifact_type: api, auth: none, https: true, cors: yes, base_url: api.coingecko.com}
FLAG: T0 — no auth, public, broad utility

# 2. GitHub
RAW: | GitHub | Repositories... | OAuth | Yes | Yes |
SYN: {name: api-github, artifact_type: api, auth: oauth, auth_env: GITHUB_TOKEN}
FLAG: KutAI already uses gh CLI for this — DUPLICATE path; expose as `call_api` only when CLI absent

# 3. Cat Facts
SYN: {name: api-cat-facts, intent_keywords: [cat, facts, joke, animal]}
FLAG: LOW value; T2; useful for telegram chitchat at most

# 4. VirusTotal
SYN: {name: api-virustotal, intent_keywords: [malware, scan, hash, security], auth: apikey}
FLAG: high value when missions touch security; T1 need-driven; auth required

# 5. Alpha Vantage
SYN: {name: api-alpha-vantage, intent_keywords: [stocks, finance, market-data], auth: apikey}
FLAG: free tier rate-limit-aggressive; existing free_apis.py likely duplicates — migration risk
```

**Adapter recommendation.** `public_apis_md` adapter: table-parser, fully mechanical, no LLM needed. **Per-row T0/T1 based on auth (no-auth=T0).** Big caveat: existing `src/tools/free_apis.py` overlaps heavily — design doc explicitly says migration deferred to v2. Recon confirms that's correct.

---

## 6. Source: GitHub topic search `claude-skill` / `agent-skill` / `claude-code-skills`

**Overview.** Three searches: 1,564 / 1,041 / 883 repos (heavy overlap). **Quality is bimodal.** ~10% are clean per-skill repos with proper `SKILL.md` + frontmatter (oaustegard/claude-skills, dpconde/claude-android-skill, matlab/skills, zarazhangrui/frontend-slides). ~30% are awesome-list re-publishers / installers (alirezarezvani/claude-skills "263+", sickn33 "1,400+", VoltAgent "1,000+", wshobson "153 skills"). ~60% are personal experiments, brand promos, non-English / niche, or wrappers.

Stargazer counts on topic search returned numbers that look implausibly large (50k+ on a 2026 repo); this is either GitHub's "topic affinity boost" weirdness or our unauthenticated API returning cached/stale rankings. **Do not use stars-from-topic-search as a trust signal.**

### 7-axis distribution (n=20, sampled top + random middle)

| Axis | Distribution |
|---|---|
| Name quality | usable 12/20 (60%); generic 3/20 ("skills", "agents"); repo-prefixed 4/20; marketing 1/20 |
| Frontmatter | yaml-fm 8/20 (the genuine per-skill repos); none 12/20 (awesome-lists, README-only repos) |
| Inputs schema | none 19/20; partial 1/20 |
| Shell blocks | mixed 6/20; prose-only 12/20; yes 2/20 (matlab, frontend-slides) |
| Intent keywords | obvious 10/20; inferable 6/20; vague 4/20 |
| Usability | runnable-as-is 5/20; needs-adaptation 10/20; blocked 5/20 (sh installers, Linux-paths, dotfile-mutation) |
| Value delta | high 3/20 (matlab, wshobson backend-architect, ctf-skills); medium 8/20; low 6/20; negative 3/20 (humanize-chinese, vibe-coding template, "ai-maestro" wrapper) |

### Raw → synthesized examples

```yaml
# 1. wshobson/agents/plugins/backend-development/agents/backend-architect.md
RAW: ---\nname: backend-architect\ndescription: Expert backend architect specializing in scalable API design...\nmodel: inherit\n---\n
SYN: {name: wshobson-backend-architect, intent_keywords: [backend, api, microservices, rest, graphql, grpc],
      kind: agent_config, mechanizable: false, model_hint: inherit}
FLAG: GREAT signal — `model:` field indicates Anthropic agentskills.io schema variant;
      maps cleanly to KutAI fatih_hoca model selection. T0 for the ~10 best wshobson agents.

# 2. matlab/skills/skills/matlab-live-script
RAW: SKILL.md + scripts/ + references/
SYN: {name: matlab-live-script, intent_keywords: [matlab, live-script, latex, equations],
      kind: prompt_skill, mechanizable: false}
FLAG: T1 — out of KutAI core scope but vendor-quality (matlab-official)

# 3. zarazhangrui/frontend-slides
SYN: {name: frontend-slides, intent_keywords: [slides, presentation, html, animation],
      kind: prompt_skill, mechanizable: needs_review (has scripts/deploy.sh, scripts/export-pdf.sh)}
FLAG: helper .sh scripts — Windows-fail risk; T1 with mechanizable=false until ports done

# 4. alirezarezvani/claude-skills (263+ skills)
SYN: AGGREGATOR — adapter must crawl, NOT vendor wholesale; each child skill gets own row
FLAG: meta-source; treat as `github_path` with deeper glob; vetting per-child

# 5. wuji-labs/nopua / voidborne-d/humanize-chinese
SYN: REJECT
FLAG: NEGATIVE — "humanize AI text" tools encode anti-LLM-fingerprinting tricks → adversarial to KutAI's quality grader
```

**Adapter recommendation.** `github_topic` adapter with **2-stage trust gate**: (1) auto-vet only if owner ∈ trusted_set (anthropics, obra, matlab, vercel-labs, cloudflare, stripe, expo, openai, figma, huggingface); (2) all others enqueue to `yalayut_pending`. LLM-fallback for description→intent_keywords on most. **Default T2 (require explicit approval per artifact).**

---

## Cross-source synthesis

### Top-20 seed-manifest priorities (`usability × value`, ranked)

Order = recommend ship in first yalayut batch. All T0 unless noted.

1. anthropics-pdf (extract/merge/split/forms — high freq in KutAI doc workflows)
2. anthropics-docx (Turkish e-commerce docs, contract drafts)
3. anthropics-xlsx (shopping comparison tables, mission reports)
4. anthropics-pptx (presentation gen)
5. anthropics-mcp-builder (we build MCPs; meta-leverage)
6. anthropics-skill-creator (recursive: yalayut can self-grow)
7. anthropics-claude-api (already in caveman plugin; cross-link not re-vendor)
8. superpowers-brainstorming (matches existing brainstorming flow)
9. superpowers-tdd (wire into coder agent reflection)
10. superpowers-systematic-debugging (wire into fixer/debugger agents)
11. superpowers-writing-plans (Z-zone plan authoring uses this discipline)
12. superpowers-subagent-driven-development (parallel-agent dispatch policy)
13. superpowers-verification-before-completion (close gap with our checkers)
14. wshobson-backend-architect (agent_config — fatih_hoca sys_prompt prepend)
15. wshobson-security-auditor (agent_config — security-review verb backstop)
16. wshobson-performance-engineer (agent_config)
17. wshobson-test-automator (agent_config)
18. cc-pypackage (`uvx cookiecutter gh:audreyfeldroy/cookiecutter-pypackage` — shell_recipe, T1)
19. cc-django (T1, audit post-gen hooks first)
20. cc-data-science (T1)

### Negative-value / explicit blocklist

- `using-superpowers` (boilerplate, refers to skill subsystem we're replacing)
- `using-git-worktrees` (conflicts with KutAI worktree convention at `.claude/worktrees/agent-<id>/`)
- `humanize-chinese`, `nopua`, similar "humanize AI text" tools (adversarial to quality grading)
- `mcp-browser-use`, `real-browser-mcp`, similar (KutAI vecihi already provides browser scraping; DUPLICATE)
- All cat/dog/joke APIs from public-apis (low-signal, pollute intent matcher)
- Aggregator repos that don't host artifacts (alirezarezvani, sickn33, VoltAgent indexes when used as direct sources — adapter should follow them, not vendor as single-row)
- Any repo whose top-level README is in marketing format ("✨ AwesomeKit2026 ⚡") without a SKILL.md

### Windows-incompat shell patterns to flag

Adapter `vet_checks` must reject (or mark `mechanizable=false` + emit warning) when body or invocation contains:

```
chmod +x ...
sudo apt-get / brew install / yum install
*.sh helper scripts referenced from SKILL.md (must have .ps1 sibling OR shebang-portable POSIX usable from git-bash)
docker run ... (acceptable but requires Docker Desktop running; flag, don't reject)
$HOME/ paths without %USERPROFILE% equivalent
/dev/null (use $null in PowerShell; flag)
ln -s (symlinks need admin or developer-mode on Windows; flag)
```

Sampled blockers: web-artifacts-builder `scripts/init-artifact.sh`, frontend-slides `scripts/deploy.sh`, several MCP server READMEs assuming `chmod +x install.sh && ./install.sh`.

### Name-rewrite failure modes (preserve `name_original`)

Canonical-rename rule should be `<source-slug>-<original-name>` to disambiguate. **Failures**:

- `pdf`, `docx`, `xlsx` — name is the file extension; rewriting to `anthropics-pdf` is correct, but matcher MUST also index `name_original` so user prompts "convert this pdf" still hit.
- `brainstorming`, `tdd`, `using-superpowers` — gerunds that are semantically the keyword. `superpowers-brainstorming` works because the prefix is short; but if we add a 2nd brainstorming skill we get collision. Resolution: prefix + tier suffix.
- `wshobson-backend-architect` — fine, but the underlying agent name `backend-architect` is already a generic role string KutAI may use elsewhere. `name_original` preservation lets fatih_hoca match either way.
- `matlab-live-script` — `matlab/` org prefix would yield `matlab-matlab-live-script` (dumb). Rule: drop prefix when org-name is already in skill-name root.
- Cookiecutter templates already start with `cookiecutter-` — strip prefix, canonical = `cc-django` not `cookiecutter-cookiecutter-django`.

Implementation: `manifest.name_original` is the on-disk SKILL.md `name:` value; `manifest.name` is the canonical `<source>-<name>` after normalization. Matcher embeds BOTH into the vector record.

### LLM synthesis token-cost projection

Per-artifact LLM normalization (description → intent_keywords + mechanizable detection):
- Input: ~500 tokens (SKILL.md frontmatter or README bullet block)
- Output: ~200 tokens (JSON manifest)
- ~700 tokens × Haiku/Sonnet-overhead pricing

Realistic monthly inflow:
- Trusted sources (anthropics, obra, wshobson, matlab, hand-picked vendors): 0 LLM cost — pure parser
- Daily cron: ~10 new artifacts/week from trusted (frontmatter only) + ~50 from awesome-mcp-servers / awesome-agent-skills / topic-search = **~280 LLM-touched artifacts/month**
- 280 × 700 tokens = **~200K tokens/month**, ~$0.20-$1 on Sonnet, ~$0.05 on Haiku
- Vet-time human-review LLM-assisted check (read manifest + flag prompt injection): +500K tokens/month

**Verdict: LLM cost is negligible at projected scale; no need to optimize for it. Use Sonnet for normalization, not Haiku — quality of keyword extraction matters more than cost.**

### Adapter list with confidence

| Adapter | Sources | Parser type | Confidence | Notes |
|---|---|---|---|---|
| `github_path` | anthropics/skills, obra/superpowers, wshobson/agents, matlab/skills, oaustegard/claude-skills | YAML-frontmatter, mechanical | **0.95** | Trivial — `python-frontmatter` library; only edge case is multi-file skills (assets/scripts) |
| `github_topic` | topic search results | mechanical-list + LLM-per-artifact | **0.6** | Stage 1 gets repo list; stage 2 needs per-repo SKILL.md probe; many topic repos lack canonical SKILL.md, fall back to README LLM-extract |
| `awesome_list_md` | awesome-mcp-servers, awesome-agent-skills, awesome-cookiecutter | regex-bullet + LLM-normalize | **0.55** | Per-bullet format varies wildly across categories; LLM fallback mandatory for ~30% |
| `public_apis_md` | public-apis/public-apis | table-parser, mechanical | **0.9** | Cleanest source after canonical SKILL.md; no LLM needed |
| `cookiecutter_template` | per-template repos | JSON-parser, mechanical | **0.85** | `cookiecutter.json` → inputs_schema directly; post-gen hooks need audit for Win-compat |
| `web_markdown` | generic SKILL.md URL | YAML-frontmatter, mechanical | **0.9** | Same shape as github_path; just different fetch |
| `clawhub_api` | ClawHub (future) | stub | n/a | Unenumerable today |

### Recommended T0/T1/T2 defaults per source

- **T0 (auto-vet, daily cron):** anthropics/skills, obra/superpowers, public-apis (no-auth rows only), matlab/skills, wshobson/agents (filtered to vetted plugin subset)
- **T1 (need-driven search + auto-vet trusted orgs):** awesome-mcp-servers (filter to orgs vercel/cloudflare/stripe/expo/figma/huggingface/openai), awesome-agent-skills index, cookiecutter individual templates from cookiecutter-org
- **T2 (human review required):** github_topic untrusted owners, awesome-cookiecutter bullets, public-apis auth-required rows, web_markdown unsigned URLs

---

## Synthesis impact on the yalayut design doc

The 2026-05-09 design doc holds up well; recon refines a few points:

1. **Manifest's `invocation.steps` is authored locally, not lifted.** Sampled SKILL.md files have NO `invocation` blocks. Our `shell_recipe` kind manifests will be **constructed by us** (or via LLM) from README scraping for cookiecutter / MCP install snippets. Confirm: that's already the design assumption — adapters synthesize manifests, they don't expect them on disk.

2. **`name_original` must be a first-class manifest field.** Not in current schema. Add: `name_original TEXT` to `yalayut_index`. Matcher embeds both name strings.

3. **Adapter count = 6, not 5.** Add `cookiecutter_template` as distinct from `github_path` (different parse logic — JSON not YAML frontmatter).

4. **Trust scoring needs an owner-allowlist field, not just source-id.** Awesome-list adapter discovers child artifacts under various owners; trust derives from `child.owner ∈ trusted_orgs` not `source_id = github:punkpeye/awesome-mcp-servers`.

5. **Stargazer count is unreliable for topic search.** Initial trust heuristic in the design ("anthropics/obra = 1.0, GitHub-topic match = 0.5") should NOT auto-bump trust based on stars. Owner allowlist + manual approval only.

6. **Add `disabled_imports` config.** `using-superpowers`, `using-git-worktrees`, `mcp-browser-use`, etc. — known-good sources still ship artifacts we don't want. Filter post-import, don't filter at adapter level (preserves traceability).

7. **Awesome-cookiecutter list is unreliable / 404 at documented URL.** Either rebuild the list locally via `topic:cookiecutter-template` search OR drop awesome-list ingestion for cookiecutter entirely and seed templates manually. Recommend the latter for v1.

No design-breaking issues. Recon validates the "lazy fetch + vet + plugin per artifact_type" core. Top-20 seed list above unblocks yalayut v1 implementation.
