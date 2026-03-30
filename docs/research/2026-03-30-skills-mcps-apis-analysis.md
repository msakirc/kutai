# Skills, MCPs, APIs & Libraries Analysis (March 2026)

> Complete analysis from the 2026-03-30 session. Covers skill system architecture,
> MCP integration, external libraries assessment, Turkish API landscape, and
> strategic decisions made.

---

## 1. Skill System — Current State

### What Exists (Working)

| Component | Status | File |
|---|---|---|
| Skills DB table | Working | `src/memory/skills.py` |
| Vector embedding (ChromaDB) | Working | Semantic collection |
| Regex + vector matching | Working (fixed) | `find_relevant_skills()` |
| Agent prompt injection | Working | `base.py` Phase 13.2 |
| Auto-extraction from tasks | Working (improved) | `orchestrator.py` |
| Success/failure tracking | Working | `record_skill_outcome()` |
| Feedback loop | Working | Orchestrator completion handler |
| 23 seed routing skills | Seeded at startup | `src/memory/seed_skills.py` |

### Fixes Applied (Session 2026-03-30)

1. **Search guard too rigid** — now accepts any data-fetching tool
   (`web_search`, `api_call`, `api_lookup`, `http_request`, `shopping_search`)
2. **No vector distance threshold** — added 0.7 cutoff, stops false matches
3. **Auto-extraction captured nothing useful** — now uses classifier metadata
   (agent_type, search_depth, shopping_sub_intent) + fixed word extraction
4. **Feedback loop missing** — agents now record injected skill names in task
   context, orchestrator tracks success/failure on completion

### Skill Categories

| Category | Count | Examples |
|---|---|---|
| API routing | 8 | currency→TCMB, weather→wttr.in, pharmacy→Nosyapi |
| Tool routing | 5 | Play Store, GitHub, PDF, shopping scrapers |
| Search strategy | 3 | sports→web_search, coding errors→StackOverflow |
| i2p workflow | 1 | competitor research→play_store+github+web_search |
| Turkish-specific | 6 | pharmacy, earthquake, fuel, gold, prayer, holidays |

### Remaining Gaps

- No skill versioning or deprecation
- No proof skills improve outcomes (needs A/B metrics)
- No skill composition (chaining skills)
- No manual skill creation via Telegram (planned for Settings menu)

---

## 2. MCP System — Current State

### Architecture

```
mcp.yaml (3 servers configured)
  ├── fetch (npx @anthropic-ai/mcp-fetch)
  ├── sequential_thinking (npx @modelcontextprotocol/server-sequential-thinking)
  └── github (npx @modelcontextprotocol/server-github)

MCPClient (src/tools/mcp_client.py)
  ├── JSON-RPC 2.0 protocol (no SDK dependency)
  ├── Stdio transport (subprocess)
  ├── SSE transport (HTTP)
  ├── Tool discovery (tools/list)
  └── TOOL_REGISTRY integration (mcp_{server}_{tool} naming)

Lazy Connection (src/tools/__init__.py)
  └── On first mcp_* tool call → parse server name → load config → connect → register
```

### Key Decision: No Auto-Connect

User explicitly requested: "no auto connects on startup. only when necessary."
All servers have `auto_connect: false`. Connection happens on-demand when an
agent first calls an `mcp_*` tool.

### What Works
- Full protocol implementation (stdio + SSE)
- Lazy on-demand connection
- Automatic tool registration into TOOL_REGISTRY
- Environment variable expansion for secrets

### What's Missing
- No reconnect on subprocess crash
- No health checks for MCP servers
- Agents don't know MCP tools exist (no capability advertising)
- No per-agent MCP tool filtering

---

## 3. External Libraries Assessment

### ClawHub / OpenClaw (13,700+ skills)

**Verdict: Prompt templates, not code. Not worth importing directly.**

- Skills are SKILL.md files with YAML frontmatter — essentially system prompts
- The "competitor-analysis" skill is just "identify competitors, analyze keywords"
- KutAI's agent prompts are already more specific and integrated

**Value as discovery source:** Category-based search can reveal APIs and tools
we don't know about. Added to `discover_new_apis()` as potential source
(not bulk scraped — category-based only).

### CrewAI Tools

**Verdict: Skip. Thin wrappers, heavy dependency (~100MB).**

- Their "tools" are Python wrappers around web search + LLM synthesis
- We already have better integration (tiered scraper, BM25, source quality)

### LangChain Toolkits

**Verdict: Skip. Same APIs, heavy framework dependency.**

- GitHubToolkit exists but requires LangChain as dependency
- We wrapped GitHub API directly in 5 lines

### Python Packages (Best ROI)

| Package | Purpose | Status |
|---|---|---|
| `google-play-scraper` | Play Store search/reviews | Installed, working |
| `PyMuPDF` | PDF text extraction | Installed, working |
| `curl_cffi` | TLS fingerprint bypass | Installed, working |
| `trafilatura` | Content extraction | Installed, working |
| `bm25s` | Relevance scoring | Installed, working |

### MCP Servers (Deferred, connect when needed)

| Server | What it adds | Priority |
|---|---|---|
| AppInsightMCP | 20 Play Store/App Store tools | Medium (we have google-play-scraper) |
| GitHub MCP | 20+ GitHub tools | Low (we have REST wrapper) |
| EnUygun MCP | Flight/bus tickets | Medium (Turkish travel) |

---

## 4. Free API Registry

### Static Registry (22 APIs)

| Category | APIs | Auth |
|---|---|---|
| Weather | wttr.in, Open-Meteo | None |
| Currency | TCMB EVDS, ExchangeRate-API, Frankfurter | TCMB needs key, others free |
| News | GNews | API key (100/day free) |
| Geo | Nominatim (OSM), OSRM routing | None |
| Time | WorldTimeAPI | None |
| Knowledge | Wikipedia EN, Wikipedia TR | None |
| Translation | LibreTranslate | None |
| Network | ipapi | None |
| Fun | JokeAPI | None |
| Health | Nosyapi Pharmacy | API key (100/day free) |
| Earthquake | Kandilli Observatory | None |
| Fuel | Turkey Fuel Prices (CollectAPI) | API key |
| Religion | Diyanet Prayer Times | None |
| Calendar | Turkey Holidays (Nager.at) | None |
| Currency | Gold Price Turkey (CollectAPI) | API key |
| Finance | BIST Stock Data (CollectAPI) | API key |
| Travel | EnUygun (MCP endpoint) | None |

### Auto-Growth Sources

| Source | Method | APIs Found |
|---|---|---|
| public-apis GitHub | Parse README markdown | 1400+ |
| free-apis.github.io | Parse JSON | 200+ |
| MCP server registry | Parse GitHub README | Hundreds |
| ClawHub | Category search (not bulk) | On-demand |

### Turkish-Specific APIs (Key Gap Filled)

Previously zero Turkish API coverage. Now:
- TCMB (official exchange rates)
- Kandilli (earthquake data)
- Nosyapi (pharmacy on duty)
- CollectAPI (fuel, gold, BIST)
- Diyanet (prayer times)
- Nager.at (Turkish holidays)
- EnUygun (travel tickets via MCP)

---

## 5. Privacy & Security Analysis

### Location Data

| Decision | Rationale |
|---|---|
| User coords in `.env` + user_preferences DB | Never committed to git, never in logs |
| OSRM for routing | OpenStreetMap-based, self-hostable, no Google tracking |
| Nominatim for geocoding | Only used for pharmacy addresses, not user location |
| Telegram GPS sharing | Best for exact location, user-initiated |
| District-level storage | Sufficient for pharmacy/weather, no exact address needed |

### API Security

| Risk | Mitigation |
|---|---|
| Phishing endpoints in discovery | HTTPS-only rule, domain validation |
| Data harvesting APIs | `verified` column — unverified APIs not shown to agents by default |
| Credential forwarding | Never send stored auth tokens to unverified APIs |
| Imported skills with prompt injection | Prompt injection scan on import |

### Data Flow

```
User query → Classifier (local) → Agent (local LLM)
  → Skill matching (local DB + ChromaDB)
  → Tool call:
    ├── API call (external — only query data sent, no user identity)
    ├── Web search (external — query + IP visible to DuckDuckGo/Brave/Google)
    └── Scraper (external — page fetch, IP visible to target site)
  → Result back to agent → Telegram response
```

No user data is ever uploaded. All discovery/import is pull-only.

---

## 6. Strategic Decisions Made

| Decision | Rationale |
|---|---|
| Don't import ClawHub skills | Prompt templates, not code — no value over our own |
| Mine ClawHub as API discovery source | The metadata reveals useful APIs we don't know about |
| Skip CrewAI/LangChain dependencies | Too heavy, we can call same APIs directly |
| Python packages > MCP for now | Faster integration, fewer moving parts |
| MCP on-demand only | No startup cost, connect when first needed |
| Seed routing skills manually | Better quality than auto-extraction for critical paths |
| Fix skill quality before adding more | Foundation must work before scaling |
| Turkish APIs are a priority | No framework covers this — it's our moat |
| Privacy-safe location only | OSRM/Nominatim, no Google, user controls sharing |

---

## 7. Implementation Status (All Done)

### Phase 1: Fix Foundations
- [x] Search guard accepts any data-fetching tool
- [x] Vector distance threshold 0.7
- [x] Auto-extraction uses classifier metadata
- [x] TCMB added to API registry

### Phase 2: Concrete Tools
- [x] Play Store tool (google-play-scraper)
- [x] GitHub search tool (REST API)
- [x] PDF extraction tool (PyMuPDF)

### Phase 3: Skill Routing
- [x] 23 seed routing skills
- [x] Feedback loop (injected skills tracked)
- [x] Idempotent seeding at startup

### Phase 4: Growth
- [x] 9 Turkish APIs in registry
- [x] 8 new routing skills for Turkish services
- [x] MCP registry as discovery source
- [x] Pharmacy on duty tool with distance calculation

---

## 8. Next Steps (Not Yet Done)

1. **Telegram menu redesign** — plan saved at `docs/superpowers/plans/2026-03-31-telegram-menu-redesign.md`
2. **Location setup flow** — GPS sharing + geocoding + user preferences
3. **Price watch scheduler** — daily at noon, decided but not implemented
4. **Discount watcher watchers** — monitors for the monitors
5. **Wrapper heartbeat** — orchestrator writes heartbeat file, wrapper detects hangs
6. **i2p rename** — idea_to_product → i2p
7. **Skill metrics** — A/B measurement of skill injection impact
8. **EnUygun MCP connection** — for flight/bus ticket search

---

## 9. Competitive Context

Full analysis at `docs/research/2026-03-30-competitive-analysis.md`.

**KutAI's unique advantages:**
- Local GPU management (swap budgets, affinity scheduling) — 8/10, leads the field
- Turkish shopping intelligence (15 scrapers) — 7/10, zero competition
- Self-improving skills — 6/10, ahead of most frameworks

**Strategy:** Don't compete on breadth (can't match LangChain's 100K stars).
Compete on depth in chosen domains and local LLM management.
