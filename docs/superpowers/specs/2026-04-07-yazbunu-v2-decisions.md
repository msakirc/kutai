# Yazbunu v2 — Decisions, Reasoning & Lessons Learned

**Date:** 2026-04-07
**Context:** Yazbunu was upgraded from a basic JSONL log viewer to a feature-rich tool. This document captures what was decided, why, and what went wrong along the way so future agents don't repeat mistakes.

---

## Core Design Constraint

**Featherweight above all.** The user's hard requirement: every feature must add zero or near-zero server RAM. The browser does the heavy lifting. Zero-dep core Python library, aiohttp is the only optional dependency (for the server). No build tools, no npm, no bundler. Vanilla JS only. This is a feature, not a limitation.

## Architecture: "Smart Browser, Dumb Server"

**Decision:** Server is a thin file reader. All intelligence (parsing, indexing, querying, aggregation, pattern detection) lives in the browser via Web Workers.

**Why:** 
- Server RAM stays at ~15MB (Python + aiohttp baseline) regardless of features added
- Browser engines are incredibly optimized — Web Workers, Canvas, etc. are essentially free
- At 30k lines/day (~3-5MB), client-side processing is totally viable
- Makes the server trivially simple to deploy and maintain

**Design-for-B:** All endpoints accept an optional `index=true` parameter (ignored today). When Phase B (sidecar index files) lands, the server uses `.idx` files for byte-range seeks without API changes. The `/api/stats` response includes `"indexed": false` so the client knows to compute its own aggregations.

## What Went Wrong: Virtual Scrolling

**Decision:** Implemented virtual scrolling with ~100 recycled DOM nodes and `position: absolute`.

**Why it seemed right:**
- Standard approach for large lists (React Virtualized, etc.)
- Only renders visible rows — theoretically more efficient
- Plan called for 100k+ line handling

**Why it was wrong for yazbunu:**
1. **Text selection breaks.** Absolute-positioned elements don't flow — you can't select text across rows like normal content. This was the #1 user complaint.
2. **Variable-height content breaks.** Exception traces, extra context fields — these expand rows beyond the fixed 28px height. Virtual scroll relies on predictable row heights for position math.
3. **Details hidden.** The old v1 viewer showed extra fields and exceptions inline in each row. Virtual scroll forced everything into a fixed 28px box, hiding the content users actually need to see.
4. **Premature optimization.** At 5000 loaded lines (the default), regular DOM handles it fine. The browser can render 5000 divs without breaking a sweat. Virtual scroll added complexity for a problem that doesn't exist at this scale.
5. **Overall experience degraded.** The user went from "yazbunu is fantastic" to "it looks like shit" after the v2 rewrite.

**Fix:** Replaced with flow-based DOM rendering (like v1). Document fragment built once, appended to DOM. New lines during tail are appended (not re-rendered). Text selection, variable-height rows, and inline details all work naturally.

**Lesson for future agents:** Don't add virtual scrolling unless the app actually has performance problems with regular DOM. Test with real user workflows before shipping. The "industry standard" approach isn't always the right one.

## What Went Wrong: Missing UX Defaults

**Problems reported:**
1. Page loaded showing the oldest logs first — user had to scroll through thousands of lines to see what just happened
2. Live tail was off by default — new logs didn't appear automatically
3. No time filtering — couldn't narrow to "last 5 minutes"

**Fixes applied:**
- Auto-scroll to bottom on load (show most recent first)
- Auto-start live tail on page load
- Default time filter set to "Last 5 min"
- New lines append to DOM instead of full re-render

**Lesson:** A log viewer's #1 job is showing you what's happening RIGHT NOW. Historical analysis is secondary. Defaults should optimize for the live monitoring use case.

## Server Decisions

### Streaming File Reads
**Decision:** Ring buffer (`collections.deque(maxlen=N)`) for `/api/logs`, line-by-line scan for `/api/tail`.

**Why:** The old code used `file.read_text()` which loaded entire files into memory. With 50MB rotation files, that's 50MB per request. The ring buffer iterates line-by-line and only keeps the last N lines in memory.

### Substring Matching (No JSON Parse)
**Decision:** `/api/stats` and `/api/context` extract fields via substring matching (`'"level":"ERROR"' in raw_line`) instead of `json.loads()`.

**Why:** JSON parsing is expensive at scale. Substring matching is O(n) on the raw string and avoids object allocation. For stats (counting levels, sources), you don't need the full parsed object — just the presence of a substring.

**Caveat:** This assumes the JSONL formatter produces consistent formatting (e.g., no spaces between key and colon). Since yazbunu controls the formatter, this is safe. External JSONL might break this assumption.

### Token Auth via /health
**Decision:** `/health` endpoint returns both liveness status AND the auth token/URL. This is the single integration point for process managers.

**Why:** The user's process manager (Yaşar Usta) already health-checks yazbunu. Instead of adding a separate auth handoff mechanism, we made `/health` pull double duty. Yaşar Usta calls `/health`, extracts the `url` field, sends it as a clickable link on Telegram. Zero extra endpoints, zero extra config.

**Security note:** `/health` is always unauthenticated (needed for liveness checks). The token is only sensitive if the server is exposed to the internet. For local use, this is fine. For remote use, TLS is recommended.

### WebSocket Tail
**Decision:** `WS /ws/tail` polls file size via `os.stat()` every 2s, seeks to new bytes when file grows.

**Why:** Replaces HTTP polling (which created a new request every 2s). WebSocket keeps a persistent connection. Server reads only new bytes (seek to last known position). Falls back to HTTP polling if WS fails.

## Browser Decisions

### Web Worker
**Decision:** Dedicated Worker thread for all CPU work: JSON parsing, inverted index, query engine, aggregation, pattern detection.

**Why:** Main thread stays responsive. No jank during search/filter. Worker builds an inverted index (field → value → line indices) on load, enabling instant faceted filtering. The query language is parsed and evaluated entirely in the Worker.

**Message protocol:** Typed messages via `postMessage`. Load in chunks of 5000 lines to avoid blocking the Worker thread.

### Query Language
**Decision:** Custom query language: `level:error src:api task:42 duration_ms:>1000 after:"5m ago"`.

**Why:** More expressive than regex for structured data. Users think in field-value pairs, not regex patterns. Supports boolean operators (AND/OR/NOT), numeric comparison, time filters, and parentheses. Bare words fall back to msg field matching (backwards-compatible with v1 regex search).

### Theming
**Decision:** 11 built-in themes as CSS variable maps. Custom themes via localStorage.

**Why:** CSS variable swapping is instant (no re-render). Themes are just JSON objects — trivial to export/import/share. Respects `prefers-color-scheme` for initial default.

### Flow-Based Rendering (Final Approach)
**Decision:** Build a DocumentFragment, append to DOM. New tail lines are appended (not re-rendered).

**Why:** Simple, correct, performant enough. Text selection, variable heights, inline details all work naturally. At 5000 lines, DOM performance is not a problem. When tailing, only new lines are appended — the existing DOM is untouched.

**Trade-off:** Full re-render on filter change (rebuilds all visible DOM nodes). At 5000 lines this takes ~50-100ms — imperceptible. If future scale requires it, virtual scroll can be re-added with proper variable-height support, but only after verifying the DOM is actually the bottleneck.

## Remote Access Decision

**Decision:** Token auth + `/health` endpoint as the integration point. Yaşar Usta sends the URL via Telegram.

**Why:** The user was using Tailscale for remote access — works but heavy. The simpler flow: Yaşar Usta calls `/health`, gets the auth URL, sends it as a clickable link on Telegram. User taps the link on their phone, authenticated via token in the URL, cookie set for 30 days.

**Fallback:** `/auth/qr` endpoint for non-Telegram setups. Self-contained HTML page with the URL prominently displayed and copy-on-click.

## Phase B: Index Files (Future)

**Not implemented yet.** The design is ready:
- Sidecar `.idx` file per `.jsonl` — byte offsets, timestamp index, bloom filter for context fields
- ~16 bytes per log line = ~480KB/day, ~3.5MB/week
- Server uses the index for byte-range seeks instead of streaming scans
- No API changes — existing endpoints become faster transparently
- Browser client unchanged

**When to implement:** When streaming scans become noticeably slow (probably at 100k+ lines or when users want to query across multiple days without loading everything).

## Repository Setup

**Decision:** Standalone repo at github.com/msakirc/yazbunu, separate from KutAI.

**Why:** yazbunu is designed to be adoptable by other projects. It has zero KutAI-specific code. The standalone repo has its own README (bilingual TR/EN), LICENSE (MIT), .gitignore, and pyproject.toml with proper classifiers.

**The KutAI-specific test** (`test_kutai_shim_reexports`) was removed from the standalone repo since it tests the re-export shim in `src.infra.logging_config` which doesn't exist outside KutAI.

## File Map

```
src/yazbunu/
├── __init__.py      # Logging API: get_logger(), init_logging(), context binding
├── __main__.py      # CLI entry: python -m yazbunu.server
├── formatter.py     # YazFormatter: JSONL output, context field promotion
├── auth.py          # Token generation, validation, QR page rendering
├── server.py        # aiohttp server: streaming reads, auth middleware, WS tail
└── static/
    ├── viewer.html  # Main app: flow-based rendering, search, detail panel, keybindings
    ├── worker.js    # Web Worker: parse, index, query, aggregate, patterns, related
    ├── themes.js    # 11 built-in themes + custom theme editor
    ├── sw.js        # Service worker for offline PWA
    └── manifest.json
```

## Key Pitfalls for Future Agents

1. **Don't add virtual scrolling** unless there's a measured DOM performance problem. Flow-based rendering is correct and fast enough for the expected scale.

2. **Don't add Python dependencies** to the core library. Zero-dep is a feature. aiohttp is the only optional dep (for the server).

3. **Don't add build tools.** No webpack, no npm, no bundler. Vanilla JS, shipped as static files.

4. **Test with real user workflows.** Open the viewer, look at logs, filter, tail — actually use it before shipping. The v2 virtual scroll rewrite was technically impressive but made the actual experience worse.

5. **Defaults matter more than features.** Auto-tail, scroll-to-bottom, default time filter — these simple defaults improved the UX more than any fancy feature.

6. **Port conflicts on Windows.** When testing, always check if port 9880 is already in use. Yaşar Usta manages the yazbunu lifecycle — if you start a test instance on the same port, Yaşar Usta's restart will fail with `OSError: [Errno 10048]`.

7. **The server substring matching** assumes yazbunu's own JSONL format. If supporting external JSONL, add JSON parse fallback.

8. **Bookmark keys** use `ts + src + msg[:80]`. This is fragile if two log lines have identical timestamps and messages. Acceptable for now but could cause phantom bookmarks on high-frequency identical messages.
