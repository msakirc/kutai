# Yazbunu v2 — Feature-Packed Featherweight Log Viewer

**Date:** 2026-04-07
**Status:** Design
**Constraint:** Every feature must add zero or near-zero server RAM. Browser does the heavy lifting.

## Overview

Yazbunu is a structured JSONL logging library + web viewer. v1 provides: zero-dep Python logger with context binding, aiohttp file server, single-page viewer with level filtering, regex search, and polling-based live tail.

v2 transforms the viewer into an industry-leading log analysis tool while keeping the server featherweight (~0 extra RAM). The philosophy: push all intelligence to the browser (Web Workers, Canvas, virtual DOM), keep the server as a thin file reader, and design APIs so a future index-file layer (Phase B) can accelerate queries without architectural changes.

**Target users:** Primary — KutAI developer (single user, local). Secondary — other projects by the same author. Tertiary — open-source adopters.

**Log volume:** ~30k lines/day, ~3-5MB/day, 5x50MB rotation = ~1 week history.

**Deployment:** Local-first, remote-capable (phone/laptop access without Tailscale).

## Architecture: Approach A (Design-for-B)

### Server (aiohttp, zero extra dependencies)

The server is a thin file reader. It never caches log data in memory. All reads are streaming (line-by-line iteration, not `file.read()`).

**Existing endpoints (unchanged):**

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Serve viewer.html |
| `GET /api/files` | List JSONL files (name, size, mtime) |
| `GET /api/logs?file=&lines=1000` | Last N lines from a file |
| `GET /api/tail?file=&after=` | Lines newer than timestamp |
| `GET /static/*` | Static assets |

**New endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /api/stats?file=` | Line count, level distribution, time range, top sources. Single streaming pass with substring matching (no full JSON parse). |
| `GET /api/context?file=&field=task&value=42` | All lines where a context field matches. Streaming grep. |
| `WS /ws/tail?file=` | WebSocket tail. Server polls file with `os.stat()` every 2s, pushes new lines. Replaces HTTP polling. Falls back to existing `GET /api/tail` if WS unavailable. |
| `GET /health` | Liveness check + auth info: `{status, version, uptime_sec, token, url}`. Integration point for process managers. |
| `GET /auth/qr` | QR code page for manual mobile connection. Inline JS QR encoder (~1KB, no server dependency). Fallback. |

**Design-for-B hooks:**
- All query endpoints accept an optional `index=true` parameter (ignored today).
- `/api/stats` response includes `"indexed": false` so the client knows to compute its own aggregations.
- When Phase B lands, the server reads `.idx` sidecar files for byte-range seeks instead of streaming scans. No API changes needed.

**Remote access:**
- Token auth: server generates random token on first start, stores in `~/.yazbunu/auth.json`. Passed as `?token=` query param or `Authorization: Bearer` header.
- Optional TLS: self-signed cert auto-generated via Python `ssl` module, stored alongside auth token.
- **Health check endpoint:** `GET /health` returns `{"status": "ok", "version": "...", "uptime_sec": ..., "token": "...", "url": "https://<host>:<port>?token=<token>"}`. This is the single integration point — any process manager (Yasar Usta, systemd, etc.) calls `/health` for both liveness AND connection info.
- **Telegram flow (KutAI):** Yasar Usta already health-checks yazbunu. It switches to `GET /health`, extracts the `url` field, and sends it as a clickable link on Telegram. User taps — instantly connected, zero manual config.
- **Generic flow (other projects):** Any process manager or script can call `/health` to get the auth URL. No Telegram dependency.
- Fallback: `GET /auth/qr` endpoint for manual connection (inline JS QR encoder, no dependency).

### Browser Client

All intelligence lives here. The server sends raw JSONL strings; the browser parses, indexes, filters, aggregates, and renders.

#### Core Engine

**Virtual Scrolling:**
- ~100 recycled DOM nodes regardless of dataset size.
- Each node is a log line row. Off-screen rows are recycled with new content on scroll.
- Current approach (1 DOM node per line) is replaced entirely.

**Web Worker:**
- Dedicated Worker thread handles all CPU-intensive work:
  - JSON parsing of raw log strings
  - Inverted index construction (field -> value -> line offsets) on load
  - Regex and structured query matching
  - Aggregation computation (sparklines, histograms, distributions)
  - Pattern detection and grouping
  - Anomaly detection (z-score on rolling error rate window)
- Main thread only handles rendering and user input. Zero jank.
- Communication via `postMessage` with typed message protocol:
  - `{type: "load", lines: string[], chunk: number}` — parse and index (sent in chunks of ~5000 lines to avoid blocking Worker)
  - `{type: "filter", query: string}` — returns matching line indices
  - `{type: "aggregate", field: string, bucket: "minute"|"hour"}` — returns counts
  - `{type: "stats"}` — returns level distribution, top sources, time range
  - `{type: "patterns"}` — returns detected message patterns with counts
  - `{type: "related", lineIndex: number, windowSec: 30}` — returns correlated lines

#### Query Language

Beyond regex. Parsed in the Worker.

```
level:error src:orchestrator task:42 after:"5m ago"
level:error OR level:warning
src:orchestrator AND NOT msg:"heartbeat"
duration_ms:>1000
(task:42 OR task:43) AND agent:researcher
```

Features:
- Field-specific filters: `field:value`, `field:>number`, `field:"quoted string"`
- Time filters: `after:"5m ago"`, `before:"2026-04-07 14:00"`, `between:"14:00".."15:00"`
- Boolean operators: `AND`, `OR`, `NOT`, parentheses
- Bare words match against `msg` field (backwards-compatible with current regex search)
- Auto-complete: field names and top values populated from Worker's inverted index
- Query history stored in `localStorage`, named saved queries
- Syntax highlighting in the search input

#### Keyboard Navigation (vim grammar)

| Key | Action |
|-----|--------|
| `j` / `k` | Move down / up one line |
| `10j` | Jump 10 lines (number prefix) |
| `g` / `G` | Jump to first / last line |
| `Ctrl-d` / `Ctrl-u` | Page down / up |
| `/` | Focus search box |
| `n` / `N` | Next / previous search match |
| `Enter` | Expand selected line (detail panel) |
| `e` | Cycle level filter |
| `f` | Toggle follow mode (live tail) |
| `t` | Time jump prompt |
| `b` | Bookmark current line |
| `B` | Open bookmarks panel |
| `s` | Toggle split view |
| `d` | Toggle dashboard mode |
| `h` | Toggle heatmap mode |
| `Tab` | Cycle focus between panels |
| `:` | Command palette |
| `?` | Keyboard shortcuts overlay |

**Command palette** (`:` prefix):
- `:level error` — set level filter
- `:task 42` — filter by task
- `:theme monokai` — switch theme
- `:export csv` — export filtered results
- `:columns +duration_ms -src` — add/remove columns
- `:preset debugging` — load column preset

#### Correlation & Drill-down

- **Pill click filtering:** Click any pill (task=42, agent=researcher) to filter. Filters stack as breadcrumbs (`task=42 > agent=researcher`). Click a breadcrumb to remove that filter.
- **Related logs:** Click a log line, see all lines sharing the same task/mission/agent within a ±30s window. Displayed in the detail panel.
- **Trace view:** Select a task ID, see a vertical timeline of that task's lifecycle across all sources. Shows duration between steps.
- **Diff view:** Select two time ranges, see what changed — new error types, missing sources, volume shifts.

#### Log Line Intelligence (Worker-computed)

- **Pattern detection:** Group similar messages (e.g., "task dispatched" x847). Collapse repeated patterns into a single row with count badge. Expand on click to see individual lines.
- **Timing analysis:** When `duration_ms` is present, show inline micro-bar charts. Highlight outliers (>2 sigma from mean for that message pattern).
- **Conversation threading:** Auto-link lines sharing task+mission into collapsible threads. View a task as a chronological "conversation" across components.

#### Analytics (all client-side, Canvas/SVG rendered)

- **Sparkline bar:** Horizontal bar above log area. One pixel-column per time bucket (auto-scales: minutes for hours of data, hours for days). Color intensity = line count, red dots = errors. Click to jump to that time range.
- **Scrollbar minimap:** Vertical strip alongside scrollbar showing error/warning density markers (like VS Code's minimap).
- **Level distribution:** Donut chart.
- **Top-N sources:** Horizontal bar chart.
- **Error rate trend:** Line chart with anomaly highlighting (z-score on rolling window).
- **Lines/minute gauge:** Current throughput.
- **Heatmap mode:** Replace log list with time x source grid. Cell color = volume. Click cell to filter.

#### Multi-panel Layout

- **Single:** Default log view (current).
- **Split:** Two log panels side-by-side — compare two files, or two filters on the same file.
- **Dashboard:** Logs + analytics panels in a configurable grid.
- Drag-to-resize panels. Layout saved to `localStorage`.
- Column presets: save/load named column configurations ("debugging", "performance", "shopping").

#### Theming

- **Built-in themes:** dark (current default), light, monokai, solarized-dark, solarized-light, nord, high-contrast.
- Themes are CSS variable maps. Switching is instant (swap `<style>` block).
- **Custom theme editor:** Color pickers for each variable (bg, surface, text, levels, accent). Live preview.
- **Export/import:** Themes as JSON. Share custom themes.
- Stored in `localStorage`. Respects `prefers-color-scheme` for initial default.

#### JSON Detail Viewer

- Click any log line → side panel shows full JSON, syntax-highlighted, collapsible nested objects.
- Copy individual fields.
- "Pin" fields to show as columns in the main log view.

#### Column Customization

- Default columns: timestamp, level, source, message.
- Add any context field as a column (task, model, duration_ms, agent, etc.).
- Drag to reorder, resize, show/hide.
- Column width auto-fit or manual.
- Named presets saved to `localStorage`.

#### Bookmarks & Annotations

- `b` to bookmark a line (stored in `localStorage` by file + line timestamp).
- Add text notes to bookmarks.
- Bookmark list panel — jump between bookmarks across files.
- Export bookmarks as markdown report.

#### Notifications

- Browser `Notification` API when a new ERROR appears while tab is backgrounded.
- Configurable: which levels, which sources trigger notifications.
- Optional audio alert (subtle beep, `AudioContext` — no sound file needed).

#### Export & Sharing

- Copy filtered results as: JSONL, CSV, formatted text, markdown table.
- Share link: full filter/time/theme state encoded in URL hash.
- "Screenshot" a log range as PNG via canvas rendering.
- Download filtered subset as file.

#### Mobile & Touch

- Responsive layout: single-column on narrow screens, bottom sheet for detail view.
- Swipe right on a line to bookmark, swipe left to copy.
- Touch-friendly filter chips replace keyboard shortcuts.
- Large tap targets for level filter, file selector.

#### Accessibility

- ARIA labels on all interactive elements.
- Screen reader announces new log lines in follow mode.
- High-contrast theme built-in.
- Respects `prefers-reduced-motion` (disables sparkline animations, scroll smoothing).
- Focus management: keyboard-navigable without mouse.

#### PWA

- Service worker caches app shell (HTML, JS, CSS, icons) for offline launch. Log data is NOT cached — it's always fetched live from the server.
- Install prompt on mobile.
- Manifest already exists — extend with service worker registration.

## File Structure (after v2)

```
yazbunu/
├── pyproject.toml
└── src/yazbunu/
    ├── __init__.py          # Logging API (unchanged)
    ├── __main__.py          # CLI entry
    ├── formatter.py         # JSONL formatter (unchanged)
    ├── server.py            # aiohttp server (add WS, stats, context, auth endpoints)
    ├── auth.py              # Token generation, validation, QR rendering
    └── static/
        ├── manifest.json
        ├── sw.js            # Service worker
        ├── viewer.html      # Main app shell
        ├── worker.js        # Web Worker (parsing, indexing, query engine)
        ├── themes.js        # Theme definitions and editor
        └── icons/           # PWA icons
```

All browser code is vanilla JS — no build step, no npm, no bundler. Ship as static files. This is a feature, not a limitation: zero build tooling means zero friction for adopters.

## RAM Budget

| Component | Steady-state RAM |
|-----------|-----------------|
| Server (aiohttp) | ~15-20MB (Python + aiohttp baseline) |
| Per-request overhead | ~0 (streaming reads, no caching) |
| Auth token storage | ~1KB on disk |
| WebSocket connections | ~1KB per connected client |
| **Total server delta from v1** | **~0** |

All new features live in the browser. Browser RAM depends on how much log data is loaded (~5MB for a full day), plus Worker memory for the inverted index (~2-3MB for 30k lines). Total browser: ~20-30MB, well within any modern browser's comfort zone.

## Phase B (future, not in this implementation)

When needed, add a sidecar `.idx` file per `.jsonl`:
- Written by the formatter on log emission (or by a background indexer)
- Contains: byte offset per line, timestamp index, bloom filter for context field values
- ~16 bytes per log line = ~480KB/day, ~3.5MB/week
- Server uses the index for byte-range seeks instead of streaming scans
- No API changes — existing endpoints become faster transparently
- Browser client unchanged

## What's NOT in scope

- No additional Python dependencies (server stays aiohttp-only)
- No database (no SQLite, no Redis, no anything)
- No build tooling for the frontend (no webpack, no npm)
- No multi-user auth (single token is sufficient)
- No log ingestion from external sources (yazbunu reads its own files)
- No alerting beyond browser notifications (no email, no webhook)
