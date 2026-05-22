# Web Preview Hosting (C10 / F1) — design

**Date:** 2026-05-22
**Status:** approved (founder)
**Context:** the i2p mission produces a static HTML prototype. `emit_preview_url`
already spawns a `cloudflared` tunnel (gated on `KUTAI_PREVIEW_PROVIDER=cloudflared`
+ binary) and `kill_preview_url` tears it down, but two defects mean no founder
ever actually saw a preview:

1. **Wrong origin:** cloudflared is pointed at `file://<dir>` — cloudflared
   proxies an HTTP origin, not a file directory, so nothing is served.
2. **Wrong / empty dir:** `emit_preview_url` serves `mission_<id>/.prototype/`,
   but the **web** track (steps 5.30a/b `generate_html_prototypes`) writes HTML
   to `mission_<id>/.web/`. Only the **mobile** track (Z5 Expo export) fills
   `.prototype/`. Web missions therefore tunnel an empty dir.

Founder is remote (Telegram, AFK) so localhost-only hosting is useless — the
deliverable is a public URL.

## Decision
Two providers behind the preview surface (founder chose "both"):
- **cloudflared** — default, instant, ephemeral live URL (primary).
- **GitHub Pages** — opt-in, persistent/shareable durable URL.

## Components

### Shared — preview-root resolution
`_resolve_preview_root(workspace_path) -> str | None`
- return `<ws>/.prototype` if it contains `index.html` (mobile/Expo bundle);
- else `<ws>/.web` if it exists and is non-empty (web HTML prototypes);
- else `None` → emitter writes the existing `pending:` placeholder.
Both providers serve this resolved root. Fixes defect #2.

### Provider 1 — cloudflared (live, ephemeral) [`emit_preview_url.py`, `kill_preview_url.py`]
- Resolve root (above); if `None` → pending (unchanged behaviour).
- Spawn a **local static server**: `python -m http.server <port> --bind 127.0.0.1
  --directory <root>` on a free ephemeral port (bind 127.0.0.1, never 0.0.0.0).
  Persist its PID to `<ws>/.httpserver.pid`.
- Spawn `cloudflared tunnel --url http://127.0.0.1:<port>` (fixes defect #1).
  Persist tunnel PID to `<ws>/.tunnel.pid` (unchanged path). Capture the
  trycloudflare URL from stdout as today.
- `kill_preview_url` terminates **both** PIDs (http server + tunnel) and clears
  both pidfiles; idempotent; Windows `CREATE_NEW_PROCESS_GROUP` / CTRL_BREAK as
  the tunnel already uses.
- Persist to `preview_log` (action="emit") + `missions.preview_url` as today.
- Fail-soft `pending:` on: no root / no cloudflared binary / no URL captured /
  spawn failure (server or tunnel). On any partial failure, tear down whatever
  was spawned so no orphan server lingers.

### Provider 2 — GitHub Pages (persistent, opt-in) [new `publish_preview_pages.py` + dispatch branch]
New mechanical verb `publish_preview_pages(mission_id, workspace_path=None)`:
- Resolve root (shared helper); `None` → `{ok: False, error: "no preview root"}`.
- Ensure the mission repo exists by reusing `init_mission_github_repo`.
- Copy the resolved root into the repo, commit to a `gh-pages` branch, push.
- Enable Pages via `gh api -X POST repos/<owner>/<repo>/pages` (source = gh-pages
  branch, path `/`). Idempotent: tolerate "already enabled".
- Return `{ok, url: "https://<owner>.github.io/<repo>/", pending: False}`.
- Persist to `preview_log` (action="pages") + `missions.preview_url`.
- Fail-soft when `gh` is absent / unauthenticated / no repo → `{ok: True,
  pending: True, reason}` so a mission never DLQs on a missing durable host.
- Founder-triggered only (fits the existing public-repo "lock").

### Telegram surface [`telegram_bot.py`]
- Keep the existing `preview:share:<mid>` button → `emit_preview_url` (cloudflared).
- Add a `[🌐 Durable link]` inline button (callback `preview:pages:<mid>`) on the
  preview-ready message → enqueue `publish_preview_pages` via Beckman (mechanical)
  → reply with the Pages URL (or a pending notice). Mirror the existing
  `preview:share:` handler shape. Keep callback_data ≤64 bytes.

## Testing (host-path, isolated; mock all subprocess/git/gh; no real network/DB)
- root resolution: `.prototype` (with index.html) preferred; `.web` fallback;
  neither → `None`.
- cloudflared: assert a local http.server is spawned over the resolved root AND
  the tunnel origin is `http://127.0.0.1:<port>` (NOT `file://`); both PIDs
  persisted; URL captured from mocked stdout.
- teardown: `kill_preview_url` kills BOTH pids and clears both pidfiles.
- pages: mock `init_mission_github_repo` + git + `gh`; assert gh-pages push +
  Pages-enable call + returned URL shape; fail-soft when `gh` missing.
- pending paths: no root / no binary / no URL → `pending:` placeholder, no orphan
  processes.
- existing `test_emit_preview_url.py` / `test_kill_preview_url.py` updated to the
  two-PID model.

## Non-goals
- No backend/dynamic hosting (prototype is static).
- No new heavy deps (cloudflared binary + `gh` are optional; absence → fail-soft).
- No auto-publish to Pages on every mission (founder-triggered to respect the
  public-repo lock).

## Build order (subagent-driven)
A. shared root resolver + cloudflared local-server fix + two-PID teardown.
B. `publish_preview_pages` verb + dispatch branch.
C. Telegram `[🌐 Durable link]` button + `preview:pages:` callback.
(C depends on B; A independent.)
