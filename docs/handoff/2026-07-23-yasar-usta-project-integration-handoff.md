# Handoff — Integrate a 2nd project into the Yaşar Usta hub

**Date:** 2026-07-23
**For:** a FRESH session. Self-contained. Goal = add the user's *other project*
as a 2nd block in the hub's `registry.yaml` so Yaşar Usta manages it alongside
KutAI (Kutay).
**Hub repo:** `C:\Users\sakir\Dropbox\Workspaces\yasar_usta` (own git + own
`.venv`; branch `main`; remotes `playground` real / `public` guarded).
**Prior handoff:** `docs/handoff/2026-07-22-yasar-usta-phase3-done-p4-handoff.md`

---

## 0. Where things stand (as of this handoff)

| Item | State |
|---|---|
| Phase 1–3 (relocate, state-dir, watchdog M4b, installer) | ✅ DONE + reviewed + published to `playground/main` |
| Phase 4 — Task Scheduler activation (`install_yasar_autostart.ps1`) | 🟡 being registered this session (elevated). **No reboot/logoff done.** |
| Optional code items (exit-42 restart, `/shutdown_hub`, retire ProcessGuard) | ✅ DONE + committed HUB `main` (`e843e9e`, `2631fb9`) — **NOT yet pushed to `playground`** |
| **Multi-project integration (THIS handoff)** | ⬜ NOT STARTED |

The hub is **already multi-project capable** — it builds one `TargetSupervisor`
per target and one dashboard row per project. Adding a project is a
`registry.yaml` edit + `/restart_hub`. No code change expected.

---

## 1. The task

Add the user's other project under `projects:` in
`yasar_usta/registry.yaml`, mirroring the existing `kutai` block
(`registry.yaml:7-57` — use it as the template). Then validate, commit,
`/restart_hub`, and confirm it appears in the Telegram `/status` dashboard.

**FIRST STEP: brainstorm / gather.** Do NOT guess the other project's shape.
Ask the user (see §3 checklist) before drafting the block.

---

## 2. Architecture facts you need (verified from source)

Loader: `yasar_usta/src/yasar_usta/registry.py::load_registry`. Config shape:
`config.py` (`GuardConfig` = per-target config — legacy name, kept; the old
`ProcessGuard` class was retired this session).

- **Tokens** resolved in string values: `${project_root}`, `${state_dir}`,
  `${env:VAR_NAME}`. An unset `${env:...}` raises `ValueError` (fail-loud).
  **Secrets go in as env NAMEs only** — never literal tokens in the yaml.
- **`state_dir` is per-project and auto-namespaced**: defaults to
  `%LOCALAPPDATA%/YasarUsta/<project_id>` (`registry.py:108-110`). So a new
  project gets its OWN state dir — no collision with kutai. Use `${state_dir}`
  for heartbeat/signal files; keep them OUT of Dropbox.
- **Telegram is hub-only.** One bot (`YASAR_USTA_BOT_TOKEN`), one chat
  (`TELEGRAM_ADMIN_CHAT_ID`). The new project does NOT get its own bot — it
  shows as another block + buttons in the same `/status` dashboard. Per-project
  `messages:` only customizes user-facing strings (i18n), not the transport.
- **`assert_consumer_imports` boot gate (hub.py):** for every project that sets
  `venv_python`, the hub probes that venv with
  `from yasar_usta import HeartbeatWriter, write_heartbeat`. **If the other
  project sets `venv_python`, its venv MUST have yasar_usta installed:**
  `<other_venv>/python.exe -m pip install -e C:\Users\sakir\Dropbox\Workspaces\yasar_usta`.
  Otherwise the hub refuses to boot (SystemExit, loud). If the project needs no
  heartbeat client, you *may* omit `venv_python` — but then no in-venv hook
  probe and you lose heartbeat-based hung detection (crash-exit restart still
  works via process exit + `auto_restart`).
- **Per-target fields** (defaults in `_build_target`): `command` (list, required),
  `cwd`, `env` (dict), `heartbeat_file` + `heartbeat_stale_seconds` (120) +
  `heartbeat_healthy_seconds` (90), `restart_exit_code` (42),
  `log_dir` ("logs"), `log_file`, `stop_timeout` (30), `auto_restart` (True),
  `backoff_steps` ([5,15,60,300]), `claude_*` (Claude Code remote), `sidecars`,
  `extra_processes` (labels of aux procs to show/clean, e.g. llama-server).
- **Hung detection needs the app to WRITE a heartbeat.** Set `heartbeat_file:
  "${state_dir}/<name>.heartbeat"` AND have the target process call
  `HeartbeatWriter`/`write_heartbeat` on a cadence < `heartbeat_stale_seconds`.
  No heartbeat writer ⇒ omit `heartbeat_file` (rely on crash-exit restart only),
  or the watchdog logic will think it's always hung.
- **Hook (optional):** `hook: "${project_root}/<hooks>.py"` runs lifecycle hooks
  (e.g. `on_exit` orphan-kill) as a SUBPROCESS in the project's own venv — the
  hub never imports project packages. See kutai's `yasar_hooks.py`.

### Known residual to respect
- **R2 — multi-target state_dir aliasing (latent):** all targets in ONE project
  share that project's `state_dir`, and `hb_paths` hardcodes
  `orchestrator.heartbeat`. **If the other project has 2+ targets that each
  heartbeat, they collide.** Single-target project ⇒ fine. Multi-target ⇒ give
  each target a DISTINCT `heartbeat_file` name and audit `hb_paths.py` first.

---

## 3. Gather from the user before drafting

- [ ] **project id** (short slug, e.g. `myapp`) + **display name**
- [ ] **repo root** absolute path
- [ ] **venv python** path (or does it share kutai's? or none?)
- [ ] **launch command** for each target process (exe + args)
- [ ] **cwd** + any required **env vars**
- [ ] **heartbeat?** does the app write one (via `HeartbeatWriter`)? If yes,
      confirm cadence; if no, we skip `heartbeat_file`
- [ ] **restart exit code** convention (default 42)
- [ ] **sidecars** (aux servers with health URLs) — any?
- [ ] **cleanup hook** on exit (orphan processes to kill)? 
- [ ] **single or multiple targets** (watch R2 if multiple)

---

## 4. Steps

1. **Brainstorm** (use `superpowers:brainstorming`) — confirm shape with user.
2. Draft the `projects.<id>:` block in `registry.yaml`, mirroring `kutai`
   (`registry.yaml:7-57`). Native `Write`/`Edit` only.
3. If `venv_python` set: install the heartbeat client into that venv —
   `<other_venv>/python.exe -m pip install -e ..\yasar_usta`.
4. **Validate parse BEFORE restarting** (hub venv; env from hub `.env`):
   ```
   cd C:\Users\sakir\Dropbox\Workspaces\yasar_usta
   .venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('.env'); from yasar_usta.registry import load_registry; h,p=load_registry('registry.yaml','.'); print([x.id for x in p])"
   ```
   Expect both project ids printed, no `ValueError`. (If `python-dotenv` absent,
   set the env vars inline instead.)
5. **Run the hub test suite** (safe — separate venv, no live dependency):
   `.venv\Scripts\python.exe -m pytest --timeout=120 -q` (baseline this session:
   **200 passed**). Add a `tests/test_registry*.py` case for the new block if one
   fits.
6. **Commit** at the milestone (conventional msg + Co-Authored-By). HUB `main`.
7. **User `/restart_hub`** in Telegram (registry is read at boot). NEVER launch/
   kill the hub from Claude.
8. **Verify:** `/status` shows the new project row + start/stop/restart buttons;
   its state files land under `%LOCALAPPDATA%\YasarUsta\<id>\`.

---

## 5. Rules (resume discipline)

- **NEVER** launch/kill the hub via Claude `!` — attaches to the transient shell
  → dies on close, taking managed children down. Restart only via Telegram
  `/restart_hub` or (once Phase 4 active) Task Scheduler.
- **NEVER** taskkill llama-server. **NEVER** run the full KUTAY pytest suite
  while the live app is up (zombie pytest holds SQLite locks → crash-loops
  KutAI). The HUB suite is safe (separate venv).
- Git Bash resets cwd to kutay between calls — `cd` to the hub each command.
- HUB git: `git push` (→ `playground`) is safe; `public` is guarded by a
  pre-push hook (needs `YASAR_USTA_PUSH=1`). `gh` CLI is logged out.
- Commit at each milestone (don't leave finished+tested work uncommitted).

## 6. Pointers

- Registry loader: `yasar_usta/src/yasar_usta/registry.py`
- Config shape: `yasar_usta/src/yasar_usta/config.py` (`GuardConfig`)
- Template block: `yasar_usta/registry.yaml:7-57` (kutai)
- Boot gates: `yasar_usta/src/yasar_usta/hub.py`
  (`assert_consumer_imports`, `assert_hub_credentials`, `assert_hub_log_dir_absolute`)
- Spec: `yasar_usta/docs/superpowers/specs/2026-07-21-yasar-usta-shared-hub-and-always-live-design.md`
- This session's HUB commits: `e843e9e` (exit-42 + `/shutdown_hub`), `2631fb9`
  (retire ProcessGuard) — LOCAL `main`, unpushed to `playground`.
- Memory: `project_yasar_relocation_alwayslive_merge_20260721.md`,
  `project_yasar_usta_multiproject_hub_20260720.md`.
