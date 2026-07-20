# Yaşar Usta Hub — Verify Checklist + Adding Project 2 (Handoff)

**Date:** 2026-07-20
**Branch:** `feat/yasar-usta-multiproject-hub` (NOT pushed, NOT merged — restart-gated)
**Owner action needed:** (1) Telegram-side verify, (2) merge to main, (3) optionally onboard Project 2.

---

## 1. Current state (what's live right now)

- **Cutover DONE.** The live system is now the **new Hub** (`kutai_wrapper.py` rewritten → loads `registry.yaml` → runs `Hub`). The old `ProcessGuard` wrapper (that had been running since 2026-07-17 16:33) was stopped; the new hub was launched and self-verified on the process/heartbeat/mutex side.
- **Verified without Telegram (done during cutover):**
  - New hub running; `logs/hub.lock` + `logs/hub.lk` created (the *new* lock — old `guard.lock` is now stale/dead).
  - `pre_boot` (M6 precise orphan-match) killed the 2 stale orphan orchestrators, did NOT over-kill sidecars.
  - Fresh orchestrator spawned (`src/app/run.py`), heartbeat fresh (<90s).
  - Sidecars **adopted** (yazbunu :9880, nerd_herd :9881 — old PIDs, not double-spawned).
  - **Mutex works:** a 2nd `kutai_wrapper.py` launch exited in ~1.1s (never-duplicates proven).
  - pytest zombies cleared (0). llama-server never touched.
- **Branch contents:** the multi-project hub (15 TDD tasks + 2 thorough review rounds, blockers fixed) **plus** the parallel session's **named-mutex singleton** (`singleton.py`, M1/M2) **plus** unrelated **m90 rider commits** (`95a62136`, `4cb833d7`, `e61e5406` — coulson/beckman/mr_roboto/workflows-engine, reviewed safe, not on main).
- **Tests:** 150 green in `packages/yasar_usta/tests/` (last full run).
- **Spec:** `docs/superpowers/specs/2026-07-17-yasar-usta-multiproject-hub-design.md`
- **Plan:** `docs/superpowers/plans/2026-07-17-yasar-usta-multiproject-hub.md`

---

## 2. PENDING — Telegram verify (could not be done without bot access)

Do these on your phone/Telegram against the running hub. Reply keyboard should be the **minimal 3-button** set: **Durum / Loglar / Claude Code** (spec R4). Per-target control is via the **inline dashboard buttons**, not the reply keyboard.

- [ ] **`/status`** (or 🔧 Durum) → dashboard shows **Kutay healthy** (💚 heartbeat) + **yazbunu / nerd_herd health lines** (📊/🟢/🟠/⚫) + per-project inline buttons. No `DUAL ORCHESTRATOR` warning (one instance).
- [ ] **♻️ restart** button → Yes/Cancel confirm → orchestrator restarts, heartbeat recovers, **no spurious crash card**.
- [ ] **⏹ stop** → confirm → orchestrator stops (parks). **▶️ start** → comes back (this exercises the B3 flag-based park→start path).
- [ ] **📊 sidecar restart** button (yazbunu / nerd_herd) → sidecar bounces cleanly.
- [ ] **`/logs 40`** → returns ~40 lines + a **Yazbunu Log Viewer** link (sidecar HTTP-alive).
- [ ] **`/restart_hub`** → mutex freed → hub re-execs → KutAI back up, **no orphaned orchestrator**.
- [ ] Optional **🛑 kill** on the target → note: kill currently **auto-restarts** (keep-alive), while its prompt says "press to start" — cosmetic message mismatch, not a bug (deferred).

### Process-side re-check (no Telegram needed)
```powershell
$root="C:\Users\sakir\Dropbox\Workspaces\kutay"
# heartbeat freshness
$hb=[double](Get-Content "$root\logs\orchestrator.heartbeat"); "hb age = $([math]::Round([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()-$hb,0))s"
# live processes (orchestrator path uses a FORWARD slash: match run\.py)
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match 'kutai_wrapper|run\.py|yazbunu|nerd_herd' } |
  Select ProcessId,@{n='role';e={ if($_.CommandLine -match 'kutai_wrapper'){'HUB'}elseif($_.CommandLine -match 'run\.py'){'orch'}elseif($_.CommandLine -match 'yazbunu'){'yazbunu'}else{'nerd_herd'} }} | ft -AutoSize
# mutex proof: this must EXIT in ~1s (not start a 2nd hub)
& "$root\.venv\Scripts\python.exe" kutai_wrapper.py
```

---

## 3. Merge + push (after verify passes)

```bash
git checkout main
git merge feat/yasar-usta-multiproject-hub
git push
```
This brings the hub + singleton + the m90 riders. (Reviewed: m90 riders are complete, self-contained, absent from main, touch zero yasar_usta files — safe to carry.)

---

## 4. Known gaps / cautions (deferred, non-blocking)

- **No watchdog / Task-Scheduler yet** (parallel session's M3/M4). The hub runs as a plain persistent process — **if it dies, nothing relaunches it**. This is what the M-series aliveness work is for.
- **Deferred sub-projects:** `healthcheck` kind (remote HTTP up/down) = **sub-project 2**; `job`/deploy kind (mobile build/release, CI deploy) = **sub-project 3**. Not built.
- Minor: kill-button prompt cosmetic mismatch; `wmic` deprecation (dual-orch detect + stale-orch sweep — fails soft); "any text while down → down_reply" affordance dropped; a stop request during backoff/-1 window can still restart.
- **Old-wrapper coexistence:** the legacy `ProcessGuard` does NOT share the mutex — during any future cutover, **stop the old wrapper first** (it won't be mutex-blocked).

---

## 5. Adding Project 2 (P2) — local `process` kind

**Only LOCAL long-running processes are supported today** (backend, telegram bot, local dev server). Remote-prod health-only + mobile build/release need sub-projects 2/3.

### Prereqs
1. KutAI-on-hub **verified (§2) and merged (§3)** first — don't stack P2 onto unverified/unmerged code.
2. P2's process(es) are long-running and can be launched by a command.

### The per-app contract (spec finding #3 — important)
- **Heartbeat (required for hang-detection):** the app MUST write `time.time()` to its `heartbeat_file` every ~15s. Easiest:
  ```python
  from yasar_usta import HeartbeatWriter   # or write_heartbeat
  # in the app's async startup:
  asyncio.create_task(HeartbeatWriter("logs/p2.heartbeat", interval=15).run())
  ```
  Without it: you still get crash/exit supervision + backoff, but **no hung-process kill**.
- **Graceful stop (optional):** the app should poll for a `shutdown.signal` file in its `log_dir` and exit cleanly when present. Without it, stop falls back to CTRL_BREAK→terminate (works, less graceful).
- **Paths are per-app:** heartbeat / shutdown-signal / state paths are relative to the app's CWD. Either set the target `cwd` so `logs/…` lines up (KutAI does this), or have the app read those paths from env/args. Do NOT assume the registry can relocate a path the app hardcodes.

### Steps
1. **Add a registry block** to `registry.yaml`:
   ```yaml
   projects:
     kutai:            # existing
       ...
     p2:
       name: MyApp
       hook_module: yasar_usta.projects.p2.hooks   # optional
       targets:
         - id: backend
           app_name: MyApp
           command: ["${project_root}/p2/.venv/Scripts/python.exe", "${project_root}/p2/app.py"]
           cwd: "${project_root}/p2"
           env:
             SOME_KEY: "value"
           heartbeat_file: "${project_root}/p2/logs/backend.heartbeat"
           heartbeat_stale_seconds: 120
           restart_exit_code: 42
           log_dir: "${project_root}/p2/logs"
           log_file: "${project_root}/p2/logs/backend.jsonl"
           sidecars: []        # optional; same shape as KutAI's
         # add a second target (e.g. a telegram bot) as another list item
   ```
   Notes: `${project_root}` resolves to the repo root; path fields are separator-normalized; unknown sidecar keys are ignored; `command`/`cwd`/`env` reach the subprocess (env merged onto `os.environ`). A single-target project's routing id = its project id; multi-target = `p2:backend`.

2. **Optional hook module** `packages/yasar_usta/src/yasar_usta/projects/p2/hooks.py` (mirror `projects/kutai/hooks.py`):
   ```python
   from yasar_usta.config import Messages
   MESSAGES = Messages(...)          # optional; else English defaults
   def pre_boot(project): ...        # optional one-time cleanup before P2 starts
   def on_exit(exit_code): ...       # optional cleanup after a P2 target exits
   ```
   Add empty `projects/p2/__init__.py`. If you set `MESSAGES`, wire it in `kutai_wrapper._apply_runtime_values` (it already loops all projects and applies each project's hook `MESSAGES` to that project's targets).

3. **Optional guard test** (mirror `tests/test_migration_kutai.py`) asserting the P2 block loads to the expected `GuardConfig`.

4. **Restart the hub** to pick up the new block: `/restart_hub` via Telegram (clean self-restart), or the manual cutover (stop hub → relaunch `kutai_wrapper.py`).

5. **Verify:** `/status` now shows **both** Kutay and MyApp with independent health + per-project buttons; crashing one must not affect the other (isolation is the point of per-target supervisors).

### Multi-project behavior reminders
- With 2+ projects, **bare** `/restart` `/stop` `/start` `/logs` are **rejected** with a hint (never guess a target) — use the **inline dashboard buttons** per project. `/status`, `/restart_hub`, `/logs N` (single-project only), and the reply keyboard stay Hub-global.
- One shared Telegram bot; one shared `.env` for the hub process (per-target `env` in the registry layers on top).
- The mutex still guarantees exactly one hub managing all projects.

---

## 6. Quick reference — key files
| File | Role |
|---|---|
| `kutai_wrapper.py` | entry point → `load_registry` → `_apply_runtime_values` → `Hub.run()` |
| `registry.yaml` | declarative projects/targets (add P2 here) |
| `packages/yasar_usta/src/yasar_usta/hub.py` | Hub: poller, lock, mutex gate, dashboard, routing, self-restart |
| `.../supervisor.py` | `TargetSupervisor`: one target's run() loop, intent API, `_park_until_wake` |
| `.../registry.py` | `load_registry` + `${}` resolution + path normalization |
| `.../singleton.py` | named-mutex single-instance authority (M1/M2) |
| `.../projects/kutai/hooks.py` | KutAI cleanup fns + Turkish MESSAGES (template for P2) |
| `.../status.py`, `.../commands.py` | dashboard text + keyboards |
