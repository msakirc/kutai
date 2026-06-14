# Llama-server wrong-port orphan — permanent fix

**Incident (2026-06-14):** SessionStart flagged "VRAM 97% full, no model loaded — possible leak."
Reality: llama-server (PID 33184) loaded + healthy on **:8080**, but `.env` says
`LLAMA_SERVER_PORT=8081`. nerd_herd probed :8081 → empty → false leak alarm; local
inference unreachable by the live stack.

## Root cause
1. **Silent `8080` fallback** in 6 sites (`os.environ.get("LLAMA_SERVER_PORT", "8080")`).
   A process spawned WITHOUT the env (orphan PID 24736, now dead, stale code) bound the
   8080 default while the rest of the stack used 8081 → split-brain. No error raised.
2. **DaLLaMa reconcile is lazy** — `kill_orphans()` runs only inside `start()` (first
   local inference). Active orchestrator hadn't needed a local model → stray persisted.
3. **`kill_orphans()` is name-based** (`taskkill /F /IM llama-server.exe`) — kills ALL
   llama-servers incl. a healthy one; and a stray on a *different* port is invisible to
   the port-specific `_port_in_use()` checks.

## Fixes (defense-in-depth, root-cause)
- **F1 — fail-loud port resolver.** New `src/infra/llama_endpoint.py::resolve_llama_port()`:
  `load_dotenv()` first (so any process reads the same `.env` source of truth), then read
  env; **raise** if still unset (no silent 8080). Wire into `local_model_manager.py:74`,
  `dlq_analyst.py:144`. Wrapper + nerd_herd `__main__` get inline equivalents (layer
  isolation — wrapper stays dep-light, package must not import `src`).
- **F4 — port-aware stray kill.** New `dallama.platform.kill_stray_servers(keep_port)`:
  kill every llama-server EXCEPT the one (if any) listening on `keep_port`. Preserves the
  good server (honors "never kill llama-server"), clears wrong-port strays + frees VRAM.
  Existing kill-all `kill_orphans()` stays only in the controlled in-`start()` relaunch.
- **F3 — eager reconcile.** `DaLLaMa.reconcile_strays()` → `kill_stray_servers(config.port)`;
  `LocalModelManager.reconcile_strays()` delegates; `run.py` calls it at boot. Clears strays
  even when no local model is loaded this session.
- **F5 — wrapper boot reconcile.** `kutai_wrapper._reconcile_stray_llama(port)` at startup
  (next to `_kill_stale_orchestrators`). Resolved port, port-aware kill.
- **F2 — loud detection in nerd_herd.** When inference metrics fetch fails AND a
  `llama-server.exe` exists but its configured URL is unreachable → log a precise ERROR
  ("stray llama-server on wrong port — not a VRAM leak; reconcile needed"). Converts the
  silent "no model + VRAM full" into an actionable signal.

## Order (TDD each)
A. `llama_endpoint.py` + tests → B. wire LMM/dlq → C. `kill_stray_servers` + platform tests
→ D. eager reconcile (dallama/LMM/run.py) → E. wrapper boot reconcile → F. nerd_herd port
fail-loud + stray detection. Run targeted suites per component; full dallama+infra suites at end.

## Verify / restart
Restart-gated. After merge: USER `/restart`. On boot the wrapper + orchestrator reconcile
kill the 8080 stray, DaLLaMa relaunches on 8081, nerd_herd probe matches, alarm clears.
