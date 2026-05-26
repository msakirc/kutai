# Handoff — hunt: unclosed aiohttp session + Telegram DNS-failure log noise (2026-05-25)

**For:** the next session investigating two recurring Telegram-window errors. This is a
hunting guide, NOT a completed investigation — nobody has grepped the code yet. Confirm
every "candidate" below before changing anything.

---

## ✅ RESOLUTION (2026-05-25, same day)

**WS-1 — FIXED. Root cause was NOT any of the §2 candidates.**

The hunt enumerated every aiohttp `ClientSession` site (761 KutAI .py files): only **4**
are non-`async with`, and all 4 have correct close paths — `yasar_usta/telegram.py`
(wrong process: wrapper, not orchestrator), `nerd_herd/client.py` (run.py:467 singleton →
leaks ≤1× at shutdown, not 14×), `vecihi/prior_art.py` (`finally: if own_session: close()`),
`mcp_client.py` (`_SSEConnection.close()`). None matched the symptom (14 *distinct-address*
sessions, orchestrator-only, clustered around `begin_call`).

**Real owner: litellm's aiohttp transport** (`litellm/llms/custom_httpx/aiohttp_handler.py`,
litellm **issue #12443**). litellm 1.81.13 defaults `disable_aiohttp_transport=False`; the
aiohttp handler's `ClientSession` close can't run from `__del__`, so aiohttp's own finalizer
logs `Unclosed client session` at ERROR on every async-client (re)creation. litellm runs only
in the orchestrator (via HaLLederiz Kadir) → guard.jsonl entries are just piped orchestrator
stderr (microsecond-identical timestamps confirm). Not the 05-25 legacy-removal work.

**Fix:** `litellm.disable_aiohttp_transport = True` in
`packages/hallederiz_kadir/src/hallederiz_kadir/caller.py` (module init, next to
`suppress_debug_info`). Reverts to litellm's pooled **httpx** transport (closed cleanly).
Guard test: `packages/hallederiz_kadir/tests/test_aiohttp_transport.py`. **Takes effect on
next orchestrator restart** (lifecycle is user-managed via Telegram).

**WS-2 — still OPEN (optional, log-noise only).** See §3. Not done; bot self-recovers.

---

## §0 — Symptoms (verbatim from the Telegram error feed, 2026-05-25)

**A. Unclosed aiohttp session** (1:45 PM):
```
🟠 [ERROR] asyncio
Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x...>
```

**B. Telegram polling connect failure** (2:04 PM, 2:46 PM — intermittent, ~hourly):
```
🟠 [ERROR] telegram.ext.Updater  Exception happened while polling for updates.
  ...
  httpcore.ConnectError: [Errno 11001] getaddrinfo failed
  (raised inside telegram/request/_httpxrequest.py do_request -> httpx -> httpcore)
```

---

## §1 — Triage already done (don't repeat)

- **B is a DNS/network failure, not a code bug.** Errno **11001 = WSAHOST_NOT_FOUND**
  (Windows) = the host could not resolve `api.telegram.org`. `python-telegram-bot` auto-
  retries polling; it recovers when DNS/network returns. The three timestamps spread over
  an hour ⇒ intermittent host network drops (WiFi / VPN / DNS server / sleep-wake).
- **NOT caused by the 2026-05-25 legacy-removal or non_goals work** — those touched
  workflow JSON, `db.py`, and engine rescue removal. Nothing in Telegram networking or
  aiohttp. Ruled out.
- **A and B are probably linked:** an `aiohttp.ClientSession` that errored mid-request
  during the network drop never reached its close path → the "Unclosed client session"
  warning. But A may also be a standalone leak that only the network blip made visible.

So there are **two workstreams**, independent, both low-severity:
- **WS-1:** find + fix the leaked `aiohttp.ClientSession` (real resource hygiene bug).
- **WS-2 (optional):** quiet the Telegram poller's DNS-failure ERROR spam (resilience/log
  noise only — the bot already recovers).

---

## §2 — WS-1: the unclosed aiohttp ClientSession

### Key discriminator
`python-telegram-bot` uses **httpx**, NOT aiohttp. So the "Unclosed client session" is
**aiohttp** = some OTHER KutAI component. The Telegram poller is a red herring for WS-1.

### How to find the owner
1. Enumerate every aiohttp session creation:
   `rg -n "ClientSession\(|aiohttp" src packages --type py`
2. For each hit classify: created with `async with aiohttp.ClientSession() as s:`
   (auto-closed, SAFE) vs assigned `self._session = aiohttp.ClientSession()` /
   `session = aiohttp.ClientSession()` with no guaranteed `await session.close()` in a
   `finally`/`__aexit__`/shutdown hook (LEAK CANDIDATE).
3. Cross-check against the 1:45 PM timing — what fired then? (price-watch checker, a
   scheduled job, a scraper, a cloud capacity probe). Grep `logs/kutai.jsonl` around
   `2026-05-25T...` 1:45 PM local for the component active just before the warning.

### Candidate owners (VERIFY — ranked by likelihood, from codebase knowledge)
- `packages/vecihi/` — the web scraper (HTTP→TLS→Stealth→Browser). HTTP tier likely uses
  aiohttp/httpx; check its session lifecycle.
- `src/tools/free_apis.py` — free API registry calls.
- `src/integrations/` / geocoding adapters (HERE/LocationIQ/Photon) — external HTTP.
- `packages/kuleden_donen_var/` — cloud provider rate-limit probes (may poll provider
  endpoints).
- `src/app/price_watch_checker.py` — daily re-scrape; fires on a timer (timing fit?).
- `src/shopping/scrapers/` — 15 scrapers; any using a module-level/long-lived session.
- `packages/hallederiz_kadir/` — litellm calls (litellm usually owns its own client, but
  check any direct aiohttp).

### Fix pattern
- Prefer `async with aiohttp.ClientSession() as session:` per call site.
- For a long-lived shared session, ensure it's closed in the component's shutdown path
  (orchestrator shutdown / `__aexit__` / a registered cleanup) with
  `await session.close()` and, for connectors, a brief `await asyncio.sleep(0)` so the
  transport finalizes (aiohttp's documented graceful-shutdown nuance).
- Add `enable_cleanup_closed=True` on the `TCPConnector` if a keep-alive socket is the leak.

### Acceptance
- The `Unclosed client session` warning stops after the suspected component runs.
- Repro (see §4) no longer emits it.
- A targeted test that constructs the component, runs one request, and asserts the session
  is closed (or that `__aexit__`/shutdown closes it). DB-isolated; `timeout` prefix.

---

## §3 — WS-2 (optional): Telegram poller DNS-failure log noise

The poller logs a full traceback at ERROR for every transient `getaddrinfo failed`. The
bot recovers on its own, so this is **noise**, not breakage. Only do this if the spam is
annoying.

### Where
- `src/app/telegram_bot.py` — where the Updater / polling is started
  (`run_polling` / `Application.builder()...`). Search `run_polling`, `Updater`,
  `get_updates`, `error_handler`, `add_error_handler`.

### Options (pick one — don't over-build, see [[feedback_no_invented_surfaces]])
1. **Register a PTB error handler** that recognizes `httpx.ConnectError` /
   `NetworkError` and logs it at WARNING (one line, no traceback) instead of ERROR.
   Smallest change.
2. Tune the httpx request via PTB's `HTTPXRequest(connect_timeout=..., pool_timeout=...)`
   + PTB's built-in polling `bootstrap_retries` / error backoff so transient DNS blips
   don't surface as ERROR at all.
- **Do NOT** build a custom DNS-retry loop or network-state machine. PTB already retries;
  the goal is only to downgrade the log level for the known-transient case.

### Acceptance
- A simulated DNS failure (block `api.telegram.org`, §4) produces a single WARNING line,
  not a repeated ERROR traceback, and the bot resumes when DNS returns.

---

## §4 — Reproduction

- **DNS failure (B / triggers A's close path):** temporarily block name resolution to
  Telegram — add `127.0.0.1 api.telegram.org` to `C:\Windows\System32\drivers\etc\hosts`
  (or pull the network for ~30s), watch the logs, then revert. Confirms the poller's error
  path + whether any aiohttp session leaks under connect failure.
- **aiohttp leak (A) directly:** run the suspected component in isolation (e.g. a one-off
  `python -c "import asyncio; from <module> import <fn>; asyncio.run(<fn>(...))"`), let it
  exit, and watch for the "Unclosed client session" warning at interpreter shutdown
  (enable `python -X dev` or `PYTHONASYNCIODEBUG=1` for the source traceback).
- **Source traceback for the leak:** set env `PYTHONASYNCIODEBUG=1` before starting KutAI
  — aiohttp then prints WHERE the session was created, which collapses the candidate list
  in §2 immediately. **This is the highest-leverage first step.**

---

## §5 — Constraints / reminders
- Orchestrator may be LIVE — DB-touching pytest deadlocks on the WAL lock; run only
  isolated/DB-free tests, always with a `timeout` prefix. Never kill llama-server /
  wrapper / orchestrator.
- Start with §4's `PYTHONASYNCIODEBUG=1` — it turns "hunt across ~7 candidates" into "read
  the one stack trace." Do that before grepping.
- Severity is LOW for both. Neither blocks missions. Don't let it balloon.
