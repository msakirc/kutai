# Yaşar Usta — Process manager / watchdog for a long-lived app

> "Yaşar Usta" is the trusted master craftsman who keeps the workshop running:
> when something breaks he fixes it and gets the shop going again, without being
> asked. Here he is the supervisor process that keeps the app alive — restarting
> it when it crashes, killing it when it hangs, and standing in over Telegram
> when it is down.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A long-running app crashes, hangs, or gets stopped — and
when it does, there is no longer anyone inside it to tell you, restart it, or even
take a command. Yaşar Usta is the outer supervisor that never goes down with the
app. It owns the app's whole lifecycle from the outside: it launches the app,
watches a heartbeat file to notice when the app has frozen (not just exited),
restarts on crash with an escalating backoff that calms a crash-loop, and keeps
its **own** Telegram bot alive the entire time — so even when the app is dead you
can still ask "what happened?" and press a button to bring it back. The point is
that the control plane survives the thing it controls.

**What it really does.** It runs one managed subprocess and an outer loop around
`wait_for_exit()`. Each time the app exits, the loop reads the exit code and acts:
crash → notify + back off + relaunch; hang (heartbeat went stale → killed, surfaced
as exit `-1`) → short restart; clean exit `0` → wait for a manual start; restart
sentinel (`42`) → relaunch immediately with no backoff. A single-instance file lock
keeps two supervisors from fighting over the same app. A Telegram poller runs for
the supervisor's whole lifetime as the down-state control plane. Optional sidecars
(observability helpers) and an optional Claude Code remote-control launcher hang off
the same loop.

**It does NOT** know or care what the managed app *is* — it is a generic supervisor
configured entirely through `GuardConfig` (the app-specific wiring, e.g. cleaning up
GPU/inference helpers on exit, lives in the consumer's `kutai_wrapper.py`, not here).
It does NOT do work *inside* the app (no task queue, no LLM calls, no model
selection), does NOT decide *when* the app should restart beyond exit-code policy,
and does NOT write its own heartbeat — the app proves liveness by calling
`HeartbeatWriter`; the supervisor only reads it.

## Public API

Two distinct surfaces. The **supervisor side** (consumed by an entry-point script
like `kutai_wrapper.py`) constructs one `ProcessGuard` from a `GuardConfig` and runs
it. The **managed-app side** imports the heartbeat helpers so the supervisor can tell
"frozen" from "busy".

```python
# ── Supervisor side (the wrapper / entry point) ──
from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig

config = GuardConfig(
    name="Yaşar Usta", app_name="MyApp",
    command=[python, "src/app/run.py"], cwd=project_root,
    telegram_token=token, telegram_chat_id=chat_id,
    backoff_steps=[5, 15, 60, 300], backoff_reset_after=600,
    heartbeat_file="logs/app.heartbeat",
    heartbeat_stale_seconds=120, heartbeat_healthy_seconds=90,
    restart_exit_code=42,
    on_exit=lambda exit_code: ...,          # cleanup hook, called after each exit
    sidecars=[SidecarConfig(name="viewer", command=[...], health_url="...")],
    messages=Messages(started="✅ up", ...), # all user-facing strings (i18n)
)
guard = ProcessGuard(config)
await guard.run()                            # blocks for the supervisor's lifetime
guard.request_shutdown()                     # ask the loop to exit (from a signal handler)

# ── Managed-app side (inside run.py) ──
from yasar_usta import HeartbeatWriter, EXIT_RESTART, EXIT_STOP

writer = HeartbeatWriter(
    "logs/app.heartbeat", interval=15.0,
    state_path="logs/app.state.json",        # optional: what was I doing?
    state_provider=lambda: {"in_flight": [...]},
)
task = asyncio.create_task(writer.run())     # tick a fresh timestamp every 15s
# ... on a requested restart: sys.exit(EXIT_RESTART)  # 42 → relaunch, no backoff
# ... on a clean stop:        sys.exit(EXIT_STOP)     # 0  → wait for manual /start
```

Top-level exports (`__all__`): `ProcessGuard`, `GuardConfig`, `Messages`,
`SidecarConfig`, `HeartbeatWriter`, the exit-code constants `EXIT_RESTART` (42) and
`EXIT_STOP` (0), and the heartbeat helpers `write_heartbeat(*paths)`,
`write_state_snapshot(path, state)`, `read_state_snapshot(path) -> dict | None`.

### Key methods (`ProcessGuard`)

| method | purpose |
|---|---|
| `ProcessGuard(config)` | construct from a `GuardConfig` (no side effects yet) |
| `await run()` | acquire the lock, start sidecars + Telegram poller + app, then loop on exit codes forever |
| `request_shutdown()` | set the stop flag so `run()` tears down cleanly — call from a SIGINT/SIGTERM/console handler |

## Architecture

The supervisor never blocks on the app: the Telegram poller and signal-file watcher
are independent asyncio tasks, while the main loop parks on the app's exit and
dispatches on the code it gets back.

```
run()
 ├─ acquire_lock(log_dir, "guard")        single instance, stale-PID recovery
 ├─ start sidecars  (optional)            detached, PID-file + HTTP health
 ├─ Telegram poller  (lifetime task)      down-state control plane, /start /restart /stop /logs
 ├─ signal watcher   (optional task)      Claude-remote signal file + sidecar re-ensure
 └─ main loop ── await subprocess.wait_for_exit() ──┐
        exit-code dispatch:                          │
          -1   hung (heartbeat stale → killed) → notify + short restart
           0   clean exit                      → wait for manual /start
          42   restart sentinel                → relaunch now, no backoff
        other  crash                           → notify + backoff(crash#) + relaunch
                                                 └─ on_exit(code) hook runs each time
```

Inside `wait_for_exit()` is the watchdog: it waits on the process in 30 s slices, and
on each timeout checks the heartbeat file. If the last tick is older than
`heartbeat_stale_seconds`, the app is frozen — the supervisor kills it (logging the
app's last state snapshot if present) and reports it as exit `-1`.

## Key Modules

| module | role |
|---|---|
| `guard.py` | `ProcessGuard` — the outer loop, exit-code policy, Telegram command handling, self-restart |
| `subprocess_mgr.py` | one managed subprocess: spawn, pipe + log output, graceful stop, **heartbeat watchdog** |
| `backoff.py` | `BackoffTracker` — escalating delay by crash count, reset after a stable run |
| `heartbeat.py` | shared liveness contract: `HeartbeatWriter`, exit-code constants, state-snapshot helpers |
| `lock.py` | cross-platform single-instance lock with stale-PID recovery (the two-file scheme) |
| `telegram.py` | minimal Bot API client (persistent `aiohttp` session): send / edit / poll / flush |
| `sidecar.py` | `SidecarManager` — detached helper processes, PID-file + HTTP health, auto-re-ensure |
| `remote.py` | Claude Code remote-control launcher: detached session, URL polling, multi-session tracking |
| `status.py` | the Telegram status-panel text + duplicate-process (dual-wrapper) detection |
| `commands.py` | reply / inline keyboard builders, JSONL log formatting for `/logs` |
| `config.py` | `GuardConfig`, `SidecarConfig`, and `Messages` (every user-facing string, for i18n) |

## Gotchas

- **The two-file lock.** `acquire_lock()` writes `guard.lock` (PID as plain text,
  **never** OS-locked, always readable) and `guard.lk` (an `msvcrt`-locked sentinel
  whose contents are irrelevant). The sentinel is the real lock; the `.lock` file
  exists so a *second* process can read the holder's PID even while the sentinel is
  held. After a power failure the sentinel can be left locked-looking — recovery
  reads the PID from `.lock`, checks `is_pid_alive()`, and only cleans up + re-locks
  if the holder is genuinely dead. A live holder still wins.
- **Windows signal pitfall.** The managed app is spawned with `CREATE_NO_WINDOW`,
  which means it has no console — so `CTRL_BREAK_EVENT` is silently lost. Graceful
  stop therefore does **not** rely on OS signals: it writes a `shutdown.signal` file
  the app polls, tries `CTRL_BREAK` for 5 s anyway, then escalates to
  `TerminateProcess` and finally `kill()`. Don't "simplify" this to a plain signal —
  it will hang on Windows.
- **Hang ≠ crash.** A frozen app never returns an exit code, so a plain `wait()`
  would block forever. The watchdog inside `wait_for_exit()` is the only thing that
  notices a freeze; it depends entirely on the app actually calling `HeartbeatWriter`.
  No heartbeat file → `is_heartbeat_stale()` returns `False` → freezes are invisible.
- **Backoff resets on stability, not on success.** `maybe_reset()` zeroes the crash
  count only after the app has *run longer than* `backoff_reset_after` (default 600 s).
  A fast crash-loop keeps escalating `5 → 15 → 60 → 300`; one long healthy run clears it.
- **Self-restart is a hard process swap.** `/restart_guard` spawns a fresh detached
  supervisor, **flushes** pending Telegram updates (so the new one doesn't re-fire the
  same callback), releases the lock, and `os._exit(0)`s. It does not return into the
  loop. The Telegram poller advances its offset *before* processing a batch for the
  same reason — a crash mid-batch must not reprocess a restart command.
- **The Telegram poller is the down-state plane.** It runs for the supervisor's whole
  lifetime, independent of the app. When the app is down, unknown messages get a
  cooldown-throttled "press start" prompt rather than silence.

## Dependencies

The package is deliberately self-contained. Its only third-party runtime dependency
is **`aiohttp`** (the Telegram Bot API client and sidecar HTTP health checks);
everything else (`msvcrt` / `fcntl`, `ctypes`, `subprocess`, `signal`) is stdlib.

It has **no hard dependency on any sibling KutAI package.** It is a generic
supervisor: the app it manages, the cleanup it runs on exit, and the sidecars it
launches are all passed in via `GuardConfig` by the consumer
(`kutai_wrapper.py`) — that script wires in KutAI specifics (the orchestrator
command, GPU/inference-helper cleanup on non-clean exit, the observability
sidecars), but none of that knowledge lives in this package. The contract back to the
managed app is the one shared seam: the app must import and run `HeartbeatWriter` and
honor the `EXIT_RESTART` / `EXIT_STOP` exit codes, or the supervisor cannot tell
frozen from busy.

Env / config of note: `telegram_token` + `telegram_chat_id` (empty → Telegram simply
disabled, the loop still supervises); `heartbeat_file` (omit → no hang detection);
`restart_exit_code` (default 42); `auto_restart` (off → crash waits for manual start).

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\yasar_usta\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> "Yaşar Usta", atölyeyi ayakta tutan güvenilir baş ustadır: bir şey bozulduğunda
> tamir eder ve dükkânı kimse söylemeden yeniden çalıştırır. Burada da, yönettiği
> uygulamayı hayatta tutan denetçi süreçtir — çöktüğünde yeniden başlatır, donduğunda
> öldürür ve uygulama kapalıyken Telegram üzerinden onun yerine durur.

### Amaç

**Ne işe yarar.** Uzun süre çalışan bir uygulama çöker, donar ya da durdurulur — ve
bu olduğunda, içeride size haber verecek, onu yeniden başlatacak, hatta bir komut
alacak kimse kalmaz. Yaşar Usta, uygulamayla birlikte asla düşmeyen dıştaki
denetçidir. Uygulamanın tüm yaşam döngüsünü dışarıdan yönetir: uygulamayı başlatır,
bir heartbeat (kalp atışı) dosyasını izleyerek uygulamanın donduğunu (yalnızca
çıktığını değil) fark eder, çökmede bir crash-loop'u yatıştıran kademeli backoff ile
yeniden başlatır ve **kendi** Telegram botunu tüm süre boyunca ayakta tutar — böylece
uygulama ölü olsa bile yine de "ne oldu?" diye sorabilir ve bir butona basıp onu geri
getirebilirsiniz. Esas mesele: kontrol düzlemi, kontrol ettiği şeyden daha uzun yaşar.

**Gerçekte ne yapar.** Yönetilen tek bir alt süreci ve `wait_for_exit()` etrafında
dönen bir dış döngüyü çalıştırır. Uygulama her çıktığında döngü çıkış kodunu okur ve
buna göre davranır: çökme → bildir + backoff uygula + yeniden başlat; donma
(heartbeat eskidi → öldürüldü, çıkış `-1` olarak yüzeye gelir) → kısa yeniden başlat;
temiz çıkış `0` → elle başlatmayı bekle; yeniden-başlat işareti (`42`) → backoff'suz
hemen yeniden başlat. Tek-örnek dosya kilidi, iki denetçinin aynı uygulama için
çatışmasını önler. Bir Telegram yoklayıcısı, denetçinin tüm yaşamı boyunca
kapalı-durum kontrol düzlemi olarak çalışır. Opsiyonel sidecar'lar (gözlemlenebilirlik
yardımcıları) ve opsiyonel bir Claude Code uzaktan-kontrol başlatıcısı aynı döngüye
bağlanır.

**Yapmadıkları**: yönetilen uygulamanın *ne olduğunu* bilmez ya da umursamaz —
tamamen `GuardConfig` ile yapılandırılan genel bir denetçidir (uygulamaya özgü bağlama,
örneğin çıkışta GPU/çıkarım yardımcılarını temizleme, burada değil tüketicinin
`kutai_wrapper.py` dosyasında yaşar). Uygulamanın *içinde* iş yapmaz (görev kuyruğu yok,
LLM çağrısı yok, model seçimi yok), çıkış-kodu politikasının ötesinde uygulamanın *ne
zaman* yeniden başlayacağına karar vermez ve kendi heartbeat'ini yazmaz — uygulama,
`HeartbeatWriter` çağırarak canlılığını kanıtlar; denetçi yalnızca onu okur.

### Genel API

İki ayrı yüzey vardır. **Denetçi tarafı** (`kutai_wrapper.py` gibi bir giriş-noktası
betiği tarafından tüketilir) bir `GuardConfig`'ten tek bir `ProcessGuard` kurar ve onu
çalıştırır. **Yönetilen-uygulama tarafı** ise heartbeat yardımcılarını import eder ki
denetçi "donmuş" ile "meşgul"ü ayırt edebilsin.

```python
# ── Denetçi tarafı (wrapper / giriş noktası) ──
from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig

config = GuardConfig(
    name="Yaşar Usta", app_name="MyApp",
    command=[python, "src/app/run.py"], cwd=project_root,
    telegram_token=token, telegram_chat_id=chat_id,
    backoff_steps=[5, 15, 60, 300], backoff_reset_after=600,
    heartbeat_file="logs/app.heartbeat",
    heartbeat_stale_seconds=120, heartbeat_healthy_seconds=90,
    restart_exit_code=42,
    on_exit=lambda exit_code: ...,          # temizlik kancası, her çıkıştan sonra çağrılır
    sidecars=[SidecarConfig(name="viewer", command=[...], health_url="...")],
    messages=Messages(started="✅ up", ...), # tüm kullanıcıya görünen metinler (i18n)
)
guard = ProcessGuard(config)
await guard.run()                            # denetçinin ömrü boyunca bloklar
guard.request_shutdown()                     # döngüden çıkmasını iste (sinyal işleyiciden)

# ── Yönetilen-uygulama tarafı (run.py içinde) ──
from yasar_usta import HeartbeatWriter, EXIT_RESTART, EXIT_STOP

writer = HeartbeatWriter(
    "logs/app.heartbeat", interval=15.0,
    state_path="logs/app.state.json",        # opsiyonel: ne yapıyordum?
    state_provider=lambda: {"in_flight": [...]},
)
task = asyncio.create_task(writer.run())     # her 15 sn'de taze bir zaman damgası at
# ... istenen yeniden başlatmada: sys.exit(EXIT_RESTART)  # 42 → yeniden başlat, backoff yok
# ... temiz durdurmada:           sys.exit(EXIT_STOP)     # 0  → elle /start bekle
```

Üst düzey export'lar (`__all__`): `ProcessGuard`, `GuardConfig`, `Messages`,
`SidecarConfig`, `HeartbeatWriter`, çıkış-kodu sabitleri `EXIT_RESTART` (42) ve
`EXIT_STOP` (0) ile heartbeat yardımcıları `write_heartbeat(*paths)`,
`write_state_snapshot(path, state)`, `read_state_snapshot(path) -> dict | None`.

#### Ana metotlar (`ProcessGuard`)

| metot | görevi |
|---|---|
| `ProcessGuard(config)` | bir `GuardConfig`'ten kurar (henüz yan etki yok) |
| `await run()` | kilidi al, sidecar'ları + Telegram yoklayıcısını + uygulamayı başlat, sonra sonsuza dek çıkış kodları üzerinde döngü kur |
| `request_shutdown()` | durdurma bayrağını ayarla ki `run()` düzgünce kapansın — SIGINT/SIGTERM/konsol işleyicisinden çağır |

### Mimari

Denetçi asla uygulamada bloklanmaz: Telegram yoklayıcısı ve sinyal-dosyası izleyicisi
bağımsız asyncio görevleridir; ana döngü ise uygulamanın çıkışında park eder ve geri
gelen koda göre dağıtım yapar.

```
run()
 ├─ acquire_lock(log_dir, "guard")        tek örnek, ölü-PID kurtarma
 ├─ sidecar'ları başlat  (opsiyonel)      detached, PID-dosyası + HTTP sağlık
 ├─ Telegram yoklayıcı  (ömür boyu görev) kapalı-durum kontrol düzlemi: /start /restart /stop /logs
 ├─ sinyal izleyici     (opsiyonel görev) Claude-remote sinyal dosyası + sidecar yeniden-sağlama
 └─ ana döngü ── await subprocess.wait_for_exit() ──┐
        çıkış-kodu dağıtımı:                          │
          -1   donmuş (heartbeat eskidi → öldürüldü) → bildir + kısa yeniden başlat
           0   temiz çıkış                           → elle /start bekle
          42   yeniden-başlat işareti                → şimdi yeniden başlat, backoff yok
        diğer  çökme                                 → bildir + backoff(çökme#) + yeniden başlat
                                                      └─ on_exit(kod) kancası her seferinde çalışır
```

`wait_for_exit()` içinde watchdog yatar: süreci 30 sn'lik dilimlerle bekler ve her
zaman aşımında heartbeat dosyasını kontrol eder. Son atış `heartbeat_stale_seconds`'tan
eskiyse uygulama donmuştur — denetçi onu öldürür (varsa uygulamanın son durum anlık
görüntüsünü loglar) ve bunu çıkış `-1` olarak bildirir.

### Ana Modüller

| modül | rolü |
|---|---|
| `guard.py` | `ProcessGuard` — dış döngü, çıkış-kodu politikası, Telegram komut işleme, kendini yeniden başlatma |
| `subprocess_mgr.py` | tek yönetilen alt süreç: başlat, çıktıyı pipe'la + logla, nazik durdur, **heartbeat watchdog** |
| `backoff.py` | `BackoffTracker` — çökme sayısına göre kademeli gecikme, kararlı çalıştan sonra sıfırlama |
| `heartbeat.py` | paylaşılan canlılık sözleşmesi: `HeartbeatWriter`, çıkış-kodu sabitleri, durum-anlık-görüntüsü yardımcıları |
| `lock.py` | ölü-PID kurtarmalı, platformlar arası tek-örnek kilit (iki-dosya şeması) |
| `telegram.py` | minimal Bot API istemcisi (kalıcı `aiohttp` oturumu): gönder / düzenle / yokla / temizle |
| `sidecar.py` | `SidecarManager` — detached yardımcı süreçler, PID-dosyası + HTTP sağlık, otomatik yeniden-sağlama |
| `remote.py` | Claude Code uzaktan-kontrol başlatıcısı: detached oturum, URL yoklama, çoklu-oturum izleme |
| `status.py` | Telegram durum-paneli metni + çift-süreç (çift-wrapper) tespiti |
| `commands.py` | yanıt / inline klavye kurucuları, `/logs` için JSONL log biçimlendirme |
| `config.py` | `GuardConfig`, `SidecarConfig` ve `Messages` (i18n için kullanıcıya görünen her metin) |

### Tuzaklar

- **İki-dosya kilidi.** `acquire_lock()`, `guard.lock` (düz metin olarak PID, **asla**
  OS-kilitli değil, daima okunabilir) ve `guard.lk` (içeriği önemsiz, `msvcrt` ile
  kilitlenen bir sentinel) yazar. Asıl kilit sentineldir; `.lock` dosyası, sentinel
  tutulurken bile *ikinci* bir sürecin tutucunun PID'ini okuyabilmesi için vardır.
  Elektrik kesintisinden sonra sentinel kilitli görünür kalabilir — kurtarma,
  `.lock`'tan PID'i okur, `is_pid_alive()` ile kontrol eder ve yalnızca tutucu
  gerçekten ölüyse temizleyip yeniden kilitler. Canlı bir tutucu yine de kazanır.
- **Windows sinyal tuzağı.** Yönetilen uygulama `CREATE_NO_WINDOW` ile başlatılır; bu
  da konsolu olmadığı anlamına gelir — dolayısıyla `CTRL_BREAK_EVENT` sessizce
  kaybolur. Bu yüzden nazik durdurma OS sinyallerine **dayanmaz**: uygulamanın
  yokladığı bir `shutdown.signal` dosyası yazar, yine de 5 sn `CTRL_BREAK` dener, sonra
  `TerminateProcess`'e ve en son `kill()`'e yükseltir. Bunu düz bir sinyale
  "sadeleştirmeyin" — Windows'ta asılı kalır.
- **Donma ≠ çökme.** Donmuş bir uygulama asla çıkış kodu döndürmez, dolayısıyla düz bir
  `wait()` sonsuza dek bloklar. Bir donmayı fark eden tek şey `wait_for_exit()`
  içindeki watchdog'tur; o da tamamen uygulamanın gerçekten `HeartbeatWriter`
  çağırmasına bağlıdır. Heartbeat dosyası yoksa → `is_heartbeat_stale()` `False`
  döner → donmalar görünmez olur.
- **Backoff kararlılıkla sıfırlanır, başarıyla değil.** `maybe_reset()`, çökme sayısını
  yalnızca uygulama `backoff_reset_after`'tan (varsayılan 600 sn) *daha uzun çalıştıktan*
  sonra sıfırlar. Hızlı bir crash-loop `5 → 15 → 60 → 300` diye yükselmeye devam eder;
  tek bir uzun sağlıklı çalışma onu temizler.
- **Kendini yeniden başlatma sert bir süreç takasıdır.** `/restart_guard`, taze bir
  detached denetçi başlatır, bekleyen Telegram güncellemelerini **temizler** (ki yenisi
  aynı callback'i tekrar tetiklemesin), kilidi bırakır ve `os._exit(0)` yapar. Döngüye
  geri dönmez. Telegram yoklayıcısı, aynı nedenle bir partiyi işlemeden *önce* offset'ini
  ilerletir — parti ortasında bir çökme, bir yeniden-başlat komutunu yeniden işlememelidir.
- **Telegram yoklayıcısı kapalı-durum düzlemidir.** Uygulamadan bağımsız olarak,
  denetçinin tüm yaşamı boyunca çalışır. Uygulama kapalıyken bilinmeyen mesajlar,
  sessizlik yerine cooldown ile sınırlanmış bir "başlat'a bas" istemi alır.

### Bağımlılıklar

Paket bilinçli olarak kendi kendine yeterlidir. Tek üçüncü-parti çalışma-zamanı
bağımlılığı **`aiohttp`**'tir (Telegram Bot API istemcisi ve sidecar HTTP sağlık
kontrolleri); geri kalan her şey (`msvcrt` / `fcntl`, `ctypes`, `subprocess`,
`signal`) standart kütüphanedir.

Hiçbir kardeş KutAI paketine **sabit bağımlılığı yoktur.** Genel bir denetçidir:
yönettiği uygulama, çıkışta çalıştırdığı temizlik ve başlattığı sidecar'lar — hepsi
tüketici (`kutai_wrapper.py`) tarafından `GuardConfig` aracılığıyla verilir; o betik
KutAI'ye özgü ayrıntıları (orkestratör komutu, temiz-olmayan çıkışta GPU/çıkarım
yardımcılarının temizlenmesi, gözlemlenebilirlik sidecar'ları) bağlar ama bu bilginin
hiçbiri bu pakette yaşamaz. Yönetilen uygulamaya geri giden tek paylaşılan dikiş şudur:
uygulama, `HeartbeatWriter`'ı import edip çalıştırmalı ve `EXIT_RESTART` / `EXIT_STOP`
çıkış kodlarına uymalıdır; yoksa denetçi donmuşu meşgulden ayırt edemez.

Önemli env / config: `telegram_token` + `telegram_chat_id` (boş → Telegram yalnızca
devre dışı kalır, döngü yine de denetler); `heartbeat_file` (atla → donma tespiti yok);
`restart_exit_code` (varsayılan 42); `auto_restart` (kapalı → çökme elle başlatmayı
bekler).

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\yasar_usta\ -q
```
