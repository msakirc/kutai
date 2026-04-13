# DaLLaMa

**Async Python manager for llama-server. Load models, swap safely, recover from crashes.**

*One dependency (httpx). Circuit breaker. Inference drain. Windows Job Objects. No queue, no routing, no opinions.*

[English](#english) | [Turkce](#turkce)

---

<a id="english"></a>

## What is DaLLaMa?

DaLLaMa manages a llama-server process so you don't have to. You tell it which model to load, it handles the rest: starting the server, building the right command line, swapping models safely, recovering from crashes, and unloading when idle.

```python
from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig

dallama = DaLLaMa(DaLLaMaConfig(
    llama_server_path="/usr/bin/llama-server",
    port=8080,
))
await dallama.start()

async with dallama.infer(ServerConfig(
    model_path="/models/qwen3-30b-q4.gguf",
    model_name="qwen3-30b",
    context_length=16384,
    thinking=True,
)) as session:
    # session.url = "http://127.0.0.1:8080"
    # llama-server is running, model is loaded, health check passed
    response = await your_http_client.post(f"{session.url}/v1/chat/completions", ...)
```

DaLLaMa swaps models automatically. If the model changes between calls, it drains in-flight requests, stops the old server, and starts the new one. If the same model is already loaded, it skips straight to inference.

## Why DaLLaMa?

| | DaLLaMa | llama-server alone | llama-swap (Go) | Ollama SDK |
|---|---|---|---|---|
| **Language** | Python (async) | CLI | Go binary | Python HTTP client |
| **Model swap** | Drain + stop + start | Kill & pray, or router mode (buggy) | Proxy with TTL | Automatic (hidden) |
| **Crash recovery** | Watchdog with circuit breaker | Manual restart | None | Daemon-level |
| **Idle unload** | Timer-based, configurable | `--sleep-idle-seconds` (leaks VRAM in router mode) | TTL-based | 5min default |
| **Per-model flags** | Full control (thinking, vision, ctx, extra) | Yes (single mode) / No (router mode, broken) | Via config | Hidden |
| **Inference drain** | Waits for in-flight requests before swap | No | Tracks count, no drain | No |
| **Windows support** | Job Objects, CTRL_BREAK, orphan cleanup | Basic | taskkill /f | Service-based |
| **Circuit breaker** | N failures -> cooldown -> auto-reset | No | No | No |
| **VRAM awareness** | Pre-load check via callback | `--fit` only | No | Hidden |

## Why not llama-server router mode?

llama-server added router mode (`--models-dir`, `--models-max`) in late 2025 with hot model swap. Three blockers prevent adoption:

- **Per-model flags broken** ([#20851](https://github.com/ggml-org/llama.cpp/issues/20851)) — all models get the same global `ctx-size` and `n-gpu-layers`. You can't run a 4B and a 32B model with different contexts.
- **Vision projector can't load dynamically** ([#20855](https://github.com/ggml-org/llama.cpp/discussions/20855)) — `--mmproj` is a blocked arg in router mode. Loading mmproj on-demand saves ~876MB VRAM.
- **Dynamic context impossible** — `POST /models/load` accepts only the model name. You can't calculate context at swap time from live VRAM readings.
- **Zombie children on crash** ([#18912](https://github.com/ggml-org/llama.cpp/issues/18912)) — crashed child processes aren't cleaned up.

DaLLaMa's swap module is designed for future migration. When these issues are fixed, the swap strategy changes from process restart to `POST /models/load` without touching your code.

## Features

### Process Lifecycle
- Start/stop llama-server with full command-line control
- Adaptive health wait: timeout scales with model file size and context length
- Graceful shutdown: CTRL_BREAK on Windows, SIGTERM on Unix, force kill on timeout
- Orphan cleanup: kills leftover llama-server processes from prior crashes at startup

### Safe Model Swapping
- **Inference drain**: waits for in-flight requests to finish before stopping the server (configurable timeout, force-drain on expiry)
- **asyncio.Lock**: only one swap at a time, concurrent requests wait
- **VRAM check**: optional callback queries free VRAM before loading — refuse if below threshold
- **Generation tracking**: orphaned inference-end calls from pre-swap requests are ignored via generation counter

### Circuit Breaker
- Counts consecutive load failures per model
- After N failures: refuse loads for a cooldown period
- Auto-resets when cooldown expires
- Different model? Not blocked — circuit breaker is per-model

### Health Watchdog
- Background task polls `/health` every 30 seconds
- **Crash detection**: process exit triggers immediate recovery swap
- **Hang detection**: 3 consecutive `/health` failures trigger stop + restart
- Skips checks during active swaps and idle unloads
- Respects circuit breaker — won't restart-loop a broken model

### Idle Unloader
- Timer-based: if no `infer()` call or `keep_alive()` within the idle window, unload
- Skips unload if inference is in-flight
- `on_ready(None, "idle_unload")` callback notifies the host

### Metrics
- Parses llama-server's Prometheus `/metrics` endpoint
- Extracts: tok/s (generation + prompt), KV cache usage, request counts
- Handles both metric name formats (`llamacpp_foo` and `llamacpp:foo`)
- Graceful degradation: returns zeros on connection failure

### Windows Safety
- **Job Objects**: `KILL_ON_JOB_CLOSE` ensures llama-server dies when the parent exits, even on crash
- **CTRL_BREAK_EVENT**: proper graceful shutdown (not `taskkill /f`)
- **CREATE_NO_WINDOW**: no console window pop-up
- **Orphan cleanup**: `taskkill /F /IM llama-server.exe` at startup

## Install

```bash
pip install dallama
```

Or from source:

```bash
pip install -e ./packages/dallama
```

## API

### DaLLaMa

```python
dallama = DaLLaMa(config: DaLLaMaConfig)

await dallama.start()    # start background tasks (watchdog, idle unloader)
await dallama.stop()     # stop server + background tasks

async with dallama.infer(config: ServerConfig) as session:
    session.url          # "http://127.0.0.1:8080"
    session.model_name   # "qwen3-30b"

dallama.keep_alive()     # reset idle timer (call between LLM calls during tool execution)
dallama.status           # ServerStatus(model_name, healthy, busy, measured_tps, context_length)
```

### DaLLaMaConfig

```python
DaLLaMaConfig(
    llama_server_path="llama-server",       # path to executable
    port=8080,                               # llama-server port
    host="127.0.0.1",                        # bind address
    idle_timeout_seconds=60.0,               # unload after this many seconds idle
    circuit_breaker_threshold=2,             # consecutive failures before cooldown
    circuit_breaker_cooldown_seconds=300.0,  # refuse loads for this long
    inference_drain_timeout_seconds=30.0,    # max wait for in-flight requests
    health_check_interval_seconds=30.0,      # watchdog poll interval
    health_fail_threshold=3,                 # consecutive /health fails before restart
    min_free_vram_mb=4096,                   # refuse load if VRAM below this
    on_ready=callback,                       # called on state changes
    get_vram_free_mb=callback,               # called before loading to check VRAM
)
```

### ServerConfig

```python
ServerConfig(
    model_path="/models/qwen3-30b-q4.gguf",  # absolute path to GGUF
    model_name="qwen3-30b",                   # human-readable name
    context_length=16384,                      # --ctx-size
    thinking=True,                             # --reasoning on/off
    vision_projector="/models/mmproj.gguf",    # --mmproj (empty = no vision)
    extra_flags=["--no-jinja"],                # appended verbatim
)
```

### Callbacks

**`on_ready(model_name: str | None, reason: str)`**

Called when DaLLaMa's state changes. Wire this to wake your task queue.

| Reason | Meaning |
|--------|---------|
| `model_loaded` | New model loaded and healthy |
| `load_failed` | Model failed to load |
| `idle_unload` | Model unloaded due to inactivity |
| `circuit_breaker_active` | Model blocked by circuit breaker |
| `circuit_breaker_reset` | Cooldown expired, model loadable again |
| `crash_recovery` | Watchdog detected crash, recovery attempted |

**`get_vram_free_mb() -> int`**

Called before loading a model. Return free VRAM in MB. If not provided, DaLLaMa skips the check and lets `--fit` handle GPU allocation.

## Architecture

```
DaLLaMa (dallama.py)
  ├── ServerProcess (server.py)   — build cmd, start/stop subprocess, poll /health
  ├── SwapManager (swap.py)       — lock, drain inflight, circuit breaker
  ├── HealthWatchdog (watchdog.py) — crash + hang detection, auto-restart
  ├── IdleUnloader (watchdog.py)   — timer-based model unload
  ├── MetricsParser (metrics.py)   — GET /metrics → tok/s, KV cache
  └── PlatformHelper (platform.py) — Job Objects, orphan cleanup, OS shutdown
```

Each module is <300 lines with a single responsibility. The main class composes them.

## What DaLLaMa Does NOT Do

- **Model selection** — DaLLaMa doesn't know which model is best for a task. You decide.
- **Request routing** — DaLLaMa doesn't route between local and cloud. You route.
- **GPU scheduling** — DaLLaMa doesn't queue requests. If it's busy, check `status.busy` and route elsewhere.
- **Inference calls** — DaLLaMa doesn't call litellm or make LLM requests. It manages the server; you make the calls.

## License

MIT

---

<a id="turkce"></a>

## DaLLaMa nedir?

DaLLaMa, llama-server surecini yoneten bir Python kutuphanesidir. Hangi modeli yukleyeceginizi soyleyin, gerisini o halleder: sunucuyu baslatir, dogru komut satirini olusturur, modelleri guvenli sekilde degistirir, cokmeleri otomatik onarir ve bosta kalinca kapatir.

```python
from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig

dallama = DaLLaMa(DaLLaMaConfig(
    llama_server_path="/usr/bin/llama-server",
    port=8080,
))
await dallama.start()

async with dallama.infer(ServerConfig(
    model_path="/models/qwen3-30b-q4.gguf",
    model_name="qwen3-30b",
    context_length=16384,
    thinking=True,
)) as session:
    # session.url = "http://127.0.0.1:8080"
    # llama-server calisiyor, model yuklendi, saglik kontrolu gecti
    response = await http_client.post(f"{session.url}/v1/chat/completions", ...)
```

## Neden DaLLaMa?

- **Guvenli model degisimi** — ucustaki isteklerin bitmesini bekler, sunucuyu durdurur, yenisini baslatir
- **Devre kesici** — ayni model ust uste bozulursa, soguma suresi boyunca yuklemeyi reddeder
- **Cokme kurtarma** — arka plan gorevi her 30 saniyede saglik kontrolu yapar, cokmeyi tespit ederse otomatik yeniden baslatir
- **Bosta kalinca kapatma** — belirli sure istek gelmezse modeli kaldirir, VRAM serbest kalir
- **Windows destegi** — Job Object ile yetim surecleri temizler, `CTRL_BREAK_EVENT` ile dugun kapatir
- **VRAM kontrolu** — yukleme oncesi serbest VRAM kontrolu (callback ile enjekte edilir)
- **Tek bagimlilik** — sadece httpx

## Neden llama-server router modu degil?

llama-server 2025 sonunda router modu ekledi. Ama uc kritik sorun var:

- Model basina farkli `ctx-size` / `n-gpu-layers` calismaz ([#20851](https://github.com/ggml-org/llama.cpp/issues/20851))
- Vision projector (`--mmproj`) dinamik olarak yuklenemez ([#20855](https://github.com/ggml-org/llama.cpp/discussions/20855))
- `POST /models/load` sadece model adi alir, baska parametre kabul etmez

DaLLaMa'nin swap modulu gelecekte router moduna gecis icin tasarlandi. Bu sorunlar duzeltildiginde, kodunuzu degistirmeden gecis yapabilirsiniz.

## Ozellikler

### Surec Yonetimi
- llama-server'i tam komut satiri kontroluyle baslat/durdur
- Model boyutu ve baglama uzunluguna gore uyarlanabilir saglik bekleme suresi
- Duzgun kapatma: Windows'ta CTRL_BREAK, Unix'te SIGTERM, zaman asiminda zorla oldurme
- Baslangicta onceki cokmelerdeki yetim surecleri temizleme

### Guvenli Model Degisimi
- **Istek tamamlama**: swap oncesi ucustaki isteklerin bitmesini bekler
- **asyncio.Lock**: ayni anda tek swap, diger istekler bekler
- **VRAM kontrolu**: yukleme oncesi serbest VRAM sorgusu
- **Nesil takibi**: swap oncesi isteklerden gelen gecikmeli yanitlar yok sayilir

### Devre Kesici
- Model basina ardisik yukleme hatalarini sayar
- N hatadan sonra: soguma suresi boyunca yuklemeyi reddeder
- Soguma suresi dolunca otomatik sifirlanir
- Farkli model? Engellenmez — devre kesici model bazlidir

### Saglik Nobeticisi
- Her 30 saniyede `/health` kontrol eder
- Cokme tespiti: surec olurse aninda kurtarma swap'i baslatir
- Takilma tespiti: 3 ardisik basarisiz `/health` kontrolunde durdurma + yeniden baslatma
- Aktif swap ve bosta kalma sirasinda kontrolleri atlar

### Bosta Kalinca Kapatma
- Zamanlayici tabanli: belirli sure icinde `infer()` veya `keep_alive()` cagrisi olmazsa modeli kaldirir
- Ucusta istek varsa kapatmayi atlar

### Metrikler
- llama-server'in Prometheus `/metrics` ciktisini cozumler
- Cikarir: token/s (uretim + istem), KV onbellek kullanimi, istek sayilari
- Her iki metrik adi formatini destekler (`llamacpp_foo` ve `llamacpp:foo`)
- Baglanti hatasinda sifir dondurur

## Kurulum

```bash
pip install dallama
```

## API

```python
dallama = DaLLaMa(DaLLaMaConfig(...))
await dallama.start()

# Cikarim — model yuklenmis mi kontrol eder, gerekirse degistirir
async with dallama.infer(ServerConfig(...)) as session:
    # session.url uzerinden llama-server'a istek gonder
    ...

dallama.keep_alive()   # bosta kalma zamanlayicisini sifirla
dallama.status         # ServerStatus(model_name, healthy, busy, measured_tps, context_length)
await dallama.stop()
```

## Mimari

```
DaLLaMa (dallama.py)
  ├── ServerProcess (server.py)   — komut olustur, surec baslat/durdur, /health kontrol
  ├── SwapManager (swap.py)       — kilit, istek tamamlama, devre kesici
  ├── HealthWatchdog (watchdog.py) — cokme + takilma tespiti, otomatik yeniden baslatma
  ├── IdleUnloader (watchdog.py)   — zamanlayici tabanli model kaldirma
  ├── MetricsParser (metrics.py)   — GET /metrics → token/s, KV onbellek
  └── PlatformHelper (platform.py) — Job Object, yetim temizleme, isletim sistemi kapatma
```

## DaLLaMa'nin Yapmadigi Seyler

- **Model secimi** — DaLLaMa hangi modelin en iyi oldugunu bilmez. Siz secersiniz.
- **Istek yonlendirme** — DaLLaMa yerel ve bulut arasinda yonlendirme yapmaz. Siz yonlendirirsiniz.
- **GPU kuyrugu** — DaLLaMa istek kuyruklama yapmaz. Mesgulse `status.busy` kontrol edip baska yere yonlendirin.
- **Cikarim cagrisi** — DaLLaMa LLM cagrisi yapmaz. Sunucuyu yonetir, cagriari siz yaparsiniz.

## Lisans

MIT
