# DaLLaMa — llama-server lifecycle keeper

> A standalone async wrapper that owns a single llama-server process: it loads
> models, swaps them safely, and keeps the server alive so the caller never has
> to babysit a subprocess.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** Driving llama.cpp's `llama-server` from Python is a swamp
of subprocess plumbing: building the right command line, polling `/health` long
enough for weights to load, draining in-flight requests before you kill the
process, cleaning up orphans after a crash, and stopping the thing reliably on
Windows (where `CTRL_BREAK` is a lie llama-server never traps). DaLLaMa absorbs
all of that. The caller says "I want *this* model loaded" and gets back a healthy
URL — or a clean failure — and nothing else to think about.

**What it really does.** It manages exactly one server process at a time. When the
requested model (or its thinking / vision settings) differs from what's loaded, it
performs a *swap*: drain in-flight inference → stop the old process → wait for CUDA
to release VRAM → start the new process → wait until `/health` passes. Around that
sit four guards: a per-model circuit breaker (stop restart-looping a broken model),
a health watchdog (auto-recover crashes and hangs), an idle unloader (free VRAM
when nobody's calling), and an optional VRAM pre-check. It also parses
llama-server's Prometheus `/metrics` for a live tokens/sec reading.

**It does NOT** pick which model to load (the caller decides), route between local
and cloud, queue or schedule requests (it manages one server — if it's busy, check
`status.busy` and route elsewhere), or make any LLM/inference call itself (the
caller POSTs to the URL DaLLaMa hands back). It is a process manager, not a client.
It has **no KutAI dependencies** — its only third-party import is `httpx`.

## Public API

Construct once, `start()` the background tasks, then drive every inference through
the `infer()` context manager. Everything below is exported from the top level.

```python
from dallama import (
    DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus,
    InferenceSession, DaLLaMaLoadError,
)

dallama = DaLLaMa(DaLLaMaConfig(
    llama_server_path=r"C:\llama\llama-server.exe",
    port=8080,
    on_ready=on_ready,            # (model_name: str | None, reason: str) -> None
    get_vram_free_mb=probe_vram,  # () -> int  (optional VRAM gate)
))
await dallama.start()             # kills orphans, launches watchdog + idle unloader

# Acquire the model for the duration of a call. Swaps only if the model,
# `thinking`, or `vision_projector` changed since the last load.
async with dallama.infer(ServerConfig(
    model_path=r"C:\models\qwen3-30b-q4.gguf",
    model_name="qwen3-30b",
    context_length=16384,
    thinking=True,
)) as session:                    # -> InferenceSession(url, model_name)
    # session.url == "http://127.0.0.1:8080" — server up, model loaded, /health green
    resp = await your_http_client.post(f"{session.url}/v1/chat/completions", ...)
    # if the swap failed, infer() raised DaLLaMaLoadError before entering the block

dallama.keep_alive()              # reset the idle timer between calls in a long tool loop
st = dallama.status               # -> ServerStatus(model_name, healthy, busy, measured_tps, context_length)
await dallama.stop()              # stop background tasks + the server
```

`__all__`: `DaLLaMa`, `DaLLaMaConfig`, `ServerConfig`, `ServerStatus`,
`InferenceSession`, `DaLLaMaLoadError`. (The metrics parser is internal — not
exported.)

### Config dataclasses

```python
DaLLaMaConfig(                      # engine settings — set once
    llama_server_path="llama-server",
    port=8080, host="127.0.0.1",
    idle_timeout_seconds=60.0,
    circuit_breaker_threshold=2,
    circuit_breaker_cooldown_seconds=300.0,
    inference_drain_timeout_seconds=30.0,
    health_check_interval_seconds=30.0,
    health_fail_threshold=3,
    min_free_vram_mb=4096,
    on_ready=None,                  # state-change callback (see reasons below)
    get_vram_free_mb=None,          # () -> int; if None, --fit handles GPU sizing
)

ServerConfig(                       # per-model job description
    model_path,                     # absolute path to the GGUF
    model_name,                     # human-readable name
    context_length,                 # --ctx-size
    thinking=False,                 # --reasoning on (only with --jinja)
    vision_projector="",            # --mmproj <path>; empty = no vision
    extra_flags=[],                 # appended to the command line verbatim
    fallback_gpu_layers=0,          # OOM-retry --n-gpu-layers value (0 = no retry)
    required_vram_mb=0,             # strip a pinned --n-gpu-layers if free VRAM < this
)

ServerStatus(model_name, healthy, busy, measured_tps, context_length)
InferenceSession(url, model_name)
DaLLaMaLoadError(model_name, reason="")   # .model_name, .reason
```

### `on_ready` reasons

The callback fires on state changes — wire it to wake your task queue. The four
reasons actually emitted:

| reason | when |
|---|---|
| `model_loaded` | a new model is loaded and healthy |
| `load_failed` | the model failed to become healthy |
| `idle_unload` | the model was unloaded after the idle window elapsed |
| `circuit_breaker_active` | the load was refused — breaker is in cooldown for this model |

(Crash recovery and hang restarts re-enter the swap path and surface as
`model_loaded` / `load_failed`; there is no separate `crash_recovery` reason.)

## Architecture

`DaLLaMa` is a thin facade composing six single-responsibility modules:

```
DaLLaMa (dallama.py)        — facade: infer() ctx-mgr, status, start/stop, tps refresh
  ├─ ServerProcess (server.py)   — build_cmd, start/stop subprocess, poll /health, stderr tail
  ├─ SwapManager (swap.py)       — lock · drain · circuit breaker · VRAM gate · OOM fallback
  ├─ HealthWatchdog (watchdog.py)— crash + hang detection → recovery swap
  ├─ IdleUnloader (watchdog.py)  — timer-based unload to free VRAM
  ├─ MetricsParser (metrics.py)  — GET /metrics → tok/s, KV-cache, request counts
  └─ PlatformHelper (platform.py)— Windows Job Object, orphan cleanup, graceful stop
```

A swap is an ordered sequence: circuit-breaker check → acquire lock → drain
in-flight (force-reset on timeout) → stop old server → sleep 2 s for VRAM release →
advisory VRAM check → strip a no-longer-fitting `--n-gpu-layers` pin → start → on
`--fit` OOM, retry once with `fallback_gpu_layers` → record outcome → notify.

## Gotchas

- **Swap triggers on more than the model name.** `infer()` re-loads when
  `model_name`, `thinking`, *or* `vision_projector` differs from the loaded config.
  Toggling thinking on the same model is a full process restart, not a flag flip.
- **Windows stop is a hard kill, not CTRL_BREAK.** llama-server installs no
  `CTRL_BREAK_EVENT` handler, so `graceful_stop` uses `terminate()`
  (TerminateProcess) directly on Windows and SIGTERM on Unix. There is no graceful
  in-process shutdown on Windows — the Job Object + TerminateProcess are the
  guarantee.
- **The VRAM check is advisory at swap time.** By the time `min_free_vram_mb` is
  checked the old server is already stopped, so a low reading only logs a warning
  and proceeds — refusing would leave nothing loaded. The hard gate is the
  separate `--n-gpu-layers` recheck (`required_vram_mb`), which strips a pinned
  offload override and falls back to `--fit` when live VRAM is short.
- **Never pass `--n-gpu-layers` casually in `extra_flags`.** `--fit` (default-on)
  auto-sizes from live VRAM; a pinned override defeats it. DaLLaMa only injects one
  as an OOM *fallback* (`fallback_gpu_layers`) and will strip a caller-pinned one if
  `required_vram_mb` says it no longer fits.
- **Generation counter guards orphaned completions.** If a drain times out,
  `force_reset_inflight` bumps a generation token; completions from the pre-swap
  generation are silently dropped so the in-flight counter can't go negative. Pass
  the token from `mark_inference_start` back to `mark_inference_end` — `infer()`
  does this for you.
- **503 is healthy.** The watchdog treats both HTTP 200 and 503 as alive (503 =
  busy loading weights), so a slow cold start does not trip the hang detector.
- **`load_timeout` raises the ceiling, it does not lower it.** The caller-supplied
  `load_timeout` is only applied when it exceeds the internal size/context estimate
  (`max`, not `min`); the internal estimate is clamped to `[60, 300]` s.
- **Stderr log path** comes from the `DALLAMA_LOG_DIR` env var (default: cwd);
  llama-server stderr lands in `<dir>/llama-server.stderr.log` and is tailed for
  OOM-signature detection.

## Dependencies

One third-party package: **`httpx`** (async `/health` and `/metrics` polling).
**No KutAI sibling packages** — DaLLaMa is fully standalone and imports nothing
from the rest of the system. The caller injects everything it needs through
`DaLLaMaConfig` callbacks (`on_ready`, `get_vram_free_mb`) and supplies the
per-model `ServerConfig` (including `fallback_gpu_layers` / `required_vram_mb`,
which the caller computes from its own VRAM/registry knowledge).

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\dallama\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Tek bir llama-server sürecinin sahipliğini üstlenen, bağımsız ve async bir
> sarmalayıcı: modelleri yükler, güvenle değiştirir ve sunucuyu ayakta tutar;
> böylece çağıran tarafın bir alt-süreçle uğraşması gerekmez.

### Amaç

**Ne işe yarar.** llama.cpp'nin `llama-server`'ını Python'dan sürmek bir alt-süreç
bataklığıdır: doğru komut satırını kurmak, ağırlıklar yüklenecek kadar uzun
`/health` yoklamak, süreci öldürmeden önce uçuştaki istekleri boşaltmak, çökme
sonrası yetim süreçleri temizlemek ve Windows'ta (llama-server'ın asla yakalamadığı
`CTRL_BREAK` yalanına rağmen) sunucuyu güvenle durdurmak. DaLLaMa bunların hepsini
soğurur. Çağıran taraf "*şu* modeli yüklü istiyorum" der ve karşılığında sağlıklı
bir URL — ya da temiz bir hata — alır, başka hiçbir şeyle uğraşmaz.

**Gerçekte ne yapar.** Aynı anda tam olarak bir sunucu sürecini yönetir. İstenen
model (ya da onun thinking / vision ayarları) yüklü olandan farklıysa bir *swap*
(değişim) yapar: uçuştaki çıkarımı boşalt → eski süreci durdur → CUDA'nın VRAM'i
bırakmasını bekle → yeni süreci başlat → `/health` geçene kadar bekle. Bunun
etrafında dört koruma oturur: model başına devre kesici (bozuk bir modeli
yeniden-başlatma döngüsüne sokma), sağlık nöbetçisi (çökme ve takılmaları otomatik
kurtar), boşta kalınca kaldırıcı (kimse çağırmazken VRAM'i serbest bırak) ve
opsiyonel VRAM ön-kontrolü. Ayrıca llama-server'ın Prometheus `/metrics` çıktısını
canlı token/sn okuması için ayrıştırır.

**Yapmadıkları**: hangi modelin yükleneceğini seçmez (çağıran taraf karar verir),
yerel ile bulut arasında yönlendirme yapmaz, istekleri kuyruğa almaz veya
zamanlamaz (tek bir sunucu yönetir — meşgulse `status.busy` kontrol edip başka yere
yönlendirin) ve hiçbir LLM/çıkarım çağrısının kendisini yapmaz (çağıran taraf
DaLLaMa'nın verdiği URL'e POST atar). Bir süreç yöneticisidir, bir istemci değil.
**Hiçbir KutAI bağımlılığı yoktur** — tek üçüncü-parti importu `httpx`'tir.

### Genel API

Bir kez kur, arka plan görevlerini `start()` ile başlat, sonra her çıkarımı
`infer()` context manager'ı üzerinden sür. Aşağıdakilerin tümü üst düzeyden
export edilir.

```python
from dallama import (
    DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus,
    InferenceSession, DaLLaMaLoadError,
)

dallama = DaLLaMa(DaLLaMaConfig(
    llama_server_path=r"C:\llama\llama-server.exe",
    port=8080,
    on_ready=on_ready,            # (model_name: str | None, reason: str) -> None
    get_vram_free_mb=probe_vram,  # () -> int  (opsiyonel VRAM kapısı)
))
await dallama.start()             # yetimleri öldürür, nöbetçi + boşta-kaldırıcı başlatır

# Modeli çağrı süresince edin. Yalnızca model, `thinking` veya `vision_projector`
# son yüklemeden bu yana değiştiyse swap yapar.
async with dallama.infer(ServerConfig(
    model_path=r"C:\models\qwen3-30b-q4.gguf",
    model_name="qwen3-30b",
    context_length=16384,
    thinking=True,
)) as session:                    # -> InferenceSession(url, model_name)
    # session.url == "http://127.0.0.1:8080" — sunucu ayakta, model yüklü, /health yeşil
    resp = await http_client.post(f"{session.url}/v1/chat/completions", ...)
    # swap başarısızsa, infer() blok'a girmeden önce DaLLaMaLoadError fırlatır

dallama.keep_alive()              # uzun bir araç döngüsünde çağrılar arası boşta zamanlayıcıyı sıfırla
st = dallama.status               # -> ServerStatus(model_name, healthy, busy, measured_tps, context_length)
await dallama.stop()              # arka plan görevlerini + sunucuyu durdur
```

`__all__`: `DaLLaMa`, `DaLLaMaConfig`, `ServerConfig`, `ServerStatus`,
`InferenceSession`, `DaLLaMaLoadError`. (Metrik ayrıştırıcı dahilidir — export
edilmez.)

#### Config dataclass'ları

```python
DaLLaMaConfig(                      # motor ayarları — bir kez verilir
    llama_server_path="llama-server",
    port=8080, host="127.0.0.1",
    idle_timeout_seconds=60.0,
    circuit_breaker_threshold=2,
    circuit_breaker_cooldown_seconds=300.0,
    inference_drain_timeout_seconds=30.0,
    health_check_interval_seconds=30.0,
    health_fail_threshold=3,
    min_free_vram_mb=4096,
    on_ready=None,                  # durum-değişimi callback'i (aşağıdaki sebeplere bak)
    get_vram_free_mb=None,          # () -> int; None ise GPU boyutlamayı --fit yapar
)

ServerConfig(                       # model başına iş tarifi
    model_path,                     # GGUF'un mutlak yolu
    model_name,                     # okunabilir ad
    context_length,                 # --ctx-size
    thinking=False,                 # --reasoning on (yalnızca --jinja ile)
    vision_projector="",            # --mmproj <yol>; boş = vision yok
    extra_flags=[],                 # komut satırına aynen eklenir
    fallback_gpu_layers=0,          # OOM-retry --n-gpu-layers değeri (0 = retry yok)
    required_vram_mb=0,             # boş VRAM bundan azsa sabitlenmiş --n-gpu-layers'ı söker
)

ServerStatus(model_name, healthy, busy, measured_tps, context_length)
InferenceSession(url, model_name)
DaLLaMaLoadError(model_name, reason="")   # .model_name, .reason
```

#### `on_ready` sebepleri

Callback durum değişimlerinde tetiklenir — bunu görev kuyruğunuzu uyandırmaya
bağlayın. Gerçekte yayınlanan dört sebep:

| sebep | ne zaman |
|---|---|
| `model_loaded` | yeni bir model yüklendi ve sağlıklı |
| `load_failed` | model sağlıklı hâle gelemedi |
| `idle_unload` | boşta penceresi dolunca model kaldırıldı |
| `circuit_breaker_active` | yükleme reddedildi — bu model için kesici soğumada |

(Çökme kurtarma ve takılma yeniden-başlatmaları swap yoluna geri girer ve
`model_loaded` / `load_failed` olarak görünür; ayrı bir `crash_recovery` sebebi
yoktur.)

### Mimari

`DaLLaMa`, tek-sorumluluklu altı modülü birleştiren ince bir cephedir:

```
DaLLaMa (dallama.py)        — cephe: infer() ctx-mgr, status, start/stop, tps tazeleme
  ├─ ServerProcess (server.py)   — build_cmd, alt-süreç başlat/durdur, /health yokla, stderr kuyruğu
  ├─ SwapManager (swap.py)       — kilit · boşaltma · devre kesici · VRAM kapısı · OOM yedeği
  ├─ HealthWatchdog (watchdog.py)— çökme + takılma tespiti → kurtarma swap'i
  ├─ IdleUnloader (watchdog.py)  — VRAM serbest bırakmak için zamanlayıcı tabanlı kaldırma
  ├─ MetricsParser (metrics.py)  — GET /metrics → token/sn, KV-cache, istek sayıları
  └─ PlatformHelper (platform.py)— Windows Job Object, yetim temizleme, düzgün durdurma
```

Bir swap sıralı bir dizidir: devre-kesici kontrolü → kilidi al → uçuştakileri
boşalt (zaman aşımında zorla sıfırla) → eski sunucuyu durdur → VRAM serbest kalması
için 2 sn bekle → tavsiye niteliğinde VRAM kontrolü → artık sığmayan
`--n-gpu-layers` pinini sök → başlat → `--fit` OOM olursa `fallback_gpu_layers` ile
bir kez yeniden dene → sonucu kaydet → bildir.

### Tuzaklar

- **Swap yalnızca model adında değil daha fazlasında tetiklenir.** `infer()`,
  `model_name`, `thinking` *ya da* `vision_projector` yüklü config'den farklıysa
  yeniden yükler. Aynı modelde thinking'i açıp kapatmak tam bir süreç
  yeniden-başlatmasıdır, bir flag değişimi değil.
- **Windows'ta durdurma CTRL_BREAK değil, sert öldürmedir.** llama-server hiçbir
  `CTRL_BREAK_EVENT` işleyicisi kurmaz; bu yüzden `graceful_stop` Windows'ta
  doğrudan `terminate()` (TerminateProcess), Unix'te SIGTERM kullanır. Windows'ta
  süreç-içi düzgün kapatma yoktur — Job Object + TerminateProcess garantidir.
- **VRAM kontrolü swap anında tavsiye niteliğindedir.** `min_free_vram_mb` kontrol
  edildiğinde eski sunucu çoktan durdurulmuştur; düşük bir okuma yalnızca uyarı
  loglar ve devam eder — reddetmek elde hiçbir şey yüklü olmaması demek olurdu. Sert
  kapı ayrı `--n-gpu-layers` yeniden-kontrolüdür (`required_vram_mb`): canlı VRAM
  yetersizse sabitlenmiş offload override'ını söküp `--fit`'e geri döner.
- **`extra_flags`'e gelişigüzel `--n-gpu-layers` koymayın.** `--fit` (öntanımlı
  açık) canlı VRAM'den otomatik boyutlandırır; sabitlenmiş bir override bunu bozar.
  DaLLaMa böyle birini yalnızca OOM *yedeği* olarak (`fallback_gpu_layers`) enjekte
  eder ve `required_vram_mb` artık sığmadığını söylüyorsa çağıranın sabitlediğini
  söker.
- **Nesil sayacı gecikmeli tamamlanmaları korur.** Boşaltma zaman aşımına uğrarsa
  `force_reset_inflight` bir nesil belirteci artırır; swap-öncesi nesilden gelen
  tamamlanmalar sessizce düşürülür, böylece uçuş sayacı eksiye gidemez.
  `mark_inference_start`'tan gelen belirteci `mark_inference_end`'e geri verin —
  `infer()` bunu sizin için yapar.
- **503 sağlıklıdır.** Nöbetçi hem HTTP 200 hem 503'ü ayakta sayar (503 = ağırlık
  yüklüyor, meşgul); böylece yavaş bir soğuk başlangıç takılma tespitçisini
  tetiklemez.
- **`load_timeout` tavanı yükseltir, düşürmez.** Çağıranın verdiği `load_timeout`
  yalnızca dahili boyut/bağlam tahminini aştığında uygulanır (`min` değil `max`);
  dahili tahmin `[60, 300]` sn'ye sıkıştırılır.
- **Stderr log yolu** `DALLAMA_LOG_DIR` env değişkeninden gelir (öntanımlı: cwd);
  llama-server stderr'i `<dir>/llama-server.stderr.log`'a düşer ve OOM imzası
  tespiti için kuyruğu okunur.

### Bağımlılıklar

Bir üçüncü-parti paket: **`httpx`** (async `/health` ve `/metrics` yoklaması).
**Hiçbir KutAI kardeş paketi yok** — DaLLaMa tamamen bağımsızdır ve sistemin geri
kalanından hiçbir şey import etmez. Çağıran taraf ihtiyaç duyduğu her şeyi
`DaLLaMaConfig` callback'leri (`on_ready`, `get_vram_free_mb`) üzerinden enjekte
eder ve model başına `ServerConfig`'i sağlar (kendi VRAM/registry bilgisinden
hesapladığı `fallback_gpu_layers` / `required_vram_mb` dahil).

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\dallama\ -q
```
