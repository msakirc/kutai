# Nerd Herd — System-state ledger & pressure brain

> The one place that knows the live state of the machine: GPU, VRAM budget,
> service health, llama-server throughput, the in-flight call list, and cloud
> rate-limit cells. Producers push facts in; consumers read a single coherent
> snapshot out — and ask it how much *pressure* a candidate model is under.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A running AI system has its state scattered everywhere:
GPU temperature lives in the driver, VRAM budget in the load policy, throughput
in llama-server, rate limits in each cloud provider, the queue in the task
master, the in-flight calls in the dispatcher. Nerd Herd is the single ledger
that collects all of it, so any consumer — a Grafana dashboard, a health
watchdog, or the model selector — reads one consistent picture instead of
polling six subsystems. Producers *push* facts in; Nerd Herd owns the merge.

**What it really does.** Two surfaces over one shared state. (1) An
**observability sidecar**: it polls the GPU (pynvml) and llama-server's native
`/metrics`, manages a 4-mode VRAM budget with an auto-detect loop that backs off
when external apps grab the GPU, tracks service health, and serves everything as
Prometheus text on `/metrics` plus a small JSON API — so Grafana scrapes it
directly with no Prometheus server. (2) A **pressure brain**: `snapshot()`
returns a `SystemSnapshot` of everything pushed in, and `SystemSnapshot.pressure_for(model, …)`
runs eleven scarcity signals (S1–S12) through three modifiers into one scalar in
`[-1, +1]` plus a diagnostic `PressureBreakdown` — the number a selector uses to
decide whether a given model is starved or abundant *right now*.

**It does NOT** pick or rank models, gate admission, or make any policy decision
— it computes the pressure scalar and hands it back; the caller decides. It does
NOT make LLM or provider HTTP calls (it only parses the metrics handed to it),
discover models, or decide swap budgets (it counts swaps in a sliding window; the
*is-this-swap-allowed?* policy lives elsewhere). Missing data yields a neutral
value, never an exception.

## Public API

Two ways to talk to Nerd Herd. **In-process** (`NerdHerd`) when you own the
object; **over HTTP** (`NerdHerdClient` + module-level helpers) when Nerd Herd
runs as a separate sidecar process and you hold a thin proxy.

```python
from nerd_herd import NerdHerd, SystemSnapshot

nh = NerdHerd(metrics_port=9881, llama_server_url="http://127.0.0.1:8080")
await nh.start()                 # metrics server + inference polling
await nh.start_auto_detect()     # GPU backoff loop (optional)

# --- producers push facts in ---
nh.push_local_state(local_state)         # the model host, on each swap
nh.push_cloud_state(cloud_provider_state)# the capacity tracker, on each response
nh.push_queue_profile(queue_profile)     # the task master, on queue change
nh.push_in_flight(list_of_inflight_calls)# the dispatcher, on begin/end
nh.record_swap(model_name)               # the dispatcher, after a swap

# --- consumers read one coherent snapshot out ---
snap: SystemSnapshot = nh.snapshot()
budget_mb = nh.get_vram_budget_mb()      # raw free VRAM (no mode cap; placement not capping)
breakdown = snap.pressure_for(model, task_difficulty=7, est_call_cost=0.002)
scalar = breakdown.scalar                # float in [-1, +1]; -1 starved, +1 abundant
```

```python
# Out-of-process: module-level singleton + HTTP client proxy.
from nerd_herd import snapshot, refresh_snapshot, push_in_flight, record_swap
from nerd_herd.client import set_default, NerdHerdClient

set_default(NerdHerdClient(port=9881))   # wire once at startup
await refresh_snapshot()                 # async fetch → caches
snap = snapshot()                        # sync read of last cached snapshot
```

**Top-level exports (`__all__`).** Facade & client: `NerdHerd`, `NerdHerdClient`,
`GPUStateProxy`. State dataclasses: `GPUState`, `SystemState`, `ExternalGPUUsage`,
`HealthStatus`, `InFlightCall`, `RateLimit`, `RateLimitMatrix`, `CloudModelState`,
`CloudProviderState`, `LocalModelState`, `QueueProfile`, `SystemSnapshot`,
`PressureBreakdown`. Collector machinery: `CollectorRegistry`, `Collector`,
`GPUCollector`, `LoadManager`, `HealthRegistry`, `InferenceCollector`,
`RingBuffer`, `SwapBudget`. Helpers: `health_summary`. Module-level singleton
funcs (used by out-of-process callers): `snapshot`, `refresh_snapshot`,
`record_swap`, `push_queue_profile`, `push_in_flight`.

### Key methods (`NerdHerd`)

| method | purpose |
|---|---|
| `snapshot() -> SystemSnapshot` | point-in-time merge of all pushed state; overlays live inference metrics onto local state |
| `push_local_state(state)` · `push_cloud_state(state)` · `push_queue_profile(p)` · `push_in_flight(calls)` | producers replace/upsert their slice of state |
| `record_swap(model_name="")` · `recent_swap_count() -> int` | sliding-window swap counter (window 300s) |
| `gpu_state() -> GPUState` | live VRAM / temp / utilization / power |
| `get_vram_budget_mb() -> int` · `get_vram_budget_fraction() -> float` | `_mb` = raw free VRAM (no cap); `_fraction` = current mode fraction (advisory only) |
| `get_load_mode() -> str` · `set_load_mode(mode, source="user") -> str` | `"full" / "heavy" / "shared" / "minimal"`; `source="user"` disables auto-detect |
| `enable_auto_management()` · `is_local_inference_allowed() -> bool` | re-enable backoff loop; `False` only in `minimal` |
| `on_mode_change(cb)` | register `(old, new, source) -> None` callback (DB persistence, notify) |
| `mark_degraded(cap)` · `mark_healthy(cap)` · `is_healthy(cap) -> bool` · `get_health_status() -> HealthStatus` | service-health registry |
| `register_collector(name, collector)` | add a custom `Collector` to the `/metrics` exposition |
| `prometheus_lines() -> str` | Prometheus text for embedding |

### `SystemSnapshot.pressure_for(...)`

```python
breakdown = snapshot.pressure_for(
    model,                       # duck-typed: .name .provider .is_free .is_local .cap_score
    task_difficulty=5,           # 1..10
    est_per_call_tokens=0,
    est_per_task_tokens=0,
    est_iterations=1,
    est_call_cost=0.0,
    cap_needed=5.0,
    consecutive_failures=0,
    fleet_consumed=None,         # {free-provider -> calls this cycle} for S12; None → S12=0
    eligible_models=None,
) -> PressureBreakdown           # .scalar in [-1, +1], .signals, .modifiers, .bucket_totals
```

## Architecture

Producers push; one snapshot merges; two consumers read. The pressure path is a
pure function over the snapshot.

```
  producers ──push──►        Nerd Herd state          ──read──► consumers
  ┌──────────────┐    ┌───────────────────────────┐
  │ model host   │──► │ LoadManager  (4 VRAM modes,│   /metrics ──► Grafana (Prometheus DS)
  │ capacity trk │──► │              auto-backoff) │   /api/*   ──► sidecar HTTP client
  │ task master  │──► │ GPUCollector (pynvml,2s)   │
  │ dispatcher   │──► │ InferenceColl(llama /metr) │   snapshot() ─┐
  └──────────────┘    │ HealthRegistry             │               ▼
                      │ SwapBudget   (300s window) │      SystemSnapshot.pressure_for()
                      │ _local / _cloud / _queue / │        S1..S12 ─► M1/M2/M3 ─► combine
                      │ _in_flight  pushed slices  │        ─► scalar ∈ [-1,+1] + breakdown
                      └───────────────────────────┘               (selector reads scalar)
```

The pressure scalar combines signals bucket-by-bucket (`combine.py`):
worst-wins inside each bucket, weighted sum across buckets, and a noisy-OR
abundance arm (`S9`, `S12`) gated by total negative pressure.

## Key Modules

| module | role |
|---|---|
| `nerd_herd.py` | `NerdHerd` facade — registers built-in collectors, holds pushed state, builds `snapshot()` |
| `client.py` | `NerdHerdClient` HTTP proxy + process-wide `get_default`/`set_default`; safe defaults when sidecar is down |
| `types.py` | all state dataclasses **and** `SystemSnapshot.pressure_for` (the signal pipeline lives here) |
| `exposition.py` | `MetricsServer` (aiohttp): `/metrics`, `/health`, `/api/*`; `API_VERSION` handshake |
| `gpu.py` / `load.py` / `inference.py` / `health.py` | the four built-in collectors |
| `registry.py` | `Collector` protocol + `CollectorRegistry` — unified `/metrics` exposition |
| `signals/` | `s1_remaining … s12_pool_balance` — pure functions, each → float `[-1, +1]` |
| `combine.py` / `modifiers.py` / `breakdown.py` | bucket combination, M1/M2/M3 reshapers, `PressureBreakdown` struct |
| `burn_log.py` | rolling per-(provider, model) burn-rate log (feeds S7); process singleton |
| `swap_budget.py` | sliding-window swap counter (data only; allow/deny policy lives elsewhere) |
| `ring_buffer.py` | fixed-capacity buffer for pre-computed inference rates |
| `health_summary.py` | resource-health rollup; **reaches into the host app** (see Dependencies) |
| `__main__.py` | `python -m nerd_herd` sidecar entry point (PID file, DB-backed mode persistence) |

## VRAM budget modes

| mode | fraction | meaning |
|---|---|---|
| `full` | 1.0 | all local capacity available |
| `heavy` | 0.9 | 90% cap, headroom for OS/desktop |
| `shared` | 0.5 | 50% cap, prefer cloud for heavy tasks |
| `minimal` | 0.0 | local inference disabled (`is_local_inference_allowed() → False`) |

The auto-detect loop **downgrades immediately** when external GPU usage rises and
**upgrades only after** `upgrade_delay` (default 300s) of sustained improvement —
asymmetric to avoid flapping. A manual `set_load_mode(..., source="user")` pins
the mode and disables auto-detect until `enable_auto_management()`.

## Gotchas

- **Producers are the source of truth.** `snapshot()` is only as fresh as the
  last push. The dispatcher must `push_in_flight` on every begin/end or pressure
  reads stale in-flight reservations. The one auto-overlay exception: live
  inference metrics (`requests_processing`, `idle_seconds`, `kv_cache_ratio`) are
  layered onto local state at snapshot time without a push.
- **In-flight is matched by *provider*, not model id.** Cloud rate-limit cells
  are often provider-aggregate (one API key shared across model ids), so
  `pressure_for` subtracts every in-flight call on the same provider. Per-model
  cells (`rpd`/`tpd`) over-subtract slightly — the safe direction (fewer
  admissions, no overshoot).
- **`pressure_for` returns a `PressureBreakdown`, not a float.** Read `.scalar`.
  The full struct (per-signal, per-bucket, modifiers) is meant to be logged for
  offline weight tuning.
- **Missing model state ⇒ neutral, not error.** No matrix, no samples, empty
  queue → signals return 0. A freshly-revived model with `<5` samples falls back
  to `provider_prior_rate`, then to neutral — never ranks as perfectly reliable.
- **HTTP client returns safe defaults silently.** When the sidecar is
  unreachable, `NerdHerdClient` methods return zeros/`"full"`/empty snapshot and
  log at debug. Use `check_version()` against `API_VERSION` to detect a stale
  sidecar and restart it.
- **`set_default` is process-wide.** The module-level `snapshot()` /
  `push_in_flight()` helpers resolve through `client.get_default()`; they no-op
  until something calls `set_default(...)`.

## Dependencies

- **Third-party**: `pynvml` (GPU; degrades gracefully with no GPU), `psutil`
  (RAM/CPU), `prometheus_client` (metric types + text format), `aiohttp` (HTTP
  server + llama-server polling), `yazbunu` (structured logging).
- **llama-server** (optional): pass `llama_server_url` to enable inference
  metrics; absent → those gauges report zero.
- **The host application** — one genuine, asymmetric coupling. The core package
  is host-independent, but `health_summary.py` deliberately reaches **into the
  host** via lazy `try/except` imports (`src.models.local_model_manager`,
  `src.models.gpu_monitor`, `src.core.router`, `src.models.model_registry`,
  `src.security.credential_store`) to assemble a full resource-health rollup. It
  is the one module that won't run standalone; every other module is
  self-contained. The state dataclasses (`CloudProviderState`, `RateLimitMatrix`,
  etc.) are the deliberate seam **producers** push through — Nerd Herd does not
  import any of them back.
- **Env**: `LLAMA_SERVER_PORT` (sidecar default llama URL), `NERD_HERD_PROJECT_ROOT`
  (sidecar `sys.path` injection for `health_summary`'s host imports).

## Runbook

Run as a standalone sidecar:

```powershell
& .\.venv\Scripts\python.exe -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080 --db-path .\data\kutai.db
```

Point Grafana at `http://localhost:9881/metrics` (datasource type: Prometheus).
The sidecar persists load mode to the `load_mode` table and restores it on boot.
Tuning the pressure signals: each `signals/sN_*.py` is a pure function with its
own test under `tests/signals/` — change one, re-run that test, and inspect the
`PressureBreakdown` it feeds. Do **not** add policy here; the scalar is advice,
the selector decides.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\nerd_herd\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Makinenin canlı durumunu bilen tek yer: GPU, VRAM bütçesi, servis sağlığı,
> llama-server hızı, uçuştaki çağrı listesi ve bulut rate-limit hücreleri.
> Üreticiler gerçekleri içeri iter; tüketiciler tutarlı tek bir anlık görüntü
> okur — ve bir aday modelin ne kadar *baskı* altında olduğunu sorar.

### Amaç

**Ne işe yarar.** Çalışan bir yapay zeka sisteminin durumu her yere dağılmıştır:
GPU sıcaklığı sürücüde, VRAM bütçesi yük politikasında, hız llama-server'da, rate
limit'ler her bulut sağlayıcısında, kuyruk görev yöneticisinde, uçuştaki çağrılar
dispatcher'da. Nerd Herd, bunların hepsini toplayan tek defterdir; böylece
herhangi bir tüketici — bir Grafana panosu, bir sağlık nöbetçisi ya da model
seçicisi — altı alt sistemi tek tek yoklamak yerine tutarlı tek bir resim okur.
Üreticiler gerçekleri *iter*; birleştirmenin sahibi Nerd Herd'dir.

**Gerçekte ne yapar.** Tek paylaşılan durumun üzerinde iki yüz. (1) Bir
**gözlemlenebilirlik yan-süreci (sidecar)**: GPU'yu (pynvml) ve llama-server'ın
yerel `/metrics`'ini yoklar, harici uygulamalar GPU'yu kaptığında geri çekilen
bir otomatik-algılama döngüsüyle 4 modlu VRAM bütçesi yönetir, servis sağlığını
izler ve her şeyi `/metrics` üzerinde Prometheus metni artı küçük bir JSON API
olarak sunar — böylece Grafana, Prometheus sunucusu olmadan doğrudan kazır. (2)
Bir **baskı beyni**: `snapshot()`, içeri itilen her şeyin bir `SystemSnapshot`'ını
döndürür ve `SystemSnapshot.pressure_for(model, …)`, on bir kıtlık sinyalini
(S1–S12) üç değiştiriciden geçirip `[-1, +1]` aralığında tek bir skalere artı bir
tanılayıcı `PressureBreakdown`'a indirger — bir seçicinin, belirli bir modelin
*şu anda* aç mı yoksa bol mu olduğuna karar vermek için kullandığı sayı.

**Yapmadıkları**: model seçmez veya sıralamaz, kabul (admission) kapısı tutmaz,
hiçbir politika kararı vermez — baskı skalerini hesaplayıp geri verir; kararı
çağıran taraf verir. LLM veya sağlayıcı HTTP çağrısı yapmaz (yalnızca kendisine
verilen metrikleri parse eder), model keşfetmez, swap bütçesine karar vermez
(kayan pencerede swap'ları sayar; *bu swap'a izin var mı?* politikası başka yerde
yaşar). Eksik veri bir istisna değil, nötr bir değer üretir.

### Genel API

Nerd Herd ile konuşmanın iki yolu. Nesneye sahipken **süreç-içi** (`NerdHerd`);
Nerd Herd ayrı bir sidecar süreç olarak çalışırken ve elinizde ince bir proxy
varken **HTTP üzerinden** (`NerdHerdClient` + modül düzeyi yardımcılar).

```python
from nerd_herd import NerdHerd, SystemSnapshot

nh = NerdHerd(metrics_port=9881, llama_server_url="http://127.0.0.1:8080")
await nh.start()                 # metrik sunucusu + çıkarım yoklaması
await nh.start_auto_detect()     # GPU geri-çekilme döngüsü (opsiyonel)

# --- üreticiler gerçekleri içeri iter ---
nh.push_local_state(local_state)         # model host, her swap'ta
nh.push_cloud_state(cloud_provider_state)# kapasite izleyici, her yanıtta
nh.push_queue_profile(queue_profile)     # görev yöneticisi, kuyruk değişiminde
nh.push_in_flight(list_of_inflight_calls)# dispatcher, her begin/end'de
nh.record_swap(model_name)               # dispatcher, bir swap'tan sonra

# --- tüketiciler tutarlı tek bir anlık görüntü okur ---
snap: SystemSnapshot = nh.snapshot()
budget_mb = nh.get_vram_budget_mb()      # ham boş VRAM (mod kısıtı yok; yerleştirme, kısıtlama değil)
breakdown = snap.pressure_for(model, task_difficulty=7, est_call_cost=0.002)
scalar = breakdown.scalar                # [-1, +1]; -1 aç, +1 bol
```

```python
# Süreç-dışı: modül düzeyi singleton + HTTP istemci proxy'si.
from nerd_herd import snapshot, refresh_snapshot, push_in_flight, record_swap
from nerd_herd.client import set_default, NerdHerdClient

set_default(NerdHerdClient(port=9881))   # başlangıçta bir kez bağla
await refresh_snapshot()                 # async getir → önbelleğe al
snap = snapshot()                        # son önbelleklenen anlık görüntünün sync okuması
```

**Üst düzey export'lar (`__all__`).** Cephe & istemci: `NerdHerd`,
`NerdHerdClient`, `GPUStateProxy`. Durum dataclass'ları: `GPUState`,
`SystemState`, `ExternalGPUUsage`, `HealthStatus`, `InFlightCall`, `RateLimit`,
`RateLimitMatrix`, `CloudModelState`, `CloudProviderState`, `LocalModelState`,
`QueueProfile`, `SystemSnapshot`, `PressureBreakdown`. Toplayıcı makinesi:
`CollectorRegistry`, `Collector`, `GPUCollector`, `LoadManager`,
`HealthRegistry`, `InferenceCollector`, `RingBuffer`, `SwapBudget`.
Yardımcılar: `health_summary`. Modül düzeyi singleton fonksiyonları (süreç-dışı
çağıranlarca kullanılır): `snapshot`, `refresh_snapshot`, `record_swap`,
`push_queue_profile`, `push_in_flight`.

#### Ana metotlar (`NerdHerd`)

| metot | görevi |
|---|---|
| `snapshot() -> SystemSnapshot` | itilen tüm durumun nokta-zaman birleşimi; canlı çıkarım metriklerini yerel duruma bindirir |
| `push_local_state` · `push_cloud_state` · `push_queue_profile` · `push_in_flight` | üreticiler kendi durum dilimlerini değiştirir/upsert eder |
| `record_swap(model_name="")` · `recent_swap_count() -> int` | kayan-pencere swap sayacı (pencere 300s) |
| `gpu_state() -> GPUState` | canlı VRAM / sıcaklık / kullanım / güç |
| `get_vram_budget_mb() -> int` · `get_vram_budget_fraction() -> float` | `_mb` = ham boş VRAM (kısıt yok); `_fraction` = geçerli mod kesri (yalnızca bilgilendirme) |
| `get_load_mode() -> str` · `set_load_mode(mode, source="user") -> str` | `"full" / "heavy" / "shared" / "minimal"`; `source="user"` otomatik-algılamayı kapatır |
| `enable_auto_management()` · `is_local_inference_allowed() -> bool` | geri-çekilme döngüsünü tekrar aç; yalnızca `minimal`'da `False` |
| `on_mode_change(cb)` | `(eski, yeni, kaynak) -> None` callback'i kaydet (DB kalıcılığı, bildirim) |
| `mark_degraded` · `mark_healthy` · `is_healthy` · `get_health_status` | servis-sağlık kayıt defteri |
| `register_collector(name, collector)` | `/metrics` sunumuna özel bir `Collector` ekle |
| `prometheus_lines() -> str` | gömme için Prometheus metni |

#### `SystemSnapshot.pressure_for(...)`

```python
breakdown = snapshot.pressure_for(
    model,                       # duck-typed: .name .provider .is_free .is_local .cap_score
    task_difficulty=5,           # 1..10
    est_per_call_tokens=0,
    est_per_task_tokens=0,
    est_iterations=1,
    est_call_cost=0.0,
    cap_needed=5.0,
    consecutive_failures=0,
    fleet_consumed=None,         # {free-sağlayıcı -> bu döngüdeki çağrılar} (S12); None → S12=0
    eligible_models=None,
) -> PressureBreakdown           # .scalar [-1, +1], .signals, .modifiers, .bucket_totals
```

### Mimari

Üreticiler iter; tek anlık görüntü birleştirir; iki tüketici okur. Baskı yolu,
anlık görüntü üzerinde saf bir fonksiyondur.

```
  üreticiler ─push─►        Nerd Herd durumu         ─read─► tüketiciler
  ┌──────────────┐    ┌───────────────────────────┐
  │ model host   │──► │ LoadManager  (4 VRAM modu, │   /metrics ──► Grafana (Prometheus DS)
  │ kapasite izl.│──► │              oto-geri-çek.)│   /api/*   ──► sidecar HTTP istemci
  │ görev yön.   │──► │ GPUCollector (pynvml, 2s)  │
  │ dispatcher   │──► │ InferenceColl(llama /metr) │   snapshot() ─┐
  └──────────────┘    │ HealthRegistry             │               ▼
                      │ SwapBudget   (300s pencere)│      SystemSnapshot.pressure_for()
                      │ _local / _cloud / _queue / │        S1..S12 ─► M1/M2/M3 ─► combine
                      │ _in_flight  itilen dilimler│        ─► skaler ∈ [-1,+1] + breakdown
                      └───────────────────────────┘               (seçici skaleri okur)
```

Baskı skaleri sinyalleri kova-kova birleştirir (`combine.py`): her kovanın
içinde en-kötü-kazanır, kovalar arası ağırlıklı toplam, ve toplam negatif baskıya
göre kapılanan bir noisy-OR bolluk kolu (`S9`, `S12`).

### Ana Modüller

| modül | rolü |
|---|---|
| `nerd_herd.py` | `NerdHerd` cephesi — yerleşik toplayıcıları kaydeder, itilen durumu tutar, `snapshot()` kurar |
| `client.py` | `NerdHerdClient` HTTP proxy + süreç-geneli `get_default`/`set_default`; sidecar kapalıyken güvenli varsayılanlar |
| `types.py` | tüm durum dataclass'ları **ve** `SystemSnapshot.pressure_for` (sinyal hattı burada yaşar) |
| `exposition.py` | `MetricsServer` (aiohttp): `/metrics`, `/health`, `/api/*`; `API_VERSION` el sıkışması |
| `gpu.py` / `load.py` / `inference.py` / `health.py` | dört yerleşik toplayıcı |
| `registry.py` | `Collector` protokolü + `CollectorRegistry` — birleşik `/metrics` sunumu |
| `signals/` | `s1_remaining … s12_pool_balance` — saf fonksiyonlar, her biri → float `[-1, +1]` |
| `combine.py` / `modifiers.py` / `breakdown.py` | kova birleşimi, M1/M2/M3 yeniden-şekillendiriciler, `PressureBreakdown` struct'ı |
| `burn_log.py` | (sağlayıcı, model) başına kayan burn-rate günlüğü (S7'yi besler); süreç singleton'ı |
| `swap_budget.py` | kayan-pencere swap sayacı (yalnızca veri; izin/ret politikası başka yerde) |
| `ring_buffer.py` | önceden hesaplanmış çıkarım oranları için sabit kapasiteli tampon |
| `health_summary.py` | kaynak-sağlık özeti; **host uygulamasına uzanır** (bkz. Bağımlılıklar) |
| `__main__.py` | `python -m nerd_herd` sidecar giriş noktası (PID dosyası, DB-destekli mod kalıcılığı) |

### VRAM bütçe modları

| mod | kesir | anlamı |
|---|---|---|
| `full` | 1.0 | tüm yerel kapasite kullanılabilir |
| `heavy` | 0.9 | %90 sınır, OS/masaüstü için pay |
| `shared` | 0.5 | %50 sınır, ağır görevlerde bulutu tercih et |
| `minimal` | 0.0 | yerel çıkarım kapalı (`is_local_inference_allowed() → False`) |

Otomatik-algılama döngüsü, harici GPU kullanımı yükseldiğinde **anında düşürür**
ve yalnızca `upgrade_delay` (varsayılan 300s) süresince sürekli iyileşmeden
**sonra yükseltir** — çırpınmayı (flapping) önlemek için asimetrik. Manuel bir
`set_load_mode(..., source="user")` modu sabitler ve `enable_auto_management()`
çağrılana dek otomatik-algılamayı kapatır.

### Tuzaklar

- **Gerçeğin kaynağı üreticilerdir.** `snapshot()` yalnızca son push kadar
  tazedir. Dispatcher her begin/end'de `push_in_flight` yapmalıdır, yoksa baskı
  bayat uçuştaki rezervasyonları okur. Tek oto-bindirme istisnası: canlı çıkarım
  metrikleri (`requests_processing`, `idle_seconds`, `kv_cache_ratio`) snapshot
  anında push olmadan yerel duruma bindirilir.
- **Uçuştakiler model id'sine değil *sağlayıcıya* göre eşleşir.** Bulut rate-limit
  hücreleri çoğu zaman sağlayıcı-toplamdır (tek bir API anahtarı model id'leri
  arasında paylaşılır), bu yüzden `pressure_for` aynı sağlayıcıdaki her uçuştaki
  çağrıyı düşer. Model-başına hücreler (`rpd`/`tpd`) hafifçe fazla düşer — güvenli
  yön (daha az kabul, taşma yok).
- **`pressure_for` bir float değil `PressureBreakdown` döndürür.** `.scalar`'ı
  okuyun. Tam struct (sinyal-başına, kova-başına, değiştiriciler) çevrimdışı
  ağırlık ayarı için loglanmak üzere tasarlanmıştır.
- **Eksik model durumu ⇒ nötr, hata değil.** Matris yok, örnek yok, boş kuyruk →
  sinyaller 0 döner. `<5` örnekli yeni-canlandırılmış bir model
  `provider_prior_rate`'e, sonra nötre geri düşer — asla kusursuz güvenilir gibi
  sıralanmaz.
- **HTTP istemci sessizce güvenli varsayılan döner.** Sidecar erişilemezken
  `NerdHerdClient` metotları sıfır/`"full"`/boş anlık görüntü döner ve debug'da
  loglar. Bayat bir sidecar'ı saptayıp yeniden başlatmak için `API_VERSION`'a
  karşı `check_version()` kullanın.
- **`set_default` süreç-genelidir.** Modül düzeyi `snapshot()` / `push_in_flight()`
  yardımcıları `client.get_default()` üzerinden çözülür; bir şey
  `set_default(...)` çağırana dek no-op'turlar.

### Bağımlılıklar

- **Üçüncü-taraf**: `pynvml` (GPU; GPU yoksa zarifçe geriler), `psutil` (RAM/CPU),
  `prometheus_client` (metrik tipleri + metin formatı), `aiohttp` (HTTP sunucusu +
  llama-server yoklaması), `yazbunu` (yapılı loglama).
- **llama-server** (opsiyonel): çıkarım metriklerini etkinleştirmek için
  `llama_server_url` geçin; yoksa o ölçümler sıfır raporlar.
- **Host uygulaması** — tek gerçek, asimetrik bağ. Çekirdek paket host'tan
  bağımsızdır, ama `health_summary.py` tam bir kaynak-sağlık özeti kurmak için
  bilinçli olarak lazy `try/except` import'larıyla **host'a uzanır**
  (`src.models.local_model_manager`, `src.models.gpu_monitor`, `src.core.router`,
  `src.models.model_registry`, `src.security.credential_store`). Standalone
  çalışmayan tek modül odur; diğer her modül kendi kendine yeter. Durum
  dataclass'ları (`CloudProviderState`, `RateLimitMatrix`, vb.) **üreticilerin**
  içinden push yaptığı bilinçli dikiş yeridir — Nerd Herd hiçbirini geri import
  etmez.
- **Env**: `LLAMA_SERVER_PORT` (sidecar varsayılan llama URL'i),
  `NERD_HERD_PROJECT_ROOT` (`health_summary`'nin host import'ları için sidecar
  `sys.path` enjeksiyonu).

### Çalıştırma

Standalone sidecar olarak çalıştırma:

```powershell
& .\.venv\Scripts\python.exe -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080 --db-path .\data\kutai.db
```

Grafana'yı `http://localhost:9881/metrics` adresine yönlendirin (veri kaynağı
tipi: Prometheus). Sidecar, yük modunu `load_mode` tablosuna kalıcılaştırır ve
açılışta geri yükler. Baskı sinyallerini ayarlama: her `signals/sN_*.py`,
`tests/signals/` altında kendi testi olan saf bir fonksiyondur — birini değiştir,
o testi yeniden çalıştır ve beslediği `PressureBreakdown`'ı incele. Buraya
politika **eklemeyin**; skaler tavsiyedir, kararı seçici verir.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\nerd_herd\ -q
```
