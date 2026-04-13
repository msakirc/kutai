# nerd herd

**Standalone observability for GPU-powered AI systems.**

*Collect metrics. Manage VRAM budgets. Serve Prometheus. No Prometheus server required.*

[English](#english) | [Turkce](#turkce)

---

<a id="english"></a>

## What is nerd herd?

nerd herd is a Python observability package that monitors GPU resources, manages VRAM budget policies, tracks service health, collects inference metrics from llama-server, pre-computes rates in ring buffers, and serves everything via a Prometheus-compatible `/metrics` endpoint. Grafana scrapes nerd herd directly — no Prometheus server needed.

```python
from nerd_herd import NerdHerd

nh = NerdHerd(
    metrics_port=9881,
    llama_server_url="http://127.0.0.1:8080",
)
await nh.start()

# DaLLaMa asks: "how much VRAM can I use?"
budget_mb = nh.get_vram_budget_mb()  # 4000 (shared mode, 50% of 8GB)

# GPU state
state = nh.gpu_state()
print(f"{state.vram_free_mb}MB free, {state.temperature_c}C")

# Health tracking
nh.mark_degraded("telegram")
nh.mark_healthy("llm")
```

Then point Grafana at `http://localhost:9881/metrics`.

## Why nerd herd?

| | nerd herd | Prometheus + Grafana | Datadog / New Relic | nvidia-smi |
|---|---|---|---|---|
| **Setup** | `pip install nerd-herd` | Docker, configs, YAML | Account, agent, $$ | Already there |
| **RAM** | ~20MB | 256MB+ (Prometheus alone) | 200MB+ (agent) | N/A |
| **VRAM budgeting** | Built-in (4 modes) | Manual alerting | Not GPU-aware | Read-only |
| **Rate computation** | Ring buffer, no TSDB | Prometheus TSDB | Cloud TSDB | N/A |
| **GPU auto-detect** | External process aware | Manual rules | Generic GPU metrics | Manual |
| **Inference metrics** | llama-server native | Custom exporter needed | Custom integration | N/A |
| **Price** | Free | Free / $$ (storage) | $$$$ | Free |

## Why not Prometheus?

Prometheus is a 256MB container that scrapes two endpoints every 15 seconds and stores time series so Grafana can compute `rate()`. For a single-GPU local AI system, that's overkill.

nerd herd replaces this by:
1. **Collecting directly** from pynvml/psutil and llama-server's native `/metrics`
2. **Pre-computing rates** in ring buffers (~10KB memory vs 256MB Prometheus)
3. **Serving `/metrics`** in Prometheus text format — Grafana scrapes it directly using its built-in Prometheus datasource type

The trade-off: you lose ad-hoc PromQL queries like `rate(counter[30s])` vs `rate(counter[5m])`. nerd herd pre-computes 1-minute rolling averages. For a local AI system with one GPU, this is more than enough.

## Features

### GPU Monitoring
- **VRAM** — total, used, free (MB)
- **Utilization** — GPU compute percentage (0-100%)
- **Temperature** — Celsius, throttling detection (>85C)
- **Power** — watts draw
- **External process detection** — detects non-self GPU consumers (games, other apps)
- **System resources** — RAM available, CPU usage
- **2-second cache** — avoids hammering pynvml

### VRAM Budget Management
- **4 load modes** — full (100%), heavy (90%), shared (50%), minimal (0% local)
- **Auto-detect loop** — watches external GPU usage, adjusts mode automatically
- **Immediate downgrade** — when external processes grab VRAM
- **Delayed upgrade** — waits 5 minutes of sustained decrease before upgrading
- **Manual override** — user sets mode, auto-detect stops until re-enabled
- **Callback-driven** — `on_mode_change(callback)` for DB persistence, notifications, etc.

### Inference Metrics
- **llama-server native** — fetches from `/metrics` endpoint (Prometheus format)
- **Pre-computed rates** — generation tokens/sec, prompt tokens/sec (1-minute rolling average)
- **Pass-through gauges** — KV cache ratio, requests processing, requests pending
- **Graceful degradation** — reports zeros when llama-server is down

### Health Registry
- **Capability tracking** — mark services as healthy or degraded
- **Boot time** — tracks when the system started
- **No opinions** — you decide what capabilities to track

### Prometheus Exposition
- **HTTP server** — aiohttp on configurable port (default 9881)
- **`/metrics`** — Prometheus text exposition format
- **`/health`** — JSON liveness check
- **Reuse address** — survives crash→restart without port conflicts

### Collector Registry
- **Protocol-based** — any object with `collect()` and `prometheus_metrics()` can register
- **Custom collectors** — register your app's metrics alongside GPU/health/inference
- **Unified exposition** — all collectors served through single `/metrics` endpoint

## Install

```bash
pip install nerd-herd
```

## Quick Start

```python
from nerd_herd import NerdHerd

async def main():
    nh = NerdHerd(
        metrics_port=9881,
        llama_server_url="http://127.0.0.1:8080",
    )
    await nh.start()

    # Start GPU auto-detection
    await nh.start_auto_detect(notify_fn=my_notify)

    # Wire mode change callback
    nh.on_mode_change(lambda old, new, src: print(f"{old} -> {new}"))

    # Register custom metrics
    nh.register_collector("myapp", MyAppCollector())

    # ... your app runs ...

    await nh.stop()
```

```
# Grafana datasource config:
# Type: Prometheus
# URL: http://localhost:9881
```

## API Reference

### NerdHerd

```python
NerdHerd(
    metrics_port=9881,           # HTTP server port
    llama_server_url=None,       # llama-server URL (None = no inference metrics)
    detect_interval=30,          # GPU auto-detect poll seconds
    upgrade_delay=300,           # seconds before auto-upgrading mode
    initial_load_mode="full",    # starting load mode
    inference_poll_interval=5,   # llama-server poll seconds
)

# Lifecycle
await nh.start()                          # start metrics server + inference polling
await nh.start_auto_detect(notify_fn)     # start GPU auto-detect loop
await nh.stop()                           # stop everything

# GPU
nh.gpu_state() -> GPUState                # VRAM, temp, utilization

# VRAM budget
nh.get_vram_budget_mb() -> int            # free VRAM * budget fraction
nh.get_vram_budget_fraction() -> float    # 0.0 - 1.0
nh.get_load_mode() -> str                 # "full" | "heavy" | "shared" | "minimal"
nh.set_load_mode(mode, source="user")     # source="user" disables auto-detect
nh.enable_auto_management()               # re-enable auto-detect
nh.is_local_inference_allowed() -> bool   # False when minimal
nh.on_mode_change(callback)               # (old_mode, new_mode, source) -> None

# Health
nh.mark_degraded(capability)
nh.mark_healthy(capability)
nh.is_healthy(capability) -> bool
nh.get_health_status() -> HealthStatus

# Custom collectors
nh.register_collector(name, collector)

# Prometheus
nh.prometheus_lines() -> str              # text format for embedding
```

### Writing a Custom Collector

```python
class MyCollector:
    name = "myapp"

    def collect(self) -> dict:
        return {"requests_total": 42, "errors_total": 3}

    def prometheus_metrics(self) -> list:
        from prometheus_client import Gauge
        # Set gauge values, return gauge objects
        ...

nh.register_collector("myapp", MyCollector())
```

## Exposed Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nerd_herd_gpu_vram_used_mb` | gauge | GPU VRAM used (MB) |
| `nerd_herd_gpu_vram_free_mb` | gauge | GPU VRAM free (MB) |
| `nerd_herd_gpu_vram_total_mb` | gauge | GPU VRAM total (MB) |
| `nerd_herd_gpu_utilization_pct` | gauge | GPU utilization (0-100) |
| `nerd_herd_gpu_temperature_c` | gauge | GPU temperature (Celsius) |
| `nerd_herd_gpu_power_draw_w` | gauge | GPU power draw (watts) |
| `nerd_herd_gpu_external_vram_mb` | gauge | External process VRAM (MB) |
| `nerd_herd_gpu_external_processes` | gauge | External GPU process count |
| `nerd_herd_system_ram_available_mb` | gauge | System RAM available (MB) |
| `nerd_herd_system_cpu_percent` | gauge | CPU usage (%) |
| `nerd_herd_load_mode` | gauge | Load mode (0-3) |
| `nerd_herd_load_mode_info{mode}` | gauge | Load mode label |
| `nerd_herd_vram_budget_fraction` | gauge | VRAM budget (0.0-1.0) |
| `nerd_herd_auto_managed` | gauge | Auto-detect active (0/1) |
| `nerd_herd_capability_healthy{name}` | gauge | Capability health (0/1) |
| `nerd_herd_inference_tokens_per_sec` | gauge | Gen tokens/sec (1m avg) |
| `nerd_herd_inference_prompt_tokens_per_sec` | gauge | Prompt tokens/sec (1m avg) |
| `nerd_herd_inference_kv_cache_ratio` | gauge | KV cache usage ratio |
| `nerd_herd_inference_requests_processing` | gauge | Requests in progress |
| `nerd_herd_inference_requests_pending` | gauge | Requests queued |

## Architecture

```
  Your App            nerd herd                     Grafana
  ┌─────────┐        ┌──────────────────┐         ┌────────────┐
  │ DaLLaMa │──asks──│ LoadManager      │         │            │
  │ budget?  │       │  4 modes, auto   │         │  Dashboards│
  │          │       │  detect loop     │         │            │
  └─────────┘        ├──────────────────┤  GET    │            │
                     │ GPUCollector     │◄────────│  /metrics  │
  llama-server       │  pynvml, psutil  │  :9881  │            │
  ┌─────────┐        ├──────────────────┤         │            │
  │ /metrics│──poll──│ InferenceCollect │         └────────────┘
  │  :8080  │  5s    │  ring buffer     │
  └─────────┘        │  rate() precomp  │
                     ├──────────────────┤
  Your App           │ HealthRegistry   │
  ┌─────────┐        │  cap tracking    │
  │ mark_   │──────>├──────────────────┤
  │ degraded│        │ CollectorRegistry│
  └─────────┘        │  + custom colls  │
                     ├──────────────────┤
                     │ MetricsServer    │
                     │  aiohttp :9881   │
                     └──────────────────┘
```

## Dependencies

| Package | Purpose |
|---------|---------|
| yazbunu | Structured logging |
| pynvml | NVIDIA GPU metrics (graceful degradation if no GPU) |
| psutil | RAM/CPU metrics |
| prometheus_client | Metric types + text format |
| aiohttp | HTTP server + llama-server polling |

## License

MIT

---

<a id="turkce"></a>

## nerd herd nedir?

nerd herd, GPU destekli yapay zeka sistemleri icin bagimsiz bir gozlemlenebilirlik paketidir. GPU kaynaklarini izler, VRAM butce politikalarini yonetir, servis sagligini takip eder, llama-server'dan cikarim metriklerini toplar ve hepsini Prometheus uyumlu `/metrics` endpointi uzerinden sunar.

```python
from nerd_herd import NerdHerd

nh = NerdHerd(
    metrics_port=9881,
    llama_server_url="http://127.0.0.1:8080",
)
await nh.start()

# DaLLaMa sorar: "ne kadar VRAM kullanabilirim?"
butce_mb = nh.get_vram_budget_mb()  # 4000 (paylasimli mod, 8GB'nin %50'si)

# GPU durumu
durum = nh.gpu_state()
print(f"{durum.vram_free_mb}MB bos, {durum.temperature_c}C")
```

Sonra Grafana'yi `http://localhost:9881/metrics` adresine yonlendirin.

## Neden nerd herd?

- **Prometheus'a gerek yok** — 256MB'lik konteyner yerine ~20MB'lik ring buffer ile oran hesaplama
- **VRAM butce yonetimi** — 4 mod (full/heavy/shared/minimal), otomatik algilama
- **GPU koruması** — harici surecler (oyunlar, diger uygulamalar) GPU'yu kullandiginda otomatik geri cekilme
- **llama-server entegrasyonu** — yerel `/metrics` endpointinden token/saniye, KV cache, istek kuyrugu
- **Saglik takibi** — servislerin durumunu (calisiyor/bozuk) izleme
- **Toplayici kayit defteri** — kendi metriklerinizi GPU/saglik/cikarim yaninda kaydedin
- **Prometheus formati** — Grafana dogrudan kazir, ek arac gerekmez

## Ozellikler

### GPU Izleme
- **VRAM** — toplam, kullanilan, bos (MB)
- **Kullanim** — GPU hesaplama yuzdesi
- **Sicaklik** — Celsius, termal kisitlama tespiti (>85C)
- **Guc** — watt cekimi
- **Harici surec tespiti** — GPU'yu kullanan diger uygulamalari algilar
- **2 saniye onbellek** — pynvml'i yormaz

### VRAM Butce Yonetimi
- **4 yuk modu** — full (%100), heavy (%90), shared (%50), minimal (%0 yerel)
- **Otomatik algilama** — harici GPU kullanimini izler, modu otomatik ayarlar
- **Aninda dusurme** — harici surecler VRAM kaptiginda
- **Gecikmeli yukseltme** — 5 dakika surekli dusus bekler
- **Manuel kontrol** — kullanici modu ayarlar, otomatik algilama durur

### Cikarim Metrikleri
- **llama-server yerel** — `/metrics` endpointinden ceker
- **Onceden hesaplanmis oranlar** — uretim token/sn, prompt token/sn (1 dakika ortalama)
- **Gecis olcumleri** — KV cache orani, islenen/bekleyen istekler

### Prometheus Sunumu
- **HTTP sunucusu** — aiohttp, varsayilan port 9881
- **`/metrics`** — Prometheus metin formati
- **`/health`** — JSON canlilik kontrolu

## Kurulum

```bash
pip install nerd-herd
```

## Hizli Baslangic

```python
from nerd_herd import NerdHerd

async def main():
    nh = NerdHerd(
        metrics_port=9881,
        llama_server_url="http://127.0.0.1:8080",
    )
    await nh.start()

    # GPU otomatik algilamayi baslat
    await nh.start_auto_detect(notify_fn=bildirim_gonder)

    # Mod degisikliginde bildirim al
    nh.on_mode_change(lambda eski, yeni, kaynak: print(f"{eski} -> {yeni}"))

    # ... uygulamaniz calisir ...

    await nh.stop()
```

```
# Grafana veri kaynagi ayari:
# Tip: Prometheus
# URL: http://localhost:9881
```

## Lisans

MIT
