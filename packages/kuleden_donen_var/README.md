# kuleden_donen_var

**Cloud LLM provider capacity tracker.**

*Zero dependencies. Tracks rate limits, quotas, and circuit breakers across cloud providers. Tells you who has capacity.*

[English](#english) | [Turkce](#turkce)

---

<a id="english"></a>

## What is Kuleden Donen Var?

Kuleden Donen Var (KDV) is a dumb pipe for cloud LLM provider health. It tracks rate limits, quotas, and circuit breakers across multiple cloud providers and fires a callback whenever capacity changes. It doesn't discover models, pick models, or make LLM calls — it just answers "can I call this provider?" and "what changed?"

```python
from kuleden_donen_var import KuledenDonenVar, KuledenConfig

def on_change(event):
    print(f"{event.provider}: {event.event_type}")
    print(f"  utilization: {event.snapshot.utilization_pct:.0f}%")

kdv = KuledenDonenVar(KuledenConfig(on_capacity_change=on_change))

# Host registers models at startup
kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
kdv.register("openai/gpt-4o", "openai", rpm=500, tpm=2000000)

# Before calling a cloud model
result = kdv.pre_call("groq/llama-8b", "groq", estimated_tokens=5000)
if result.allowed:
    # make the call...
    kdv.post_call("groq/llama-8b", "groq", headers=response_headers, token_count=3200)
elif result.daily_exhausted:
    # pick a different model
    pass
else:
    # wait result.wait_seconds or pick another model
    pass

# On failure
kdv.record_failure("groq/llama-8b", "groq", "rate_limit")  # adaptive limit reduction
kdv.record_failure("openai/gpt-4o", "openai", "server_error")  # circuit breaker

# Check status anytime
for provider, status in kdv.status.items():
    print(f"{provider}: {status.utilization_pct:.0f}% used, cb={status.circuit_breaker_open}")
```

## Why KDV?

KDV exists to enforce a hard package boundary. Without it, agents fixing a shopping bug could reach into rate limiter internals, or an agent investigating a timeout could modify circuit breaker logic. Package extraction makes wrong fixes impossible.

## What It Does

### Rate Limiting
- **Two-tier**: per-model AND per-provider aggregate limits
- **RPM + TPM + RPD**: requests per minute, tokens per minute, requests per day
- **Adaptive 429 handling**: automatically reduces limits when getting rate limited, gradually restores after 10 minutes of no 429s
- **Header-derived limits**: parses provider response headers to discover real limits (overrides configured defaults)

### Provider Support
| Provider | Header Format | Daily Limits |
|----------|--------------|--------------|
| OpenAI | `x-ratelimit-*` | No |
| Groq | `x-ratelimit-*` | No |
| Anthropic | `anthropic-ratelimit-*` | No |
| Gemini | `x-ratelimit-*` | Yes (RPD) |
| Cerebras | `x-ratelimit-*-minute/day` | Yes (RPD) |
| SambaNova | `x-ratelimit-*-minute/day` | Yes (RPD) |

New providers automatically fall back to OpenAI-style header parsing.

### Circuit Breaker
- Per-provider failure tracking with sliding window
- Configurable threshold (default: 3 failures in 5 minutes)
- Automatic cooldown (default: 10 minutes)
- Resets on first successful call
- Cloud-only — DaLLaMa has its own circuit breaker for local models

### Capacity Events
The `on_capacity_change` callback fires on:

| Event | When |
|-------|------|
| `capacity_restored` | Response headers show significant headroom improvement |
| `limit_hit` | 429 received, limits adaptively reduced |
| `circuit_breaker_tripped` | Provider disabled after repeated failures |
| `circuit_breaker_reset` | Provider re-enabled after successful call |
| `daily_exhausted` | Daily request limit reached |

Each event includes a full `ProviderStatus` snapshot with utilization %, remaining counts, and reset timers.

## API

### `KuledenDonenVar(config)`

Main class. Compose with `KuledenConfig`:

```python
KuledenConfig(
    circuit_breaker_threshold=3,        # failures before tripping
    circuit_breaker_window_seconds=300,  # sliding window for failures
    circuit_breaker_cooldown_seconds=600,  # cooldown after trip
    on_capacity_change=callback,        # fires on state changes
)
```

### `register(model_id, provider, rpm, tpm, ...)`

Register a cloud model. Host pushes config — KDV never pulls from a registry.

### `pre_call(model_id, provider, estimated_tokens=0) -> PreCallResult`

Non-blocking capacity check. Returns:
- `allowed` — safe to call
- `wait_seconds` — if not allowed, estimated wait (0 if circuit breaker)
- `daily_exhausted` — daily limit hit, skip this model entirely

### `post_call(model_id, provider, headers, token_count)`

Record a successful call. Parses response headers, updates limits, records RPM/TPM usage, resets circuit breaker.

### `record_failure(model_id, provider, error_type)`

Record a failed call. Error types:
- `"rate_limit"` — adaptive limit reduction
- `"server_error"` — circuit breaker failure count
- `"timeout"` — circuit breaker failure count
- `"auth"` — ignored (permanent, not transient)

### `restore_limits()`

Gradually restore adaptive limit reductions. Called periodically by the host watchdog.

### `status -> dict[str, ProviderStatus]`

Current state of all providers. Each `ProviderStatus` contains:
- `circuit_breaker_open`, `utilization_pct`, `rpm/tpm/rpd_remaining`, `reset_in_seconds`
- `models: dict[str, ModelStatus]` — per-model breakdown

## What It Is Not

- **Not a model registry** — doesn't know what models exist
- **Not a router** — doesn't pick which model to use
- **Not an LLM client** — doesn't make litellm calls
- **Not DaLLaMa** — doesn't manage local models or GPU

KDV is to cloud providers what DaLLaMa is to llama-server: a backend manager. DaLLaMa manages a process and yields a URL. KDV manages capacity state and yields go/no-go decisions.

## Installation

```bash
pip install -e packages/kuleden_donen_var
```

Zero dependencies — stdlib only.

---

<a id="turkce"></a>

## Kuleden Donen Var nedir?

Kuleden Donen Var (KDV), bulut LLM saglayicilarinin kapasite durumunu takip eden bir pakettir. Rate limit, kota ve circuit breaker yonetimini tek bir arayuzde toplar. Saglayicilarin kapasitesi degistiginde bir callback ile bildirir.

Modelleri kesfetmez, model secmez, LLM cagrisi yapmaz. Sadece "bu saglayiciyi cagirabilir miyim?" sorusuna cevap verir ve "ne degisti?" bilgisini iletir.

Ismi havaalanlarindaki "kuleden donen var" anonsu gibidir: bir ucak piste inmek icin musait oldugunda kule haber verir. KDV de ayni sekilde bir bulut saglayicinin kapasitesi acildiginda host uygulamaya haber verir.

### Ne yapar?

- **Rate limiting**: Model basina ve saglayici basina iki katmanli limit takibi (RPM, TPM, RPD)
- **Adaptif 429 yonetimi**: Rate limit yediginde limitleri otomatik dusurur, 10 dakika sorunsuz gecince kademeli geri yukselir
- **Header parsing**: API yanit header'larindan gercek limitleri okur (OpenAI, Anthropic, Groq, Gemini, Cerebras, SambaNova)
- **Circuit breaker**: Saglayici basina hata takibi, esik asildinda gecici devre disi birakma
- **Kapasite olaylari**: `on_capacity_change` callback'i ile kapasite degisikliklerini anlik bildirir

### Ne yapmaz?

- Model kesfetmez (registry'nin isi)
- Model secmez (router'in isi)
- LLM cagrisi yapmaz (HaLLederiz Kadir'in isi)
- Yerel model yonetmez (DaLLaMa'nin isi)

### Kullanim

```python
from kuleden_donen_var import KuledenDonenVar, KuledenConfig

kdv = KuledenDonenVar(KuledenConfig(
    on_capacity_change=lambda evt: print(f"{evt.provider}: {evt.event_type}")
))

# Modelleri kaydet
kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)

# Cagri oncesi kontrol
sonuc = kdv.pre_call("groq/llama-8b", "groq")
if sonuc.allowed:
    # cagiriver...
    kdv.post_call("groq/llama-8b", "groq", headers=yanitlar, token_count=3200)

# Hata durumunda
kdv.record_failure("groq/llama-8b", "groq", "rate_limit")

# Durum sorgusu
for saglayici, durum in kdv.status.items():
    print(f"{saglayici}: %{durum.utilization_pct:.0f} kullanildi")
```

### Bagimliliklari

Sifir. Sadece Python stdlib kullanir.
