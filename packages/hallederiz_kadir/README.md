# hallederiz_kadir

**LLM call execution hub.**

*Handles everything between "I picked a model" and "here's the response" — litellm calls, streaming, retries, response parsing, quality checks, and metrics.*

[English](#english) | [Turkce](#turkce)

---

<a id="english"></a>

## What is HaLLederiz Kadir?

HaLLederiz Kadir is the LLM call execution layer. The dispatcher picks a model and says "call this." HaLLederiz Kadir takes it from there: builds the completion kwargs, acquires the GPU (local) or checks capacity (cloud), streams the response, retries on transient errors, parses the result, checks quality, and records metrics.

Named after a TV character who talked his way through everything. "Hallederiz" means "we'll handle it" — with the double-L nodding to LLM.

```python
from hallederiz_kadir import call, CallResult, CallError

result = await call(
    model=model_info,           # ModelInfo from router scoring
    messages=messages,          # Already redacted by dispatcher
    tools=tool_defs,            # Or None
    timeout=120.0,              # Computed by dispatcher
    task="shopping_advisor",    # For sampling profile + logging
    needs_thinking=True,        # Enable thinking for this call
    estimated_output_tokens=500,
)

if isinstance(result, CallResult):
    print(result.content)       # The response
    print(result.tool_calls)    # Tool calls, if any
    print(result.thinking)      # Thinking content, if any
    print(f"{result.latency:.1f}s, ${result.cost:.4f}")
else:
    # CallError — dispatcher tries next candidate
    print(f"{result.category}: {result.message}")
    if result.partial_content:
        print(f"Salvaged: {result.partial_content[:100]}...")
```

## Why HaLLederiz Kadir?

Before extraction, `router.py` was 1,700 lines doing two unrelated jobs: model scoring AND call execution. An agent fixing a retry bug could break scoring logic. HaLLederiz Kadir enforces the boundary — call execution lives here, scoring stays in the router.

## What It Does

### Call Execution
- Builds `completion_kwargs` from `ModelInfo` (sampling, api_base, api_key, tools, response_format)
- `litellm.acompletion()` with `asyncio.wait_for` timeout enforcement
- Local models: streaming with partial content accumulation
- Cloud models: direct completion

### Two Backend Paths
| Backend | Flow |
|---------|------|
| **Local** (llama-server) | DaLLaMa GPU acquire → health check → swap detection → stream → release |
| **Cloud** (Groq, OpenAI, etc.) | KDV pre_call → capacity check → completion → KDV post_call |

Caller doesn't know or care which path runs.

### Retry & Error Handling
- Per-model retry loop (2 retries local, 3 cloud)
- Error classification: `timeout`, `rate_limited`, `auth_failure`, `gpu_busy`, `daily_exhausted`, `loading`, `circuit_breaker`, `no_model`, `connection_error`, `server_error`, `quality_failure`
- Partial content salvage on timeout (streaming buffer preserved)
- Auth and daily_exhausted errors skip retries (not transient)

### Response Parsing
- Content, tool_calls, thinking extraction from litellm responses
- Think-tag stripping (`<think>...</think>`) when thinking not requested
- Rescue: reasoning_content promoted to content when content is empty
- Malformed tool call arguments handled gracefully (empty dict fallback)
- Cost calculation (0.0 for local, litellm.completion_cost for cloud)

### Quality Checks
- Degenerate output detection via Dogru mu Samet
- Streaming abort on repetitive content (monitored during generation)
- Quality failures return `CallError(category="quality_failure")` with the content for potential salvage

### Metrics & Observability
- Speed measurement: tok/s → model registry + Nerd Herd
- Prometheus metrics via `track_model_call_metrics`
- Audit trail recording
- Structured logging via Yazbunu

## API

### `call(model, messages, tools, timeout, task, needs_thinking, estimated_output_tokens) -> CallResult | CallError`

The only entry point. Dispatcher calls this once per candidate model.

### `CallResult`
```python
@dataclass
class CallResult:
    content: str                # Response text
    tool_calls: list[dict] | None
    thinking: str | None        # Thinking content (if enabled)
    usage: dict                 # prompt_tokens, completion_tokens
    cost: float                 # 0.0 for local
    latency: float              # seconds
    model: str                  # litellm name
    model_name: str             # human-readable name
    is_local: bool
    provider: str               # "local" or provider name
    task: str
```

### `CallError`
```python
@dataclass
class CallError:
    category: str               # error classification
    message: str
    retryable: bool             # hint for dispatcher's candidate loop
    partial_content: str | None # salvaged from streaming on timeout
```

## What It Is Not

- **Not a model selector** — doesn't know about scoring or candidates
- **Not a policy layer** — doesn't manage swap budgets or categorize calls
- **Not a process manager** — doesn't start/stop llama-server (DaLLaMa's job)
- **Not a capacity tracker** — doesn't track rate limits (KDV's job)

HaLLederiz Kadir is to LLM calls what DaLLaMa is to llama-server: a focused executor. DaLLaMa manages a process and yields a URL. HaLLederiz Kadir takes that URL (or a cloud model name) and handles the complete call lifecycle.

## Package Structure

```
packages/hallederiz_kadir/
  src/hallederiz_kadir/
    __init__.py      # exports: call, CallResult, CallError
    caller.py        # main call() — local/cloud routing, kwargs, streaming
    response.py      # response parsing, think-tag stripping, cost
    retry.py         # retry loop, error classification, partial salvage
    types.py         # CallResult, CallError dataclasses
  tests/
    test_caller.py
    test_response.py
    test_retry.py
    test_types.py
```

## Installation

```bash
pip install -e packages/hallederiz_kadir
```

Depends on `litellm>=1.40.0`.

---

<a id="turkce"></a>

## HaLLederiz Kadir nedir?

HaLLederiz Kadir, LLM cagri yurutme katmanidir. Dispatcher modeli secer ve "bunu cagir" der. HaLLederiz Kadir geri kalanini halleder: completion parametrelerini olusturur, GPU'yu alir (yerel) veya kapasiteyi kontrol eder (bulut), yaniti stream eder, gecici hatalarda tekrar dener, sonucu ayristirir, kalite kontrol yapar ve metrikleri kaydeder.

Ismi her seyi konusarak halleden bir TV karakterinin lakabindan gelir. "Hallederiz" icerisindeki buyuk "LL" harfleri LLM'e gonderme yapar.

### Ne yapar?

- **Cagri yurutme**: ModelInfo'dan completion kwargs olusturur, litellm.acompletion ile cagri yapar
- **Iki backend yolu**: Yerel modeller icin DaLLaMa GPU semafor → streaming; bulut icin KDV kapasite kontrol → completion
- **Yeniden deneme**: Yerel modellerde 2, bulutta 3 deneme; hata siniflandirma (timeout, rate_limit, auth, gpu_busy, vb.)
- **Yanit ayristirma**: Content, tool_calls, thinking cikarma; think-tag temizleme; maliyet hesaplama
- **Kalite kontrol**: Dogru mu Samet ile dejenere cikti tespiti; streaming sirasinda tekrar izleme
- **Metrikler**: tok/s olcum, Prometheus, audit trail, Yazbunu loglama

### Ne yapmaz?

- Model secmez (router'in isi)
- Swap butcesi yonetmez (dispatcher'in isi)
- llama-server sureci yonetmez (DaLLaMa'nin isi)
- Rate limit takip etmez (KDV'nin isi)

### Bagimliliklari

`litellm>=1.40.0`
