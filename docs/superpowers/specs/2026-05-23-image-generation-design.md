# Image Generation — `clair_obscur` + `paintress`

**Date:** 2026-05-23
**Status:** design approved (brainstorming), pending spec review → plan
**Strategic lock:** `project_z1_strategic_locks_20260509` lock #5 — "Image generation
= provider abstraction. New Turkish-named package, parallel shape to
hallederiz_kadir (call) + fatih_hoca (selection). Providers: local SDXL + cloud.
Selection scored on cost/latency/quality/VRAM. Local fallback preserves M3."

---

## 1. Goal

Give KutAI real image generation, replacing the placeholder images Z1 emits in
HTML prototypes. A founder-driven `/image <prompt>` test surface and an i2p
prototype-phase step that swaps placeholder `<img>` for generated assets.

The work is **not** a new subsystem with its own brain. It slots into the
existing singular architecture: one task selector (Beckman), one model/provider
selector (Fatih Hoca), one dumb-pipe dispatcher, and thin per-concern execution
wrappers (DaLLaMa / HaLLederiz Kadir / KDV). Image generation reuses every one
of those and adds **only the two execution wrappers that genuinely have no
image equivalent yet.**

> **Naming note:** lock #5 said "Turkish-named package". The founder overrode
> that here with an *Expedition 33* theme — `clair_obscur` (chiaroscuro =
> light/dark, the rendering engine) and `paintress` (the one who paints the
> image). Intentional, not a mistake.

## 2. Architecture — concern mapping

The LLM stack is decomposed by **concern**, and each box is a thin wrapper
around exactly one external thing. The dispatcher is a dumb pipe that wires
selection → URL → interaction:

| Concern | LLM (today) | Image (this work) |
|---|---|---|
| admit task | `general_beckman` | **reused** |
| select model/provider | `fatih_hoca` | **reused** — `output_modality="image"` filter |
| local backend = process + VRAM, hands back a URL | `dallama` (wraps llama-server) | **`clair_obscur`** — wraps ComfyUI/A1111 |
| cloud capacity / rate-limit state | `kuleden_donen_var` (KDV) | **reused** — extended with image providers |
| interaction = make the call, parse, return | `hallederiz_kadir` (wraps litellm) | **`paintress`** — custom HTTP per image provider |
| orchestrate (ask→load→call→retry) | `LLMDispatcher` | **reused** — gains a modality branch |

Two genuinely-new packages: **`clair_obscur`** (local image-server wrapper,
DaLLaMa-parallel) and **`paintress`** (interaction caller, HK-parallel).
Everything else is reuse.

### Why these two are separate (not one package)

DaLLaMa and HaLLederiz Kadir are separate for a reason: process/VRAM lifecycle
is a different concern from making a call. The same split holds for images. A
single merged package would re-create the "one box doing three jobs" anti-pattern.

### Why `paintress` is not just "extend HK"

HaLLederiz Kadir is a **litellm** wrapper. litellm's `image_generation` covers
OpenAI/Vertex/Bedrock but **not** the chosen free providers — Pollinations (a
plain GET URL) and Hugging Face serverless inference. Those need custom HTTP.
Bolting non-litellm HTTP onto HK would muddy its single concern, so the image
interaction caller is its own thin wrapper.

## 3. Request flow

```
caller (/image cmd, i2p step)
  → general_beckman.enqueue(image spec, runner="direct")     # admit
  → beckman.next_task(): fatih_hoca.select(modality="image")  # pick provider
  → orchestrator pump dispatches runner="direct" image task
  → LLMDispatcher.dispatch(spec)  [dumb pipe, modality=image branch]
       if pick.is_local:  clair_obscur.ensure_server() → URL on model.api_base
       if pick cloud:     (KDV pre-call gate happens inside paintress)
  → paintress.generate(pick, spec)  → ImageResult(bytes/path, cost, provider)
  → result flows back to caller (PNG written under mission workspace)
```

This is the **same lane** as the LLM `raw_dispatch` path, with `image_call` in
place of `llm_call` and `paintress.generate` in place of `hallederiz_kadir.call`.

## 4. Package: `clair_obscur` (local image-server wrapper, DaLLaMa-parallel)

**Single concern:** lifecycle of a local image-generation HTTP server, handing
back a URL. The image-world `llama-server` is **ComfyUI** (`--listen`,
`/prompt` queue API) or **AUTOMATIC1111** (`--api` → `/sdapi/v1/txt2img`).

```
packages/clair_obscur/src/clair_obscur/
  __init__.py        # public API
  server.py          # start / stop / health-poll / base_url
  config.py          # backend choice (comfyui|a1111), exe path, port, weights dir
  vram.py            # VRAM arbitration contract with DaLLaMa
```

Responsibilities:
- **start / stop / status / health-poll** the local image server process (mirror
  `dallama.DaLLaMa.start/stop/status`).
- **hand back `base_url`** so `paintress` can call it (mirror how DaLLaMa makes
  `model.api_base` exist).
- **VRAM arbitration**: SDXL (~6–7 GB) and llama-server fight for the same 8 GB
  GPU. `clair_obscur` does **not** own arbitration — the dispatcher does (it
  "acquires GPU slots" per its docstring). Before bringing the image server up,
  the dispatcher calls `dallama.stop()` to free llama-server VRAM; DaLLaMa lazily
  reloads on the next LLM `infer()`. `clair_obscur.vram` exposes the required-MB
  estimate so the dispatcher can decide.

**MVP scope:** scaffolded. `available()` returns `False` until a backend
(ComfyUI/A1111) + weights are installed and configured. Process management,
config, health-poll, and the VRAM contract are built and unit-tested against a
mock server; no real GPU process is launched in CI. Local generation lights up
when the founder installs a backend later. (Founder decision: scaffold-now,
install-later.)

## 5. Package: `paintress` (interaction caller, HK-parallel)

**Single concern:** given a picked provider (cloud endpoint, or local
`base_url` from `clair_obscur`), make the image call, parse, return bytes + cost.

```
packages/paintress/src/paintress/
  __init__.py        # async generate(pick, spec) -> ImageResult
  types.py           # ImageSpec, ImageResult
  providers/
    base.py          # ImageProvider Protocol: name, available(), async generate(spec, *, base_url=None)
    pollinations.py  # GET image.pollinations.ai/prompt/{prompt}?model=flux  (no key)
    huggingface.py   # POST HF serverless inference, FLUX.1-schnell (HF_TOKEN)
    local_server.py  # call clair_obscur's base_url (ComfyUI/A1111 API)
```

Mirrors `hallederiz_kadir.caller.call(model, ...)`:
- dispatched by `pick.provider` to the right adapter (like HK routes by
  `model.provider`).
- **KDV pre/post** for cloud providers — `paintress` calls `kdv.pre_call` /
  `record_attempt` / `post_call` exactly as HK does (rate-limit gating lives in
  KDV, not here).
- adapters **never raise** — map errors to `ImageResult.error` (same convention
  as `fatih_hoca/cloud/providers/base.py` and HK's `CallError`).
- returns `ImageResult` { `bytes`/`path`, `provider`, `model`, `cost`,
  `latency`, `error` }.

**No selection, no rate-tracking state, no server management** inside `paintress`.

### `ImageSpec`
`prompt: str`, `negative_prompt: str|None`, `width/height: int`,
`steps: int|None`, `seed: int|None`, `quality_tier: "fast"|"quality"`,
`out_dir: str` (mission workspace), `filename_hint: str|None`.

## 6. Reused-box extensions

### `fatih_hoca` — selection (no new brain)
- `fatih_hoca/cloud/types.py` already carries `output_modality: "text"|"image"`;
  cloud discovery already tags image models (`gemini._infer_modality`).
- Register the MVP image providers as catalog entries with
  `output_modality="image"` and cost/latency/quality/vram profiles:
  - `pollinations/flux` — cloud, no key, default.
  - `huggingface/FLUX.1-schnell` — cloud, `HF_TOKEN`, quality tier.
  - local SDXL (`is_local=True`, served by `clair_obscur`) — registered but
    `available()`-gated (scaffold).
- Add `modality` to requirements + **one eligibility line** mirroring the
  existing `needs_vision` filter (`selector.py:459`): keep only image entries
  when `reqs.modality == "image"`. Existing scoring (cost/speed/utilization/
  stickiness) already covers cost/latency/vram — no new scoring math.
- `select(modality="image", ...)` returns a `Pick` whose `model` is an image
  provider.

### `kuleden_donen_var` (KDV) — cloud rate-limits
- Extend provider tracking to Pollinations + HF so `paintress`'s pre/post calls
  gate image cloud capacity the same way LLM cloud is gated. Pollinations has no
  documented quota — treat as best-effort/no-limit but still route through KDV
  for uniform in-flight accounting.

### `LLMDispatcher` — dumb pipe, modality branch
- `dispatch(spec)` reads `context.image_call` (parallel to `context.llm_call`).
- On the image branch: if `pick.is_local`, call `clair_obscur.ensure_server()`
  (which may trigger `dallama.stop()` for VRAM) and set `model.api_base`; then
  call `paintress.generate(pick, spec)` instead of `hallederiz_kadir.call`.
- `pick_log` still fires (record provider + success/fail), reusing the existing
  `_record_pick` path so image-selection telemetry lands like LLM picks.
- Retry stays where it already is (no new retry surface in either new package).

## 7. Consumers

### `/image <prompt>` (Telegram)
Enqueues an image spec via `beckman.enqueue` (honors
`feedback_singular_dispatcher_caller` — no direct dispatcher call), then replies
with the generated photo. Direct test surface, decoupled from i2p.

### i2p prototype placeholder swap (mr_roboto mechanical)
- New mr_roboto executor (`action == "swap_placeholder_images"`): scans the
  prototype-phase HTML for placeholder `<img>` (convention to be confirmed
  against the current prototype generator — likely `via.placeholder.*` src or a
  `data-gorsel`/`data-image` marker), generates one image per placeholder using
  its `alt` / surrounding section text as the prompt, writes assets under
  `mission_{id}/assets/`, and rewrites `src`.
- Each individual generation goes through the **dispatcher image lane** (enqueue
  → admit → select → paintress), not a mechanical→dispatcher shortcut. The
  mechanical orchestrates scan + rewrite; generation stays on the singular lane.
- Wired as a mechanical step in the i2p prototype phase.

### `coulson`
Untouched — raw image generation has no ReAct loop or prompt intelligence.

## 8. Providers (MVP)

| Provider | Tier | Key | Notes |
|---|---|---|---|
| Pollinations | default | none | `image.pollinations.ai/prompt/{p}?model=flux`, unlimited free (rate-limited), zero signup |
| Hugging Face | quality | `HF_TOKEN` (free) | FLUX.1-schnell serverless inference; better quality/reliability when token present |
| local SDXL via `clair_obscur` | cost-ceiling | none | scaffolded; lights up after ComfyUI/A1111 install |

Founder decision: cheapest/biggest-free. Pollinations is the literal biggest
free plan; HF is the free-token quality upgrade; local SDXL is the M3
cost-ceiling fallback.

## 9. Config / secrets

- `.env`: `HF_TOKEN` (optional — HF tier off when absent), `KUTAI_IMAGE_CLOUD_BURST`
  flag (opt-in cloud), `CLAIR_OBSCUR_BACKEND` (`comfyui|a1111`),
  `CLAIR_OBSCUR_URL`/port, weights dir.
- Absent local backend → `clair_obscur.available()` is `False`; hoca filters it
  out; no crash.

## 10. Testing

- **Host-path tests** for the i2p step and `/image` (recurring lesson: unit-green
  ≠ wired; the unit suites have passed *with* real bugs before).
- `paintress` adapters tested against recorded provider responses (no live
  network in CI).
- `clair_obscur` tested against a mock HTTP server (no GPU process in CI).
- hoca modality-filter table tests; dispatcher modality-branch test with mocked
  `paintress`/`clair_obscur`.
- PNG-exists / min-bytes assertion on generated assets.

## 11. MVP scope vs deferred

**In MVP:** both packages; hoca modality filter; KDV image-provider extension;
dispatcher modality branch; Pollinations (default) + HF (upgrade) working
cloud generation; `/image` command; i2p placeholder-swap step; `clair_obscur`
scaffolded (stub).

**Deferred:** real ComfyUI/A1111 install + live local generation + live VRAM
swap-war validation; additional cloud providers (Replicate/Together/Cloudflare);
LLM prompt-enrichment as a separate agent step (keeps `paintress` LLM-free).

## 12. Risks / open questions

- **VRAM swap-war** (local): stopping llama-server for each image generation is
  slow and disruptive. Deferred with local scaffold, but the dispatcher
  arbitration contract must be designed now so it's correct when local lands.
- **Placeholder convention**: the exact marker the prototype generator emits
  must be confirmed during implementation (the swap step depends on it).
- **Pollinations reliability**: public free service; HF token tier is the
  reliability hedge. Both behind the same `paintress` interface.
- **Package naming**: `paintress` module = "The Paintress"; `clair_obscur`
  module = "Clair-Obscur" (hyphen invalid in Python identifiers → underscore).
  Confirm `paintress` vs `the_paintress` at review.
```
