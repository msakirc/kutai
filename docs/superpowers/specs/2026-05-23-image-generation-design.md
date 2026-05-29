# Image Generation — `clair_obscur` + `paintress` + `renoir`

**Date:** 2026-05-23 (deep-dive 05-24/05-25)
**Status:** design approved; Plan 1 v2, Plan 2 v2, Plan 3 v2 written (2026-05-25). Plan 1 v2 must merge before Plans 2/3 (file overlap on dispatcher + beckman). Pre-existing 4-file working-tree diff (`record_verdict.py`, `llm_dispatcher.py`, `db.py`, `pick_log.py`) must be resolved before execution since Plan 1 v2 / Plan 2 v2 both edit `llm_dispatcher.py`.
**Strategic lock:** `project_z1_strategic_locks_20260509` #5 — image generation =
provider abstraction, parallel shape to existing wrappers, providers local + cloud,
selection scored on cost/latency/quality/VRAM, local preserves the M3 cost ceiling.

---

## 1. Goal

Real image generation for KutAI, replacing the placeholder images Z1 emits in HTML
prototypes. Two consumers: a `/image <prompt>` Telegram test surface, and an i2p
prototype-phase step that swaps placeholder `<img>` for generated assets.

## 2. Guiding principle — no invented surfaces

Image generation is **not** a special subsystem with its own brain or its own
control surfaces. It is **tasks flowing through the existing singular lifecycle**.
Before adding anything, map the need onto what already exists:

| Need | Owner (existing) |
|---|---|
| select provider (local/cloud, fit, cost, budget, vram) | `fatih_hoca` |
| observe/report system state (VRAM, in-flight, residency) | `nerd_herd` |
| admit **or hold** a task by urgency + pool pressure | `general_beckman` |
| cloud capacity / concurrency cap / rate-limit | `kuleden_donen_var` (kdv) + beckman admission |
| retry / reselect on failure | beckman availability-retry (failed pick in `failures`) |
| orchestrate ask→load→call (dumb pipe) | `LLMDispatcher` |

No bespoke hold-timers, escalation paths, GPU "lanes", semaphores, retry loops, or
rate-limiters. Holding a task in queue is normal; beckman re-evaluates every tick,
so stale picks self-correct. The only genuinely-new code is **three thin wrapper
packages** (each a parallel to an existing wrapper) and **one new scoring path in
hoca** for image-models.

## 3. The three new packages

Each wraps exactly one external thing, mirroring an existing thin wrapper.

### 3a. `clair_obscur` — local image-server wrapper (≈ `dallama`)
Wraps a local image-gen HTTP server — the image-world `llama-server`: **ComfyUI**
(`--listen`, `/prompt`) or **AUTOMATIC1111** (`--api`). Hands back a `base_url`.
```
packages/clair_obscur/src/clair_obscur/
  __init__.py    # public API
  server.py      # start / stop / status / health-poll / base_url
  config.py      # backend (comfyui|a1111), exe path, port, model, weights dir
```
- `available()` = backend installed + reachable (the VRAM-**fit** decision is hoca's, via nerd_herd — not a probe here).
- **PID-lock + boot reconcile** (mirrors dallama's stale-lock recovery): on `start()` write `image_server.lock` (PID); on boot, if a stale lock points at a live image-server process from a prior crash, kill it (safe — it is NOT llama-server) and verify `get_vram_free_mb` recovered. **Never touches llama-server's PID** (CLAUDE.md rule).
- MVP: **scaffolded**. Server lifecycle, config, health-poll, orphan-reconcile built and unit-tested against a mock server. No GPU process launched in CI. Lights up when the founder installs ComfyUI/A1111 + a footprint-friendly model (SDXL-Turbo / SD1.5, so it fits 8GB after the llama unload).

### 3b. `paintress` — image interaction caller (≈ `hallederiz_kadir`)
Given hoca's picked provider (cloud endpoint, or local `base_url`), makes the call,
validates, returns bytes/path + cost. **LLM-free.**
```
packages/paintress/src/paintress/
  __init__.py    # async generate(pick, spec) -> ImageResult
  types.py       # ImageSpec, ImageResult
  providers/
    base.py          # ImageProvider Protocol: name, available(), generate(spec, *, base_url=None)
    pollinations.py  # GET image.pollinations.ai/prompt/{prompt}?model=flux&seed=...
    huggingface.py   # POST HF serverless inference, FLUX.1-schnell (HF_TOKEN); handle 503 "model loading"
    local_server.py  # call clair_obscur's base_url (ComfyUI/A1111 API)
```
- routed by `pick.model.provider` (like HK routes by `model.provider`).
- **kdv pre/post** for cloud providers (rate-limit + in-flight, exactly as HK does).
- adapters **never raise** → map errors to `ImageResult.error` (same convention as `fatih_hoca/cloud/providers/base.py` and HK's `CallError`).
- calls `renoir.assess(bytes)` before returning; bad image → `CallError(retryable)`.
- **heartbeat keepalive** around the call (image gen is 10–60s+; the 300s no-progress watchdog must stay satisfied — same `heartbeat.keepalive()` wrap HK uses).
- writes the PNG under the **mission workspace** (`mission_{id}/assets/`, via `get_mission_workspace` — never repo root; cf. the marketing_copy leak). Result carries the **path, not bytes** (don't push MB through the task DB).
- sanitizes `filename_hint` (path-traversal) and URL-encodes the prompt (pollinations puts it in the URL).

### 3c. `renoir` — image quality judge (≈ `dogru_mu_samet`)
```
packages/renoir/src/renoir/
  __init__.py    # assess(bytes) -> ImageVerdict {ok, reason}
```
Catches the real free-provider failure mode (HTTP 200 with garbage): magic-bytes +
decodes-as-image + trivial heuristics (not all-one-color, min size/entropy). Room to
grow (perceptual/NSFW) later. Called by paintress, parallel to how HK calls
`dogru_mu_samet`.

## 4. `fatih_hoca` — the image scorer (the one new brain path)

A purpose-built scorer for image-models — sibling `fatih_hoca/image_select.py`, NOT
the 15-dim text engine bent onto images. `hoca.select(modality="image")` dispatches
to it.

**Dimensions** (drop text-only: thinking, fn-calling, context, tok/s):

| Dim | Static/dynamic | Source |
|---|---|---|
| cost ($/img, vs remaining budget) | static | provider profile |
| quality rank (per quality_tier) | static (hand-set MVP) | profile |
| reliability (recent success rate) | dynamic | success tracking (like kdv `recent_success_rate`) |
| latency (warm vs cold) | semi | profile + state |
| cloud capacity | dynamic | kdv |
| **eviction cost** (local only) | dynamic | **nerd_herd** |

**Pipeline** (mirrors hoca's shape): eligibility gate (provider available, cloud not
daily-exhausted, **local VRAM-fits per nerd_herd**, honors `failures`) → base score
→ state adjustment → argmax.

**Eviction-cost formula** (the novel core, reads nerd_herd):
```
if image_server_resident:          eviction = 0      # already warm → batch case
elif llm_in_flight > 0:            eviction = HUGE    # would stall live LLM work
elif llm_loaded or llm_queue > 0:  eviction = HIGH    # pipeline stall + reload cost
else (GPU idle):                   eviction = LOW     # clean take
```
Behavior falls out for free: cloud wins under LLM load, local wins idle, and once
the image server is warm `eviction=0` so a 10-placeholder batch does **1 eviction
then 9 cheap local gens** — emergent batching, no special batch logic.

Quality rank is hand-set per provider for MVP (no image-benchmark infra). Image
providers are **statically registered** in hoca's catalog with
`output_modality="image"` (they are NOT in cloud `/models` discovery); the
benchmark-enrichment pipeline must tolerate entries with no benchmark data.

## 5. Data taxonomy

**No `BaseModelInfo` subclassing.** Initial spec proposed shared base + Text/Image subclasses, but Plan 1 v2's recon confirmed `ModelInfo` has 4 required fields with no defaults at the top of its dataclass (`registry.py:53-56`). Any defaulted parent would crash compile with "non-default argument follows default argument." Cleanest: **`ImageModelInfo` is an independent `@dataclass`; the dispatcher branches on `isinstance(pick.model, ImageModelInfo)`.** Zero touch to the 40-field `ModelInfo` hot path; `Pick.model` stays loosely typed (already `model: object` per `fatih_hoca/types.py:9`).

```
@dataclass
class ImageModelInfo:
    name, provider, location, endpoint, api_base, quality_rank,
    cost_per_image, vram_mb, supports_seed, max_width, max_height,
    is_loaded, tier, litellm_name  # litellm_name carried for telemetry parity
    @property is_local              # location in ("local", "ollama")
    @property supports_image_generation  # always True; ModelInfo overrides False
```

`ImageSpec`: prompt, negative_prompt, width/height, steps, **seed: int|None** (None=random),
quality_tier ("fast"|"quality"), out_dir, filename_hint.
`ImageResult`: path, provider, model, cost, latency, **seed_used: int|None**, error.

`ImageSpec`: prompt, negative_prompt, width/height, steps, **seed: int|None** (None=random),
quality_tier ("fast"|"quality"), out_dir, filename_hint.
`ImageResult`: path, provider, model, cost, latency, **seed_used: int|None**, error.

## 6. GPU handover & lifecycle (generic mechanics)

- **No killing live LLM calls.** A local-image task waits for the current local-LLM task to finish (natural boundary). It's a normal queued task; beckman admits-or-holds by urgency + pool pressure.
- **Priority arbitrates the handover.** When the local slot frees, beckman admits the highest-priority waiting *local* task (text or image); a higher-prio image holds the slot against lower-prio LLM work. Cloud tasks (either modality) bypass the local slot entirely.
- **Dispatcher minor touch** on a local-image dispatch: `dallama.unload()` (then poll `get_vram_free_mb` until the image model fits) → `clair_obscur.start()` → `paintress.generate()`. dallama lazy-reloads on the next LLM task. The *decision* was beckman's; this is mechanical follow-through. Counts as **one swap** against hoca's swap budget (so per-image eviction would correctly trip the thrash guard — batching rewarded).
- **Warm across a batch:** beckman keeps clair_obscur warm while consecutive higher-prio image tasks run; on lane switch beckman calls `clair_obscur.record_release_hint()` (NOT direct `stop()`). The backstop in `ImageServer._arm_idle_backstop` then fires the actual `stop()` after `idle_release_seconds`. The dispatcher's idempotent `start()` clears the pending hint, so a back-to-back image arriving mid-window reuses the warm server. **No dead code** — the backstop is the normal-path stop trigger, not just safety.
- **Telemetry parity.** The image lane mirrors the LLM lane's full envelope inside `_dispatch_image`: `begin_call/end_call` (in-flight registry), `_record_pick` on success+failure (`model_pick_log`), `record_call_tokens` (zero tokens for image — row still writes for rollup), `record_call_cost`. All recon-confirmed generic (no LLM-only fields blocking). The local handover + `paintress.generate` are both wrapped in **`heartbeat.keepalive()`** so the 30-60s+ unload+poll+start window never trips the 300s watchdog.

## 7. Retry (generic)

Image task = single-shot. paintress failure → dispatcher surfaces
`CallError(retryable)` → beckman's availability-retry reselects via hoca (failed
provider added to `failures`; the image scorer's eligibility gate excludes it) →
re-dispatch the next provider. Shared backoff ladder + attempt cap, same as OVERHEAD
LLM. Full exhaustion → task fails → consumers degrade: i2p keeps the placeholder,
`/image` reports failure.

One plumbing note: an image task triggers a swap like MAIN_WORK but is single-shot
like OVERHEAD — `CallCategory` gets a third value, `IMAGE`.

**Shared win.** Plan 1 v2's audit + recon found the LLM path **also** lacked inter-task `failed_models` → `failures=` propagation across re-admissions. `on_task_finished` writes `task.context["failed_models"]` (`orchestrator.py:824-828`) but `next_task()` never read it back. Plan 1 v2 introduces `_select_for_admission(spec)` in beckman that reads `failed_models` and forwards it as `failures=` to `fatih_hoca.select(...)` — **benefits both text and image retries**. Without this, a re-admitted text task could re-pick the just-failed model; for image (single-shot, no ReAct loop) the gap was fatal.

## 8. Consumers

- **Prompt-writing task** (quality): a **beckman-admitted, full-lifecycle coulson task** (`prompt_writer` agent, pure-config single-call) reads design context (design tokens, screen/section plan, brand voice) and emits an enriched diffusion prompt per placeholder, scaffolded by **templates** + **few-shot exemplars** to help small/local LLMs. Output feeds the image tasks. (paintress stays LLM-free.) JSON shape enforced via **`artifact_schema` on the i2p step** → `workflow_engine.constrained_emit.maybe_apply` runs a post-emit structured pass with `response_format=json_schema` when the cheap-tier LLM emits malformed JSON. Without the artifact_schema, prompt_writer at `default_tier="cheap"` parses garbage too often to be useful.
- **`/image <prompt>`**: `beckman.enqueue` an image spec (no direct dispatcher call — `feedback_singular_dispatcher_caller`); reply with the photo. No dedicated rate-limit — beckman + kdv already pace the queue (single-user system); optional deep-queue warning.
- **i2p prototype swap**: a mr_roboto mechanical (`action == "swap_placeholder_images"`) **recursively** walks `mission_{id}/.web/**/*.html` (multi-screen prototypes) for placeholder `<img>` matching `^https?://placehold\.co/` (i2p_v3.json step 5.30a convention), enqueues a prompt-writing task then per-placeholder image tasks through beckman, writes assets under `mission_{id}/.web/assets/<placeholder_id>.png` (inside the web-preview served root), rewrites `src` to relative `assets/<id>.png`. Best-effort: per-placeholder failures keep the original `placehold.co` URL (`done_when` accepts `skipped_count > 0`). **Sibling `verify_swap_placeholder_images_shape` post-hook** (Z2/Z3 verify-shape pattern, mirrors `verify_charter_shape`) gates `emit_preview_url` — the preview URL only surfaces with a verifier-passed swap (asserts `replaced_count` agrees with disappeared `placehold.co` URLs, errors-margin for graceful degrade).
- **`coulson`**: untouched except as the runtime for the prompt-writing task (a normal LLM task).

## 9. Providers (MVP) — founder bar: cheapest / biggest free

| Provider | Tier | Key | Notes |
|---|---|---|---|
| Pollinations | default | none | unlimited free (rate-limited), zero signup; validate real image (200-with-garbage) |
| Hugging Face | quality | `HF_TOKEN` (free) | FLUX.1-schnell; handle 503 "model loading"; gated-model 403 → unavailable |
| local SDXL via `clair_obscur` | cost-ceiling | none | scaffolded; footprint-friendly model so it fits 8GB |

## 10. Config / secrets
`.env`: `HF_TOKEN` (HF tier off when absent), `KUTAI_IMAGE_CLOUD_BURST` opt-in flag,
`CLAIR_OBSCUR_BACKEND` (`comfyui|a1111`), URL/port, model, weights dir. Absent local
backend → `clair_obscur.available()` False → hoca filters it out, no crash.

## 11. Testing
- **Host-path tests** for the i2p step and `/image` (unit-green ≠ wired — the recurring lesson).
- `paintress` adapters against recorded responses (no live network in CI); `renoir` against fixture images (good/blank/garbage).
- `clair_obscur` against a mock HTTP server + simulated orphan lock (no GPU in CI).
- hoca image-scorer table tests incl. the eviction-cost matrix; dispatcher modality-branch test with mocked paintress/clair_obscur.

## 12. MVP scope vs deferred
**In MVP:** the 3 packages (clair_obscur scaffolded), hoca image scorer, kdv image
providers + concurrency cap, nerd_herd `image_server_resident` + `vram_mb` fields,
dispatcher modality branch + unload touch wrapped in `keepalive()`, **independent
`ImageModelInfo` + `isinstance` branching** (no BaseModelInfo refactor), Pollinations
+ HF cloud working, `prompt_writer` agent + diffusion template + `artifact_schema`
constrained-emit pass, `/image`, i2p `swap_placeholder_images` + `verify_swap_placeholder_images_shape` posthook, **shared beckman inter-task `failed_models` propagation** (text + image).
**Deferred:** real ComfyUI/A1111 install + live local generation + VRAM swap-war
validation; more cloud providers (Replicate/Together/Cloudflare); perceptual/NSFW in
renoir.

## 13. Risks / open
- **Local may be practically unusable on 8GB** (desktop+browser+Ollama overhead) — reinforces cloud-first; hoca's VRAM-fit gate handles it honestly.
- **Asset serving for preview**: the i2p swap's rewritten `src` must resolve in the web-preview host (now shipped — `project_web_preview_hosting_20260522`); wire `mission_{id}/assets/` into the preview root.
- **Placeholder marker convention**: confirm against the current prototype generator at impl time.
- **Decisions log:** D1 orphan=clair_obscur PID-lock · D2 VRAM-fit=hoca/nerd_herd · D3 cloud concurrency=kdv+beckman · D4 quality=`renoir` pkg · D5 prompts=beckman coulson task + templates · D6 seed=capture+expose, fresh default · D7 /image=no limiter · D8 held task=just a held task.
- **Naming:** `clair_obscur` / `paintress` / `renoir` — Expedition-33 / painter theme, overriding lock #5's "Turkish-named" (founder's call). Confirm `paintress` vs `the_paintress` at review.
