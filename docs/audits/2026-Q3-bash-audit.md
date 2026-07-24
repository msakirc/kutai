# Bash audit — 2026-Q3

_Quarterly check: what does each scaffolding layer do that bash + Claude can't? Mini-SWE-agent showed 65% SWE-bench in 100 LOC + bash — every layer below is on trial._

## Per-layer inventory

| Layer | Kind | LOC | Pub-syms | Tests | Deps | Last touched | Rationale |
|---|---|---:|---:|---:|---:|---|---|
| `c21_paraflow_diff` | package | 413 | 3 | 1 | 6 | 2026-05-10 | c21_paraflow_diff — Z1 Tier 7B (C21). |
| `clair_obscur` | package | 454 | 10 | 6 | 13 | 2026-06-24 | clair_obscur — local image-server wrapper. Parallel to dallama (LLM-side). |
| `coulson` | package | 6,170 | 98 | 25 | 40 | 2026-06-26 | Runtime — multi-call orchestration for LLM tasks. |
| `dallama` | package | 1,159 | 14 | 9 | 18 | 2026-06-14 | DaLLaMa — Python async llama-server process manager. |
| `db` | package | 7,280 | 180 | 2 | 17 | 2026-06-23 | **The DB engine.** Owns the SQLite layer for KutAI: the single long-lived |
| `dogru_mu_samet` | package | 240 | 9 | 5 | 10 | 2026-06-20 | Doğru mu Samet — detect degenerate LLM output. |
| `fatih_hoca` | package | 8,925 | 120 | 45 | 40 | 2026-06-26 | Fatih Hoca — model manager: scoring, selection, swap budget. |
| `finch` | package | 396 | 9 | 7 | 11 | 2026-06-26 | Finch — leaf package owning prompt/profile content + build API. |
| `general_beckman` | package | 13,124 | 122 | 69 | 35 | 2026-06-27 | General Beckman — the task master. |
| `hallederiz_kadir` | package | 1,325 | 7 | 6 | 21 | 2026-06-25 | HaLLederiz Kadir — LLM call execution hub. |
| `husam` | package | 428 | 2 | 10 | 16 | 2026-06-25 | husam (kumandan hüsamettin) — the non-agentic single-call worker. |
| `intersect` | package | 604 | 10 | 7 | 7 | 2026-06-24 | Intersect — thin per-task match+expose layer over the yalayut catalog. |
| `kara_kutu` | package | 595 | 11 | 1 | 16 | 2026-06-25 | kara_kutu — flight recorder. |
| `kuleden_donen_var` | package | 2,594 | 29 | 13 | 21 | 2026-06-25 | Kuleden Dönen Var — cloud LLM provider capacity tracker. |
| `mr_roboto` | package | 40,180 | 277 | 108 | 75 | 2026-06-27 | Mr. Roboto — mechanical dispatcher: non-LLM task executors. |
| `nerd_herd` | package | 3,200 | 63 | 42 | 25 | 2026-06-22 | Nerd Herd — standalone observability package. |
| `paintress` | package | 281 | 7 | 4 | 15 | 2026-06-08 | paintress — image interaction caller. Given hoca's pick, calls the provider, |
| `renoir` | package | 37 | 2 | 1 | 3 | 2026-06-07 | renoir — image quality judge. Parallel to dogru_mu_samet (text). |
| `sade_kalsin` | package | 455 | 9 | 3 | 10 | 2026-05-10 | Sade Kalsin (Turkish: "stay simple") — quarterly bash-audit harness. |
| `safety_guard` | package | 179 | 11 | 3 | 7 | 2026-05-18 | Pre-action safety guard: reversibility tag resolution + collision guards. |
| `src/agents` | src_module | 254 | 4 | 16 | 15 | 2026-06-26 | BaseAgent — profile interface for the runtime. |
| `src/app` | src_module | 19,737 | 111 | 35 | 58 | 2026-06-26 | Phase 12.1 — FastAPI REST API Server |
| `src/collaboration` | src_module | 407 | 9 | 0 | 7 | 2026-06-21 | Phase 13 — Agent Collaboration. |
| `src/comms` | src_module | 144 | 3 | 0 | 5 | 2026-06-08 | SP4b Plan 3 — crisis/incident/press_kit CPS producers (LLM out of mr_roboto). |
| `src/context` | src_module | 872 | 13 | 3 | 14 | 2026-06-16 | Intelligent code context assembly and repository mapping. |
| `src/core` | src_module | 2,344 | 42 | 22 | 40 | 2026-06-21 | Back-compat shim. |
| `src/founder_actions` | src_module | 440 | 12 | 5 | 6 | 2026-06-12 | Founder Actions — Z6 T1B real-world bridge queue. |
| `src/growth` | src_module | 423 | 9 | 3 | 4 | 2026-05-15 | Z9 Growth — post-launch analytics, hypothesis, and lifecycle tooling. |
| `src/infra` | src_module | 2,428 | 67 | 32 | 34 | 2026-06-25 | Phase 9.3 — Alerting |
| `src/integrations` | src_module | 1,603 | 17 | 8 | 23 | 2026-05-17 | External service integrations — base classes, registry, and HTTP driver. |
| `src/languages` | src_module | 166 | 8 | 0 | 7 | 2026-03-17 | Multi-language toolkit for coding pipeline quality (Phase 10.1). |
| `src/memory` | src_module | 3,957 | 77 | 3 | 28 | 2026-06-21 | Memory & Knowledge System — vector store, RAG, episodic memory. |
| `src/models` | src_module | 2,023 | 34 | 3 | 30 | 2026-06-14 | CLI tool for benchmark operations and registry diagnostics. |
| `src/ops` | src_module | 1,123 | 33 | 14 | 9 | 2026-05-19 | Z8 T4B — per-verb action cooldowns for the on-call agent. |
| `src/parsing` | src_module | 944 | 12 | 0 | 11 | 2026-03-26 | Multi-language code parsing — tree-sitter with regex/ast fallbacks. |
| `src/research` | src_module | 639 | 1 | 1 | 11 | 2026-06-15 | Research-domain orchestration (prior-art, idea validation). |
| `src/reviews` | src_module | 126 | 2 | 1 | 4 | 2026-06-05 | SP4b — reviews CPS producers (LLM extracted out of mr_roboto). |
| `src/runtime` | src_module | 63 | 0 | 1 | 1 | 2026-06-10 | Backward-compat shim — runtime relocated to packages/coulson/. |
| `src/security` | src_module | 1,121 | 26 | 7 | 20 | 2026-05-15 | Security utilities — sensitivity detection, data scanning. |
| `src/shopping` | src_module | 20,096 | 225 | 26 | 57 | 2026-06-16 | Product cache with SQLite and TTL-based invalidation. |
| `src/telemetry` | src_module | 187 | 3 | 2 | 4 | 2026-06-18 | Cross-cutting telemetry helpers (B10+ rework metric, future signals). |
| `src/tools` | src_module | 8,877 | 128 | 8 | 87 | 2026-06-16 | Tool registry — every tool an agent can invoke. |
| `src/util` | src_module | 138 | 4 | 1 | 2 | 2026-05-15 | src/util/lang.py — Z7 T2B Multilingual base utilities. |
| `src/workflows` | src_module | 7,116 | 101 | 29 | 28 | 2026-06-27 | Workflow engine package for loading, validating, and executing workflow defin... |
| `src/workspace` | src_module | 0 | 0 | 0 | 0 | — | — |
| `vecihi` | package | 274 | 10 | 2 | 10 | 2026-06-05 | Vecihi — auto-escalating web scraper. |
| `workflow_engine` | package | 174 | 2 | 0 | 7 | 2026-06-15 | > One job: take a workflow step that just finished and move the mission forward |
| `yalayut` | package | 4,877 | 122 | 13 | 31 | 2026-06-25 | Yalayut — vetted catalog of external skills, APIs, MCP servers. |
| `yasar_usta` | package | 2,086 | 24 | 10 | 31 | 2026-06-22 | Yaşar Usta — Telegram-controlled process manager. |

## Aggregate LOC by category

| Category | Layers | LOC | Tests |
|---|---:|---:|---:|
| package | 24 | 95,450 | 392 |
| src_module | 25 | 75,228 | 220 |

## Hot-spots (LOC x age x inverse-tests)

Top candidates for the four-question interrogation below.

| Rank | Layer | Score | LOC | Tests | Last touched |
|---:|---|---:|---:|---:|---|
| 1 | `src/parsing` | 8.68 | 944 | 0 | 2026-03-26 |
| 2 | `src/languages` | 6.61 | 166 | 0 | 2026-03-17 |
| 3 | `src/collaboration` | 6.18 | 407 | 0 | 2026-06-21 |
| 4 | `workflow_engine` | 5.39 | 174 | 0 | 2026-06-15 |
| 5 | `src/comms` | 5.28 | 144 | 0 | 2026-06-08 |
| 6 | `c21_paraflow_diff` | 3.44 | 413 | 1 | 2026-05-10 |
| 7 | `src/research` | 3.37 | 639 | 1 | 2026-06-15 |
| 8 | `kara_kutu` | 3.24 | 595 | 1 | 2026-06-25 |
| 9 | `db` | 3.03 | 7,280 | 2 | 2026-06-23 |
| 10 | `src/util` | 2.78 | 138 | 1 | 2026-05-15 |
| 11 | `src/reviews` | 2.59 | 126 | 1 | 2026-06-05 |
| 12 | `src/runtime` | 2.19 | 63 | 1 | 2026-06-10 |
| 13 | `src/memory` | 2.13 | 3,957 | 3 | 2026-06-21 |
| 14 | `vecihi` | 2.00 | 274 | 2 | 2026-06-05 |
| 15 | `src/models` | 1.99 | 2,023 | 3 | 2026-06-14 |

## The four audit questions

1. What does this layer do that bash + Claude can't?
2. Last time we changed this layer for a model-capability reason vs an integration reason — when?
3. If we deleted it tomorrow, what test would catch it?
4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?

## Per-hot-spot interrogation

### `src/parsing`

- **Path:** `src/parsing`
- **LOC:** 944  •  **Tests:** 0  •  **Last touched:** 2026-03-26
- **Rationale:** Multi-language code parsing — tree-sitter with regex/ast fallbacks.

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/parsing for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `src/languages`

- **Path:** `src/languages`
- **LOC:** 166  •  **Tests:** 0  •  **Last touched:** 2026-03-17
- **Rationale:** Multi-language toolkit for coding pipeline quality (Phase 10.1).

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/languages for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `src/collaboration`

- **Path:** `src/collaboration`
- **LOC:** 407  •  **Tests:** 0  •  **Last touched:** 2026-06-21
- **Rationale:** Phase 13 — Agent Collaboration.

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/collaboration for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `workflow_engine`

- **Path:** `packages/workflow_engine`
- **LOC:** 174  •  **Tests:** 0  •  **Last touched:** 2026-06-15
- **Rationale:** > One job: take a workflow step that just finished and move the mission forward

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed workflow_engine for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `src/comms`

- **Path:** `src/comms`
- **LOC:** 144  •  **Tests:** 0  •  **Last touched:** 2026-06-08
- **Rationale:** SP4b Plan 3 — crisis/incident/press_kit CPS producers (LLM out of mr_roboto).

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/comms for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_
