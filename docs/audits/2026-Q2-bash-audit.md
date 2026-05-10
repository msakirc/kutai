# Bash audit — 2026-Q2

_Quarterly check: what does each scaffolding layer do that bash + Claude can't? Mini-SWE-agent showed 65% SWE-bench in 100 LOC + bash — every layer below is on trial._

## Per-layer inventory

| Layer | Kind | LOC | Pub-syms | Tests | Deps | Last touched | Rationale |
|---|---|---:|---:|---:|---:|---|---|
| `c21_paraflow_diff` | package | 413 | 3 | 1 | 6 | — | c21_paraflow_diff — Z1 Tier 7B (C21). |
| `coulson` | package | 4,181 | 52 | 0 | 29 | 2026-05-10 | Runtime — multi-call orchestration for LLM tasks. |
| `dallama` | package | 1,035 | 13 | 7 | 18 | 2026-05-01 | DaLLaMa — Python async llama-server process manager. |
| `dogru_mu_samet` | package | 215 | 8 | 4 | 10 | 2026-04-17 | Doğru mu Samet — detect degenerate LLM output. |
| `fatih_hoca` | package | 6,966 | 74 | 19 | 31 | 2026-05-08 | Fatih Hoca — model manager: scoring, selection, swap budget. |
| `general_beckman` | package | 3,268 | 49 | 10 | 17 | 2026-05-09 | General Beckman — the task master. |
| `hallederiz_kadir` | package | 1,198 | 6 | 4 | 18 | 2026-05-10 | HaLLederiz Kadir — LLM call execution hub. |
| `kuleden_donen_var` | package | 2,126 | 20 | 10 | 18 | 2026-05-08 | Kuleden Dönen Var — cloud LLM provider capacity tracker. |
| `mr_roboto` | package | 10,478 | 76 | 34 | 31 | 2026-05-10 | Mr. Roboto — mechanical dispatcher: non-LLM task executors. |
| `nerd_herd` | package | 2,513 | 52 | 25 | 22 | 2026-05-08 | Nerd Herd — standalone observability package. |
| `sade_kalsin` | package | 455 | 9 | 3 | 10 | — | Sade Kalsin (Turkish: "stay simple") — quarterly bash-audit harness. |
| `safety_guard` | package | 50 | 2 | 1 | 3 | 2026-05-08 | Pre-action safety guard: reversibility tag resolution + collision guards. |
| `salako` | package | 0 | 0 | 0 | 0 | 2026-05-08 | — |
| `src/agents` | src_module | 2,022 | 23 | 2 | 33 | 2026-05-08 | Analyst agent — performs structured analysis, data interpretation, |
| `src/app` | src_module | 7,473 | 16 | 3 | 41 | 2026-05-10 | Phase 12.1 — FastAPI REST API Server |
| `src/collaboration` | src_module | 375 | 9 | 0 | 7 | 2026-03-26 | Phase 13 — Agent Collaboration. |
| `src/context` | src_module | 872 | 13 | 2 | 13 | 2026-04-07 | Intelligent code context assembly and repository mapping. |
| `src/core` | src_module | 3,977 | 51 | 10 | 36 | 2026-05-08 | Code review post-hook — LLM judges a build step's emitted code. |
| `src/infra` | src_module | 6,001 | 197 | 6 | 31 | 2026-05-10 | Forensic logger for admission-gate violations. |
| `src/integrations` | src_module | 394 | 4 | 1 | 12 | 2026-03-17 | External service integrations — base classes, registry, and HTTP driver. |
| `src/languages` | src_module | 166 | 8 | 0 | 7 | 2026-03-17 | Multi-language toolkit for coding pipeline quality (Phase 10.1). |
| `src/memory` | src_module | 3,720 | 72 | 0 | 25 | 2026-05-07 | Memory & Knowledge System — vector store, RAG, episodic memory. |
| `src/models` | src_module | 2,727 | 49 | 0 | 31 | 2026-05-07 | Auto-tuner: blends profile scores, benchmark scores, and empirical |
| `src/parsing` | src_module | 944 | 12 | 0 | 11 | 2026-03-26 | Multi-language code parsing — tree-sitter with regex/ast fallbacks. |
| `src/runtime` | src_module | 68 | 0 | 1 | 1 | 2026-05-04 | Backward-compat shim — runtime relocated to packages/coulson/. |
| `src/security` | src_module | 394 | 11 | 1 | 12 | 2026-04-18 | Security utilities — sensitivity detection, data scanning. |
| `src/shopping` | src_module | 20,100 | 225 | 25 | 56 | 2026-05-01 | Product cache with SQLite and TTL-based invalidation. |
| `src/telemetry` | src_module | 121 | 2 | 1 | 2 | 2026-05-09 | Cross-cutting telemetry helpers (B10+ rework metric, future signals). |
| `src/tools` | src_module | 7,672 | 113 | 1 | 80 | 2026-05-10 | Tool registry — every tool an agent can invoke. |
| `src/workflows` | src_module | 5,819 | 69 | 3 | 26 | 2026-05-10 | Workflow engine package for loading, validating, and executing workflow defin... |
| `src/workspace` | src_module | 0 | 0 | 0 | 0 | — | — |
| `vecihi` | package | 881 | 11 | 3 | 18 | 2026-05-10 | Vecihi — auto-escalating web scraper. |
| `workflow_engine` | package | 169 | 2 | 0 | 5 | 2026-04-25 | — |
| `yasar_usta` | package | 2,081 | 24 | 9 | 31 | 2026-05-02 | Yaşar Usta — Telegram-controlled process manager. |

## Aggregate LOC by category

| Category | Layers | LOC | Tests |
|---|---:|---:|---:|
| package | 16 | 36,029 | 130 |
| src_module | 18 | 62,845 | 56 |

## Hot-spots (LOC x age x inverse-tests)

Top candidates for the four-question interrogation below.

| Rank | Layer | Score | LOC | Tests | Last touched |
|---:|---|---:|---:|---:|---|
| 1 | `coulson` | 8.34 | 4,181 | 0 | 2026-05-10 |
| 2 | `src/memory` | 8.29 | 3,720 | 0 | 2026-05-07 |
| 3 | `src/models` | 7.96 | 2,727 | 0 | 2026-05-07 |
| 4 | `src/parsing` | 7.69 | 944 | 0 | 2026-03-26 |
| 5 | `src/collaboration` | 6.67 | 375 | 0 | 2026-03-26 |
| 6 | `src/languages` | 5.87 | 166 | 0 | 2026-03-17 |
| 7 | `workflow_engine` | 5.34 | 169 | 0 | 2026-04-25 |
| 8 | `c21_paraflow_diff` | 4.52 | 413 | 1 | — |
| 9 | `src/tools` | 4.47 | 7,672 | 1 | 2026-05-10 |
| 10 | `src/integrations` | 3.43 | 394 | 1 | 2026-03-17 |
| 11 | `src/security` | 3.16 | 394 | 1 | 2026-04-18 |
| 12 | `src/agents` | 2.54 | 2,022 | 2 | 2026-05-08 |
| 13 | `src/context` | 2.46 | 872 | 2 | 2026-04-07 |
| 14 | `src/telemetry` | 2.41 | 121 | 1 | 2026-05-09 |
| 15 | `sade_kalsin` | 2.30 | 455 | 3 | — |

## The four audit questions

1. What does this layer do that bash + Claude can't?
2. Last time we changed this layer for a model-capability reason vs an integration reason — when?
3. If we deleted it tomorrow, what test would catch it?
4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?

## Per-hot-spot interrogation

### `coulson`

- **Path:** `packages/coulson`
- **LOC:** 4,181  •  **Tests:** 0  •  **Last touched:** 2026-05-10
- **Rationale:** Runtime — multi-call orchestration for LLM tasks.

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed coulson for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `src/memory`

- **Path:** `src/memory`
- **LOC:** 3,720  •  **Tests:** 0  •  **Last touched:** 2026-05-07
- **Rationale:** Memory & Knowledge System — vector store, RAG, episodic memory.

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/memory for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

### `src/models`

- **Path:** `src/models`
- **LOC:** 2,727  •  **Tests:** 0  •  **Last touched:** 2026-05-07
- **Rationale:** Auto-tuner: blends profile scores, benchmark scores, and empirical

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/models for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_

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

### `src/collaboration`

- **Path:** `src/collaboration`
- **LOC:** 375  •  **Tests:** 0  •  **Last touched:** 2026-03-26
- **Rationale:** Phase 13 — Agent Collaboration.

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed src/collaboration for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_
