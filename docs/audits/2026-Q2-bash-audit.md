# Bash audit — 2026-Q2

_Quarterly check: what does each scaffolding layer do that bash + Claude can't? Mini-SWE-agent showed 65% SWE-bench in 100 LOC + bash — every layer below is on trial._

## Per-layer inventory

| Layer | Kind | LOC | Pub-syms | Tests | Deps | Last touched | Rationale |
|---|---|---:|---:|---:|---:|---|---|
| `c21_paraflow_diff` | package | 413 | 3 | 1 | 6 | 2026-05-10 | c21_paraflow_diff — Z1 Tier 7B (C21). |
| `coulson` | package | 4,864 | 60 | 3 | 32 | 2026-05-12 | Runtime — multi-call orchestration for LLM tasks. |
| `dallama` | package | 1,035 | 13 | 7 | 18 | 2026-05-01 | DaLLaMa — Python async llama-server process manager. |
| `dogru_mu_samet` | package | 215 | 8 | 4 | 10 | 2026-04-17 | Doğru mu Samet — detect degenerate LLM output. |
| `fatih_hoca` | package | 6,983 | 74 | 20 | 31 | 2026-05-11 | Fatih Hoca — model manager: scoring, selection, swap budget. |
| `general_beckman` | package | 6,133 | 64 | 16 | 21 | 2026-05-12 | General Beckman — the task master. |
| `hallederiz_kadir` | package | 1,235 | 6 | 4 | 18 | 2026-05-12 | HaLLederiz Kadir — LLM call execution hub. |
| `kuleden_donen_var` | package | 2,147 | 21 | 10 | 19 | 2026-05-10 | Kuleden Dönen Var — cloud LLM provider capacity tracker. |
| `mr_roboto` | package | 21,139 | 133 | 56 | 49 | 2026-05-12 | Mr. Roboto — mechanical dispatcher: non-LLM task executors. |
| `nerd_herd` | package | 2,513 | 52 | 25 | 22 | 2026-05-08 | Nerd Herd — standalone observability package. |
| `sade_kalsin` | package | 455 | 9 | 3 | 10 | 2026-05-10 | Sade Kalsin (Turkish: "stay simple") — quarterly bash-audit harness. |
| `safety_guard` | package | 50 | 2 | 1 | 3 | 2026-05-08 | Pre-action safety guard: reversibility tag resolution + collision guards. |
| `salako` | package | 0 | 0 | 0 | 0 | 2026-05-08 | — |
| `src/agents` | src_module | 2,259 | 26 | 7 | 36 | 2026-05-12 | Analyst agent — performs structured analysis, data interpretation, |
| `src/app` | src_module | 10,099 | 30 | 19 | 46 | 2026-05-12 | Phase 12.1 — FastAPI REST API Server |
| `src/collaboration` | src_module | 375 | 9 | 0 | 7 | 2026-03-26 | Phase 13 — Agent Collaboration. |
| `src/context` | src_module | 872 | 13 | 2 | 13 | 2026-04-07 | Intelligent code context assembly and repository mapping. |
| `src/core` | src_module | 4,049 | 51 | 11 | 36 | 2026-05-12 | Code review post-hook — LLM judges a build step's emitted code. |
| `src/founder_actions` | src_module | 464 | 11 | 3 | 5 | 2026-05-12 | Founder Actions — Z6 T1B real-world bridge queue. |
| `src/infra` | src_module | 9,526 | 253 | 25 | 34 | 2026-05-12 | Forensic logger for admission-gate violations. |
| `src/integrations` | src_module | 774 | 9 | 8 | 18 | 2026-05-12 | External service integrations — base classes, registry, and HTTP driver. |
| `src/languages` | src_module | 166 | 8 | 0 | 7 | 2026-03-17 | Multi-language toolkit for coding pipeline quality (Phase 10.1). |
| `src/memory` | src_module | 3,941 | 77 | 1 | 25 | 2026-05-12 | Memory & Knowledge System — vector store, RAG, episodic memory. |
| `src/models` | src_module | 2,727 | 49 | 0 | 31 | 2026-05-07 | Auto-tuner: blends profile scores, benchmark scores, and empirical |
| `src/ops` | src_module | 876 | 29 | 11 | 8 | 2026-05-12 | Z8 T4B — per-verb action cooldowns for the on-call agent. |
| `src/parsing` | src_module | 944 | 12 | 0 | 11 | 2026-03-26 | Multi-language code parsing — tree-sitter with regex/ast fallbacks. |
| `src/runtime` | src_module | 68 | 0 | 1 | 1 | 2026-05-04 | Backward-compat shim — runtime relocated to packages/coulson/. |
| `src/security` | src_module | 1,043 | 25 | 7 | 20 | 2026-05-11 | Security utilities — sensitivity detection, data scanning. |
| `src/shopping` | src_module | 20,100 | 225 | 25 | 56 | 2026-05-01 | Product cache with SQLite and TTL-based invalidation. |
| `src/telemetry` | src_module | 121 | 2 | 1 | 2 | 2026-05-09 | Cross-cutting telemetry helpers (B10+ rework metric, future signals). |
| `src/tools` | src_module | 8,511 | 125 | 7 | 84 | 2026-05-12 | Tool registry — every tool an agent can invoke. |
| `src/workflows` | src_module | 6,427 | 76 | 11 | 27 | 2026-05-12 | Workflow engine package for loading, validating, and executing workflow defin... |
| `src/workspace` | src_module | 0 | 0 | 0 | 0 | — | — |
| `vecihi` | package | 911 | 11 | 3 | 19 | 2026-05-12 | Vecihi — auto-escalating web scraper. |
| `workflow_engine` | package | 177 | 2 | 0 | 5 | 2026-05-10 | — |
| `yasar_usta` | package | 2,081 | 24 | 9 | 31 | 2026-05-02 | Yaşar Usta — Telegram-controlled process manager. |

## Aggregate LOC by category

| Category | Layers | LOC | Tests |
|---|---:|---:|---:|
| package | 16 | 50,351 | 162 |
| src_module | 20 | 73,342 | 139 |

## Hot-spots (LOC x age x inverse-tests)

Top candidates for the four-question interrogation below.

| Rank | Layer | Score | LOC | Tests | Last touched |
|---:|---|---:|---:|---:|---|
| 1 | `src/models` | 8.04 | 2,727 | 0 | 2026-05-07 |
| 2 | `src/parsing` | 7.76 | 944 | 0 | 2026-03-26 |
| 3 | `src/collaboration` | 6.74 | 375 | 0 | 2026-03-26 |
| 4 | `src/languages` | 5.93 | 166 | 0 | 2026-03-17 |
| 5 | `workflow_engine` | 5.22 | 177 | 0 | 2026-05-10 |
| 6 | `src/memory` | 4.15 | 3,941 | 1 | 2026-05-12 |
| 7 | `c21_paraflow_diff` | 3.05 | 413 | 1 | 2026-05-10 |
| 8 | `src/context` | 2.49 | 872 | 2 | 2026-04-07 |
| 9 | `src/telemetry` | 2.44 | 121 | 1 | 2026-05-09 |
| 10 | `src/runtime` | 2.17 | 68 | 1 | 2026-05-04 |
| 11 | `coulson` | 2.13 | 4,864 | 3 | 2026-05-12 |
| 12 | `safety_guard` | 1.99 | 50 | 1 | 2026-05-08 |
| 13 | `vecihi` | 1.71 | 911 | 3 | 2026-05-12 |
| 14 | `sade_kalsin` | 1.55 | 455 | 3 | 2026-05-10 |
| 15 | `src/founder_actions` | 1.54 | 464 | 3 | 2026-05-12 |

## The four audit questions

1. What does this layer do that bash + Claude can't?
2. Last time we changed this layer for a model-capability reason vs an integration reason — when?
3. If we deleted it tomorrow, what test would catch it?
4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?

## Per-hot-spot interrogation

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

### `workflow_engine`

- **Path:** `packages/workflow_engine`
- **LOC:** 177  •  **Tests:** 0  •  **Last touched:** 2026-05-10
- **Rationale:** —

  1. What does this layer do that bash + Claude can't?
     - _answer:_
  2. Last time we changed workflow_engine for a model-capability reason vs an integration reason — when?
     - _answer:_
  3. If we deleted it tomorrow, what test would catch it?
     - _answer:_
  4. Is the abstraction earning its keep, or did we lock in 2024-era constraints?
     - _answer:_
