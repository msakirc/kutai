# Package README Standard — Design

**Date:** 2026-05-29
**Status:** Approved (guideline doc, no enforcement tooling)
**Scope:** All sub-packages under `packages/` except `c21_paraflow_diff` (paraflow),
`sade_kalsin`, and `safety_guard`.

## Goal

A single written standard that defines what a good package README is — what must
exist, what must not, and how the bilingual (English + Turkish) structure works.
This is a **guideline**, not a linted or test-enforced template. It relies on
discipline plus the good/bad examples below, not CI.

## Audience & Purpose

A package README serves two readers:

1. **Future founder** returning to a package months later.
2. **Claude subagents** doing subagent-driven development on the package.

The README is the package's **contract surface**. A reader should be able to use
or modify the package without reading all of its internals. Concretely, it answers
the three isolation questions:

- **What does it do?**
- **How do you use it?**
- **What does it depend on?**

The core sections map directly to those questions.

## File Layout — Bilingual, Single File

One `README.md` per package. English first, then a horizontal rule, then a
`## Türkçe` section carrying the **same** structure in Turkish.

```
# Name — Role

...English (core sections + any optional sections)...

---
## Türkçe

...Turkish: the same sections, mirrored, with PROPER Turkish characters...
```

Rationale for single-file over split `README.tr.md`: matches the existing pattern
in the 5 already-bilingual packages, avoids file sprawl. Trade-off accepted: the
two halves can drift; mitigate by editing both halves in the same change.

## Sections — Tiered by Package Size

### CORE — every package, both languages

| Section | Content |
|---|---|
| **Title + 1-liner** | `# Name — Role`. Include the Turkish-nickname gloss if the package has one (e.g. Mr. Roboto's "steadfast worker who just does the grunt work"). |
| **Purpose** | 2–4 sentences. What the package owns and why it exists. The *why*, not a feature list. |
| **Public API** | Code block of what callers import / call. This is the contract — keep it accurate. |
| **Tests** | Exact command to run the package's tests, cross-platform safe. |

### OPTIONAL — complex packages add as warranted

- **Architecture / flow** — ASCII layer or data-flow diagram when the package is
  multi-stage (e.g. Fatih Hoca's `select → rank → utilization` layer block).
- **Key Modules table** — `module → role` navigation map when there are many files.
- **Runbook / Tuning** — a "how to safely change this" procedure for packages with
  delicate behavior (e.g. Fatih Hoca's tuning checklist + "don't tune by eyeball").
- **Gotchas / Notes** — load-bearing couplings, footguns, honest TODOs (e.g.
  Mr. Roboto's note that it still imports from `src.infra.db`).
- **Dependencies** — non-obvious needs: DB, Nerd Herd snapshot, env vars.

A small package (Mr. Roboto, ~58 lines) stays short; a deep one (Fatih Hoca,
~111 lines) grows. Don't force ceremony on tiny packages.

## Turkish Rules

- **Real Turkish characters are mandatory.** ASCII-fied Turkish is banned.
  Examples of the bug to fix:
  - `surecini yoneten` → `sürecini yöneten`
  - `cagri yurutme` → `çağrı yürütme`
  - `degistirir` → `değiştirir`
  - `gorevleri` → `görevleri`
  - `saglayicilarin` → `sağlayıcılarının`
- The Turkish section **mirrors** the English sections (same structure), not a
  loose paraphrase.
- Package nicknames spelled correctly with Turkish characters: **Yaşar Usta**,
  **Doğru mu Samet**, **Küleden Dönen Var**, **HaLLederiz Kadir**.

## SHOULD NOT Exist — Evergreen Rules

A README should not contain anything that rots or restates source:

- ❌ **Dates / phase numbers in the body** (`Phase 2d 2026-04-20`). They rot and
  force the reader to know phase history. Link a design doc for history instead.
- ❌ **Hard test counts** (`378 tests`). Stale the moment the count changes. Say
  "run the suite," give no number.
- ❌ **Changelog / history.** Git log and `docs/` own that.
- ❌ **Platform-broken commands.** This is a Windows 11 repo; no bare
  `PYTHONPATH=x cmd` (bash-only). Give a PowerShell-safe form or note both shells.
- ❌ **Line-by-line restating of internals.** README is the contract, not a source
  dump. If you must read internals to change a unit, that's a boundary smell — not
  the README's job to paper over.
- ❌ **Stale or aspirational API / action tables.** Tables list what *is* wired,
  verified at write time. (Mr. Roboto's README lists 2 actions but the dispatcher
  also routes `clarify` and `notify_user` — exactly this drift bug.)
- ❌ **Marketing fluff, ASCII-art banners.**

## Length Guide

Small package ~30–50 lines per language; complex ~100–120 per language. Mr. Roboto
(58) and Fatih Hoca (111) are about right.

## Examples to Embed in the Standard

Use real in-repo READMEs — they're credible:

- **Fatih Hoca** — *good* structure (layer diagram, Key Modules table, tuning
  runbook) but *bad* drift (dated phase headers, hard `378 tests` count).
- **Mr. Roboto** — *good* size and honest TODO note, but *bad* stale action table
  (missing `clarify` / `notify_user`).
- **ASCII-fied Turkish before/after** — e.g. DaLLaMa's
  `surecini yoneten` → `sürecini yöneten`.

## Out of Scope (Tracked Separately)

The standard itself does not fix existing files. Two follow-up work streams:

1. **Repair** the 5 ASCII-fied Turkish sections (dallama, general_beckman,
   hallederiz_kadir, kuleden_donen_var, nerd_herd) to use proper Turkish characters.
2. **Write** missing READMEs for the 7 packages without one (dogru_mu_samet,
   intersect, salako, vecihi, workflow_engine, yalayut, yasar_usta), bilingual
   per this standard.

## Current State Snapshot

| Package | README? | Turkish section? | Turkish chars OK? |
|---|---|---|---|
| coulson | ✅ | EN-only | n/a |
| dallama | ✅ | ✅ appended | ❌ ascii-fied |
| fatih_hoca | ✅ | EN-only | n/a |
| general_beckman | ✅ | ✅ appended | ❌ ascii-fied |
| hallederiz_kadir | ✅ | ✅ appended | ❌ ascii-fied |
| kuleden_donen_var | ✅ | ✅ appended | ❌ ascii-fied |
| mr_roboto | ✅ | EN-only | n/a |
| nerd_herd | ✅ | ✅ appended | ❌ ascii-fied |
| dogru_mu_samet | ❌ | — | — |
| intersect | ❌ | — | — |
| salako | ❌ | — | — |
| vecihi | ❌ | — | — |
| workflow_engine | ❌ | — | — |
| yalayut | ❌ | — | — |
| yasar_usta | ❌ | — | — |

Excluded from scope: `c21_paraflow_diff`, `sade_kalsin`, `safety_guard`.
