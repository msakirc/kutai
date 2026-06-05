# Package README Gold Standard — Design

**Date:** 2026-05-29
**Status:** Approved (guideline doc, no enforcement tooling)
**Scope:** All sub-packages under `packages/` except `c21_paraflow_diff` (paraflow),
`sade_kalsin`, and `safety_guard`. 15 packages in scope.
**Companion artifact:** `docs/package-readme-template.md` — the copyable skeleton.

## Goal

Define the **gold standard** for a package README: an objective bar, anchored by a
self-contained annotated template plus a binary checklist. This is a **guideline**,
not linted or test-enforced — it relies on discipline, the template, and the
good/bad examples below. **No real package is blessed as "the reference"** (any
current README may be stale; e.g. KDV has drifted since its last update). Every
package is built to the same template + checklist equally.

## Audience & Purpose

A README serves two readers: the **future founder** returning months later, and
**Claude subagents** doing subagent-driven development. The README is the package's
**contract surface** — use or modify the package without reading all internals.

## The Gold Standard

### The one-sentence test

A README is gold if a fresh reader (or subagent) can, **without opening any source
file**:

1. say what the package is for **and what it deliberately won't do**,
2. make a correct call against its public API,
3. run its tests,
4. know the 1–2 things that will bite them.

If any of those requires reading code, it is not gold.

### Canonical section order

```
# Name — Role (nickname gloss)     ← what, one line
## Purpose                         ← why + explicit NON-GOALS
## Public API                      ← contract: real symbols + signatures
## [Architecture]   (optional)     ← flow diagram, multi-stage only
## [Key Modules]    (optional)     ← module→role table, many-file only
## [Dependencies]   (optional)     ← non-obvious needs
## [Gotchas]        (optional)     ← footguns, load-bearing couplings
## [Runbook/Tuning] (optional)     ← how to change safely
## Tests                           ← exact, PowerShell-safe command
---
## Türkçe                          ← full mirror, proper chars
```

CORE sections (always): Title+1-liner · Purpose · Public API · Tests.
OPTIONAL sections: added only when the package's complexity warrants.

### The bar per CORE section

- **Title role** is a *job*, not a noun category. "Model-selection brain" beats
  "model utilities". Include the Turkish-nickname gloss if the package has one.
- **Purpose** is 2–4 sentences and MUST include at least one explicit non-goal
  sentence ("It does NOT …"). Boundaries are first-class. KDV's "doesn't discover
  models, pick models, or make LLM calls" is the pattern to copy.
- **Public API** shows the 2–5 entry points a caller actually uses (not every
  export). Every symbol is importable today, the signature matches source today,
  return types are named. A caller can copy-paste from the block.
- **Tests** is a command that runs as-written on Windows PowerShell (the repo
  host) — no bash-only constructs.

### Acceptance checklist (binary — all must pass)

1. Title role is a job description, not a noun category.
2. Purpose states at least one explicit non-goal.
3. Every Public API symbol verified against code (not remembered), signature current.
4. Test command runs as written on this host (Windows PowerShell).
5. Turkish section present, mirrors EN sections, uses proper Turkish characters,
   nickname spelled correctly.
6. Zero banned items (see Evergreen Rules below).
7. A reader can use **and** modify the package without opening source.

## Bilingual Layout — Single File

One `README.md`. English first, then a horizontal rule, then `## Türkçe` carrying
the **same** structure in Turkish. Single-file (over split `README.tr.md`) matches
the existing pattern in the 5 already-bilingual packages and avoids file sprawl.
Trade-off accepted: the halves can drift; mitigate by editing both in one change.

## Turkish Rules

- **Real Turkish characters are mandatory.** ASCII-fied Turkish is banned:
  - `surecini yoneten` → `sürecini yöneten`
  - `cagri yurutme` → `çağrı yürütme`
  - `degistirir` → `değiştirir`
  - `gorevleri` → `görevleri`
  - `saglayicilarin` → `sağlayıcılarının`
- The Turkish section **mirrors** the English sections (same structure), not a loose
  paraphrase.
- **Restore chars by meaning, not by pattern.** Don't mechanically add Turkish
  diacritics to every vowel — match the intended word. `kule` (tower) ≠ `küle`
  (ash); the KDV idiom is the control-tower "kuleden dönen var," so it is
  `Kuleden` (plain u) with only `Dönen` taking the ö. The opposite failure of
  ascii-fication, and just as wrong.
- Nicknames spelled correctly: **Yaşar Usta**, **Doğru mu Samet**,
  **Kuleden Dönen Var** (*kule* = tower, not *küle* = ash), **HaLLederiz Kadir**.

## Evergreen Rules — SHOULD NOT exist

(Checklist item 6.) A README must not contain anything that rots or restates source:

- ❌ Dates / phase numbers in the body (`Phase 2d 2026-04-20`) → link a design doc.
- ❌ Hard test counts (`378 tests`) → say "run the suite," no number.
- ❌ Changelog / history → git log and `docs/` own that.
- ❌ Platform-broken commands → Windows host; no bash-only `PYTHONPATH=x cmd`.
- ❌ Line-by-line restating of internals → README is the contract, not a source dump.
- ❌ Stale or aspirational API / action tables → tables list what *is* wired,
  verified at write time.
- ❌ Marketing fluff, ASCII-art banners.

## Length Guide

Small package ~30–50 lines per language; complex ~100–120 per language.

## Examples (embed in the standard)

Real in-repo READMEs, credible:

- **Fatih Hoca** — *good* structure (layer diagram, Key Modules table, tuning
  runbook); *bad* drift (dated phase headers, hard `378 tests` count).
- **Mr. Roboto** — *good* size and honest TODO note; *bad* stale action table
  (missing `clarify` / `notify_user`).
- **ASCII-fied Turkish before/after** — DaLLaMa's `surecini yoneten` →
  `sürecini yöneten`.

## Execution Methodology — 2-Stage, Fresh Agent per Package

The standard does not fix files by itself. Building the READMEs uses a two-stage,
fan-out approach so each README is grounded in current code, not memory:

**Stage 1 — Analysis (one fresh agent per in-scope package).** Reads only its
package's source and current README. Emits a structured **README source brief**:

- **Does** — real behavior, derived from code (not docstrings).
- **Doesn't** — boundaries / non-goals.
- **Edges** — gotchas, footguns, load-bearing couplings, error handling.
- **Public API** — actual exported symbols + current signatures.
- **Drift** — where the current README disagrees with current code (e.g.
  Mr. Roboto's missing actions, KDV's post-upgrade changes).
- **Dependencies** + a **verified** test command.

**Stage 2 — Write.** Each brief + this standard + the template are turned into the
bilingual README. Fact-finding is separated from prose so Turkish quality is
controlled centrally and briefs are reviewable before any README is written.

## Per-Package Work Classification

| Package | README? | Turkish | Stage-2 work |
|---|---|---|---|
| coulson | ✅ | EN-only | add Türkçe mirror; verify EN vs code |
| dallama | ✅ | ascii-fied | repair TR chars; verify drift |
| fatih_hoca | ✅ | EN-only | add Türkçe; strip date/count drift |
| general_beckman | ✅ | ascii-fied | repair TR chars; verify drift |
| hallederiz_kadir | ✅ | ascii-fied | repair TR chars; verify drift |
| kuleden_donen_var | ✅ | ascii-fied | repair TR + refresh (post-upgrade drift) |
| mr_roboto | ✅ | EN-only | add Türkçe; fix stale action table |
| nerd_herd | ✅ | ascii-fied | repair TR chars; verify drift |
| dogru_mu_samet | ❌ | — | write full bilingual README |
| intersect | ❌ | — | write full bilingual README |
| vecihi | ❌ | — | write full bilingual README |
| workflow_engine | ❌ | — | write full bilingual README |
| yalayut | ❌ | — | write full bilingual README |
| yasar_usta | ❌ | — | write full bilingual README |

Excluded from scope: `c21_paraflow_diff`, `sade_kalsin`, `safety_guard`.

`salako` struck from this table: it was renamed to `mr_roboto` (commit
e16c75d6, full cutover, no shim). `packages/salako/` survives only as orphaned,
gitignored `.pyc` cruft — nothing tracked, not importable. No README; the work it
implied is `mr_roboto`'s. The orphan directory was removed.
