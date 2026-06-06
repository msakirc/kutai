<!--
GOLD-STANDARD PACKAGE README TEMPLATE
Copy this into packages/<pkg>/README.md and fill it in.
Delete every HTML comment (like this one) before committing.
Delete OPTIONAL sections the package doesn't need. Keep CORE sections always.
The bar: a fresh reader can use AND modify the package WITHOUT opening source.
-->

# Name — Role
<!-- One line. Role is a JOB, not a category. "Model-selection brain" > "model utilities".
     Add the Turkish-nickname gloss in parentheses if the package has one. -->

## Purpose
<!-- 2-4 sentences. What problem it owns and why it exists (the WHY, not a feature list).
     MUST include at least one explicit NON-GOAL sentence: "It does NOT ..."
     Boundaries are first-class — they're how callers know what NOT to expect here. -->

## Public API
<!-- The contract. The 2-5 entry points a caller actually uses — not every export.
     Every symbol shown MUST be importable today and its signature MUST match source.
     Name the return types. A caller should be able to copy-paste from this block. -->
```python
from name import primary_entrypoint, ResultType

result = await primary_entrypoint(arg)   # -> ResultType(status, ...)
```

<!-- ===== OPTIONAL SECTIONS — keep only what the package needs ===== -->

## Architecture
<!-- OPTIONAL — only for multi-stage packages. ASCII flow / layer diagram. -->

## Key Modules
<!-- OPTIONAL — only when there are many files. module -> role table. -->
| module | role |
|---|---|
| `x.py` | ... |

## Dependencies
<!-- OPTIONAL — non-obvious needs: DB, Nerd Herd snapshot, env vars, sibling packages. -->

## Gotchas
<!-- OPTIONAL — footguns, load-bearing couplings, honest TODOs.
     This is where you record the 1-2 things that will bite a future editor. -->

## Runbook / Tuning
<!-- OPTIONAL — a "how to change this safely" procedure for delicate packages. -->

<!-- ===== CORE — always keep ===== -->

## Tests
<!-- Exact command that runs AS-WRITTEN on Windows PowerShell (the repo host).
     No bash-only constructs (no `VAR=x cmd`, no bare `PYTHONPATH=x cmd`). -->
```powershell
python -m pytest packages/name/
```

---
## Türkçe
<!-- Full mirror of the sections above, SAME structure.
     PROPER Turkish characters are mandatory: ş ı ç ğ ö ü İ — never ascii-fied.
     Package nicknames spelled correctly: Yaşar Usta, Doğru mu Samet, Kuleden Dönen Var. -->

### Amaç
<!-- Purpose, mirrored. Include the non-goal sentence ("... yapmaz"). -->

### Genel API
<!-- Public API, mirrored. Same code block. -->

### Testler
<!-- Tests, mirrored. -->
