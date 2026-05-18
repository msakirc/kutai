# Brand Voice Document Schema

Each file in this directory defines a brand voice profile. The `brand_voice_lint` posthook
reads the active brand voice for a product (identified by `product_id`) and validates
produced copy against these rules.

## File naming

`<slug>.md` — e.g. `marketing.md`, `support.md`, `technical.md`.
The `marketing.md` profile is the default when no explicit profile is declared on the step.

## Required sections

Every brand voice file MUST contain the following YAML front-matter block:

```yaml
---
slug: marketing               # matches filename stem; used as profile key
display_name: Marketing Voice
version: "1.0"

# ── Vocabulary rules ──────────────────────────────────────────────────────────
prohibited_terms:             # list of exact strings or regex patterns
  - "cheap"
  - "guaranteed returns"
  - /(?i)unlimited\s+free/    # regex: prefix with /…/ to activate regex mode

# ── Readability targets ───────────────────────────────────────────────────────
target_avg_sentence_length_words: 18   # sentences longer than 1.5× this → warning
flesch_kincaid_reading_level_max: 10   # FK grade level ceiling (inclusive)
                                       # FK formula: 0.39×(words/sentences) + 11.8×(syllables/words) - 15.59

# ── Pronoun ratio ─────────────────────────────────────────────────────────────
# Ratio = we_count / (we_count + you_count); range [0.0, 1.0]
# we_ratio_max: brand is reader-centric (prefer "you"); 0.3 means ≤30% "we"
we_ratio_max: 0.3

# ── Tone keywords ─────────────────────────────────────────────────────────────
# Required tone signals: at least one must appear per 200-word block.
tone_required_signals:
  - confident
  - helpful
  - clear
# Forbidden tone signals: none of these may appear.
tone_forbidden_signals:
  - fear
  - urgency_artificial
---
```

## Body (optional)

After the front-matter, add free-form guidance for human editors and the LLM:

```markdown
## Voice principles

- Speak to the reader as a knowledgeable peer, not an authority figure.
- Use active voice. Passive constructions dilute impact.
- One idea per sentence; no compound sentences joined by semicolons.

## Examples

Good: "You can export your data in three steps."
Avoid: "Data can be exported by users via the export function."
```

## Lint pass/fail logic (implemented by brand_voice_lint.py)

1. Each `prohibited_terms` entry is checked against the produced text (regex or exact).
2. `flesch_kincaid_reading_level_max` is computed over the full text block.
3. `target_avg_sentence_length_words` is checked; sentences > 1.5× threshold are flagged.
4. `we_ratio_max` is checked on the pronoun counts.
5. `tone_required_signals` must fire at least once per 200-word window (keyword presence check).
6. `tone_forbidden_signals` must not appear anywhere.

Any violation in category 1, 5, or 6 → FAIL (blocker).
Categories 2, 3, 4 → WARNING if within 20% of threshold; FAIL if exceeded.
