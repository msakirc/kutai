# Channel Rules Document Schema

Each file in this directory defines the submission rules for one external channel.
The `copy_compliance_review` posthook reads the applicable channel rule file
(matched by the `channel` value on the workflow step context) and validates
produced copy before it is written to `external_comms_log`.

## File naming

`<channel>.md` where `<channel>` matches the `channel` column in `external_comms_log`.

Valid channel values: `email`, `tweet`, `reddit_post`, `hn_post`, `ph_post`,
`linkedin_post`, `press_release`, `blog_post`, `sms`, `webhook`.

Example files provided: `hn.example.md`, `producthunt.example.md`.

## Required sections

Every channel rules file MUST contain the following YAML front-matter block:

```yaml
---
channel: hn_post              # must match filename stem (minus .md)
display_name: Hacker News Show HN
version: "1.0"

# ── Length limits ─────────────────────────────────────────────────────────────
max_title_chars: 80           # hard limit; lint FAILs if exceeded
max_body_chars: 2000          # 0 = no limit (e.g. blog_post)
max_total_chars: 0            # combined limit; 0 = no limit

# ── Banned words / patterns ───────────────────────────────────────────────────
# Exact strings or /regex/ patterns (same syntax as brand_voices).
banned_words:
  - /(?i)launch(ing|ed)\s+today/   # HN hates these
  - "check out"
  - /(?i)excited\s+to\s+(announce|share)/

# ── Required disclosures ──────────────────────────────────────────────────────
# Each entry is a label + pattern. ALL must be present in the submitted text.
required_disclosures: []      # e.g. [{label: "affiliate", pattern: /(?i)affiliate/}]

# ── Image requirements ────────────────────────────────────────────────────────
image_required: false
image_min_width_px: 0
image_min_height_px: 0
image_max_size_kb: 0          # 0 = no limit
image_allowed_formats: []     # empty = any format accepted
---
```

## Body (optional)

After the front-matter, add human-readable guidance for editors and LLM prompts.
Include examples of passing and failing copy.

## Lint pass/fail logic (implemented by copy_compliance_review.py)

1. `max_title_chars` / `max_body_chars` / `max_total_chars`: hard limits → FAIL.
2. `banned_words`: any match → FAIL.
3. `required_disclosures`: any missing entry → FAIL.
4. `image_required = true` with no image URL in context → FAIL.
5. Image dimension / size / format checks (when image_required = true) → FAIL.

All violations are surfaced as a list in the posthook result; the step retries
with the violation list injected into the agent's next-iteration context.
