---
channel: hn_post
display_name: Hacker News Show HN
version: "1.0"

max_title_chars: 80
max_body_chars: 2000
max_total_chars: 0

banned_words:
  - /(?i)launch(ing|ed)\s+today/
  - "check out"
  - /(?i)excited\s+to\s+(announce|share)/
  - /(?i)game[\s-]changer/
  - "disrupting"
  - /(?i)revolutionary/

required_disclosures: []

image_required: false
image_min_width_px: 0
image_min_height_px: 0
image_max_size_kb: 0
image_allowed_formats: []
---

## HN Show HN guidelines

HN readers value directness. They dislike marketing language, superlatives,
and anything that sounds like a press release.

### Title rules

- Start with "Show HN:" prefix.
- State what the project does, not how great it is.
- Avoid adjectives: "fast", "simple", "easy" — show, don't tell.
- Max 80 chars including the "Show HN:" prefix.

### Body rules

- What does it do? (one sentence)
- Why did you build it? (technical motivation, not business pitch)
- What is the current state? (alpha/beta/stable, known limitations)
- What feedback are you looking for?

### Examples

Good title: "Show HN: Open-source CLI to diff SQLite schemas across migrations"
Avoid: "Show HN: The revolutionary database tool that's changing how teams work"

Good body opener: "I built this because I kept forgetting which columns changed
between migration files and wanted a quick diff without spinning up a full DB."
Avoid: "We are excited to announce the launch of our new product today!"
