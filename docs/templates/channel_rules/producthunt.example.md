---
channel: ph_post
display_name: Product Hunt
version: "1.0"

max_title_chars: 60
max_body_chars: 260           # tagline field
max_total_chars: 0

banned_words:
  - /(?i)best\s+[\w\s]+\s+ever/
  - /(?i)world[\s-]?class/
  - /(?i)industry[\s-]?leading/
  - /(?i)guaranteed/

required_disclosures: []

image_required: true
image_min_width_px: 240
image_min_height_px: 240
image_max_size_kb: 5120       # 5 MB
image_allowed_formats:
  - png
  - jpg
  - jpeg
  - gif
---

## Product Hunt submission guidelines

### Title

- 60 chars max.
- Sentence case: only capitalise first word and proper nouns.
- No punctuation at the end.
- No "the", "a", "an" at the start unless part of the name.

### Tagline (max 260 chars)

- One sentence that tells a user what the product does.
- Start with a verb: "Turn...", "Build...", "Track...".
- Avoid "We", "Our", "I" — focus on what the user gets.
- No period at the end.

### Thumbnail image

- Square, minimum 240×240 px.
- PNG or JPG preferred; GIF allowed for animated demos.
- Max 5 MB.
- No text overlay covering more than 20% of the image.

### Examples

Good title: "Schema Diff CLI"
Avoid: "The Best Database Schema Comparison Tool Ever Built"

Good tagline: "Compare SQLite schemas across migration files in one command"
Avoid: "We are excited to share our revolutionary world-class database tool"
