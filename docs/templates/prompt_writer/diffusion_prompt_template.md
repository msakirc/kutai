# Diffusion-prompt template (prompt_writer)

You will receive the design context and a list of placeholder intents.
Fill the JSON envelope below with one enriched prompt per placeholder.

## Inputs

DESIGN TOKENS (palette + type — bias generated images toward these colors):
{design_tokens}

BRAND VOICE (mood / tone — bias subject choice and composition):
{brand_voice}

SECTION INTENTS (screen role per placeholder):
{section_intent}

PLACEHOLDERS (the list to enrich):
{placeholders}

## Few-shot exemplars

EXAMPLE 1
Input placeholder:
  placeholder_id: hero_1
  alt: "smiling barista handing over a takeaway cup"
  width: 390
  height: 220
  section: hero
Brand voice: "warm, neighborhood coffee shop — third-wave"
Design tokens: { primary: "#E07A5F" (warm coral), surface: "#F4F1DE" (cream) }
Expected prompt:
  "Warm candid photo of a smiling young barista handing a takeaway cup, soft morning light through cafe window, warm coral apron accent against cream-toned interior, shallow depth of field, eye-level wide composition."

EXAMPLE 2
Input placeholder:
  placeholder_id: feature_2
  alt: "ai-powered task triage dashboard"
  width: 260
  height: 180
  section: feature
Brand voice: "calm, professional productivity tool"
Design tokens: { primary: "#3D405B" (slate indigo), accent: "#81B29A" (muted sage) }
Expected prompt:
  "Minimal isometric illustration of a clean dashboard with sorted task cards, slate indigo header bar, muted sage progress accents on soft white background, flat vector style, centered composition, no text on screen."

EXAMPLE 3
Input placeholder:
  placeholder_id: avatar_3
  alt: "user portrait"
  width: 64
  height: 64
  section: testimonial
Brand voice: "diverse community, real people"
Design tokens: { primary: "#264653" (deep teal) }
Expected prompt:
  "Friendly close-up headshot of a person against soft deep-teal blurred background, natural diffuse lighting, eye contact, neutral expression, square composition."

## Now emit the JSON

Return ONLY the final_answer JSON envelope — no prose, no markdown
fences around it. Every placeholder_id from the input MUST appear in
`prompts`. Each prompt MUST be <=220 characters and MUST embed at least
one design-token color cue.
