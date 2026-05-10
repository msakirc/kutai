# Uizard — Competitor Research

**Date:** 2026-05-09
**Cluster:** AI design tools
**Confidence:** Medium (official site + Miro acquisition post + third-party reviews; post-acquisition roadmap is partially opaque)

## TL;DR

Uizard pioneered AI-driven UI design (founded 2018 in Copenhagen). Its hallmark "Autodesigner" generates multi-screen, themed, editable prototypes from text/sketch/screenshot. Acquired by **Miro on May 27, 2024**; standalone product still runs but is increasingly framed as "Uizard by Miro Labs." Reached ~3M users and ~$3.5M ARR pre-acquisition.

## 1. Input Flow

- **Text prompt** (Autodesigner): "Generate a fitness app with onboarding, home, workout, profile."
- **Hand-drawn sketch** (Wireframe Scanner): photo → editable wireframe.
- **Screenshot** (Screenshot Scanner): existing app screen → editable mockup.
- Combined: sketch + prompt for refinements.
- Iteration via natural-language prompts on individual components.

## 2. Output Type

- **Multi-screen, editable prototypes** with themes applied.
- Wireframe or mid-fidelity mockup, depending on input.
- Clickable interactive prototypes.
- Themed components (buttons, inputs, cards) using the project palette.

## 3. Iteration Loop

- **Autodesigner 2.0** — conversational AI lets you edit designs section-by-section by chat.
- Per-component prompt edits ("make this card larger, add a CTA").
- Theme regeneration applies new style globally.
- Manual edits via canvas drag-and-drop.

## 4. Style + Theming

- **Theme generator** — describe the brand vibe, get a full theme (palette, fonts, components).
- Custom **brand kits** with fonts, colors, button styles.
- Theme swap regenerates project styling instantly.

## 5. Multi-Screen Consistency

- Autodesigner generates a **complete user flow** (multiple screens) with consistent theming.
- Shared component library across screens within a project.
- Linking between screens for prototype navigation.

## 6. Charter / Spec / PRD Generation

- None. Autodesigner is direct prompt → UI screens.
- No upstream PRD/charter artifact.

## 7. Image Generation

- Built-in icon and image library (limited).
- No generative imagery as a headline feature; relies on stock + uploaded assets.

## 8. Export

- **Figma**: no direct export — workaround is SVG export, then re-import (a long-standing complaint).
- **Code (Developer Handoff)**: React + CSS handoff on Pro tier.
- PNG / PDF / image export.
- Live shareable prototype links.

## 9. Pricing (2026)

- **Free**: 3 AI generations/month, 2 active projects, 5 screens/project, Autodesigner 1.5 (older model), 10 templates, 400 components.
- **Pro**: ~$12/month annual — 500 AI generations/month, Autodesigner 2.0, React/CSS handoff, 100 projects, unlimited viewers/commenters.
- **Business**: ~$39/month — team collaboration, priority support.
- **Enterprise**: custom — unlimited generations, design system setup, AI Data SLA, white-glove onboarding.

## 10. Underlying Model

- Not publicly disclosed. Autodesigner is built in-house with proprietary models trained on UI datasets (Uizard's research roots are in pix2code / Tony Beltramelli's PhD work). Some recent generation likely augments with third-party LLMs but not confirmed.

## 11. Recent Updates (2024-2026)

- **May 27, 2024**: Acquired by Miro. Uizard team continues; no immediate product changes.
- **2024-2025**: Autodesigner 2.0 release (conversational section-level editing).
- **2025+**: Deeper Miro integrations rolled out — "How To Use Miro With Uizard" workflow guide published. Brand evolving to "Uizard by Miro Labs."
- **Notable**: No public roadmap signal that Uizard standalone gets sunset; positioning is complementary to Miro's whiteboard.

## 12. Limitations

- **No direct Figma export** — biggest complaint, requires SVG workaround.
- AI output "sometimes looks amateur" — quality below Stitch and Visily on visual polish.
- Free tier extremely restrictive (1 active project effectively).
- Limited generative imagery / illustration.
- Code export limited to React + CSS, no Flutter/SwiftUI/etc.
- Underlying model not transparent.
- Post-acquisition roadmap unclear — risk of feature stagnation while Miro integration takes priority.

## What i2p Should Notice

1. **Autodesigner 2.0's section-level conversational editing** is the right granularity for an agent loop — not "regenerate the whole screen" and not "change one pixel," but "rework this card." i2p's visual-review phase (Z4) should target section-level, not screen-level, regen.
2. **Pix2code DNA** — Uizard's wireframe/screenshot scanners come from research on translating images to code. i2p's "real-world bridge" Z6 can borrow this for ingest of competitor screenshots as design priors.
3. **Theme generator as a first-class artifact** — generating the theme separately from the screens lets you swap-and-regenerate. i2p's pre-code Z1 should produce a theme artifact independent of screen artifacts.
4. **Acquisition risk signal**: even successful AI design tools end up acquired into broader collab platforms (Miro, Figma, Adobe). i2p's defensibility is the *idea-to-product pipeline*, not the design step alone.

## Sources

- [Uizard Joins Miro (announcement)](https://uizard.io/blog/uizard-joins-miro/)
- [Uizard official site](https://uizard.io/)
- [Crunchbase — Miro acquires Uizard](https://www.crunchbase.com/acquisition/mirohq-acquires-uizard-technologies--3f732879)
- [Silicon Canals — Uizard acquired by Miro](https://siliconcanals.com/uizard-acquired-by-miro/)
- [Indie Hackers — Uizard's path to $3.5M ARR](https://www.indiehackers.com/post/tech/growing-the-first-ever-ai-product-design-tool-to-3-5m-arr-and-getting-acquired-1z993H3c4Es50z3DTAA2)
- [Banani — Uizard AI Review](https://www.banani.co/blog/uizard-ai-review)
- [How To Use Miro With Uizard](https://uizard.io/blog/how-to-use-miro-with-uizard/)
- [Uizard Pricing](https://uizard.io/pricing/)
- [byFounders — Uizard by Miro Labs](https://www.byfounders.vc/portfolio/uizard)
