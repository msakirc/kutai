# Paraflow

> Research date: 2026-05-09. Domain confirmed: **paraflow.com** (the design tool). `paraflow.ai` and `paraflowai.com` are an unrelated VC-OS product — not the same company.

## What it is (one paragraph)

Paraflow is a canvas-based "AI product design agent" that converts a single natural-language product idea into a connected bundle of artifacts on one infinite canvas: PRD, user personas, user flows, screen plans, hi-fi screens, interactive prototype, style guide, and front-end code, plus a Figma file or GitHub repo for handoff. The pitch is "two AI minds — a product manager and a designer — collaborating on the same canvas," with bidirectional links so that edits to specs propagate to designs and vice versa. ([paraflow.com](https://paraflow.com/), [Product Hunt](https://www.producthunt.com/products/paraflow), [funblocks.net](https://www.funblocks.net/aitools/reviews/paraflow))

## Input flow (concrete)

The actual onboarding is light, not a wizard:

1. **Sign up → Dashboard → "New Project."**
2. **Single chat box** — describe the product goal and key scenarios in natural language, OR pick / rewrite a **preset prompt** (template). ([toolnavs.com](https://toolnavs.com/en/article/400-what-if-you-dont-know-how-to-write-a-prd-paraflow-allows-you-to-easily-produce-s))
3. **Optional context attachments**: image, Markdown file, or Notion link. HTML snapshots and screenshots are also accepted to anchor visual style. ([toolnavs.com](https://toolnavs.com/en/article/400-what-if-you-dont-know-how-to-write-a-prd-paraflow-allows-you-to-easily-produce-s), [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
4. **Visual style preferences** — collected as part of the brief (e.g. mood/tone reference) rather than a separate questionnaire step. ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
5. **AI generates a "To-Do List"** of what it intends to produce; the user **confirms** the list before the agent starts generating Specs. This is the only structured gate. ([toolnavs.com](https://toolnavs.com/en/article/400-what-if-you-dont-know-how-to-write-a-prd-paraflow-allows-you-to-easily-produce-s), [Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e))

There is **no fixed N-question wizard**. The flow is "one prompt + optional attachments + confirm a generated to-do list." Match against the user's recollection ("stated the idea, then a few refinement rounds"): consistent.

## Refinement loop

- **Conversational, on-canvas, per-asset.** The user "points to a reference asset and describes the change. The agent updates screens, components, and the style guide." ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
- **Bidirectional sync** — PRDs, flows, and prototypes are described as staying aligned in real time, so editing a spec propagates to the matching screen and vice versa. ([Product Hunt](https://www.producthunt.com/products/paraflow))
- **Stage cadence observed in a third-party walkthrough**:
  1. Specs refinement (~2-4 minutes for initial PRD)
  2. Flow + UI adjustments (manual edits for terminology like status labels)
  3. Code integration (~30-50 % time savings on boilerplate, but wiring still needed) ([automateed.com](https://www.automateed.com/paraflow-review))
- Refinement is **not** form-based (no "regen with these constraints" dialog) and **not** bulk-bundle regen — it is per-target chat edits.

## Output bundle

Confirmed artifacts (cited):

| Artifact | Source |
|---|---|
| Product charter / objectives | [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/) |
| User personas | [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/) |
| PRD | [paraflow.com](https://paraflow.com/), [Product Hunt](https://www.producthunt.com/products/paraflow) |
| User flow diagrams | [paraflow.com](https://paraflow.com/) |
| Screen plans (scope per screen) | [paraflow.com](https://paraflow.com/) |
| Style guide (typography, color, spacing) | [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/) |
| Hi-fi screens (mobile + web) | [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/) |
| Interactive prototype | [paraflow.com](https://paraflow.com/) |
| Front-end code (HTML/CSS/JS) | [Product Hunt](https://www.producthunt.com/products/paraflow), [Medium / Eyad Kelleh](https://medium.com/@kelleheyad/how-paraflow-changed-my-solo-developer-workflow-in-just-one-week-b54b5b7210cd) |
| Figma file export | [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/) |
| GitHub repo push (new repos only) | [Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e) |
| `project.json` config + assets in ZIP | [Medium / Eyad Kelleh](https://medium.com/@kelleheyad/how-paraflow-changed-my-solo-developer-workflow-in-just-one-week-b54b5b7210cd) |

**Cross-check vs TruthRate folder** (charter, 2 personas, PRD, user_flow.md, 24 screen plans, 24 HTML prototypes, light+dark style guide): every artifact in TruthRate matches a documented Paraflow output. The light+dark style-guide pairing is **not** explicitly documented as a built-in feature in any source we found — unknown whether it is automatic from the brief or required a refinement round.

## Style + image generation

- **Style guide derivation**: described as "AI-generated style guides [that] ensure consistent design across all screens and components" — inferred from the product description and any visual references provided. The user can attach style references; otherwise the agent picks. No source describes a structured palette-input form. ([aipure.ai](https://aipure.ai/products/paraflow), [canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
- **Semantic color tokens** (e.g. TruthRate's `fact_primary`): unknown whether Paraflow auto-tags semantic roles. No source mentions named-token generation. Likely emerges from the agent's PRD-aware naming since the same agent writes both spec and tokens, but this is **inferred, not documented**.
- **Image generation**: unknown source. One first-hand user describes Paraflow producing "a 3D charging cable with a smile emoji against a rich purple gradient. Particle effects and all" — implying built-in image generation, but the underlying model/source is **not disclosed** by any source we examined. ([Medium / Eyad Kelleh](https://medium.com/@kelleheyad/how-paraflow-changed-my-solo-developer-workflow-in-just-one-week-b54b5b7210cd))

## Multi-surface / multi-screen consistency

- **Surfaces**: mobile **and** web are explicitly listed as outputs ("mobile and web screens"). Desktop-app surfaces are not mentioned. ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
- **Multi-screen consistency**: claimed via the **shared canvas + AI-generated style guide** that all screens reference. The PM agent owns the spec; the design agent reuses the same component/token library across screens. No source describes the consistency mechanism in technical detail (component instancing? prompt-time style injection?). ([aipure.ai](https://aipure.ai/products/paraflow))
- **Responsive breakpoints**: not addressed in any source. Unknown.

## Export + deploy

- **Figma**: editable file export for designer handoff. ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
- **GitHub**: push the front-end project to a **new** repo. Pushing into an **existing** repo is not supported (cited limitation). Backend code generation is "under testing" / not production. ([Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e), [Product Hunt](https://www.producthunt.com/products/paraflow))
- **HTML import**: existing site → import → Paraflow suggests improvements. ([Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e))
- **Hosted preview / shareable workspace**: a "shareable collaborative workspace" exists; whether prototypes get a public URL is unclear. ([automateed.com](https://www.automateed.com/paraflow-review))
- **ZIP / interactive HTML files**: confirmed by a first-hand user who received markdown docs, interactive HTML, exported visual assets, CSS, and `project.json`. ([Medium / Eyad Kelleh](https://medium.com/@kelleheyad/how-paraflow-changed-my-solo-developer-workflow-in-just-one-week-b54b5b7210cd))

## Pricing

Two plans, credit-based:

| Plan | Price | Credits |
|---|---|---|
| Free | $0 / month | 200 daily credits, capped at 500/month |
| Pro | $25 / month | 2,500/month + 200 daily (cap 500/month) bonus |

Yearly subscription advertised at 20 % savings. **Per-action credit cost is not published**; the FAQ references it but the answers are not on the public page. ([paraflow.com/pricing](https://paraflow.com/pricing), [futurepedia.io](https://www.futurepedia.io/tool/paraflow))

Note: an older summary cited "400 monthly credits" on the free plan and "2,500 + 400 bonus" on Pro — the live pricing page now shows "200 daily / 500 monthly" caps, suggesting the model was rebalanced. ([Product Hunt](https://www.producthunt.com/products/paraflow), [paraflow.com/pricing](https://paraflow.com/pricing))

## Underlying tech (if disclosed)

**Not disclosed.** No source examined names a specific LLM (no GPT-4 / Claude / Gemini / Qwen attribution). Marketing copy is consistent: "two AI minds — a PM and a designer." Whether these are two roles played by one model with different system prompts, or genuinely separate models, is unstated. ([futurepedia.io](https://www.futurepedia.io/tool/paraflow), [aipure.ai](https://aipure.ai/products/paraflow), [Product Hunt](https://www.producthunt.com/products/paraflow))

## Differentiators (claimed)

1. **Single canvas with bidirectional links** between PRD, flow, screens, prototype, and code — not a sequence of disconnected tools. ([paraflow.com](https://paraflow.com/), [Product Hunt](https://www.producthunt.com/products/paraflow))
2. **Two-agent simulation** (PM + Designer) on the same artifact graph rather than one generalist agent. ([funblocks.net](https://www.funblocks.net/aitools/reviews/paraflow))
3. **End-to-end bundle from a single prompt** — charter through code, no per-stage tool switching. ([Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e))
4. **Conversational asset-pointer edits** ("change this, like that") rather than form-driven regen. ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/))
5. **HTML import for brownfield** improvement suggestions. ([Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e))

## Limitations (observed)

- **Front-end only** — backend code generation "under testing," not shipped. ([Product Hunt](https://www.producthunt.com/products/paraflow))
- **GitHub push to new repos only** — cannot target an existing repo. ([Medium / Digital Works](https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e))
- **Generic outputs for niche/regulated domains** — B2B-enterprise or medical/regulated products produce generic specs that need heavy manual correction. Accessibility audits not built in. ([canadiantechnologymagazine.com](https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/), search synthesis above)
- **Manual UX-copy polish required** — terminology (status labels, microcopy) typically needs a refinement pass; it is not production-ready out of the box. ([automateed.com](https://www.automateed.com/paraflow-review))
- **Code is scaffolding, not production** — exported front-end needs engineering review and wiring. ([automateed.com](https://www.automateed.com/paraflow-review))
- **Integration limits** — does not seamlessly connect to all third-party design-team tools (no Jira ticket creation, etc.; suggested as a roadmap item). ([futurepedia.io](https://www.futurepedia.io/tool/paraflow), [funblocks.net](https://www.funblocks.net/aitools/reviews/paraflow))
- **No public Reddit / HN complaint threads found** — the review surface is dominated by AI-tool aggregators and Medium posts. Real adversarial user feedback is thin, so the limitations list is partly inferred from positive reviews' caveats.
- **Per-action credit cost not published** — pricing transparency is incomplete. ([paraflow.com/pricing](https://paraflow.com/pricing))
- **No disclosed model attribution** — vendor-lock and quality drift risk for users who care which LLM produces their specs.

## Sources (URLs)

- https://paraflow.com/
- https://paraflow.com/pricing
- https://www.producthunt.com/products/paraflow
- https://medium.com/@digitalworks2020/paraflow-the-ai-agent-that-turns-ideas-into-product-design-52f7be3d474e
- https://medium.com/@kelleheyad/how-paraflow-changed-my-solo-developer-workflow-in-just-one-week-b54b5b7210cd
- https://www.futurepedia.io/tool/paraflow
- https://www.automateed.com/paraflow-review
- https://canadiantechnologymagazine.com/ai-product-design-agent-idea-to-prototype/
- https://www.funblocks.net/aitools/reviews/paraflow
- https://aipure.ai/products/paraflow
- https://toolnavs.com/en/article/400-what-if-you-dont-know-how-to-write-a-prd-paraflow-allows-you-to-easily-produce-s
- https://chatgate.ai/post/paraflow

Sources we could **not** retrieve (cited for completeness):
- https://www.youtube.com/watch?v=em7Luc83mPs ("Paraflow AI Tutorial 2025/2026: Build a Student Dashboard") — only footer scraped.
- https://www.toolify.ai/tool/paraflow — 403.

## Implications for KutAI i2p

1. **The "single prompt + confirm a generated to-do list" gate beats a long wizard.** Paraflow does not interrogate the founder; it drafts the to-do list and asks "ok?" Our mission-preflight (z0) currently leans toward more upfront questions — Paraflow's evidence says the cheaper bet is *generate a plan from one prompt, then let the user accept/edit it*. Brainstorming + plan-confirm beats requirements-gathering.
2. **Per-asset conversational refinement, not per-bundle regen.** Paraflow's UX is "point + describe change" on individual artifacts, with the PM/Designer split keeping spec+design in sync. Our i2p has the dual-role separation (planner/coder/visual-reviewer) but lacks the **bidirectional propagation** — editing a step produces no auto-update to siblings. That's the missing primitive: a *spec-change → downstream-asset diff* loop.
3. **Style guide is inferred, not asked.** Paraflow does not solicit palette input; it infers from the brief and any reference assets. Our i2p should treat style as *derived state*, not user-collected, with an optional reference attachment slot. Semantic-token naming (`fact_primary`) likely falls out for free when the same agent owns spec + tokens — worth replicating by giving one agent both the PRD and the style-guide outputs.
4. **GitHub-new-repo-only is a real ceiling.** Paraflow can't extend an existing codebase. KutAI's brownfield story (HTML import → improvement suggestions exists in Paraflow but is shallow) is wide open as a differentiator if we make i2p work *into* an existing repo with diffs, not just greenfield scaffolds.
5. **Model attribution opacity is a wedge.** Paraflow hides which LLM it uses. KutAI's local + cloud + 15-dim selection (Fatih Hoca) is the opposite — transparent, tunable, swap-aware. For founders who care about cost or privacy, "you can see exactly which model wrote your PRD and swap it" is a real differentiator that Paraflow structurally cannot match without re-architecting.
