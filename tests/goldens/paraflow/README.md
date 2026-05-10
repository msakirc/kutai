# Paraflow goldens — Z1 Tier 7B (C21)

Reference output bundles produced by Paraflow (the SaaS) for archetypal
products. Used by `packages/c21_paraflow_diff/` to regression-test KutAI
mission output for **coverage**, **coherence**, and **design fitness**.

## Layout

```
goldens/paraflow/
  MANIFEST.json                       — index of archetypes
  README.md                           — this file
  truthrate/                          — universal review platform
    charter.md
    personas.md
    prd.md
    user_flow.md
    style_guide_light.md
    style_guide_dark.md
    screen_plans/  (subset: 5 of 24)
    screens/       (subset: 5 of 24)
```

## How these were captured

`truthrate/` content was copied **live** on 2026-05-10 from:

```
C:/Users/sakir/Dropbox/Workspaces/Bilinc/main/paraflow/
```

The 5-screen subset (home, search, sign_in, profile, write_review) is
representative of the 24-screen full bundle and keeps repo bloat down.

## Adding a new archetype

1. Create `goldens/paraflow/<archetype>/` with the same artifact layout.
2. Copy charter / personas / PRD / user_flow / 5 screen plans + HTMLs /
   light + dark style guides.
3. Add an entry in `MANIFEST.json` — record `paraflow_source`,
   `captured_at`, and which screens were sampled.
4. Re-run `pytest packages/c21_paraflow_diff/tests/` to confirm the new
   archetype loads cleanly.

## Stub fallback

If the paraflow source path is unavailable on a given machine (CI, fresh
clone), drop placeholder files containing `# TODO: real paraflow content`
into the archetype directory. The diff harness still runs against stubs;
verdicts will skew toward `paraflow_partial` until real content lands.
