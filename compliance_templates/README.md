# Compliance template root (Z1 Tier 5A — P6)

Hand-curated templates only. **Never** agent-generated. Each template
has a sibling `.meta.json` carrying `{"version": "...", "last_reviewed": "YYYY-MM-DD"}`.

## Layout

```
compliance_templates/
  <jurisdiction>/<lang>/<doc_type>.md.j2
  <jurisdiction>/<lang>/<doc_type>.meta.json
  default/en/<doc_type>.md.j2
  default/en/<doc_type>.meta.json
```

## Lookup

`src/tools/compliance_templates.py::compliance_template_render` resolves
in this order:

1. `<jurisdiction>/<lang>/<doc_type>.md.j2`
2. `default/<lang>/<doc_type>.md.j2`
3. `default/en/<doc_type>.md.j2`

## Stale policy

`STALE_DAYS = 180`. Older templates render but the result is flagged
`stale=true`. Reviewers decide whether to block — the renderer never blocks.

## Doc types

- `privacy_policy`
- `cookie_banner`
- `dpa` (Data Processing Agreement)
- `tos`
- `retention_policy`
- `age_gate`
- `accessibility_statement`
- `data_processing_record`

## Adding a template

1. Hand-write the `.md.j2` Jinja template using fingerprint fields as variables.
2. Drop a `.meta.json` next to it with version + last_reviewed.
3. Reference the doc type from `1.11a compliance_overlay`'s
   `compliance_template_render` calls.
4. Bump `last_reviewed` after every legal review.
