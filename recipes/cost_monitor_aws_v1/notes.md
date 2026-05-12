# cost_monitor (aws)

`cost_pull` calls Cost Explorer for yesterday's total. Skipped at the
executor level when AWS creds aren't present (gate via
`needs_real_tools`).
