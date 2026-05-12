# cve_scan (docker)

V1 delegates to `trivy image` when on PATH; otherwise the executor
returns `skipped=true`. OSV.dev query is best-effort fallback once V2
parses base-image OS package lists.
