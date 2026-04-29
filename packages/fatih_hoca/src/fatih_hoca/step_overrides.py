"""Static curated step-level token overrides.

Hand-seeded from 2026-04-28 telemetry sweep. Only known-heavy steps get
an entry. Entries override AGENT_REQUIREMENTS defaults until the learned
B-table (step_token_stats) has >= MIN_SAMPLES for that step.

Source: docs/research/2026-04-28-token-distribution.md §6 outliers.
"""
from __future__ import annotations

from fatih_hoca.estimates import Estimates


STEP_TOKEN_OVERRIDES: dict[str, Estimates] = {
    # i2p_v3 — known-heavy artifact-emit steps
    "4.5b":   Estimates(in_tokens=10_000, out_tokens=100_000, iterations=12),  # openapi_spec
    "5.4b":   Estimates(in_tokens=6_000,  out_tokens=92_000,  iterations=8),   # forms_and_states
    "3.5":    Estimates(in_tokens=10_000, out_tokens=58_000,  iterations=24),  # integration_requirements
    "4.15a1": Estimates(in_tokens=20_000, out_tokens=44_000,  iterations=6),   # backend_core_design
    "5.11b":  Estimates(in_tokens=28_000, out_tokens=43_000,  iterations=8),   # design_handoff_document
    "3.6":    Estimates(in_tokens=11_000, out_tokens=27_000,  iterations=8),   # platform_and_accessibility_requirements
    "4.5a":   Estimates(in_tokens=10_000, out_tokens=25_000,  iterations=8),   # api_resource_model
    "5.11a":  Estimates(in_tokens=13_000, out_tokens=25_000,  iterations=8),   # design_system_handoff
    "5.7":    Estimates(in_tokens=5_000,  out_tokens=23_000,  iterations=8),   # component_specs
    "3.7":    Estimates(in_tokens=10_000, out_tokens=23_000,  iterations=8),   # business_rules_extraction
}
