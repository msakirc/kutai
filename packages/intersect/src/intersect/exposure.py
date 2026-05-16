"""Exposure-class decision. Thresholds tunable via ops; defaults strict."""
from __future__ import annotations

# θ_preempt > θ_inject > θ_tool > θ_min. Conservative defaults; lowered
# later based on yalayut_usage success-rate telemetry.
THETA_PREEMPT: float = 0.80
THETA_INJECT: float = 0.55
THETA_TOOL: float = 0.45
THETA_MIN: float = 0.30
