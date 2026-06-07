# router.py
"""
Model Router — now a thin back-compat shim. The legacy select_model()
scorer was deleted 2026-06-07 (zero prod callers; live selection is
fatih_hoca/ranking.py::rank_candidates). The two live residents were
relocated to their proper homes the same day:
  - ModelCallFailed -> src/core/exceptions.py
  - get_kdv()       -> src/infra/rate_limiter.py
This module re-exports all four names (plus the ModelRequirements /
ScoredModel selection types from fatih_hoca) so existing
`from src.core.router import ...` call sites keep working.
"""

from __future__ import annotations

from src.core.exceptions import ModelCallFailed
from src.infra.rate_limiter import get_kdv
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.ranking import ScoredModel

__all__ = ["ModelCallFailed", "get_kdv", "ModelRequirements", "ScoredModel"]
