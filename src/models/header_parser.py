# src/models/header_parser.py
"""Shim — delegates to kuleden_donen_var package.

All real logic lives in packages/kuleden_donen_var/.
This file preserves import paths during migration.
"""
from kuleden_donen_var.header_parser import (  # noqa: F401
    RateLimitSnapshot,
    parse_rate_limit_headers,
)
