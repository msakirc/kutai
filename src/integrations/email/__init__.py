"""Z7 T2A — Email-send shared service package.

Provider-abstracted, per-product email service.
Free-tier-first: Brevo (300/day) and Resend (3k/mo) are the default adapters.
Paid adapters (Postmark, SES) are stubs that raise NotImplementedError
until a product earns revenue.
"""
from __future__ import annotations
