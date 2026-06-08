"""Single owner of local-model context sizing (moved out of the dispatcher).

need_ctx = clamp(ceil_2048(min_context or estimate or MIN_CTX), MIN_CTX, model_ctx)

MIN_CTX=8192 is evidence-backed (kutai.jsonl 05-29/30: smallest genuine task
need 4412; bottom cluster 4412-10207; 8192 covers it with margin). Override
via env LLAMA_MIN_CTX.
"""
from __future__ import annotations

import os

MIN_CTX = int(os.environ.get("LLAMA_MIN_CTX", "8192"))
_BLOCK = 2048


def _ceil_block(n: int) -> int:
    if n <= 0:
        return 0
    return ((n + _BLOCK - 1) // _BLOCK) * _BLOCK


def compute_need_ctx(*, min_context: int, est_in: int, est_out: int, model_ctx: int) -> int:
    """Return the exact context window to load for a local model."""
    need = min_context
    if need <= 0 and (est_in or est_out):
        need = int((est_in + est_out) * 1.3) + 512
    need = _ceil_block(need) if need > 0 else MIN_CTX
    need = max(MIN_CTX, need)
    if model_ctx > 0:
        need = min(need, model_ctx)
    return need
