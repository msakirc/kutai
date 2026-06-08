# context_sizing.py — local-load context-window sizing POLICY (KutAI app-side).
"""How big a context window a local model load actually needs.

This is KutAI policy, NOT generic llama-server lifecycle — which is why it
lives here and not in the DaLLaMa package. The MIN_CTX floor below is tuned
from KutAI's own task corpus (kutai.jsonl); pushing that into the published,
generic DaLLaMa loader would pollute it with one consumer's tuning.
``LocalModelManager.ensure_model`` calls ``need_ctx()`` and hands the result
to DaLLaMa as a plain ServerConfig.context_length — DaLLaMa stays generic.

need-ctx (2026-05-31): a local model loads at the call's REAL need, not at a
VRAM-derived size bumped to a fixed floor. ``--fit`` (default-on) then fits
GPU layers to that ctx + live VRAM, so a small need fits comfortably even
under a transient VRAM spike — no VRAM math here to get wrong. This inverts
the old model (ctx derived from VRAM, then floored to 16384) that OOM'd a 9B
and starved the researcher (mission_79).
Direction + rationale: docs/handoff/2026-05-31-load-mode-redesign-ideas.md.

  need_ctx = clamp(ceil_2048(task_min_ctx or MIN_CTX), MIN_CTX, model.context_length)

Why MIN_CTX = 8192 (kutai.jsonl 05-29/30): smallest genuine task need = 4412,
bottom cluster 4412–10207; 8192 covers it, 4096 has zero margin. The only
reloads ever observed were upward from ≥16384 (18k–28k tasks that carry their
own min_context) — handled by the reload-on-expansion guard in ensure_model.
"""
import os

MIN_CTX = int(os.environ.get("LLAMA_MIN_CTX", "8192"))


def need_ctx(task_min_ctx: int, model_ctx_ceiling: int) -> int:
    """Context a local load actually needs: the task requirement floored to
    MIN_CTX, rounded up to a 2048-block (KV alignment / anti-churn), capped at
    the model's trained window. No VRAM math — ``--fit`` owns layer fitting.
    """
    need = max(task_min_ctx if task_min_ctx > 0 else MIN_CTX, MIN_CTX)
    need = ((need + 2047) // 2048) * 2048
    return min(need, model_ctx_ceiling) if model_ctx_ceiling > 0 else need
