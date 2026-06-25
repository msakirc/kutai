# infra/tracing.py — back-compat shim.
#
# Per-task execution tracing moved to the `kara_kutu` leaf package
# (flight recorder). This shim is kept ONLY for src/ core callers still on the
# old path (telegram_bot). New code: import `kara_kutu`.
# Sub-packages must NOT import this — they import `kara_kutu` directly.
from kara_kutu.tracing import (  # noqa: F401
    append_trace,
    get_trace,
    format_trace,
)
