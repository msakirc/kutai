"""Core exceptions shared across the dispatch/selection path."""

from __future__ import annotations


class ModelCallFailed(RuntimeError):
    """All model candidates exhausted — no backpressure, no retry.

    Raised by the dispatcher when every candidate fails. The caller
    (process_task) catches this and puts the task to sleep, waiting
    for a signal that capacity has changed.
    """

    def __init__(self, call_id: str, last_error: str, error_category: str):
        super().__init__(f"All models failed for '{call_id}': {last_error}")
        self.call_id = call_id
        self.last_error = last_error
        self.error_category = error_category
