"""In-flight tracker for cloud calls.

Tracks calls dispatched but not yet confirmed (header-derived rate-limit
update hasn't landed). Purpose: close the t=0 burst-admission window
for pools near depletion.

TTL safety net: if dispatcher dies mid-call (crash, SIGKILL, power loss)
and `end_call` never fires, the handle is pruned lazily on next
`begin_call` or `count` after `ttl_s` elapses. Default 180s — 2x the
worst observed legitimate cloud call duration.

State is in-process only. Intended usage via the module-level singleton
exposed from `kuleden_donen_var.__init__`.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

DEFAULT_TTL_S = float(os.environ.get("KDV_INFLIGHT_TTL_S", "180"))


@dataclass(frozen=True)
class InFlightHandle:
    provider: str
    model: str
    started_at: float
    ttl_s: float
    token: str


class InFlightTracker:
    """Per-(provider, model) handle lists with lazy TTL pruning."""

    def __init__(self) -> None:
        self._handles: dict[tuple[str, str], list[InFlightHandle]] = {}

    def _prune(self, key: tuple[str, str]) -> None:
        bucket = self._handles.get(key)
        if not bucket:
            return
        now = time.time()
        self._handles[key] = [h for h in bucket if h.started_at + h.ttl_s > now]

    def begin_call(
        self,
        provider: str,
        model: str,
        ttl_s: float = DEFAULT_TTL_S,
    ) -> InFlightHandle:
        key = (provider, model)
        self._prune(key)
        handle = InFlightHandle(
            provider=provider,
            model=model,
            started_at=time.time(),
            ttl_s=ttl_s,
            token=str(uuid.uuid4()),
        )
        self._handles.setdefault(key, []).append(handle)
        return handle

    def end_call(self, handle: InFlightHandle) -> None:
        key = (handle.provider, handle.model)
        bucket = self._handles.get(key, [])
        # Filter by token — idempotent: second end_call on same handle is a no-op.
        self._handles[key] = [h for h in bucket if h.token != handle.token]

    def count(self, provider: str, model: str) -> int:
        key = (provider, model)
        self._prune(key)
        return len(self._handles.get(key, []))
