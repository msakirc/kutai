# rate_limiter.py
"""
Two-tier rate limiting: per-model + per-provider.

Providers like Groq enforce per-model RPM/TPM limits AND
per-account aggregate limits. We track both layers.

Supports:
- Adaptive limit discovery from 429 errors
- Dynamic limit updates from API response headers
- Daily limit tracking (Cerebras, SambaNova, Gemini)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .header_parser import RateLimitSnapshot

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Tracks request and token usage within a sliding window."""
    rpm_limit: int
    tpm_limit: int
    _request_timestamps: list[float] = field(default_factory=list)
    _token_log: list[tuple[float, int]] = field(default_factory=list)
    _rate_limit_hits: int = 0
    _last_429_at: float = 0.0
    _original_rpm: int = 0
    _original_tpm: int = 0

    # Header-derived live state
    _header_rpm_remaining: int | None = field(default=None, repr=False)
    _header_tpm_remaining: int | None = field(default=None, repr=False)
    _header_rpm_reset_at: float | None = field(default=None, repr=False)
    _header_tpm_reset_at: float | None = field(default=None, repr=False)
    _limits_discovered: bool = field(default=False, repr=False)
    _last_header_update: float = field(default=0.0, repr=False)

    # Daily limits (Cerebras, SambaNova, Gemini)
    rpd_limit: int | None = field(default=None, repr=False)
    rpd_remaining: int | None = field(default=None, repr=False)
    rpd_reset_at: float | None = field(default=None, repr=False)

    # Extended axis stubs — populated by header parser as providers expose them.
    # Adapter reads these via getattr(state, f"{axis}_limit", None).
    rph_limit: int | None = field(default=None, repr=False)
    rph_remaining: int | None = field(default=None, repr=False)
    rph_reset_at: float | None = field(default=None, repr=False)
    rpw_limit: int | None = field(default=None, repr=False)
    rpw_remaining: int | None = field(default=None, repr=False)
    rpw_reset_at: float | None = field(default=None, repr=False)
    rpmonth_limit: int | None = field(default=None, repr=False)
    rpmonth_remaining: int | None = field(default=None, repr=False)
    rpmonth_reset_at: float | None = field(default=None, repr=False)
    tph_limit: int | None = field(default=None, repr=False)
    tph_remaining: int | None = field(default=None, repr=False)
    tph_reset_at: float | None = field(default=None, repr=False)
    tpd_limit: int | None = field(default=None, repr=False)
    tpd_remaining: int | None = field(default=None, repr=False)
    tpd_reset_at: float | None = field(default=None, repr=False)
    tpw_limit: int | None = field(default=None, repr=False)
    tpw_remaining: int | None = field(default=None, repr=False)
    tpw_reset_at: float | None = field(default=None, repr=False)
    tpmonth_limit: int | None = field(default=None, repr=False)
    tpmonth_remaining: int | None = field(default=None, repr=False)
    tpmonth_reset_at: float | None = field(default=None, repr=False)
    itpm_limit: int | None = field(default=None, repr=False)
    itpm_remaining: int | None = field(default=None, repr=False)
    itpm_reset_at: float | None = field(default=None, repr=False)
    itpd_limit: int | None = field(default=None, repr=False)
    itpd_remaining: int | None = field(default=None, repr=False)
    itpd_reset_at: float | None = field(default=None, repr=False)
    otpm_limit: int | None = field(default=None, repr=False)
    otpm_remaining: int | None = field(default=None, repr=False)
    otpm_reset_at: float | None = field(default=None, repr=False)
    otpd_limit: int | None = field(default=None, repr=False)
    otpd_remaining: int | None = field(default=None, repr=False)
    otpd_reset_at: float | None = field(default=None, repr=False)
    cpd_limit: int | None = field(default=None, repr=False)
    cpd_remaining: int | None = field(default=None, repr=False)
    cpd_reset_at: float | None = field(default=None, repr=False)
    cpmonth_limit: int | None = field(default=None, repr=False)
    cpmonth_remaining: int | None = field(default=None, repr=False)
    cpmonth_reset_at: float | None = field(default=None, repr=False)

    def __post_init__(self):
        self._original_rpm = self.rpm_limit
        self._original_tpm = self.tpm_limit

    # ── Persistence ──────────────────────────────────────────────────────
    # snapshot_state captures the fields that should survive a process
    # restart: adapted limits + 429 history (so maybe_restore_limits decay
    # continues), daily counters (provider-enforced), and header-derived
    # quotas (still meaningful until reset_at). Per-minute timestamps and
    # token logs are NOT persisted — stale 60s windows lead to over-throttle.
    _PERSISTED_FIELDS = (
        "rpm_limit", "tpm_limit",
        "_rate_limit_hits", "_last_429_at",
        "_original_rpm", "_original_tpm",
        "_header_rpm_remaining", "_header_tpm_remaining",
        "_header_rpm_reset_at", "_header_tpm_reset_at",
        "_limits_discovered", "_last_header_update",
        "rpd_limit", "rpd_remaining", "rpd_reset_at",
        # Daily axes added by the parallel agent's groq parser (12d5889).
        # Without persisting these, a Groq response-then-restart cycle
        # loses the provider-reported daily-bucket position until the
        # next response.
        "tpd_limit", "tpd_remaining", "tpd_reset_at",
        "itpd_limit", "itpd_remaining", "itpd_reset_at",
        "otpd_limit", "otpd_remaining", "otpd_reset_at",
    )

    def snapshot_state(self) -> dict:
        return {k: getattr(self, k) for k in self._PERSISTED_FIELDS}

    def restore_state(self, snap: dict) -> None:
        # rpm_limit / tpm_limit are determined fresh at boot by registration
        # (discovery + static seed). Persisted values are stale whenever
        # the seed changes — production triage 2026-04-30: gemini-3-flash-
        # preview kept the pre-fix detect_cloud_model defaults (rpm=15,
        # tpm=1M) restored over the new static seed (5, 250K) for hours
        # after the fix shipped. Pool pressure saw inflated headroom,
        # selector kept picking, KDV kept refusing.
        #
        # Only restore the ADAPTIVE state (rate_limit_hits, last_429_at,
        # etc.) — persistence's job is "remember adapted reduction across
        # restarts so we don't re-discover the throttled limit". When
        # _rate_limit_hits == 0, there's no adaptive reduction to keep
        # and `rpm_limit` / `tpm_limit` should reflect the freshly
        # registered values. When > 0, persisted limits represent the
        # adapted (smaller) state and are still safer than registration's
        # pristine values.
        hits = int(snap.get("_rate_limit_hits", 0) or 0)
        skip_keys: set[str] = set()
        if hits == 0:
            # rpd_limit / tpd_limit also belong here. Production triage
            # 2026-05-01: register_cloud_from_discovered → ModelInfo.
            # rate_limit_rpd → state.rpd_limit set to 1500 at boot for
            # gemma-4-31b-it. restore_state then wrote back the persisted
            # rpd_limit=None (from the older state predating the rpd
            # propagation fix). Result: matrix.rpd cell empty → S1
            # never saw daily quota → selector kept picking saturated
            # gemini models.
            skip_keys = {
                "rpm_limit", "tpm_limit",
                "rpd_limit", "rpd_remaining", "rpd_reset_at",
                "tpd_limit", "tpd_remaining", "tpd_reset_at",
                "_original_rpm", "_original_tpm",
            }
        for k in self._PERSISTED_FIELDS:
            if k in skip_keys:
                continue
            if k in snap:
                setattr(self, k, snap[k])

    def _cleanup(self, now: float | None = None) -> None:
        """Remove entries older than 60 seconds."""
        now = now or time.time()
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < 60
        ]
        self._token_log = [
            (t, c) for t, c in self._token_log if now - t < 60
        ]

    @property
    def current_rpm(self) -> int:
        self._cleanup()
        return len(self._request_timestamps)

    @property
    def current_tpm(self) -> int:
        self._cleanup()
        return sum(c for _, c in self._token_log)

    @property
    def rpm_headroom(self) -> int:
        return max(0, self.rpm_limit - self.current_rpm)

    @property
    def tpm_headroom(self) -> int:
        return max(0, self.tpm_limit - self.current_tpm)

    # ── Adapter-facing `remaining` properties ────────────────────────────
    # The nerd_herd adapter reads `getattr(state, f"{axis}_remaining", ...)`
    # to populate matrix.rpm.remaining / matrix.tpm.remaining for S1
    # depletion arm + S2/S3 burden. Without these properties, the adapter
    # got None for every Gemini-class provider (Gemini emits nothing in
    # response headers, only on 429 body) — matrix cells stayed empty and
    # S1 returned 0.0 via the exhausted-neutral path on time_bucketed
    # pools. Selector saw no signal, kept picking the saturated model,
    # 429'd. Pool pressure was inert.
    #
    # Precedence: fresh provider header (authoritative when reset_at >
    # now and last_header_update < 5s) → sliding-window-derived
    # (rpm_limit - current_rpm). When neither is available, fall back to
    # rpm_limit so a never-called model still reports SOMETHING — the
    # full bucket — instead of None.
    # Post-429 cooldown window: how long after a 429 hit we keep
    # reporting 0 remaining regardless of sliding-window recovery.
    # Without this, the sliding window decays the oldest timestamp and
    # frees a slot ~60s after the burst — selector sees frac=1/30=0.033,
    # S1 fires -0.89 (under the strict -1.0 floor), threshold at high
    # urgency lets the model through, and we 429 again. Production
    # triage 2026-05-01: groq/llama-4-scout 1276 fails / 45 success
    # (3%), groq/llama-3.3 811/2 (0%) — selector kept thrashing on
    # marginal capacity. 60s matches the typical RPM window, longer
    # than the keepalive interval, shorter than DLQ retry.
    POST_429_COOLDOWN_SECONDS: float = 60.0

    @property
    def in_post_429_cooldown(self) -> bool:
        """True if a 429 fired within the cooldown window. Used to gate
        rpm_remaining at 0 even when the sliding window has decayed
        enough to suggest free capacity."""
        if self._last_429_at <= 0:
            return False
        return (time.time() - self._last_429_at) < self.POST_429_COOLDOWN_SECONDS

    def _header_rpm_valid(self, now: float) -> bool:
        """Provider header value is authoritative when the bucket window
        it described hasn't elapsed yet. Past reset_at means the bucket
        refilled; old `remaining` is stale. Without reset_at, fall back
        to a 60s safety window matching the rpm bucket — after that
        without confirmation, assume stale. RateLimitManager.record_request
        decrements `_header_rpm_remaining` on each admission so the
        projection stays accurate between provider responses, removing
        the prior 5s freshness limit which made every burst ride
        sliding-window estimates instead of provider's authoritative view."""
        if self._header_rpm_remaining is None:
            return False
        if self._header_rpm_reset_at is not None:
            return now < self._header_rpm_reset_at
        return (now - self._last_header_update) < 60.0

    def _header_tpm_valid(self, now: float) -> bool:
        """Mirror of `_header_rpm_valid` for the token axis."""
        if self._header_tpm_remaining is None:
            return False
        if self._header_tpm_reset_at is not None:
            return now < self._header_tpm_reset_at
        return (now - self._last_header_update) < 60.0

    @property
    def rpm_remaining(self) -> int | None:
        if self.rpm_limit <= 0:
            return None
        # Hard cooldown after a recent 429 — overrides both fresh
        # provider headers and sliding-window math. The provider just
        # told us we're over; sliding-window slot recovery is misleading.
        if self.in_post_429_cooldown:
            return 0
        now = time.time()
        if self._header_rpm_valid(now):
            return int(self._header_rpm_remaining)
        return max(0, self.rpm_limit - self.current_rpm)

    @property
    def tpm_remaining(self) -> int | None:
        if self.tpm_limit <= 0:
            return None
        now = time.time()
        if self._header_tpm_valid(now):
            return int(self._header_tpm_remaining)
        return max(0, self.tpm_limit - self.current_tpm)

    @property
    def rpm_reset_at(self) -> float | None:
        """Real-time recovery hint: when does the RPM bucket clear next?

        Provider header wins when present (authoritative). Otherwise
        compute from oldest sliding-window entry: that entry expires at
        `oldest + 60s`, which is when the next call slot opens. Pool
        pressure consumers (S9 perishability, S1 time-decay) use this
        to know HOW LONG until recovery — answers the user's question
        without a quarantine timer.

        When no calls and no header → None (no info, signal stays
        neutral).
        """
        if self._header_rpm_reset_at is not None:
            return float(self._header_rpm_reset_at)
        if self._request_timestamps:
            return float(min(self._request_timestamps)) + 60.0
        return None

    @property
    def tpm_reset_at(self) -> float | None:
        if self._header_tpm_reset_at is not None:
            return float(self._header_tpm_reset_at)
        if self._token_log:
            return float(min(t for t, _ in self._token_log)) + 60.0
        return None

    def has_capacity(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made without waiting."""
        now = time.time()
        # Daily limit exhaustion is absolute. Both RPD and TPD axes
        # gate independently — Groq free-tier llama-3.3-70b-versatile
        # has TPD=100K with RPD effectively unbounded for token-heavy
        # workloads, so an RPD-only check left TPD-bound calls free
        # to bash the wall (production triage 2026-05-08 task #14618:
        # 97775/100000 TPD used, next request 5797 tokens, KDV said
        # OK because RPD wasn't hit, Groq returned 429 with 51m26s
        # cooldown). Proactive TPD gate prevents the hit.
        if self.rpd_remaining is not None and self.rpd_remaining <= 0:
            if self.rpd_reset_at and now < self.rpd_reset_at:
                return False
        if self.tpd_remaining is not None:
            # Reject when remaining can't fit the estimate (mirrors
            # tpm gate logic). Strict < admits cold-start calls where
            # remaining == estimate.
            if self.tpd_remaining < max(estimated_tokens, 1):
                if self.tpd_reset_at and now < self.tpd_reset_at:
                    return False

        # Use header-derived remaining when valid (within reset window).
        # Local-decrement in record_request keeps `_header_rpm_remaining`
        # accurate between responses, so the projection stays sharp
        # without reverting to sliding-window math after 5 seconds.
        if self._header_rpm_valid(now):
            rpm_ok = self._header_rpm_remaining > 1
        else:
            rpm_ok = self.rpm_headroom > 1

        # Use >= so cold-start calls where headroom EQUALS the estimate are
        # admitted (e.g. qwen3-32b free tier tpm=6000, researcher estimate=6000).
        # Strict > made any tight-limit model unusable for the largest typical call.
        if self._header_tpm_valid(now):
            tpm_ok = self._header_tpm_remaining >= estimated_tokens
        else:
            tpm_ok = self.tpm_headroom >= estimated_tokens

        return rpm_ok and tpm_ok

    def utilization_pct(self) -> float:
        """How close to limits we are (0-100). Includes daily axes
        so pool-pressure scarcity downranks models nearing TPD/RPD
        before the gate slams shut. Pre-2026-05-08 only minute axes
        were considered; selector kept ranking groq/llama-3.3-70b
        first while it sat at 97% TPD."""
        rpm_pct = (self.current_rpm / self.rpm_limit * 100) if self.rpm_limit else 0
        tpm_pct = (self.current_tpm / self.tpm_limit * 100) if self.tpm_limit else 0
        rpd_pct = 0.0
        if self.rpd_limit and self.rpd_remaining is not None:
            rpd_pct = (self.rpd_limit - self.rpd_remaining) / self.rpd_limit * 100
        tpd_pct = 0.0
        if self.tpd_limit and self.tpd_remaining is not None:
            tpd_pct = (self.tpd_limit - self.tpd_remaining) / self.tpd_limit * 100
        return max(rpm_pct, tpm_pct, rpd_pct, tpd_pct)

    async def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Wait until rate limit allows a request.
        Returns seconds waited (0 if no wait needed).
        Returns -1.0 if daily limit exhausted (caller should skip model).
        """
        waited = 0.0
        now = time.time()
        self._cleanup(now)

        # Daily limit check — if exhausted, signal skip. Both axes
        # gate independently; TPD gate added 2026-05-08 (Groq free
        # tier llama-3.3-70b-versatile hit TPD wall while RPD had
        # plenty of headroom — caller had no proactive signal).
        if self.rpd_remaining is not None and self.rpd_remaining <= 0:
            if self.rpd_reset_at and self.rpd_reset_at > now:
                logger.warning(
                    f"Rate limiter: RPD daily exhausted, "
                    f"resets in {self.rpd_reset_at - now:.0f}s"
                )
                return -1.0
        if self.tpd_remaining is not None and self.tpd_remaining < max(estimated_tokens, 1):
            if self.tpd_reset_at and self.tpd_reset_at > now:
                logger.warning(
                    f"Rate limiter: TPD daily exhausted "
                    f"(remaining={self.tpd_remaining}, est={estimated_tokens}), "
                    f"resets in {self.tpd_reset_at - now:.0f}s"
                )
                return -1.0

        header_fresh = (now - self._last_header_update) < 10.0

        # RPM check — prefer header reset time if available
        if header_fresh and self._header_rpm_remaining is not None and self._header_rpm_remaining <= 1:
            if self._header_rpm_reset_at and self._header_rpm_reset_at > now:
                rpm_wait = self._header_rpm_reset_at - now + 0.5
                logger.info(
                    f"Rate limiter: RPM wait {rpm_wait:.1f}s (from headers, "
                    f"remaining={self._header_rpm_remaining})"
                )
                await asyncio.sleep(rpm_wait)
                waited += rpm_wait
                self._header_rpm_remaining = None  # stale after wait
            elif len(self._request_timestamps) >= self.rpm_limit:
                oldest = self._request_timestamps[0]
                rpm_wait = 60 - (now - oldest) + 0.5
                if rpm_wait > 0:
                    logger.info(
                        f"Rate limiter: RPM wait {rpm_wait:.1f}s "
                        f"({self.current_rpm}/{self.rpm_limit})"
                    )
                    await asyncio.sleep(rpm_wait)
                    waited += rpm_wait
                    self._cleanup()
        elif len(self._request_timestamps) >= self.rpm_limit:
            oldest = self._request_timestamps[0]
            rpm_wait = 60 - (now - oldest) + 0.5
            if rpm_wait > 0:
                logger.info(
                    f"Rate limiter: RPM wait {rpm_wait:.1f}s "
                    f"({self.current_rpm}/{self.rpm_limit})"
                )
                await asyncio.sleep(rpm_wait)
                waited += rpm_wait
                self._cleanup()

        # TPM check
        if estimated_tokens > 0 and self.current_tpm + estimated_tokens > self.tpm_limit:
            if header_fresh and self._header_tpm_reset_at and self._header_tpm_reset_at > time.time():
                tpm_wait = self._header_tpm_reset_at - time.time() + 0.5
            elif self._token_log:
                oldest_token_ts = self._token_log[0][0]
                tpm_wait = 60 - (time.time() - oldest_token_ts) + 0.5
            else:
                tpm_wait = 0
            if tpm_wait > 0:
                logger.info(
                    f"Rate limiter: TPM wait {tpm_wait:.1f}s "
                    f"({self.current_tpm}/{self.tpm_limit})"
                )
                await asyncio.sleep(tpm_wait)
                waited += tpm_wait
                self._cleanup()

        self._request_timestamps.append(time.time())
        return waited

    def record_tokens(self, token_count: int) -> None:
        """Record actual token usage after a call completes."""
        self._token_log.append((time.time(), token_count))

    def record_429(self) -> None:
        """
        Record a rate limit hit. Adaptively lower limits.

        If we're getting 429s below our configured limit, the provider's
        actual limit is lower than what we think. Reduce by 20% each hit,
        minimum 50% of original.
        """
        self._rate_limit_hits += 1
        self._last_429_at = time.time()

        # Adaptive reduction
        min_rpm = max(1, self._original_rpm // 2)
        min_tpm = max(1000, self._original_tpm // 2)

        new_rpm = max(min_rpm, int(self.rpm_limit * 0.8))
        new_tpm = max(min_tpm, int(self.tpm_limit * 0.8))

        if new_rpm < self.rpm_limit or new_tpm < self.tpm_limit:
            logger.warning(
                f"Rate limit adapted: RPM {self.rpm_limit}→{new_rpm}, "
                f"TPM {self.tpm_limit}→{new_tpm} "
                f"(hit #{self._rate_limit_hits})"
            )
            self.rpm_limit = new_rpm
            self.tpm_limit = new_tpm

    def maybe_restore_limits(self) -> None:
        """
        If we haven't hit a 429 in 10 minutes, gradually restore limits.
        Called periodically by the watchdog.
        """
        if self._rate_limit_hits == 0:
            return

        now = time.time()
        if now - self._last_429_at < 600:  # 10 minutes
            return

        # Restore 10% toward original
        if self.rpm_limit < self._original_rpm:
            self.rpm_limit = min(
                self._original_rpm,
                self.rpm_limit + max(1, self._original_rpm // 10),
            )
        if self.tpm_limit < self._original_tpm:
            self.tpm_limit = min(
                self._original_tpm,
                self.tpm_limit + max(100, self._original_tpm // 10),
            )

        logger.debug(
            f"Rate limits partially restored: "
            f"RPM={self.rpm_limit}/{self._original_rpm}, "
            f"TPM={self.tpm_limit}/{self._original_tpm}"
        )

    def update_from_snapshot(self, snap: RateLimitSnapshot) -> None:
        """Update state from parsed response headers."""
        now = time.time()
        self._last_header_update = now

        # Update limits if provider reports them
        if snap.rpm_limit is not None and snap.rpm_limit != self.rpm_limit:
            logger.info(
                f"Rate limit discovered: RPM {self.rpm_limit}→{snap.rpm_limit}"
            )
            self.rpm_limit = snap.rpm_limit
            self._original_rpm = snap.rpm_limit
        if snap.tpm_limit is not None and snap.tpm_limit != self.tpm_limit:
            logger.info(
                f"Rate limit discovered: TPM {self.tpm_limit}→{snap.tpm_limit}"
            )
            self.tpm_limit = snap.tpm_limit
            self._original_tpm = snap.tpm_limit

        # If we discovered real limits, clear any adaptive reductions
        if snap.rpm_limit is not None or snap.tpm_limit is not None:
            self._limits_discovered = True
            if self._rate_limit_hits > 0:
                self._rate_limit_hits = 0
                self._last_429_at = 0.0

        # Store remaining counts (ground truth from provider)
        if snap.rpm_remaining is not None:
            self._header_rpm_remaining = snap.rpm_remaining
        if snap.tpm_remaining is not None:
            self._header_tpm_remaining = snap.tpm_remaining

        # Store reset timestamps
        if snap.rpm_reset_at is not None:
            self._header_rpm_reset_at = snap.rpm_reset_at
        if snap.tpm_reset_at is not None:
            self._header_tpm_reset_at = snap.tpm_reset_at

        # Daily limits
        if snap.rpd_limit is not None:
            self.rpd_limit = snap.rpd_limit
        if snap.rpd_remaining is not None:
            self.rpd_remaining = snap.rpd_remaining
        if snap.rpd_reset_at is not None:
            self.rpd_reset_at = snap.rpd_reset_at

        # Token-day limits (Groq/Gemini paid tiers)
        if snap.tpd_limit is not None:
            self.tpd_limit = snap.tpd_limit
        if snap.tpd_remaining is not None:
            self.tpd_remaining = snap.tpd_remaining
        if snap.tpd_reset_at is not None:
            self.tpd_reset_at = snap.tpd_reset_at

        # Input/output token splits (Anthropic exposes minute; tiers expose day)
        for axis in ("itpm", "itpd", "otpm", "otpd"):
            for suffix in ("limit", "remaining", "reset_at"):
                attr = f"{axis}_{suffix}"
                v = getattr(snap, attr, None)
                if v is not None:
                    setattr(self, attr, v)

        # `Retry-After` header is the provider's authoritative "do not call
        # before T" hint. Treat it as an rpm-bucket floor: force remaining=0
        # and raise reset_at to max(existing, now+retry_after). Stricter
        # signals win — a bucket reset_at = 5s does not override a provider
        # retry-after of 30s. This matters for degraded 429s where bucket
        # headers are absent or stale (Cerebras 429 with retry-after=12s but
        # no x-ratelimit-reset-* fields, observed during burst spikes).
        if snap.retry_after_seconds is not None and snap.retry_after_seconds > 0:
            floor = now + snap.retry_after_seconds
            if self._header_rpm_reset_at is None or floor > self._header_rpm_reset_at:
                self._header_rpm_reset_at = floor
            self._header_rpm_remaining = 0


class RateLimitManager:
    """
    Two-tier rate limiting manager.

    Tier 1: Per-model limits (e.g., groq/llama-8b has its own RPM)
    Tier 2: Per-provider aggregate (e.g., all Groq models share account quota)

    A request must pass BOTH tiers to proceed.
    """

    def __init__(self):
        self.model_limits: dict[str, RateLimitState] = {}
        self._provider_limits: dict[str, RateLimitState] = {}

    def register_model(
        self,
        litellm_name: str,
        provider: str,
        rpm: int,
        tpm: int,
        provider_aggregate_rpm: int | None = None,
        provider_aggregate_tpm: int | None = None,
    ) -> None:
        """
        Register rate limits for a model.

        Args:
            litellm_name: unique model identifier
            provider: provider name (for aggregate tracking)
            rpm: per-model requests per minute
            tpm: per-model tokens per minute
            provider_aggregate_rpm: shared provider-level RPM (optional)
            provider_aggregate_tpm: shared provider-level TPM (optional)
        """
        if litellm_name not in self.model_limits:
            self.model_limits[litellm_name] = RateLimitState(
                rpm_limit=rpm, tpm_limit=tpm,
            )

        # Provider aggregate — only create once per provider. When BOTH
        # aggregate values are None (caller signal: provider does not enforce
        # an account-wide cap), skip creation entirely so per-model buckets
        # do the gating. has_capacity/record_*/update_from_headers all guard
        # on `if provider_state:` so a missing entry short-circuits cleanly.
        if provider not in self._provider_limits:
            if provider_aggregate_rpm is None and provider_aggregate_tpm is None:
                pass
            else:
                agg_rpm = provider_aggregate_rpm or rpm * 3  # heuristic
                agg_tpm = provider_aggregate_tpm or tpm * 3
                self._provider_limits[provider] = RateLimitState(
                    rpm_limit=agg_rpm, tpm_limit=agg_tpm,
                )

    def has_capacity(
        self,
        litellm_name: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> bool:
        """Check if both model and provider have capacity."""
        model_state = self.model_limits.get(litellm_name)
        provider_state = self._provider_limits.get(provider)

        model_ok = model_state.has_capacity(estimated_tokens) if model_state else True
        provider_ok = provider_state.has_capacity(estimated_tokens) if provider_state else True

        return model_ok and provider_ok

    async def wait_and_acquire(
        self,
        litellm_name: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> float:
        """
        Wait for both model and provider limits, then record request.
        Returns total seconds waited, or -1.0 if daily limit exhausted.
        """
        total_waited = 0.0

        # Wait on model limit first
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            result = await model_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0
            total_waited += result

        # Then wait on provider aggregate
        provider_state = self._provider_limits.get(provider)
        if provider_state:
            result = await provider_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0
            total_waited += result

        return total_waited

    def record_request(
        self,
        litellm_name: str,
        provider: str,
    ) -> None:
        """Record a request timestamp for RPM tracking AND decrement
        the daily RPD counter when a static cap is known.

        Daily decrement runs locally so providers that don't return
        rpd headers (notably Gemini's free tier — 20 req/day with no
        x-ratelimit-* response headers) still surface daily exhaustion
        BEFORE the first 429 body parse. Production 2026-05-02
        15:00-17:00 UTC: every gemini admit hit a 429 because
        is_daily_exhausted only flipped after the response body parser
        ran — by which time the burst had already committed many calls
        to the wall.

        Per-minute (`_header_rpm_remaining`) is also decremented on each
        admission so the projection stays accurate between provider
        responses. Header from response is authoritative; we maintain it
        until the next response replaces it. Replaced an earlier ghost-
        timestamp backfill (reverted c02e3ee) — local decrement is the
        symmetric counterpart to TPM reservation already in place.

        Reset window: 24h rolling from first call of the day. When
        rpd_reset_at falls in the past, both rpd_remaining and the
        reset clock are refreshed. Header-derived updates from
        update_from_headers still take precedence (line 492-495 in
        update_from_headers) so providers that DO send headers
        (groq) override our local count with their authoritative one.
        """
        import time
        now = time.time()
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state._request_timestamps.append(now)
            self._decrement_header_rpm(model_state)
            self._tick_rpd(model_state, now)

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state._request_timestamps.append(now)
            self._decrement_header_rpm(provider_state)
            self._tick_rpd(provider_state, now)

    @staticmethod
    def _decrement_header_rpm(state) -> None:
        """Locally project RPM consumption between provider responses.
        Provider's response sets `_header_rpm_remaining` authoritatively
        via update_from_snapshot; we decrement on each admission so the
        next admission sees the projected value without waiting for the
        next response. Floors at 0 — has_capacity reads the value via
        the `rpm_remaining` property which gates admission appropriately.
        Cap is the runtime guard; provider's next response is the truth."""
        if state._header_rpm_remaining is not None and state._header_rpm_remaining > 0:
            state._header_rpm_remaining -= 1

    @staticmethod
    def _next_utc_midnight(now: float) -> float:
        """Compute the next UTC midnight after `now` as a unix timestamp.

        Calendar-based daily windows align with most provider quota
        boundaries (gemini, openai). Rolling 24h-from-first-call drifts
        by hours each day; midnight UTC stays anchored. If a provider
        actually resets at a different time-of-day, the 429 body parser
        refines `rpd_reset_at` from the server's `retryDelay` on the
        first wall-hit and the local count corrects.
        """
        import datetime as _dt
        d = _dt.datetime.utcfromtimestamp(now).date()
        next_midnight = _dt.datetime.combine(d, _dt.time(0, 0)) + _dt.timedelta(days=1)
        return next_midnight.replace(tzinfo=_dt.timezone.utc).timestamp()

    @staticmethod
    def _tick_rpd(state, now: float) -> None:
        """Decrement rpd_remaining by 1 for a single attempt.

        No-op when rpd_limit is unknown — providers without a daily
        cap leave both fields None and we can't manufacture one.
        Refreshes the day window when rpd_reset_at has passed.
        Calendar-based reset (next UTC midnight) replaces the earlier
        rolling-24h approach so windows align with provider quotas.
        """
        if state.rpd_limit is None:
            return
        if state.rpd_reset_at and now >= state.rpd_reset_at:
            state.rpd_remaining = state.rpd_limit
            state.rpd_reset_at = RateLimitManager._next_utc_midnight(now)
        if state.rpd_remaining is None:
            state.rpd_remaining = state.rpd_limit
        if state.rpd_reset_at is None:
            state.rpd_reset_at = RateLimitManager._next_utc_midnight(now)
        if state.rpd_remaining > 0:
            state.rpd_remaining -= 1

    def record_tokens(
        self,
        litellm_name: str,
        provider: str,
        token_count: int,
    ) -> None:
        """Record actual token usage after a call completes."""
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.record_tokens(token_count)

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.record_tokens(token_count)

    def record_429(
        self,
        litellm_name: str,
        provider: str,
    ) -> None:
        """Record a rate limit error — adaptively reduces limits."""
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.record_429()

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.record_429()

    def mark_daily_exhausted(
        self,
        litellm_name: str,
        provider: str,
        axis: str = "tpd",
        retry_seconds: float = 3600.0,
    ) -> None:
        """Slam the daily axis shut for this model based on a 429 body
        that named the daily limit. has_capacity / wait_if_needed will
        refuse subsequent admissions until reset_at elapses.

        Without this, KDV had no way to learn from the 429 itself —
        only header-derived state could populate rpd/tpd_remaining,
        and headers don't always arrive before the wall (cold start,
        edge cases). Production triage 2026-05-08: Groq's 429 body
        named "tokens per day (TPD)" with a precise reset duration,
        but the recovery path threw all of that away."""
        model_state = self.model_limits.get(litellm_name)
        if model_state is None:
            return
        reset_at = time.time() + max(retry_seconds, 60.0)
        if axis == "rpd":
            model_state.rpd_remaining = 0
            model_state.rpd_reset_at = reset_at
        else:
            # TPD: zero out remaining so has_capacity refuses any
            # estimate; reset_at gates the wait_if_needed skip.
            model_state.tpd_remaining = 0
            model_state.tpd_reset_at = reset_at

    def update_from_headers(
        self,
        litellm_name: str,
        provider: str,
        snapshot: RateLimitSnapshot,
    ) -> None:
        """
        Update rate limit state from parsed response headers.
        Updates both model-level and provider-level state.
        """
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.update_from_snapshot(snapshot)

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.update_from_snapshot(snapshot)

    def restore_limits(self) -> None:
        """Periodically called to gradually restore adapted limits."""
        for state in self.model_limits.values():
            state.maybe_restore_limits()
        for state in self._provider_limits.values():
            state.maybe_restore_limits()

    def get_utilization(self, litellm_name: str) -> float:
        """Get utilization percentage for a model (0-100)."""
        state = self.model_limits.get(litellm_name)
        return state.utilization_pct() if state else 0.0

    def get_provider_utilization(self, provider: str) -> float:
        """Get utilization percentage for a provider (0-100)."""
        state = self._provider_limits.get(provider)
        return state.utilization_pct() if state else 0.0

    def is_daily_exhausted(self, litellm_name: str) -> bool:
        """Check if a model's daily request limit is exhausted."""
        state = self.model_limits.get(litellm_name)
        if not state:
            return False
        if state.rpd_remaining is not None and state.rpd_remaining <= 0:
            if state.rpd_reset_at and time.time() < state.rpd_reset_at:
                return True
        return False

    def is_rpm_cooldown(self, litellm_name: str) -> bool:
        """True iff a provider Retry-After / x-ratelimit-reset has installed
        a future floor on rpm with remaining=0. Reads raw `_header_*` fields,
        not the freshness-windowed `rpm_remaining` property — retry-after
        values can exceed the 5s freshness window, after which the property
        falls back to sliding-window math and the cooldown floor would
        become invisible to the selector. Authoritative cooldown signal,
        siblings `is_daily_exhausted`."""
        state = self.model_limits.get(litellm_name)
        if not state:
            return False
        if (
            state._header_rpm_remaining is not None
            and state._header_rpm_remaining <= 0
            and state._header_rpm_reset_at is not None
            and state._header_rpm_reset_at > time.time()
        ):
            return True
        return False

    def get_status(self) -> dict:
        """Full status for diagnostics."""
        models = {}
        for name, state in self.model_limits.items():
            models[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
                "discovered": state._limits_discovered,
            }

        providers = {}
        for name, state in self._provider_limits.items():
            providers[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
                "discovered": state._limits_discovered,
            }

        return {"models": models, "providers": providers}

    @property
    def provider_limits(self):
        return self._provider_limits
