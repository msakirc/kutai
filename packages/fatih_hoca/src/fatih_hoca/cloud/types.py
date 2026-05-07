"""Shared dataclasses for cloud discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderStatus = Literal["ok", "auth_fail", "server_error", "network_error", "rate_limited"]


@dataclass
class DiscoveredModel:
    """One model surfaced by a provider's /models endpoint.

    Carries litellm_name plus opportunistically scraped fields. Adapter
    populates whatever the provider response actually contains; missing
    fields stay None and downstream code falls back to litellm db / defaults.
    """
    litellm_name: str
    raw_id: str
    active: bool = True
    # Output modality: "text" (chat/completion), "image" (text→image gen),
    # "audio" (TTS / speech), "embedding" (vector), "video". The 15-dim
    # capability vector is text-task oriented; non-text models would score
    # high on irrelevant dims and be picked for coder/reviewer tasks. Adapter
    # detects modality from /models response (supportedGenerationMethods,
    # name patterns) so registry can skip non-text registrations.
    output_modality: str = "text"
    context_length: int | None = None
    max_output_tokens: int | None = None
    # Per-request input-token cap, distinct from context_length. Some
    # provider tiers gate single-request input below the model's full
    # context window — Groq free-tier compound/compound-mini accept
    # ~131K context but reject single requests over ~6-12K with
    # HTTP 413 `request_too_large`. None = no known cap (use
    # context_length as the ceiling). Adapter populates from a
    # provider-specific table when the cap is documented.
    max_input_tokens: int | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    rate_limit_rpd: int | None = None
    rate_limit_tpd: int | None = None
    supports_function_calling: bool | None = None
    sampling_defaults: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """Outcome of one provider's discovery probe."""
    provider: str
    status: ProviderStatus
    auth_ok: bool
    models: list[DiscoveredModel] = field(default_factory=list)
    error: str | None = None
    served_from_cache: bool = False
    fetched_at: str | None = None
