"""Content quality assessment orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detectors import (
    HARD_CAP,
    check_header_repetition,
    check_paragraph_repetition,
    check_size,
    check_token_entropy,
)


@dataclass
class ContentQualityResult:
    """Structured result from assess()."""

    size: int
    max_size: int
    repetition_ratio: float
    paragraph_repetition: float
    token_entropy: float
    is_degenerate: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.is_degenerate:
            return f"ok ({self.size} chars)"
        parts = []
        for reason in self.reasons:
            if reason == "size_exceeded":
                parts.append(f"size_exceeded ({self.size} > {self.max_size})")
            elif reason == "header_repetition":
                parts.append(f"header_repetition ({self.repetition_ratio:.2f})")
            elif reason == "paragraph_repetition":
                parts.append(f"paragraph_repetition ({self.paragraph_repetition:.2f})")
            elif reason == "low_entropy":
                parts.append(f"low_entropy ({self.token_entropy:.2f} bits)")
            else:
                parts.append(reason)
        return "degenerate: " + ", ".join(parts)


def assess(text, max_size: int = 20_000) -> ContentQualityResult:
    if isinstance(text, dict):
        import json
        text = json.dumps(text, ensure_ascii=False)
    elif not isinstance(text, str):
        text = str(text)
    effective_max = min(max_size, HARD_CAP)
    reasons: list[str] = []

    size_val, size_breached, size_reason = check_size(text, effective_max)
    if size_breached and size_reason:
        reasons.append(size_reason)

    header_ratio, header_breached, header_reason = check_header_repetition(text)
    if header_breached and header_reason:
        reasons.append(header_reason)

    para_ratio, para_breached, para_reason = check_paragraph_repetition(text)
    if para_breached and para_reason:
        reasons.append(para_reason)

    entropy_val, entropy_breached, entropy_reason = check_token_entropy(text)
    if entropy_breached and entropy_reason:
        reasons.append(entropy_reason)

    # Size alone is not degenerate — large but unique content (e.g. shopping
    # search aggregating 16 scrapers) is legitimate.  Only flag degenerate
    # when there is at least one *quality* signal (repetition / low entropy),
    # or when size exceeds the absolute HARD_CAP.
    quality_reasons = [r for r in reasons if r != "size_exceeded"]
    is_deg = bool(quality_reasons) or (size_val > HARD_CAP)

    return ContentQualityResult(
        size=size_val,
        max_size=effective_max,
        repetition_ratio=header_ratio,
        paragraph_repetition=para_ratio,
        token_entropy=entropy_val,
        is_degenerate=is_deg,
        reasons=reasons,
    )
