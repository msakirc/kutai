"""Z7 T2A — Email provider registry.

Maps provider name (as stored in product_email_config.provider) to
its adapter class.  Import is lazy so unused adapters never load boto3 etc.

Usage:
    from src.integrations.email.registry import get_provider_class
    cls = get_provider_class("brevo")
    provider = cls(api_key="...", from_domain="example.com")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.integrations.email.base import EmailProvider

# Lazy registry — each entry is a (module_path, class_name) tuple.
_REGISTRY: dict[str, tuple[str, str]] = {
    "brevo": (
        "src.integrations.email.providers.brevo",
        "BrevoProvider",
    ),
    "resend": (
        "src.integrations.email.providers.resend",
        "ResendProvider",
    ),
    "postmark": (
        "src.integrations.email.providers.postmark",
        "PostmarkProvider",
    ),
    "ses": (
        "src.integrations.email.providers.ses",
        "SESProvider",
    ),
}


def get_provider_class(provider: str) -> type["EmailProvider"]:
    """Return the adapter class for *provider*.

    Raises KeyError if the provider is unknown.
    Raises ImportError if the module cannot be loaded.
    """
    if provider not in _REGISTRY:
        raise KeyError(
            f"Unknown email provider '{provider}'. "
            f"Known providers: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[provider]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
