"""Z6 T3C — wave 1 vendor config sanity checks.

Each new config must:
- parse as JSON
- declare service_name + base_url + auth_type
- register an HttpIntegration on auto-discovery
- expose the documented actions with method/path/required_params

These are static structural tests — no HTTP requests, no credentials.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "integrations" / "configs"


WAVE1: dict[str, dict] = {
    "stripe": {
        "base_url": "https://api.stripe.com",
        "auth_type": "bearer",
        "auth_token_field": "secret_key",
        "actions": {
            "list_products", "create_product", "create_price",
            "create_checkout_session", "list_subscriptions",
            "retrieve_balance",
        },
    },
    "sendgrid": {
        "base_url": "https://api.sendgrid.com",
        "auth_type": "bearer",
        "auth_token_field": "api_key",
        "actions": {
            "send_mail", "list_templates", "verify_domain",
            "list_suppressions",
        },
    },
    "cloudflare": {
        "base_url": "https://api.cloudflare.com/client/v4",
        "auth_type": "bearer",
        "auth_token_field": "api_token",
        "actions": {
            "list_zones", "list_dns_records", "create_dns_record",
            "delete_dns_record",
        },
    },
    "sentry": {
        "base_url": "https://sentry.io/api/0",
        "auth_type": "bearer",
        "auth_token_field": "auth_token",
        "actions": {
            "list_projects", "list_issues", "get_issue", "list_releases",
        },
    },
    "supabase": {
        "base_url": "https://api.supabase.com/v1",
        "auth_type": "bearer",
        "auth_token_field": "service_role_key",
        "actions": {
            "list_projects", "run_migration", "list_buckets",
            "create_signed_url",
        },
    },
}


@pytest.mark.parametrize("service", sorted(WAVE1.keys()))
def test_config_file_parses(service):
    path = _CONFIG_DIR / f"{service}.json"
    assert path.exists(), f"missing config: {path}"
    with open(path) as f:
        cfg = json.load(f)
    assert cfg["service_name"] == service
    assert cfg["base_url"] == WAVE1[service]["base_url"]
    assert cfg["auth_type"] == WAVE1[service]["auth_type"]


@pytest.mark.parametrize("service", sorted(WAVE1.keys()))
def test_actions_have_required_shape(service):
    path = _CONFIG_DIR / f"{service}.json"
    with open(path) as f:
        cfg = json.load(f)
    actions = cfg.get("actions") or {}
    expected = WAVE1[service]["actions"]
    # Wave-1 set is the floor — later tiers (T5/T6) may extend with more
    # actions (e.g. stripe gained list_disputes, list_tax_transactions,
    # create_customer, cancel_subscription, confirm_test_payment). The
    # test guarantees the original wave-1 contract is intact.
    missing = expected - set(actions.keys())
    assert not missing, (
        f"{service} missing wave-1 actions: {sorted(missing)}; "
        f"got {sorted(actions.keys())}"
    )
    for name, spec in actions.items():
        assert "method" in spec, f"{service}.{name} missing method"
        assert "path" in spec, f"{service}.{name} missing path"
        assert spec["method"] in (
            "GET", "POST", "PUT", "PATCH", "DELETE",
        )
        assert spec["path"].startswith("/"), (
            f"{service}.{name} path must be absolute: {spec['path']!r}"
        )
        assert "required_params" in spec
        assert isinstance(spec["required_params"], list)


@pytest.mark.parametrize("service", sorted(WAVE1.keys()))
def test_http_integration_loads(service):
    from src.integrations.http_integration import HttpIntegration

    integration = HttpIntegration.from_service_name(service)
    assert integration.service_name == service
    caps = integration.capabilities()
    # Wave-1 floor; tiers may extend the action set.
    missing = WAVE1[service]["actions"] - set(caps)
    assert not missing, (
        f"{service} HttpIntegration missing wave-1 caps: {sorted(missing)}; "
        f"got {sorted(caps)}"
    )


def test_registry_picks_up_wave1_configs():
    """Fresh registry auto-discovers all wave-1 configs alongside legacy ones."""
    from src.integrations.registry import IntegrationRegistry

    reg = IntegrationRegistry(auto_discover=True)
    services = reg.list_services()
    for service in WAVE1.keys():
        assert service in services, (
            f"{service} not auto-discovered. Got: {services}"
        )
        adapter = reg.get(service)
        assert adapter is not None
        assert adapter.service_name == service


def test_required_params_referenced_in_path_or_body():
    """Sanity: every {placeholder} in path is in required_params."""
    import re
    for service in WAVE1.keys():
        path = _CONFIG_DIR / f"{service}.json"
        with open(path) as f:
            cfg = json.load(f)
        for name, spec in cfg["actions"].items():
            placeholders = set(re.findall(r"\{([a-zA-Z_]+)\}", spec["path"]))
            required = set(spec.get("required_params") or [])
            missing = placeholders - required
            assert not missing, (
                f"{service}.{name} path placeholders {missing} not in "
                f"required_params {required}"
            )
