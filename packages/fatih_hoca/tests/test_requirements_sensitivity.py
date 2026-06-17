# test_requirements_sensitivity.py
"""The sensitivity-keyword heuristic that auto-sets ``local_only`` must match
whole words, not raw substrings.

Live regression (2026-06-16): i2p step 5.0a design_tokens_generation has a
description mentioning the ambition-tier enum value ``private_beta``. The old
substring scan saw ``"private" in "...private_beta..."`` -> forced
``local_only=True`` -> the analyst task could only run on local models. With
the local GPU busy / unloadable, every such task crash-looped with "No model
candidates available" / "Failed to load local model" while cloud sat idle.

A word containing a sensitivity token as a substring (private_beta, homepage,
army) must NOT trip the heuristic; a genuine standalone token (my password,
personal data, home address) must.
"""
from __future__ import annotations

import asyncio

from fatih_hoca.requirements_builder import requirements_for


def _run(coro):
    return asyncio.run(coro)


def test_private_beta_substring_does_not_force_local_only():
    task = {
        "id": 1,
        "title": "[5.0a] design_tokens_generation",
        "description": (
            "Generate design tokens conditioned on the brand. If the "
            "ambition tier is private_beta or public_launch, scale the "
            "palette accordingly."
        ),
        "priority": 5,
    }
    reqs = _run(requirements_for(task, {"is_workflow_step": True}, agent_name="analyst"))
    assert reqs.local_only is False


def test_homepage_substring_does_not_force_local_only():
    task = {
        "id": 2,
        "title": "build_homepage",
        "description": "Lay out the homepage hero and footer.",
        "priority": 5,
    }
    reqs = _run(requirements_for(task, {}, agent_name="analyst"))
    assert reqs.local_only is False


def test_genuine_sensitive_word_still_forces_local_only():
    task = {
        "id": 3,
        "title": "vault_task",
        "description": "Summarize my password vault entries.",
        "priority": 5,
    }
    reqs = _run(requirements_for(task, {}, agent_name="analyst"))
    assert reqs.local_only is True


def test_personal_data_phrase_still_forces_local_only():
    task = {
        "id": 4,
        "title": "pii_task",
        "description": "Redact personal data from the export.",
        "priority": 5,
    }
    reqs = _run(requirements_for(task, {}, agent_name="analyst"))
    assert reqs.local_only is True


def test_workflow_step_security_subject_matter_does_not_force_local_only():
    """Live regression 2026-06-18: i2p step [4.6] auth_system_design *discusses*
    password hashing as SUBJECT MATTER. The user's actual idea is a benign todo
    app. A workflow step's template-generated title/description must NOT be
    keyword-scanned — that mass-false-marks public engineering work local_only
    (5 i2p steps live: auth_system_design, html_prototypes lucide:home icon,
    compliance_checks 'personal info', ...).
    """
    task = {
        "id": 459204,
        "title": "[4.6] auth_system_design",
        "description": (
            "Design authentication: login, logout, password reset, "
            "password hashing (bcrypt/scrypt), refresh token mechanism."
        ),
        "priority": 5,
    }
    ctx = {
        "is_workflow_step": True,
        "workflow_context": {
            "raw_idea": "an improved todo app, evolved into a habit builder",
            "product_name": "an improved todo app",
        },
    }
    reqs = _run(requirements_for(task, ctx, agent_name="architect"))
    assert reqs.local_only is False


def test_workflow_step_inherits_local_only_from_sensitive_user_idea():
    """The other half of the user directive: 'skip for workflow steps, BUT mark
    them if they really contain sensitive info.' Sensitivity is a property of the
    user's *actual* idea (propagated as workflow_context), not the generated step
    text — so a benign step description under a sensitive mission still goes
    local_only.
    """
    task = {
        "id": 9,
        "title": "[4.6] auth_system_design",
        "description": "Design the login screen layout and navigation flow.",
        "priority": 5,
    }
    ctx = {
        "is_workflow_step": True,
        "workflow_context": {
            "raw_idea": "an app to store my password vault and personal records",
            "product_name": "password vault",
        },
    }
    reqs = _run(requirements_for(task, ctx, agent_name="architect"))
    assert reqs.local_only is True
