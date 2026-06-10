"""Tests: grading + code_review posthook builders produce correct messages via Foundry.

Each test verifies:
1. The posthook builder calls build_messages("grading"/"code_review", ...) and
   the result matches the EXACT messages the old hard-coded constants would have
   produced (message-equivalence proof).
2. Key substrings are present (semantic smoke check).
"""
from __future__ import annotations

import json
from prompt_foundry import build_messages


# ── OLD CONSTANTS (kept here for equivalence verification only) ──────────────

_GRADING_SYSTEM = (
    "You are a strict SEMANTIC evaluator. Required fields and sections are "
    "ALREADY verified deterministically by a schema gate before you run — "
    "structural completeness is NOT your job. Judge ONLY semantic quality: "
    "relevance, content adequacy, coherence, structural soundness. NEVER FAIL "
    "an output for a missing, extra, or renamed field or section — that is "
    "checked elsewhere. Reply ONLY with the requested fields, one per line. "
    "Do not add explanation or commentary."
)

_GRADING_PROMPT = """Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

Field presence is verified deterministically upstream. Judge ONLY whether the
content semantically solves the task. DO NOT JUDGE field/section presence, and
do not penalise fields named in the Description but absent from the output.

Reply with EXACTLY these fields, one per line:
RELEVANT: YES or NO (does the content address THIS task, not a different one)
COMPLETE: YES or NO (does the CONTENT substantively solve the task — adequate depth, no stubs or hand-waving; this is semantic adequacy, NOT field presence)
VERDICT: PASS or FAIL
WELL_FORMED: PASS or FAIL (no repeated sections, no garbage, structurally sound)
COHERENT: PASS or FAIL (output makes logical sense end-to-end)
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
PREFERENCE: one-line user preference signal observed in this task, or NONE
INSIGHT: one-line reusable learning from this task, or NONE"""

_CODE_REVIEW_SYSTEM = (
    "You are a strict code reviewer. Inspect the emitted code below for "
    "correctness, security, error handling, and completeness against the "
    "task's stated requirements. Be concrete and actionable. Do not approve "
    "code that is a stub, scaffold, or placeholder when real implementation "
    "was requested. Reply ONLY in the requested format."
)

_CODE_REVIEW_PROMPT = """Review this build-step output.

Task: {title}
Description: {description}
Declared files (the agent claims to have produced these): {produces}
Output:
{response}

Reply with EXACTLY this format, in this order:

ISSUES:
- <one concrete issue per bullet, including file path + line/symbol + suggested fix>
- <another>
- (use the literal word NONE if no issues found)

VERDICT: PASS or FAIL

Notes:
- VERDICT must be FAIL if ANY of these are true: missing implementation,
  hardcoded secret, SQL injection, missing auth check, broken imports,
  syntax error, returning fake/placeholder data ("TODO", "abc123", "uuid"
  literal in body), claimed file not actually written.
- Otherwise VERDICT may be PASS even with low/medium issues; severity is
  the source's retry feedback, not your verdict.
"""


# ── GRADING ──────────────────────────────────────────────────────────────────

class TestGradingFoundryMessages:
    def _old_msgs(self, title, description, response):
        return [
            {"role": "system", "content": _GRADING_SYSTEM},
            {"role": "user", "content": _GRADING_PROMPT.format(
                title=title, description=description, response=response,
            )},
        ]

    def test_grading_uses_foundry_build(self):
        msgs = build_messages("grading", {"title": "T", "description": "D", "response": "R"})
        assert "SEMANTIC" in msgs[0]["content"]
        assert "T" in msgs[1]["content"]
        assert "R" in msgs[1]["content"]

    def test_grading_system_char_exact(self):
        msgs = build_messages("grading", {"title": "T", "description": "D", "response": "R"})
        assert msgs[0]["content"] == _GRADING_SYSTEM

    def test_grading_message_equivalence_typical(self):
        """Character-exact equivalence against old hard-coded constants."""
        title = "Implement OAuth2"
        description = "Add OAuth2 login flow with refresh tokens."
        response = "def oauth2_login(): return {'access_token': 'abc'}"
        new_msgs = build_messages("grading", {
            "title": title, "description": description, "response": response,
        })
        old_msgs = self._old_msgs(title, description, response)
        assert new_msgs == old_msgs, (
            f"system equal: {new_msgs[0] == old_msgs[0]}, "
            f"user equal: {new_msgs[1] == old_msgs[1]}"
        )

    def test_grading_message_equivalence_empty_fields(self):
        """Equivalence with empty-string fields (boundary case)."""
        new_msgs = build_messages("grading", {"title": "", "description": "", "response": ""})
        old_msgs = self._old_msgs("", "", "")
        assert new_msgs == old_msgs

    def test_grading_user_contains_verdict_prompt(self):
        msgs = build_messages("grading", {"title": "x", "description": "y", "response": "z"})
        user = msgs[1]["content"]
        assert "VERDICT: PASS or FAIL" in user
        assert "RELEVANT:" in user
        assert "COMPLETE:" in user

    def test_grading_message_structure(self):
        msgs = build_messages("grading", {"title": "T", "description": "D", "response": "R"})
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


# ── CODE REVIEW ──────────────────────────────────────────────────────────────

class TestCodeReviewFoundryMessages:
    def _old_msgs(self, title, description, produces, response):
        return [
            {"role": "system", "content": _CODE_REVIEW_SYSTEM},
            {"role": "user", "content": _CODE_REVIEW_PROMPT.format(
                title=title, description=description,
                produces=produces, response=response,
            )},
        ]

    def test_code_review_uses_foundry_build(self):
        msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": "[]", "response": "R",
        })
        assert "code reviewer" in msgs[0]["content"]
        assert "VERDICT" in msgs[1]["content"]

    def test_code_review_system_char_exact(self):
        msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": "[]", "response": "R",
        })
        assert msgs[0]["content"] == _CODE_REVIEW_SYSTEM

    def test_code_review_message_equivalence_typical(self):
        """Character-exact equivalence against old hard-coded constants."""
        title = "Build auth middleware"
        description = "Implement JWT auth middleware"
        produces = json.dumps(["src/middleware/auth.py"])
        response = "def auth_middleware(req): return check_jwt(req.headers)"
        new_msgs = build_messages("code_review", {
            "title": title, "description": description,
            "produces": produces, "response": response,
        })
        old_msgs = self._old_msgs(title, description, produces, response)
        assert new_msgs == old_msgs, (
            f"system equal: {new_msgs[0] == old_msgs[0]}, "
            f"user equal: {new_msgs[1] == old_msgs[1]}"
        )

    def test_code_review_message_equivalence_empty_produces(self):
        """Equivalence with empty produces list (common case)."""
        produces = json.dumps([])
        new_msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": produces, "response": "R",
        })
        old_msgs = self._old_msgs("T", "D", produces, "R")
        assert new_msgs == old_msgs

    def test_code_review_user_contains_issues_section(self):
        msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": "[]", "response": "R",
        })
        user = msgs[1]["content"]
        assert "ISSUES:" in user
        assert "VERDICT: PASS or FAIL" in user
        assert "Declared files" in user

    def test_code_review_message_structure(self):
        msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": "[]", "response": "R",
        })
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_code_review_produces_json_braces_safe(self):
        """JSON value in produces field must not confuse {field} substitution."""
        produces = json.dumps([{"file": "auth.py", "type": "module"}])
        msgs = build_messages("code_review", {
            "title": "T", "description": "D", "produces": produces, "response": "R",
        })
        assert produces in msgs[1]["content"]
