"""Manifest parsing + canonical-name + validation tests."""
import pytest

from yalayut.manifest import (
    parse_manifest_yaml, validate_manifest, canonical_name,
)
from yalayut.contracts import Manifest

SAMPLE = """
name: cc-django
name_original: cookiecutter-django
version: 1.0.0
artifact_type: skill
kind: shell_recipe
source: github:cookiecutter/cookiecutter-django
owner: cookiecutter
license: BSD-3-Clause
mechanizable: true
intent_keywords: [django, web-app, fullstack]
inputs_schema:
  project_name:
    type: string
    bind_from: [task.title]
invocation:
  steps:
    - cmd: "uvx cookiecutter gh:cookiecutter/cookiecutter-django"
"""


def test_parse_yaml_manifest():
    m = parse_manifest_yaml(SAMPLE)
    assert isinstance(m, Manifest)
    assert m.name == "cc-django"
    assert m.kind == "shell_recipe"
    assert m.mechanizable is True
    assert m.intent_keywords == ["django", "web-app", "fullstack"]
    assert m.inputs_schema["project_name"]["bind_from"] == ["task.title"]


def test_validate_rejects_missing_required():
    bad = Manifest(name="", name_original="x", version="", artifact_type="")
    errs = validate_manifest(bad)
    assert any("name" in e for e in errs)
    assert any("version" in e for e in errs)
    assert any("artifact_type" in e for e in errs)


def test_validate_rejects_bad_artifact_type():
    m = Manifest(name="a", name_original="a", version="1", artifact_type="bogus")
    errs = validate_manifest(m)
    assert any("artifact_type" in e for e in errs)


def test_validate_passes_good_manifest():
    assert validate_manifest(parse_manifest_yaml(SAMPLE)) == []


@pytest.mark.parametrize("source_slug,original,expected", [
    ("anthropics", "pdf", "anthropics-pdf"),
    ("matlab", "matlab-live-script", "matlab-live-script"),  # drop dup prefix
    ("cookiecutter", "cookiecutter-django", "cc-django"),     # cc-* special
    ("superpowers", "brainstorming", "superpowers-brainstorming"),
])
def test_canonical_name(source_slug, original, expected):
    assert canonical_name(source_slug, original) == expected
