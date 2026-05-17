"""Tests for yalayut shell-safety tokenizer + allowlist."""
import pytest

from yalayut.shell_safety import (
    tokenize_cmd,
    check_shell_bin,
    windows_incompat_reason,
    ShellSafetyError,
)


def test_tokenize_simple():
    assert tokenize_cmd("uvx cookiecutter gh:cookiecutter/cookiecutter-django") == [
        "uvx", "cookiecutter", "gh:cookiecutter/cookiecutter-django",
    ]


def test_tokenize_quoted_arg():
    assert tokenize_cmd('cookiecutter --no-input project_name="My App"') == [
        "cookiecutter", "--no-input", "project_name=My App",
    ]


def test_tokenize_empty_raises():
    with pytest.raises(ShellSafetyError):
        tokenize_cmd("   ")


def test_allowlisted_bins_pass():
    for binary in ("uvx", "npx", "npm", "pip", "git", "cookiecutter", "python"):
        assert check_shell_bin(binary) is True


def test_unknown_bin_rejected():
    assert check_shell_bin("curl") is False
    assert check_shell_bin("rm") is False


def test_windows_incompat_chmod():
    assert windows_incompat_reason("chmod +x install.sh") == "chmod"


def test_windows_incompat_sudo():
    assert windows_incompat_reason("sudo apt-get install foo") in ("sudo", "apt")


def test_windows_incompat_bare_sh():
    assert windows_incompat_reason("./install.sh") == "bare_sh"


def test_windows_compat_clean_cmd():
    assert windows_incompat_reason("uvx cookiecutter gh:foo/bar") is None
