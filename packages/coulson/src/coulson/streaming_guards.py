"""Streaming post-processor pipeline (B3).

Z1 Tier 5C. Token-stream interceptor pipeline. Each guard is a rule-based
(NOT LLM — speed-critical) check applied to chunks as they stream from the
LLM. Guards are stateful where needed (brace tracking, fence tracking) and
return one of three actions:

- ``"fix"``  — token rewritten in place; downstream sees the fixed text
- ``"warn"`` — token passes through, but a row is logged to
  ``streaming_guard_log`` for telemetry
- ``"halt"`` — abort the stream immediately (caller treats like a finish)

Pipeline shape::

    pipeline = StreamingGuardPipeline()
    for chunk in stream:
        outcome = pipeline.process(chunk)
        if outcome.halt:
            break
        emit(outcome.text)
        for note in outcome.warnings:
            log(note)

Guards are registered in :data:`_GUARD_REGISTRY` (order matters — earliest
guard wins on collisions). Add a new guard by appending its callable to the
list; it must implement :class:`Guard` (or duck-type the same shape).

Opt-out
-------
Set ``KUTAI_STREAMING_GUARDS=off`` to bypass the pipeline (default: on).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# Use the same logger surface as the rest of coulson when available;
# otherwise fall back to a stdlib logger so this module is import-safe in
# unit tests that don't have the full app wiring.
try:
    from yazbunu import get_logger as _get_logger
except Exception:  # pragma: no cover
    import logging

    def _get_logger(name: str) -> Any:  # type: ignore[misc]
        return logging.getLogger(name)


logger = _get_logger("coulson.streaming_guards")


# ─── Public dataclasses ──────────────────────────────────────────────────


@dataclass
class GuardResult:
    """Outcome of a single guard's check on a single token."""

    action: str  # "fix" | "warn" | "halt" | "pass"
    patched_token: str | None = None
    note: str = ""
    guard_name: str = ""


@dataclass
class PipelineOutcome:
    """Outcome of the full pipeline for one streamed token."""

    text: str
    halt: bool = False
    warnings: list[GuardResult] = field(default_factory=list)
    fixes: list[GuardResult] = field(default_factory=list)


# ─── Guard protocol ──────────────────────────────────────────────────────


class Guard:
    """Stateful per-stream guard. Subclass and override :meth:`check`."""

    name = "guard"

    def reset(self) -> None:
        """Reset internal state (called at start of each new stream)."""

    def check(self, token: str, accumulated: str) -> GuardResult:
        return GuardResult(action="pass", guard_name=self.name)


# ─── 1. Malformed-JSON guard ─────────────────────────────────────────────


_FENCE_OPEN = re.compile(r"```json\b", re.IGNORECASE)
_FENCE_CLOSE = "```"


class MalformedJsonGuard(Guard):
    """Track brace/bracket balance inside ```json fences.

    State machine:
      - Outside fence: noop.
      - Inside fence: track balance of `{}` and `[]`. Track string state
        with `"..."` and the escape backslash.

    A token that *closes* the fence while balance != 0 emits a ``"warn"``
    (we don't try to inject braces inline — too risky). A token that
    appears to terminate the JSON value cleanly while a string is still
    open also emits ``"warn"``.
    """

    name = "malformed_json"

    def __init__(self) -> None:
        self.in_fence = False
        self.in_string = False
        self.escape = False
        self.brace = 0
        self.bracket = 0

    def reset(self) -> None:
        self.in_fence = False
        self.in_string = False
        self.escape = False
        self.brace = 0
        self.bracket = 0

    def _scan(self, text: str) -> tuple[bool, bool]:
        """Returns (entered_fence_this_call, closed_fence_unbalanced)."""
        i = 0
        entered = False
        closed_unbalanced = False
        while i < len(text):
            ch = text[i]
            if not self.in_fence:
                # Look for ```json opening
                m = _FENCE_OPEN.search(text, i)
                if m:
                    self.in_fence = True
                    entered = True
                    i = m.end()
                    self.in_string = False
                    self.escape = False
                    self.brace = 0
                    self.bracket = 0
                    continue
                break
            # Inside fence
            if text.startswith(_FENCE_CLOSE, i):
                if self.brace != 0 or self.bracket != 0 or self.in_string:
                    closed_unbalanced = True
                self.in_fence = False
                i += 3
                continue
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
            else:
                if ch == '"':
                    self.in_string = True
                elif ch == "{":
                    self.brace += 1
                elif ch == "}":
                    self.brace -= 1
                elif ch == "[":
                    self.bracket += 1
                elif ch == "]":
                    self.bracket -= 1
            i += 1
        return entered, closed_unbalanced

    def check(self, token: str, accumulated: str) -> GuardResult:
        _, closed_unbalanced = self._scan(token)
        if closed_unbalanced:
            return GuardResult(
                action="warn",
                note=(
                    f"json fence closed with unbalanced "
                    f"braces={self.brace} brackets={self.bracket} "
                    f"in_string={self.in_string}"
                ),
                guard_name=self.name,
            )
        return GuardResult(action="pass", guard_name=self.name)


# ─── 2. Broken-markdown-fence guard ──────────────────────────────────────


class BrokenFenceGuard(Guard):
    """Track ``` fence open/close balance across the stream.

    On stream completion (signaled by ``token == ""`` with the
    ``finalize=True`` shape — we treat any non-trivial accumulated text
    with odd fence count at end-of-stream as a candidate for auto-close).
    The pipeline calls :meth:`finalize` before yielding the last chunk.
    """

    name = "broken_fence"

    _FENCE = "```"

    def __init__(self) -> None:
        self.open_fences = 0

    def reset(self) -> None:
        self.open_fences = 0

    def check(self, token: str, accumulated: str) -> GuardResult:
        # Count fences that appear in this token. A simple count toggles
        # state (open ↔ closed). This is a conservative tracker; nested
        # fences are unsupported by markdown so a flat count matches reality.
        n = token.count(self._FENCE)
        if n:
            self.open_fences = (self.open_fences + n) % 2
        return GuardResult(action="pass", guard_name=self.name)

    def finalize(self) -> GuardResult | None:
        """Called once at stream end. Auto-close an unclosed fence."""
        if self.open_fences % 2 == 1:
            return GuardResult(
                action="fix",
                patched_token="\n```\n",
                note="auto-closed unclosed markdown fence at stream end",
                guard_name=self.name,
            )
        return None


# ─── 3. Hallucinated import-path guard ───────────────────────────────────


# Conservative stdlib + common-deps allowlist. Real-world wiring should
# extend this from the active project's pyproject.toml/requirements.txt;
# kept inline here for unit-test determinism.
_DEFAULT_ALLOWLIST = {
    # Python stdlib (subset)
    "os", "sys", "json", "re", "io", "math", "time", "typing", "asyncio",
    "pathlib", "subprocess", "datetime", "collections", "functools",
    "itertools", "logging", "hashlib", "uuid", "random", "string",
    "dataclasses", "enum", "abc", "contextlib", "tempfile", "shutil",
    "argparse", "configparser", "csv", "sqlite3", "urllib", "http",
    "email", "html", "xml", "warnings", "traceback", "inspect", "copy",
    "pickle", "base64", "binascii", "struct", "array", "heapq", "bisect",
    "weakref", "operator", "ast",
    # Common Python deps in this project
    "anyio", "aiohttp", "aiosqlite", "litellm", "openai", "anthropic",
    "pytest", "pydantic", "fastapi", "starlette", "httpx", "requests",
    "numpy", "pandas", "scipy", "torch", "transformers",
    "sentence_transformers", "chromadb", "telegram", "telegram_bot",
    "yaml", "toml", "tomli", "click", "rich", "jinja2", "bs4",
    "beautifulsoup4", "lxml", "selectolax", "playwright",
    # JS stdlib-ish + common
    "react", "vue", "next", "express", "lodash", "axios", "fs", "path",
    "crypto", "events", "stream", "util", "child_process",
    # Project-internal namespaces
    "src", "mr_roboto", "general_beckman", "fatih_hoca", "coulson",
    "hallederiz_kadir", "dogru_mu_samet", "kuleden_donen_var",
    "nerd_herd", "vecihi", "yasar_usta", "dallama", "yazbunu",
}


# Match `from X import Y` or `import X` (Python) / `import ... from 'X'` (JS)
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
    re.MULTILINE,
)
_JS_IMPORT_RE = re.compile(
    r"""(?:from|import)\s*\(?\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)


class HallucinatedImportGuard(Guard):
    """Flag suspicious top-level imports against an allowlist.

    Only checks once per import statement (dedup via seen-set). Returns
    ``"warn"`` (never ``"halt"``) — false positives are common, so the
    signal is for telemetry, not blocking.
    """

    name = "hallucinated_import"

    def __init__(self, allowlist: Iterable[str] | None = None) -> None:
        self.allowlist = set(allowlist) if allowlist is not None else set(_DEFAULT_ALLOWLIST)
        self.seen: set[str] = set()
        self._buffer = ""

    def reset(self) -> None:
        self.seen.clear()
        self._buffer = ""

    def _root(self, mod: str) -> str:
        return mod.split(".", 1)[0].split("/", 1)[0].lstrip("@")

    def check(self, token: str, accumulated: str) -> GuardResult:
        # Buffer until newline; imports are line-anchored.
        self._buffer += token
        if "\n" not in self._buffer:
            return GuardResult(action="pass", guard_name=self.name)
        lines, _, rest = self._buffer.rpartition("\n")
        self._buffer = rest
        suspicious: list[str] = []
        # JS imports first (handle the `from 'X'` shape) so relative paths
        # are skipped before the Python regex sees an `import x` prefix.
        js_lines: set[int] = set()
        for m in _JS_IMPORT_RE.finditer(lines):
            mod = (m.group(1) or "").strip()
            if not mod:
                continue
            line_idx = lines.count("\n", 0, m.start())
            js_lines.add(line_idx)
            root = self._root(mod)
            if root in self.seen:
                continue
            self.seen.add(root)
            # Skip relative imports
            if mod.startswith(".") or mod.startswith("/"):
                continue
            if root and root not in self.allowlist:
                suspicious.append(f"js:{mod}")
        for m in _PY_IMPORT_RE.finditer(lines):
            mod = (m.group(1) or m.group(2) or "").strip()
            if not mod:
                continue
            line_idx = lines.count("\n", 0, m.start())
            if line_idx in js_lines:
                # This is a JS-shaped import that already matched above.
                continue
            root = self._root(mod)
            if root in self.seen:
                continue
            self.seen.add(root)
            if root and root not in self.allowlist:
                suspicious.append(f"py:{mod}")
        if suspicious:
            return GuardResult(
                action="warn",
                note=f"suspicious imports: {suspicious}",
                guard_name=self.name,
            )
        return GuardResult(action="pass", guard_name=self.name)


# ─── 4. Common-typo guard ────────────────────────────────────────────────


def _load_typo_map(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = Path(__file__).parent / "streaming_typos.txt"
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" not in line:
                continue
            wrong, right = line.split("=>", 1)
            wrong = wrong.rstrip()
            right = right.lstrip()
            if wrong and right:
                out[wrong] = right
    except Exception:  # pragma: no cover
        pass
    return out


class TypoGuard(Guard):
    """Inline-fix unambiguous code typos via a small lookup table.

    Buffer-based: accumulates tokens until a typo span won't be split
    across the boundary. Conservative — only fixes literal substrings.
    """

    name = "typo"

    def __init__(self, typo_map: dict[str, str] | None = None) -> None:
        self.map = typo_map if typo_map is not None else _load_typo_map()
        self._max_key = max((len(k) for k in self.map), default=0)
        self._tail = ""  # carry-over to handle splits across token boundaries
        # Pre-compute set of typo prefixes for fast partial-match detection.
        self._prefixes: set[str] = set()
        for k in self.map:
            for i in range(1, len(k)):
                self._prefixes.add(k[:i])

    def reset(self) -> None:
        self._tail = ""

    def _split_point(self, text: str) -> int:
        """Return index N such that text[:N] is safe to emit, text[N:] is the
        smallest tail that could still complete a typo across the boundary."""
        if not self._prefixes or self._max_key < 2:
            return len(text)
        # Largest k such that text endswith a typo-prefix of length k
        max_check = min(len(text), self._max_key - 1)
        for k in range(max_check, 0, -1):
            if text[-k:] in self._prefixes:
                return len(text) - k
        return len(text)

    def check(self, token: str, accumulated: str) -> GuardResult:
        if not self.map:
            return GuardResult(action="pass", guard_name=self.name)
        combined = self._tail + token
        fixed = combined
        any_fix = False
        notes: list[str] = []
        for wrong, right in self.map.items():
            if wrong in fixed:
                fixed = fixed.replace(wrong, right)
                any_fix = True
                notes.append(f"{wrong}->{right}")
        # Smart split: only retain a tail that could still match a typo prefix.
        split = self._split_point(fixed)
        emit = fixed[:split]
        self._tail = fixed[split:]
        if any_fix:
            return GuardResult(
                action="fix",
                patched_token=emit,
                note=f"typo fixes: {notes}",
                guard_name=self.name,
            )
        # No fix; if we buffered something we still have to emit the safe
        # portion via the "fix" channel (it carries patched_token). When the
        # whole token is safe (split==len) and tail is empty, return pass
        # so unrelated guards see the original text uninstrumented.
        if not self._tail and emit == token:
            return GuardResult(action="pass", guard_name=self.name)
        return GuardResult(
            action="fix",
            patched_token=emit,
            note="",
            guard_name=self.name,
        )

    def flush(self) -> str:
        out = self._tail
        self._tail = ""
        return out


# ─── Pipeline ────────────────────────────────────────────────────────────


# Default registry. Order matters: typo fixes happen first (so JSON guard
# sees corrected text), then JSON, then fence, then import.
_GUARD_REGISTRY: list[Callable[[], Guard]] = [
    TypoGuard,
    MalformedJsonGuard,
    BrokenFenceGuard,
    HallucinatedImportGuard,
]


def _opt_out() -> bool:
    return (os.environ.get("KUTAI_STREAMING_GUARDS") or "").strip().lower() in {
        "off",
        "0",
        "false",
        "no",
    }


class StreamingGuardPipeline:
    """Stateful pipeline of guards run against streamed tokens.

    Use::

        pipeline = StreamingGuardPipeline()
        async for chunk in stream:
            outcome = pipeline.process(chunk)
            for w in outcome.warnings:
                log_warn(w)
            yield outcome.text
            if outcome.halt:
                break
        # always call .finalize() to flush trailing state and auto-fixes
        tail = pipeline.finalize()
        if tail.text:
            yield tail.text
    """

    def __init__(
        self,
        guards: list[Guard] | None = None,
        sink: Callable[[GuardResult], None] | None = None,
    ) -> None:
        self.disabled = _opt_out()
        if guards is not None:
            self.guards = guards
        else:
            self.guards = [factory() for factory in _GUARD_REGISTRY]
        self._accumulated = ""
        self.sink = sink or self._default_sink

    @staticmethod
    def _default_sink(result: GuardResult) -> None:
        if result.action in {"warn", "halt"}:
            logger.warning(
                f"[streaming_guard:{result.guard_name}] "
                f"{result.action}: {result.note}"
            )

    def reset(self) -> None:
        for g in self.guards:
            g.reset()
        self._accumulated = ""

    def process(self, token: str) -> PipelineOutcome:
        """Run guards on a single token. Returns the (possibly fixed) text
        plus any warnings/halt signal."""
        if self.disabled or not token:
            self._accumulated += token or ""
            return PipelineOutcome(text=token or "", halt=False)
        text = token
        warnings: list[GuardResult] = []
        fixes: list[GuardResult] = []
        halt = False
        for guard in self.guards:
            res = guard.check(text, self._accumulated)
            res.guard_name = res.guard_name or guard.name
            if res.action == "halt":
                self.sink(res)
                warnings.append(res)
                halt = True
                break
            if res.action == "fix":
                if res.patched_token is not None:
                    text = res.patched_token
                fixes.append(res)
                if res.note:
                    self.sink(res)
            elif res.action == "warn":
                warnings.append(res)
                self.sink(res)
            # "pass" → no-op
        self._accumulated += text
        return PipelineOutcome(
            text=text, halt=halt, warnings=warnings, fixes=fixes
        )

    def finalize(self) -> PipelineOutcome:
        """Flush trailing state — auto-close fences, drain typo tail."""
        if self.disabled:
            return PipelineOutcome(text="", halt=False)
        text_parts: list[str] = []
        warnings: list[GuardResult] = []
        fixes: list[GuardResult] = []
        for guard in self.guards:
            # Drain typo carry-over first so its tail rides through other
            # guards' state.
            if isinstance(guard, TypoGuard):
                tail = guard.flush()
                if tail:
                    text_parts.append(tail)
        for guard in self.guards:
            if hasattr(guard, "finalize"):
                res = guard.finalize()  # type: ignore[attr-defined]
                if res is None:
                    continue
                if res.action == "fix" and res.patched_token:
                    text_parts.append(res.patched_token)
                    fixes.append(res)
                elif res.action == "warn":
                    warnings.append(res)
                    self.sink(res)
        return PipelineOutcome(
            text="".join(text_parts), halt=False,
            warnings=warnings, fixes=fixes,
        )


__all__ = [
    "Guard",
    "GuardResult",
    "PipelineOutcome",
    "StreamingGuardPipeline",
    "MalformedJsonGuard",
    "BrokenFenceGuard",
    "HallucinatedImportGuard",
    "TypoGuard",
]
