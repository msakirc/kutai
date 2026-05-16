"""Yalayut contracts — plugin protocols + cross-component dataclasses.

These types are in-process only. Anything crossing a serialization boundary
(task["skills"], discovery task specs) is a plain dict, per the spec's
interface contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Manifest:
    """Typed metadata for one catalog artifact (synthesized or seeded)."""
    name: str                       # canonical '<source-slug>-<original>'
    name_original: str              # upstream raw name
    version: str
    artifact_type: str              # 'skill' | 'api' | 'mcp'
    kind: str | None = None         # skill sub-type
    source: str = ""
    owner: str | None = None
    license: str | None = None
    mechanizable: bool = False
    model_hint: str | None = None
    applies_to: str = "execution"   # v1: always 'execution'
    intent_keywords: list[str] = field(default_factory=list)
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    invocation: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    disabled_imports_check: bool = True
    mcp: dict[str, Any] = field(default_factory=dict)
    auth_env: str | None = None


@dataclass
class Issue:
    """One vetting finding. max_tier caps the artifact at this tier."""
    check: str
    max_tier: int
    detail: str = ""


@dataclass
class ArtifactRef:
    """A discovered-but-not-yet-fetched artifact."""
    source_id: str
    name: str
    fetch_url: str
    owner: str | None = None
    raw_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceConfig:
    """A row of yalayut_sources, hydrated for an adapter."""
    source_id: str
    source_type: str
    endpoint: str
    auth_env: str | None = None
    trusted: bool = False
    discovery_mode: str = "on_demand"
    min_interval_s: int | None = None


@dataclass
class IndexRow:
    """A hydrated row of yalayut_index."""
    id: int
    artifact_type: str
    kind: str | None
    source: str
    owner: str | None
    name: str
    name_original: str | None
    version: str
    manifest_path: str | None
    body_excerpt: str | None
    vet_tier: int | None
    exposure_class: str | None
    applies_to: str | None
    mechanizable: bool
    model_hint: str | None
    enabled: bool


@dataclass
class TaskContext:
    """The subset of a KutAI task dict yalayut needs for query/binding."""
    task_id: str = ""
    title: str = ""
    description: str = ""
    agent_type: str = ""
    recipe_hint: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: dict) -> "TaskContext":
        """Build a TaskContext from a raw KutAI task dict."""
        ctx = task.get("context") or {}
        return cls(
            task_id=str(task.get("id", "")),
            title=task.get("title", "") or task.get("name", ""),
            description=task.get("description", "") or task.get("goal", ""),
            agent_type=task.get("agent_type", ""),
            recipe_hint=ctx.get("recipe_hint"),
            payload=ctx.get("payload", {}) or {},
        )

    def query_text(self) -> str:
        """The text embedded for similarity search."""
        parts = [self.title, self.description, self.recipe_hint or ""]
        return " ".join(p for p in parts if p).strip()


@dataclass
class SkillApplication:
    """Structured (NOT rendered) result of matching one artifact to a task.
    Phase 3's intersect produces these; consumers render them."""
    artifact_id: int
    name: str
    exposure_class: str             # 'inject' | 'tool' | 'preempt'
    applies_to: str = "execution"
    render: str = "prose"           # 'prose' | 'prebind'
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class Artifact:
    """A ranked query() result."""
    artifact_id: int
    name: str
    name_original: str | None
    artifact_type: str
    kind: str | None
    vet_tier: int
    score: float
    exposure_class: str | None
    applies_to: str | None
    mechanizable: bool
    body_excerpt: str | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Outcome of a recipe execution."""
    ok: bool
    detail: str = ""
    artifacts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiscoveryPlugin(Protocol):
    """Per-artifact-type parse + vet. Lives in yalayut/plugins/."""
    artifact_type: str

    def parse_manifest(self, raw: bytes, source_meta: dict) -> Manifest: ...
    def vet_checks(self, manifest: Manifest, body_path: Path) -> list[Issue]: ...


@runtime_checkable
class AccessPlugin(Protocol):
    """Per-artifact-type query + binding. Does NOT render prompts."""
    artifact_type: str

    def to_application(
        self, row: IndexRow, task_ctx: TaskContext
    ) -> SkillApplication: ...
    def bind_args(
        self, row: IndexRow, task_ctx: TaskContext
    ) -> dict | None: ...
    def execute(
        self, row: IndexRow, task_ctx: TaskContext, inputs: dict
    ) -> Result: ...


@runtime_checkable
class SourceAdapter(Protocol):
    """Per-source-type discovery + fetch."""
    source_type: str

    async def discover(
        self, source_cfg: SourceConfig
    ) -> list[ArtifactRef]: ...
    async def fetch(self, ref: ArtifactRef) -> Path: ...
