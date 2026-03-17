# model_profiles.py
"""
Known model family profiles — the knowledge base.

Each family has:
  - base_capabilities: scores at the anchor_size (typically the flagship)
  - anchor_params_b: the param count those scores are calibrated for
  - specialties, vision support, function calling support markers
  - variants for sub-families (e.g., qwen2.5-coder vs qwen2.5 base)

All base scores assume FP16 / unquantized at anchor size.
The registry scales these based on actual size + quantization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.infra.logging_config import get_logger


@dataclass
class FamilyProfile:
    """Profile for a model family at its anchor size."""
    base_capabilities: dict[str, float]
    anchor_params_b: float = 30.0           # param count scores are calibrated for
    specialty: str = ""                     # "coding", "reasoning", "vision", etc.
    has_vision: bool = False
    function_calling: bool = False          # native tool-call format support
    thinking_capable: bool = False          # built-in CoT/thinking mode
    context_default: int = 32768


# ─── Pattern Matching ────────────────────────────────────────────────────────
# Ordered most-specific → least-specific. First match wins.
# Each entry: (pattern_substrings, family_key)

FAMILY_PATTERNS: list[tuple[list[str], str]] = [
    # Coder variants (must come before base families)
    (["qwen3", "coder"],                    "qwen3_coder"),
    (["qwen2.5", "coder"],                  "qwen25_coder"),
    (["codellama"],                          "codellama"),
    (["deepseek", "coder"],                  "deepseek_coder"),
    (["starcoder"],                          "starcoder"),

    # Vision models
    (["llava"],                              "llava"),
    (["moondream"],                          "moondream"),
    (["internvl"],                           "internvl"),
    (["minicpm", "v"],                       "minicpm_v"),
    (["qwen2", "vl"],                        "qwen2_vl"),
    (["gemma3"],                             "gemma3"),      # gemma3 has vision

    # GLM models (before base to match flash variant first)
    (["glm-4", "flash"],                     "glm4_flash"),
    (["glm-4"],                              "glm4"),
    (["glm4"],                               "glm4"),

    # Reasoning / thinking models
    (["qwq"],                                "qwq"),
    (["deepseek", "r1"],                     "deepseek_r1"),

    # Base families
    (["qwen3.5"],                            "qwen35"),
    (["qwen3"],                              "qwen3"),
    (["qwen2.5"],                            "qwen25"),
    (["qwen2"],                              "qwen2"),
    (["llama-3.3"],                          "llama33"),
    (["llama-3.2"],                          "llama32"),
    (["llama-3.1"],                          "llama31"),
    (["llama3"],                             "llama3"),
    (["phi-4"],                              "phi4"),
    (["phi-3"],                              "phi3"),
    (["phi4-mini"],                          "phi4_mini"),
    (["phi"],                                "phi3"),
    (["gemma-2"],                            "gemma2"),
    (["gemma2"],                             "gemma2"),
    (["gemma"],                              "gemma"),
    (["mistral"],                            "mistral"),
    (["mixtral"],                            "mixtral"),
    (["deepseek-v3"],                        "deepseek_v3"),
    (["deepseek-v2"],                        "deepseek_v2"),
    (["deepseek"],                           "deepseek"),
    (["command-r"],                          "command_r"),
    (["yi"],                                 "yi"),
    (["internlm"],                           "internlm"),
]


# ─── Family Profiles ────────────────────────────────────────────────────────

FAMILY_PROFILES: dict[str, FamilyProfile] = {
    # ════════════════════════════════════════════════════════
    # Qwen family
    # ════════════════════════════════════════════════════════
    "qwen3": FamilyProfile(
        anchor_params_b=32,
        specialty="",
        function_calling=True,
        thinking_capable=True,
        context_default=40960,
        base_capabilities={
            "reasoning":             8.5,
            "planning":              8.0,
            "analysis":              8.0,
            "code_generation":       8.0,
            "code_reasoning":        7.5,
            "system_design":         7.5,
            "prose_quality":         7.5,
            "instruction_adherence": 8.0,
            "domain_knowledge":      8.0,
            "context_utilization":   7.5,
            "structured_output":     8.0,
            "tool_use":              8.0,
            "vision":                0.0,
            "conversation":          7.5,
        },
    ),
    "qwen35": FamilyProfile(
        anchor_params_b=32,
        specialty="",
        function_calling=True,
        thinking_capable=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             9.0,
            "planning":              8.5,
            "analysis":              8.5,
            "code_generation":       8.5,
            "code_reasoning":        8.0,
            "system_design":         8.0,
            "prose_quality":         8.0,
            "instruction_adherence": 8.5,
            "domain_knowledge":      8.5,
            "context_utilization":   8.5,
            "structured_output":     8.5,
            "tool_use":              8.5,
            "vision":                0.0,
            "conversation":          8.0,
        },
    ),
    "qwen3_coder": FamilyProfile(
        anchor_params_b=32,
        specialty="coding",
        function_calling=True,
        thinking_capable=True,
        context_default=40960,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              6.5,
            "analysis":              7.0,
            "code_generation":       9.5,
            "code_reasoning":        9.0,
            "system_design":         7.0,
            "prose_quality":         5.5,
            "instruction_adherence": 8.0,
            "domain_knowledge":      6.5,
            "context_utilization":   7.5,
            "structured_output":     8.0,
            "tool_use":              7.5,
            "vision":                0.0,
            "conversation":          4.5,
        },
    ),
    "qwen25": FamilyProfile(
        anchor_params_b=32,
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              7.0,
            "analysis":              7.0,
            "code_generation":       7.0,
            "code_reasoning":        6.5,
            "system_design":         7.0,
            "prose_quality":         7.5,
            "instruction_adherence": 7.5,
            "domain_knowledge":      7.5,
            "context_utilization":   7.0,
            "structured_output":     7.5,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          7.0,
        },
    ),
    "qwen25_coder": FamilyProfile(
        anchor_params_b=32,
        specialty="coding",
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              5.5,
            "analysis":              6.5,
            "code_generation":       9.5,
            "code_reasoning":        9.0,
            "system_design":         6.5,
            "prose_quality":         5.0,
            "instruction_adherence": 7.5,
            "domain_knowledge":      6.0,
            "context_utilization":   7.0,
            "structured_output":     7.5,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          4.0,
        },
    ),
    "qwen2": FamilyProfile(
        anchor_params_b=72,
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              6.5,
            "code_generation":       6.5,
            "code_reasoning":        6.0,
            "system_design":         6.0,
            "prose_quality":         7.0,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.0,
            "context_utilization":   6.5,
            "structured_output":     7.0,
            "tool_use":              6.5,
            "vision":                0.0,
            "conversation":          6.5,
        },
    ),
    "qwen2_vl": FamilyProfile(
        anchor_params_b=7,
        specialty="vision",
        has_vision=True,
        function_calling=False,
        context_default=32768,
        base_capabilities={
            "reasoning":             6.0,
            "planning":              5.0,
            "analysis":              6.5,
            "code_generation":       4.0,
            "code_reasoning":        4.0,
            "system_design":         3.5,
            "prose_quality":         6.0,
            "instruction_adherence": 6.5,
            "domain_knowledge":      6.0,
            "context_utilization":   5.5,
            "structured_output":     6.0,
            "tool_use":              3.0,
            "vision":                8.0,
            "conversation":          6.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Llama family
    # ════════════════════════════════════════════════════════
    "llama33": FamilyProfile(
        anchor_params_b=70,
        function_calling=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              7.0,
            "analysis":              7.0,
            "code_generation":       7.0,
            "code_reasoning":        6.5,
            "system_design":         6.5,
            "prose_quality":         8.0,
            "instruction_adherence": 7.5,
            "domain_knowledge":      7.5,
            "context_utilization":   7.5,
            "structured_output":     7.0,
            "tool_use":              7.5,
            "vision":                0.0,
            "conversation":          8.0,
        },
    ),
    "llama32": FamilyProfile(
        anchor_params_b=3,
        function_calling=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             4.5,
            "planning":              4.0,
            "analysis":              4.0,
            "code_generation":       3.5,
            "code_reasoning":        3.5,
            "system_design":         3.0,
            "prose_quality":         5.0,
            "instruction_adherence": 5.0,
            "domain_knowledge":      4.5,
            "context_utilization":   5.0,
            "structured_output":     5.0,
            "tool_use":              4.5,
            "vision":                0.0,
            "conversation":          5.5,
        },
    ),
    "llama31": FamilyProfile(
        anchor_params_b=70,
        function_calling=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              6.5,
            "code_generation":       6.5,
            "code_reasoning":        6.0,
            "system_design":         6.0,
            "prose_quality":         7.5,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.0,
            "context_utilization":   7.0,
            "structured_output":     6.5,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          7.5,
        },
    ),
    "llama3": FamilyProfile(
        anchor_params_b=70,
        function_calling=True,
        context_default=8192,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              6.0,
            "analysis":              6.0,
            "code_generation":       6.0,
            "code_reasoning":        5.5,
            "system_design":         5.5,
            "prose_quality":         7.0,
            "instruction_adherence": 6.5,
            "domain_knowledge":      6.5,
            "context_utilization":   5.5,
            "structured_output":     6.0,
            "tool_use":              5.5,
            "vision":                0.0,
            "conversation":          7.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Phi family
    # ════════════════════════════════════════════════════════
    "phi4": FamilyProfile(
        anchor_params_b=14,
        function_calling=True,
        context_default=16384,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.0,
            "analysis":              6.5,
            "code_generation":       6.5,
            "code_reasoning":        6.0,
            "system_design":         5.5,
            "prose_quality":         6.0,
            "instruction_adherence": 6.5,
            "domain_knowledge":      6.5,
            "context_utilization":   6.0,
            "structured_output":     6.5,
            "tool_use":              5.5,
            "vision":                0.0,
            "conversation":          5.5,
        },
    ),
    "phi4_mini": FamilyProfile(
        anchor_params_b=3.8,
        function_calling=True,
        context_default=16384,
        base_capabilities={
            "reasoning":             5.5,
            "planning":              4.5,
            "analysis":              5.0,
            "code_generation":       5.0,
            "code_reasoning":        4.5,
            "system_design":         3.5,
            "prose_quality":         4.5,
            "instruction_adherence": 5.5,
            "domain_knowledge":      5.0,
            "context_utilization":   4.5,
            "structured_output":     5.5,
            "tool_use":              4.0,
            "vision":                0.0,
            "conversation":          4.5,
        },
    ),
    "phi3": FamilyProfile(
        anchor_params_b=14,
        function_calling=False,
        context_default=4096,
        base_capabilities={
            "reasoning":             6.0,
            "planning":              5.0,
            "analysis":              5.5,
            "code_generation":       5.5,
            "code_reasoning":        5.0,
            "system_design":         4.5,
            "prose_quality":         5.5,
            "instruction_adherence": 5.5,
            "domain_knowledge":      5.5,
            "context_utilization":   4.5,
            "structured_output":     5.0,
            "tool_use":              3.0,
            "vision":                0.0,
            "conversation":          5.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Gemma family
    # ════════════════════════════════════════════════════════
    "gemma3": FamilyProfile(
        anchor_params_b=27,
        has_vision=True,
        function_calling=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              6.5,
            "analysis":              7.0,
            "code_generation":       7.0,
            "code_reasoning":        6.5,
            "system_design":         6.0,
            "prose_quality":         7.0,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.0,
            "context_utilization":   7.0,
            "structured_output":     7.0,
            "tool_use":              6.5,
            "vision":                7.0,
            "conversation":          7.0,
        },
    ),
    "gemma2": FamilyProfile(
        anchor_params_b=27,
        function_calling=False,
        context_default=8192,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              5.5,
            "analysis":              6.0,
            "code_generation":       6.0,
            "code_reasoning":        5.5,
            "system_design":         5.0,
            "prose_quality":         6.5,
            "instruction_adherence": 6.0,
            "domain_knowledge":      6.5,
            "context_utilization":   5.5,
            "structured_output":     5.5,
            "tool_use":              3.5,
            "vision":                0.0,
            "conversation":          6.0,
        },
    ),
    "gemma": FamilyProfile(
        anchor_params_b=7,
        function_calling=False,
        context_default=8192,
        base_capabilities={
            "reasoning":             5.0,
            "planning":              4.0,
            "analysis":              4.5,
            "code_generation":       5.0,
            "code_reasoning":        4.0,
            "system_design":         3.5,
            "prose_quality":         5.0,
            "instruction_adherence": 4.5,
            "domain_knowledge":      5.0,
            "context_utilization":   4.0,
            "structured_output":     4.0,
            "tool_use":              2.5,
            "vision":                0.0,
            "conversation":          4.5,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Mistral family
    # ════════════════════════════════════════════════════════
    "mistral": FamilyProfile(
        anchor_params_b=24,
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              6.5,
            "code_generation":       7.0,
            "code_reasoning":        6.5,
            "system_design":         6.5,
            "prose_quality":         7.5,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.0,
            "context_utilization":   7.0,
            "structured_output":     7.0,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          7.5,
        },
    ),
    "mixtral": FamilyProfile(
        anchor_params_b=47,   # 47B total, ~12B active
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              6.5,
            "code_generation":       7.0,
            "code_reasoning":        6.0,
            "system_design":         6.0,
            "prose_quality":         7.0,
            "instruction_adherence": 6.5,
            "domain_knowledge":      7.0,
            "context_utilization":   6.5,
            "structured_output":     6.5,
            "tool_use":              6.5,
            "vision":                0.0,
            "conversation":          7.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # GLM family
    # ════════════════════════════════════════════════════════
    "glm4": FamilyProfile(
        anchor_params_b=9,
        specialty="",
        function_calling=True,
        thinking_capable=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              7.0,
            "analysis":              7.0,
            "code_generation":       7.0,
            "code_reasoning":        6.5,
            "system_design":         6.5,
            "prose_quality":         7.0,
            "instruction_adherence": 7.5,
            "domain_knowledge":      7.0,
            "context_utilization":   7.5,
            "structured_output":     7.0,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          7.5,
        },
    ),
    "glm4_flash": FamilyProfile(
        anchor_params_b=9,
        specialty="",
        function_calling=True,
        thinking_capable=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              6.5,
            "code_generation":       6.5,
            "code_reasoning":        6.0,
            "system_design":         6.0,
            "prose_quality":         6.5,
            "instruction_adherence": 7.0,
            "domain_knowledge":      6.5,
            "context_utilization":   7.0,
            "structured_output":     6.5,
            "tool_use":              6.5,
            "vision":                0.0,
            "conversation":          7.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # DeepSeek family
    # ════════════════════════════════════════════════════════
    "deepseek_v3": FamilyProfile(
        anchor_params_b=671,   # MoE, ~37B active
        function_calling=True,
        context_default=65536,
        base_capabilities={
            "reasoning":             8.5,
            "planning":              8.0,
            "analysis":              8.0,
            "code_generation":       8.5,
            "code_reasoning":        8.0,
            "system_design":         7.5,
            "prose_quality":         7.5,
            "instruction_adherence": 7.5,
            "domain_knowledge":      8.5,
            "context_utilization":   7.5,
            "structured_output":     7.5,
            "tool_use":              7.0,
            "vision":                0.0,
            "conversation":          7.0,
        },
    ),
    "deepseek_r1": FamilyProfile(
        anchor_params_b=671,
        specialty="reasoning",
        function_calling=True,
        thinking_capable=True,
        context_default=65536,
        base_capabilities={
            "reasoning":             9.5,
            "planning":              8.5,
            "analysis":              9.0,
            "code_generation":       8.0,
            "code_reasoning":        9.0,
            "system_design":         8.0,
            "prose_quality":         6.5,
            "instruction_adherence": 7.0,
            "domain_knowledge":      8.0,
            "context_utilization":   7.0,
            "structured_output":     6.5,
            "tool_use":              6.0,
            "vision":                0.0,
            "conversation":          5.5,
        },
    ),
    "deepseek_coder": FamilyProfile(
        anchor_params_b=33,
        specialty="coding",
        function_calling=True,
        context_default=16384,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              5.5,
            "analysis":              6.0,
            "code_generation":       8.5,
            "code_reasoning":        8.0,
            "system_design":         6.0,
            "prose_quality":         4.5,
            "instruction_adherence": 6.5,
            "domain_knowledge":      5.5,
            "context_utilization":   6.0,
            "structured_output":     6.5,
            "tool_use":              5.5,
            "vision":                0.0,
            "conversation":          3.5,
        },
    ),
    "deepseek": FamilyProfile(
        anchor_params_b=67,
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.5,
            "planning":              7.0,
            "analysis":              7.0,
            "code_generation":       7.5,
            "code_reasoning":        7.0,
            "system_design":         6.5,
            "prose_quality":         6.5,
            "instruction_adherence": 6.5,
            "domain_knowledge":      7.0,
            "context_utilization":   6.5,
            "structured_output":     6.5,
            "tool_use":              6.0,
            "vision":                0.0,
            "conversation":          6.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Thinking / reasoning specialists
    # ════════════════════════════════════════════════════════
    "qwq": FamilyProfile(
        anchor_params_b=32,
        specialty="reasoning",
        function_calling=True,
        thinking_capable=True,
        context_default=40960,
        base_capabilities={
            "reasoning":             9.0,
            "planning":              8.0,
            "analysis":              8.5,
            "code_generation":       7.5,
            "code_reasoning":        8.5,
            "system_design":         7.5,
            "prose_quality":         6.0,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.5,
            "context_utilization":   7.0,
            "structured_output":     6.5,
            "tool_use":              6.0,
            "vision":                0.0,
            "conversation":          5.5,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Vision specialists
    # ════════════════════════════════════════════════════════
    "llava": FamilyProfile(
        anchor_params_b=13,
        specialty="vision",
        has_vision=True,
        context_default=4096,
        base_capabilities={
            "reasoning":             5.0,
            "planning":              3.5,
            "analysis":              5.5,
            "code_generation":       3.0,
            "code_reasoning":        3.0,
            "system_design":         2.5,
            "prose_quality":         5.0,
            "instruction_adherence": 5.0,
            "domain_knowledge":      5.0,
            "context_utilization":   4.0,
            "structured_output":     4.0,
            "tool_use":              2.0,
            "vision":                7.5,
            "conversation":          5.5,
        },
    ),
    "moondream": FamilyProfile(
        anchor_params_b=2,
        specialty="vision",
        has_vision=True,
        context_default=2048,
        base_capabilities={
            "reasoning":             3.0,
            "planning":              2.0,
            "analysis":              4.0,
            "code_generation":       1.5,
            "code_reasoning":        1.5,
            "system_design":         1.0,
            "prose_quality":         3.5,
            "instruction_adherence": 4.0,
            "domain_knowledge":      3.0,
            "context_utilization":   2.5,
            "structured_output":     3.5,
            "tool_use":              1.0,
            "vision":                6.0,
            "conversation":          3.0,
        },
    ),
    "internvl": FamilyProfile(
        anchor_params_b=26,
        specialty="vision",
        has_vision=True,
        function_calling=False,
        context_default=8192,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              5.0,
            "analysis":              7.0,
            "code_generation":       4.0,
            "code_reasoning":        4.0,
            "system_design":         3.5,
            "prose_quality":         5.5,
            "instruction_adherence": 6.0,
            "domain_knowledge":      6.5,
            "context_utilization":   5.5,
            "structured_output":     5.5,
            "tool_use":              3.0,
            "vision":                8.5,
            "conversation":          5.5,
        },
    ),
    "minicpm_v": FamilyProfile(
        anchor_params_b=8,
        specialty="vision",
        has_vision=True,
        context_default=8192,
        base_capabilities={
            "reasoning":             5.5,
            "planning":              4.5,
            "analysis":              6.0,
            "code_generation":       3.5,
            "code_reasoning":        3.5,
            "system_design":         3.0,
            "prose_quality":         5.0,
            "instruction_adherence": 5.5,
            "domain_knowledge":      5.5,
            "context_utilization":   5.0,
            "structured_output":     5.0,
            "tool_use":              2.5,
            "vision":                7.5,
            "conversation":          5.0,
        },
    ),

    # ════════════════════════════════════════════════════════
    # Other families
    # ════════════════════════════════════════════════════════
    "codellama": FamilyProfile(
        anchor_params_b=34,
        specialty="coding",
        context_default=16384,
        base_capabilities={
            "reasoning":             5.5,
            "planning":              4.5,
            "analysis":              5.0,
            "code_generation":       7.5,
            "code_reasoning":        7.0,
            "system_design":         5.0,
            "prose_quality":         3.5,
            "instruction_adherence": 5.5,
            "domain_knowledge":      4.5,
            "context_utilization":   5.5,
            "structured_output":     5.0,
            "tool_use":              4.0,
            "vision":                0.0,
            "conversation":          3.0,
        },
    ),
    "starcoder": FamilyProfile(
        anchor_params_b=15,
        specialty="coding",
        context_default=8192,
        base_capabilities={
            "reasoning":             4.5,
            "planning":              3.5,
            "analysis":              4.0,
            "code_generation":       7.0,
            "code_reasoning":        6.0,
            "system_design":         4.0,
            "prose_quality":         2.5,
            "instruction_adherence": 4.5,
            "domain_knowledge":      3.5,
            "context_utilization":   4.5,
            "structured_output":     4.5,
            "tool_use":              3.0,
            "vision":                0.0,
            "conversation":          2.0,
        },
    ),
    "command_r": FamilyProfile(
        anchor_params_b=35,
        function_calling=True,
        context_default=131072,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.5,
            "analysis":              7.0,
            "code_generation":       6.0,
            "code_reasoning":        5.5,
            "system_design":         6.0,
            "prose_quality":         7.5,
            "instruction_adherence": 7.0,
            "domain_knowledge":      7.5,
            "context_utilization":   8.0,
            "structured_output":     7.0,
            "tool_use":              7.5,
            "vision":                0.0,
            "conversation":          7.5,
        },
    ),
    "yi": FamilyProfile(
        anchor_params_b=34,
        function_calling=False,
        context_default=32768,
        base_capabilities={
            "reasoning":             6.5,
            "planning":              5.5,
            "analysis":              6.0,
            "code_generation":       6.0,
            "code_reasoning":        5.5,
            "system_design":         5.5,
            "prose_quality":         6.5,
            "instruction_adherence": 6.0,
            "domain_knowledge":      6.5,
            "context_utilization":   6.5,
            "structured_output":     5.5,
            "tool_use":              3.5,
            "vision":                0.0,
            "conversation":          6.0,
        },
    ),
    "internlm": FamilyProfile(
        anchor_params_b=20,
        function_calling=True,
        context_default=32768,
        base_capabilities={
            "reasoning":             7.0,
            "planning":              6.0,
            "analysis":              6.5,
            "code_generation":       6.5,
            "code_reasoning":        6.0,
            "system_design":         5.5,
            "prose_quality":         6.0,
            "instruction_adherence": 6.5,
            "domain_knowledge":      6.5,
            "context_utilization":   6.0,
            "structured_output":     6.5,
            "tool_use":              6.0,
            "vision":                0.0,
            "conversation":          6.0,
        },
    ),
}


# ─── Cloud Model Profiles (by litellm name substring) ────────────────────────
# These are full-res profiles — no size scaling needed.

CLOUD_PROFILES: dict[str, dict] = {
    "claude-sonnet-4": {
        "capabilities": {
            "reasoning": 9.5, "planning": 9.5, "analysis": 9.5,
            "code_generation": 9.5, "code_reasoning": 9.5, "system_design": 9.5,
            "prose_quality": 9.5, "instruction_adherence": 9.5,
            "domain_knowledge": 9.0, "context_utilization": 9.0,
            "structured_output": 9.0, "tool_use": 9.5,
            "vision": 8.5, "conversation": 9.0,
        },
        "thinking_model": False,
        "has_vision": True,
    },
    "claude-3-5-sonnet": {
        "capabilities": {
            "reasoning": 9.0, "planning": 9.0, "analysis": 9.0,
            "code_generation": 9.0, "code_reasoning": 9.0, "system_design": 9.0,
            "prose_quality": 9.0, "instruction_adherence": 9.0,
            "domain_knowledge": 8.5, "context_utilization": 8.5,
            "structured_output": 8.5, "tool_use": 9.0,
            "vision": 8.0, "conversation": 8.5,
        },
        "thinking_model": False,
        "has_vision": True,
    },
    "gpt-4o": {
        "capabilities": {
            "reasoning": 8.5, "planning": 8.5, "analysis": 8.5,
            "code_generation": 8.5, "code_reasoning": 8.0, "system_design": 8.5,
            "prose_quality": 9.0, "instruction_adherence": 8.5,
            "domain_knowledge": 9.0, "context_utilization": 8.0,
            "structured_output": 8.5, "tool_use": 9.0,
            "vision": 8.5, "conversation": 9.0,
        },
        "thinking_model": False,
        "has_vision": True,
    },
    "gpt-4o-mini": {
        "capabilities": {
            "reasoning": 7.0, "planning": 6.5, "analysis": 6.5,
            "code_generation": 7.0, "code_reasoning": 6.5, "system_design": 6.0,
            "prose_quality": 7.0, "instruction_adherence": 7.5,
            "domain_knowledge": 7.0, "context_utilization": 6.5,
            "structured_output": 7.5, "tool_use": 7.5,
            "vision": 6.5, "conversation": 7.0,
        },
        "thinking_model": False,
        "has_vision": True,
    },
    "o4-mini": {
        "capabilities": {
            "reasoning": 9.5, "planning": 9.0, "analysis": 9.0,
            "code_generation": 9.0, "code_reasoning": 9.0, "system_design": 8.5,
            "prose_quality": 7.0, "instruction_adherence": 8.5,
            "domain_knowledge": 8.5, "context_utilization": 7.5,
            "structured_output": 8.0, "tool_use": 8.5,
            "vision": 7.0, "conversation": 6.0,
        },
        "thinking_model": True,
        "has_vision": True,
    },
    "gemini-2.5-flash": {
        "capabilities": {
            "reasoning": 9.0, "planning": 8.5, "analysis": 8.5,
            "code_generation": 8.5, "code_reasoning": 8.0, "system_design": 8.0,
            "prose_quality": 8.0, "instruction_adherence": 8.5,
            "domain_knowledge": 8.5, "context_utilization": 9.0,
            "structured_output": 8.5, "tool_use": 8.5,
            "vision": 8.0, "conversation": 8.0,
        },
        "thinking_model": True,
        "has_vision": True,
    },
    "gemini-2.0-flash": {
        "capabilities": {
            "reasoning": 7.5, "planning": 7.5, "analysis": 7.5,
            "code_generation": 7.5, "code_reasoning": 7.0, "system_design": 7.0,
            "prose_quality": 7.5, "instruction_adherence": 7.5,
            "domain_knowledge": 7.5, "context_utilization": 8.5,
            "structured_output": 7.5, "tool_use": 7.5,
            "vision": 7.5, "conversation": 7.5,
        },
        "thinking_model": False,
        "has_vision": True,
    },
    "llama-3.3-70b": {
        "capabilities": {
            "reasoning": 7.5, "planning": 7.0, "analysis": 7.0,
            "code_generation": 7.0, "code_reasoning": 6.5, "system_design": 6.5,
            "prose_quality": 8.0, "instruction_adherence": 7.5,
            "domain_knowledge": 7.5, "context_utilization": 7.5,
            "structured_output": 7.0, "tool_use": 7.5,
            "vision": 0.0, "conversation": 8.0,
        },
        "thinking_model": False,
        "has_vision": False,
    },
    "llama-3.1-8b": {
        "capabilities": {
            "reasoning": 5.0, "planning": 4.5, "analysis": 4.5,
            "code_generation": 4.5, "code_reasoning": 4.0, "system_design": 3.5,
            "prose_quality": 5.0, "instruction_adherence": 5.5,
            "domain_knowledge": 5.0, "context_utilization": 5.0,
            "structured_output": 5.0, "tool_use": 5.0,
            "vision": 0.0, "conversation": 5.5,
        },
        "thinking_model": False,
        "has_vision": False,
    },
    "Qwen3-32B": {
        "capabilities": {
            "reasoning": 8.5, "planning": 8.0, "analysis": 8.0,
            "code_generation": 8.0, "code_reasoning": 7.5, "system_design": 7.5,
            "prose_quality": 7.5, "instruction_adherence": 8.0,
            "domain_knowledge": 8.0, "context_utilization": 7.5,
            "structured_output": 8.0, "tool_use": 8.0,
            "vision": 0.0, "conversation": 7.5,
        },
        "thinking_model": True,
        "has_vision": False,
    },
    "deepseek-r1": {
        "capabilities": {
            "reasoning": 9.5, "planning": 8.5, "analysis": 9.0,
            "code_generation": 8.0, "code_reasoning": 9.0, "system_design": 8.0,
            "prose_quality": 6.5, "instruction_adherence": 7.0,
            "domain_knowledge": 8.0, "context_utilization": 7.0,
            "structured_output": 6.5, "tool_use": 6.0,
            "vision": 0.0, "conversation": 5.5,
        },
        "thinking_model": True,
        "has_vision": False,
    },
}


# ─── Quantization Quality Impact ────────────────────────────────────────────
# Fraction of FP16 quality retained at each quantization level.
# These are rough averages from perplexity studies.

QUANTIZATION_RETENTION: dict[str, float] = {
    "F32":     1.00,
    "F16":     1.00,
    "FP16":    1.00,
    "BF16":    1.00,
    "Q8_0":    0.98,
    "Q8":      0.98,
    "Q6_K":    0.96,
    "Q6_K_L":  0.96,
    "Q5_K_M":  0.94,
    "Q5_K_S":  0.93,
    "Q5_K":    0.94,
    "Q5_0":    0.92,
    "Q5":      0.93,
    "Q4_K_XL": 0.935,
    "Q4_K_L":  0.93,
    "Q4_K_M":  0.92,
    "Q4_K_S":  0.90,
    "Q4_K":    0.92,
    "Q4_0":    0.88,
    "Q4":      0.90,
    "Q3_K_M":  0.86,
    "Q3_K_L":  0.87,
    "Q3_K_S":  0.83,
    "Q3_K":    0.86,
    "Q3":      0.85,
    "Q2_K":    0.78,
    "Q2_K_S":  0.75,
    "Q2":      0.76,
    "IQ4_XS":  0.91,
    "IQ4_NL":  0.91,
    "IQ3_M":   0.85,
    "IQ3_S":   0.83,
    "IQ3_XS":  0.80,
    "IQ3_XXS": 0.78,
    "IQ2_M":   0.73,
    "IQ2_S":   0.70,
    "IQ2_XS":  0.67,
    "IQ2_XXS": 0.63,
    "IQ1_M":   0.55,
    "IQ1_S":   0.50,
}

# ─── Size Scaling ────────────────────────────────────────────────────────────
# Interpolation table: (param_count_b, multiplier)
# Multiplier = 1.0 at anchor size. Scores = base_score * size_mult * quant_mult.
# Built from empirical benchmark scaling curves across model families.

SIZE_SCALING_TABLE: list[tuple[float, float]] = [
    (0.5,   0.35),
    (1.0,   0.45),
    (1.5,   0.52),
    (2.0,   0.58),
    (3.0,   0.65),
    (4.0,   0.70),
    (7.0,   0.78),
    (8.0,   0.80),
    (13.0,  0.88),
    (14.0,  0.89),
    (20.0,  0.94),
    (27.0,  0.98),
    (30.0,  1.00),    # ← reference point
    (32.0,  1.00),
    (34.0,  1.01),
    (40.0,  1.03),
    (47.0,  1.05),
    (65.0,  1.08),
    (70.0,  1.09),
    (72.0,  1.09),
    (110.0, 1.12),
    (180.0, 1.14),
    (400.0, 1.16),
    (671.0, 1.18),
]


def interpolate_size_multiplier(params_b: float) -> float:
    """
    Linear interpolation over SIZE_SCALING_TABLE.
    Returns multiplier relative to 30B reference.
    """
    if params_b <= 0:
        return 0.35

    table = SIZE_SCALING_TABLE

    # Clamp to table bounds
    if params_b <= table[0][0]:
        return table[0][1]
    if params_b >= table[-1][0]:
        return table[-1][1]

    # Find surrounding entries and interpolate
    for i in range(len(table) - 1):
        lo_p, lo_m = table[i]
        hi_p, hi_m = table[i + 1]
        if lo_p <= params_b <= hi_p:
            t = (params_b - lo_p) / (hi_p - lo_p)
            return lo_m + t * (hi_m - lo_m)

    return 1.0  # fallback


def get_quant_retention(quant_str: str) -> float:
    """
    Look up quality retention for a quantization string.
    Handles fuzzy matching (e.g., "Q4_K_M" from "q4_k_m-00001").
    """
    if not quant_str:
        return 0.92  # assume Q4_K_M as safe default

    q = quant_str.upper().strip()

    # Direct match
    if q in QUANTIZATION_RETENTION:
        return QUANTIZATION_RETENTION[q]

    # Try progressively shorter prefixes
    # e.g., "Q4_K_M_00001" → "Q4_K_M" → "Q4_K" → "Q4"
    for known_q, retention in sorted(
        QUANTIZATION_RETENTION.items(), key=lambda x: -len(x[0])
    ):
        if known_q in q:
            return retention

    # Last resort: guess from first chars
    if "Q8" in q or "F16" in q or "FP16" in q:
        return 0.98
    if "Q6" in q:
        return 0.96
    if "Q5" in q:
        return 0.93
    if "Q4" in q:
        return 0.90
    if "Q3" in q:
        return 0.85
    if "Q2" in q:
        return 0.76
    if "IQ4" in q:
        return 0.91
    if "IQ3" in q:
        return 0.83
    if "IQ2" in q:
        return 0.70
    if "IQ1" in q:
        return 0.52

    return 0.90  # conservative default


def detect_family(name_lower: str) -> str | None:
    """
    Match a model name/filename against FAMILY_PATTERNS.
    Returns family key or None.
    """
    for pattern_parts, family_key in FAMILY_PATTERNS:
        if all(part in name_lower for part in pattern_parts):
            return family_key
    return None


def get_default_profile() -> FamilyProfile:
    """Fallback profile for completely unknown model families."""
    return FamilyProfile(
        anchor_params_b=7,
        base_capabilities={
            "reasoning":             5.0,
            "planning":              4.0,
            "analysis":              4.5,
            "code_generation":       4.5,
            "code_reasoning":        4.0,
            "system_design":         3.5,
            "prose_quality":         4.5,
            "instruction_adherence": 4.5,
            "domain_knowledge":      4.5,
            "context_utilization":   4.0,
            "structured_output":     4.0,
            "tool_use":              3.0,
            "vision":                0.0,
            "conversation":          4.0,
        },
    )


# ─── YAML Override Loading ─────────────────────────────────────────────────
# Hardcoded dicts above are defaults. YAML entries override them at import time.

_YAML_PATH = Path(__file__).parent / "model_families.yaml"
if _YAML_PATH.exists():
    try:
        import yaml
        from datetime import datetime as _dt

        _yaml_data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        _logger = get_logger("models.model_profiles")
        _now = _dt.now()

        if _yaml_data:
            for key, data in (_yaml_data.get("families") or {}).items():
                _verified = data.pop("last_verified", None)
                if _verified:
                    try:
                        _vdate = _dt.strptime(str(_verified), "%Y-%m")
                        if (_now - _vdate).days > 180:
                            _logger.warning(
                                f"Model family '{key}' last verified {_verified} "
                                f"(>6 months ago) — consider updating"
                            )
                    except ValueError:
                        pass
                FAMILY_PROFILES[key] = FamilyProfile(**data)

            for key, data in (_yaml_data.get("cloud") or {}).items():
                _verified = data.pop("last_verified", None)
                if _verified:
                    try:
                        _vdate = _dt.strptime(str(_verified), "%Y-%m")
                        if (_now - _vdate).days > 180:
                            _logger.warning(
                                f"Cloud profile '{key}' last verified {_verified} "
                                f"(>6 months ago) — consider updating"
                            )
                    except ValueError:
                        pass
                CLOUD_PROFILES[key] = data

    except ImportError:
        pass  # pyyaml not installed — YAML overrides disabled
    except Exception as _e:
        logging.getLogger(__name__).warning(f"Failed to load model_families.yaml: {_e}")


# ─── Task Parameter Profiles ──────────────────────────────────────────────────
# Per-task-type generation parameters. Applied in router.py after model selection.
# Thinking/reasoning models always override temperature to None (model controls it).

_TASK_PARAM_DEFAULTS: dict[str, dict] = {
    # Low temp for precise, deterministic output
    "coding":      {"temperature": 0.15, "top_p": 0.9},
    "debugging":   {"temperature": 0.10, "top_p": 0.9},
    "testing":     {"temperature": 0.15, "top_p": 0.9},
    # Medium temp for analysis + planning
    "analysis":    {"temperature": 0.30, "top_p": 0.95},
    "planning":    {"temperature": 0.35, "top_p": 0.95},
    "research":    {"temperature": 0.30, "top_p": 0.95},
    "review":      {"temperature": 0.25, "top_p": 0.95},
    # Higher temp for creative/generative tasks
    "writing":     {"temperature": 0.65, "top_p": 0.95},
    "creative":    {"temperature": 0.80, "top_p": 0.95},
    # Conservative for factual/tool tasks
    "tool_use":    {"temperature": 0.10, "top_p": 0.9},
    "extraction":  {"temperature": 0.10, "top_p": 0.9},
    # Defaults
    "general":     {"temperature": 0.30, "top_p": 0.95},
}

_DEFAULT_PARAMS: dict = {"temperature": 0.30, "top_p": 0.95}


def get_task_params(task_type: str | None) -> dict:
    """
    Return generation parameters for a given task type.

    Usage in router:
        params = get_task_params(reqs.effective_task)
        completion_kwargs["temperature"] = params["temperature"]

    Callers should skip setting temperature entirely for thinking models
    (check model.thinking_model first).
    """
    if not task_type:
        return dict(_DEFAULT_PARAMS)
    return dict(_TASK_PARAM_DEFAULTS.get(task_type, _DEFAULT_PARAMS))
