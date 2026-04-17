# Shopping Pipeline Automation — Hybrid Workflow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shopping research runs as a workflow (same engine as i2p) with mechanical steps executed by Python (`ShoppingPipeline`) and judgment steps by LLM agents. No LLM waste on steps that are just "call a tool and return results."

**Architecture:** Follows the existing `CodingPipeline` pattern — workflow steps with `agent: "shopping_pipeline"` get delegated to `ShoppingPipeline.run()` which executes Python functions directly. LLM agents only handle deal analysis and recommendation synthesis.

**Remaining tasks:**
- Task A: Create ShoppingPipeline class
- Task B: Wire into orchestrator (same pattern as CodingPipeline)
- Task C: Update workflow JSONs — mechanical steps use `shopping_pipeline` agent
- Task D: Tests
