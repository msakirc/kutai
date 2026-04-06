"""
Yazbunu — structured JSONL logging for the KutAI ecosystem.

Usage:
    from yazbunu import get_logger, init_logging

    init_logging(log_dir="./logs", project="kutai")
    logger = get_logger("core.orchestrator")
    logger.info("task dispatched", task="42", mission="m-7")
"""

from yazbunu.formatter import YazFormatter

__all__ = ["get_logger", "init_logging", "YazFormatter"]
