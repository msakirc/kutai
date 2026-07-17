"""Optional per-project lifecycle hooks. A hook module may define:
  pre_boot(project)   -> runs once, after the hub lock, before supervisors start
  on_exit(exit_code)  -> passed to each target's GuardConfig.on_exit
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("yasar_usta.hooks")


def load_hook(module_path: str | None):
    if not module_path:
        return None
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logger.error("Hook module %s failed to import: %s", module_path, e)
        return None


def run_pre_boot(hook, project) -> None:
    if hook is None or not hasattr(hook, "pre_boot"):
        return
    try:
        hook.pre_boot(project)
    except Exception as e:
        # Surfaced, not swallowed (spec: KutAI pre_boot failure must be visible)
        logger.error("pre_boot for %s failed: %s", project.id, e)
        raise
