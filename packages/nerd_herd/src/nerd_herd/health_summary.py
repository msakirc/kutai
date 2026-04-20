"""health_summary — ported from general_beckman.watchdog.check_resources.

Returns a structured dict describing the current resource health state.
All telegram.send_notification calls are dropped — alerts reach Telegram
via the nerd_herd_health_alert cron marker which spawns salako.notify_user.
"""
from __future__ import annotations

from nerd_herd.health import HealthRegistry

_registry = HealthRegistry()


async def health_summary() -> dict:
    """Check system resource health and return a structured summary.

    Returns:
        {
            "alerts": [str, ...],           # high-severity strings for Telegram alerting
            "issues": [str, ...],           # full issue list for logs / /status
            "providers_degraded": [...],    # list of degraded cloud provider names
            "vram_leak_suspected": bool,
            "gpu_throttling": bool,
            "low_ram": bool,
            "credentials_expired": [...],   # list of expired credential service names
            "local_model_healthy": bool,
        }
    """
    resource_issues: list[str] = []
    alerts: list[str] = []
    providers_degraded: list[str] = []
    vram_leak_suspected = False
    gpu_throttling = False
    low_ram = False
    credentials_expired: list[str] = []
    local_model_healthy = True

    # ── Check llama-server health ──
    try:
        from src.models.local_model_manager import get_local_manager
        manager = get_local_manager()
        if manager.current_model and not manager.is_loaded:
            msg = f"llama-server unhealthy (model: {manager.current_model})"
            resource_issues.append(msg)
            local_model_healthy = False
    except Exception as e:
        import logging
        logging.getLogger("nerd_herd.health_summary").warning(
            f"Local model check failed: {e}"
        )

    # ── Check GPU health ──
    try:
        from src.models.gpu_monitor import get_gpu_monitor
        from src.models.local_model_manager import get_local_manager

        gpu_state = get_gpu_monitor().get_state()

        if gpu_state.gpu.available:
            if gpu_state.gpu.is_throttling:
                msg = (
                    f"GPU thermal throttling! "
                    f"Temp: {gpu_state.gpu.temperature_c}°C"
                )
                resource_issues.append(msg)
                alerts.append(msg)
                gpu_throttling = True

            mgr = get_local_manager()
            if gpu_state.gpu.vram_usage_pct > 95 and not mgr.is_loaded:
                msg = (
                    f"VRAM nearly full ({gpu_state.gpu.vram_usage_pct:.0f}%) "
                    f"but no model loaded — possible leak"
                )
                resource_issues.append(msg)
                alerts.append(msg)
                vram_leak_suspected = True

        if gpu_state.ram_available_mb < 2048:
            msg = f"Low RAM: {gpu_state.ram_available_mb}MB available"
            resource_issues.append(msg)
            alerts.append(msg)
            low_ram = True

    except Exception as e:
        import logging
        logging.getLogger("nerd_herd.health_summary").warning(
            f"GPU health check failed: {e}"
        )

    # ── Check local model status ──
    try:
        from src.models.local_model_manager import get_local_manager
        mgr = get_local_manager()
        mgr_status = mgr.get_status()
        if not mgr_status.get("healthy", True) and mgr_status.get("loaded_model"):
            resource_issues.append("Local model unhealthy")
            local_model_healthy = False
    except Exception as e:
        import logging
        logging.getLogger("nerd_herd.health_summary").warning(
            f"Local model status check failed: {e}"
        )

    # ── Check circuit breakers (cloud providers) ──
    try:
        from src.core.router import get_kdv
        kdv_status = get_kdv().status
        degraded = [
            p for p, prov_status in kdv_status.items()
            if prov_status.circuit_breaker_open
        ]
        if degraded:
            providers_degraded = degraded
            msg = f"Degraded providers: {', '.join(degraded)}"
            resource_issues.append(msg)

            from src.models.model_registry import get_registry
            registry = get_registry()
            all_cloud_providers = set(
                m.provider for m in registry.cloud_models()
            )
            if all_cloud_providers and all_cloud_providers.issubset(set(degraded)):
                critical_msg = (
                    "ALL cloud providers are degraded! "
                    "Only local inference available."
                )
                resource_issues.append(critical_msg)
                alerts.append(critical_msg)
    except Exception as e:
        import logging
        logging.getLogger("nerd_herd.health_summary").warning(
            f"Circuit breaker check failed: {e}"
        )

    # ── Restore rate limits ──
    try:
        from src.core.router import get_kdv
        get_kdv().restore_limits()
    except Exception:
        pass

    # ── Check expiring credentials ──
    try:
        from src.security.credential_store import list_credentials, get_credential
        services = await list_credentials()
        for svc in services:
            cred = await get_credential(svc)
            if cred is None:
                msg = f"Credential '{svc}' has expired. Refresh with /credential add."
                resource_issues.append(msg)
                alerts.append(msg)
                credentials_expired.append(svc)
    except Exception as e:
        import logging
        logging.getLogger("nerd_herd.health_summary").warning(
            f"Credential expiry check failed: {e}"
        )

    return {
        "alerts": alerts,
        "issues": resource_issues,
        "providers_degraded": providers_degraded,
        "vram_leak_suspected": vram_leak_suspected,
        "gpu_throttling": gpu_throttling,
        "low_ram": low_ram,
        "credentials_expired": credentials_expired,
        "local_model_healthy": local_model_healthy,
    }
