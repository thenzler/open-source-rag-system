"""
Metrics API Router
Provides Prometheus metrics endpoint and monitoring information
"""

import logging

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse

from ..middleware.metrics_middleware import (get_db_metrics, get_doc_metrics,
                                             get_llm_metrics,
                                             get_query_metrics)
from ..services.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Get Prometheus metrics in standard format

    Returns metrics in Prometheus exposition format for scraping by Prometheus server.
    """
    try:
        metrics_service = get_metrics_service()

        if not metrics_service.enabled:
            raise HTTPException(
                status_code=503,
                detail="Metrics collection is disabled (Prometheus client not available)",
            )

        metrics_data = metrics_service.get_metrics()

        return Response(
            content=metrics_data, media_type=metrics_service.get_content_type()
        )

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@router.get("/health")
async def get_metrics_health():
    """
    Get metrics system health status

    Returns basic health information about the metrics collection system.
    """
    try:
        metrics_service = get_metrics_service()
        stats = metrics_service.get_stats_summary()

        return {
            "status": "healthy" if stats.get("metrics_enabled", False) else "disabled",
            "metrics_enabled": stats.get("metrics_enabled", False),
            **stats,
        }

    except Exception as e:
        logger.error(f"Failed to get metrics health: {e}")
        return {"status": "error", "metrics_enabled": False, "error": str(e)}


@router.get("/stats")
async def get_metrics_stats():
    """
    Get detailed metrics statistics

    Returns comprehensive statistics about the metrics collection system
    and current system state.
    """
    try:
        metrics_service = get_metrics_service()

        if not metrics_service.enabled:
            return {
                "metrics_enabled": False,
                "message": "Metrics collection is disabled",
            }

        # Get basic stats
        stats = metrics_service.get_stats_summary()

        # Add collector information
        stats["collectors"] = {
            "query_metrics": get_query_metrics() is not None,
            "database_metrics": get_db_metrics() is not None,
            "document_metrics": get_doc_metrics() is not None,
            "llm_metrics": get_llm_metrics() is not None,
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get metrics stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics statistics")


@router.post("/update-system")
async def update_system_metrics():
    """
    Manually trigger system metrics update

    Forces an immediate update of system resource metrics (CPU, memory, disk).
    Normally these are updated automatically when metrics are requested.
    """
    try:
        metrics_service = get_metrics_service()

        if not metrics_service.enabled:
            raise HTTPException(
                status_code=503, detail="Metrics collection is disabled"
            )

        metrics_service.update_system_metrics()

        return {"status": "success", "message": "System metrics updated"}

    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to update system metrics")


@router.get("/config")
async def get_metrics_config():
    """
    Get metrics configuration information

    Returns configuration details about the metrics collection system.
    """
    try:
        metrics_service = get_metrics_service()

        config = {
            "metrics_enabled": metrics_service.enabled,
            "prometheus_available": hasattr(metrics_service, "registry"),
            "endpoints": {
                "prometheus_metrics": "/metrics",
                "health_check": "/metrics/health",
                "statistics": "/metrics/stats",
                "manual_update": "/metrics/update-system",
            },
        }

        if metrics_service.enabled:
            config["registry_info"] = {
                "collectors_count": len(metrics_service.registry._collector_to_names),
                "metric_families": list(
                    metrics_service.registry._collector_to_names.values()
                ),
            }

        return config

    except Exception as e:
        logger.error(f"Failed to get metrics config: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get metrics configuration"
        )


@router.get("/sample")
async def get_sample_metrics():
    """
    Get sample of current metrics

    Returns a parsed sample of current metrics for debugging and monitoring.
    This is NOT in Prometheus format - use /metrics for that.
    """
    try:
        metrics_service = get_metrics_service()

        if not metrics_service.enabled:
            raise HTTPException(
                status_code=503, detail="Metrics collection is disabled"
            )

        # Get raw metrics
        raw_metrics = metrics_service.get_metrics()

        # Parse into a more readable format
        sample = {
            "timestamp": metrics_service.start_time,
            "uptime_seconds": metrics_service.get_stats_summary().get(
                "uptime_seconds", 0
            ),
            "metrics_preview": [],
        }

        # Extract first few lines as preview
        lines = raw_metrics.split("\n")
        for line in lines[:20]:  # First 20 lines
            if line.strip() and not line.startswith("#"):
                sample["metrics_preview"].append(line.strip())

        sample["total_metrics_lines"] = len(
            [line for line in lines if line.strip() and not line.startswith("#")]
        )

        return sample

    except Exception as e:
        logger.error(f"Failed to get sample metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sample metrics")
