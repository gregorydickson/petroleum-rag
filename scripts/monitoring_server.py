"""FastAPI monitoring server for RAG system health checks and metrics.

This server provides:
- /health - Basic health check endpoint
- /health/live - Kubernetes liveness probe
- /health/ready - Kubernetes readiness probe with dependency checks
- /metrics - Prometheus metrics endpoint
- /stats - Human-readable statistics
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Monitoring Server",
    description="Health checks and metrics for petroleum RAG benchmark system",
    version="1.0.0",
)


# -------------------------------------------------------------------------
# Health Check Endpoints
# -------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint.

    Returns:
        Dictionary with status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe.

    This endpoint checks if the application is alive and should not be restarted.
    It performs minimal checks to avoid false positives.

    Returns:
        Dictionary with status
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness() -> dict[str, Any]:
    """Kubernetes readiness probe with dependency checks.

    This endpoint checks if the application is ready to serve traffic by
    validating connections to critical dependencies:
    - Embedder (OpenAI API)
    - Storage backends (Chroma, Weaviate, FalkorDB)

    Returns:
        Dictionary with overall status and individual component checks
    """
    from embeddings.embedder import UnifiedEmbedder
    from storage.chroma_store import ChromaStore
    from storage.falkordb_store import FalkorDBStore
    from storage.weaviate_store import WeaviateStore

    health_status: dict[str, str] = {
        "embedder": "unknown",
        "chroma": "unknown",
        "weaviate": "unknown",
        "falkordb": "unknown",
    }

    # Check embedder
    try:
        embedder = UnifiedEmbedder()
        is_valid = await embedder.validate_connection()
        health_status["embedder"] = "healthy" if is_valid else "unhealthy"
        await embedder.close()
    except Exception as e:
        health_status["embedder"] = f"unhealthy: {str(e)[:100]}"
        logger.warning(f"Embedder health check failed: {e}")

    # Check Chroma
    try:
        chroma = ChromaStore()
        await chroma.initialize()
        is_healthy = await chroma.health_check()
        health_status["chroma"] = "healthy" if is_healthy else "unhealthy"
    except Exception as e:
        health_status["chroma"] = f"unhealthy: {str(e)[:100]}"
        logger.warning(f"Chroma health check failed: {e}")

    # Check Weaviate
    try:
        weaviate = WeaviateStore()
        await weaviate.initialize()
        is_healthy = await weaviate.health_check()
        health_status["weaviate"] = "healthy" if is_healthy else "unhealthy"
    except Exception as e:
        health_status["weaviate"] = f"unhealthy: {str(e)[:100]}"
        logger.warning(f"Weaviate health check failed: {e}")

    # Check FalkorDB
    try:
        falkordb = FalkorDBStore()
        await falkordb.initialize()
        is_healthy = await falkordb.health_check()
        health_status["falkordb"] = "healthy" if is_healthy else "unhealthy"
    except Exception as e:
        health_status["falkordb"] = f"unhealthy: {str(e)[:100]}"
        logger.warning(f"FalkorDB health check failed: {e}")

    # Determine overall readiness
    # At minimum, embedder and at least one storage backend must be healthy
    embedder_healthy = health_status["embedder"] == "healthy"
    storage_healthy = any(
        health_status[backend] == "healthy"
        for backend in ["chroma", "weaviate", "falkordb"]
    )

    all_ready = embedder_healthy and storage_healthy

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": health_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# -------------------------------------------------------------------------
# Metrics Endpoint
# -------------------------------------------------------------------------


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint.

    This endpoint exposes all Prometheus metrics in the text-based
    exposition format that Prometheus can scrape.

    Returns:
        Response with Prometheus metrics in text format
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# -------------------------------------------------------------------------
# Statistics Endpoint
# -------------------------------------------------------------------------


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Human-readable statistics endpoint.

    This endpoint provides a summary of system statistics in JSON format,
    making it easy to inspect the system state without Prometheus.

    Returns:
        Dictionary with system statistics
    """
    # Import here to avoid circular dependencies
    from prometheus_client import REGISTRY

    # Collect current metric values
    stats_data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Get metric samples
    try:
        for metric in REGISTRY.collect():
            metric_name = metric.name
            if metric_name.startswith("rag_"):
                # Store metric samples
                samples = []
                for sample in metric.samples:
                    samples.append(
                        {
                            "name": sample.name,
                            "labels": sample.labels,
                            "value": sample.value,
                        }
                    )
                stats_data[metric_name] = {
                    "type": metric.type,
                    "documentation": metric.documentation,
                    "samples": samples[:10],  # Limit to first 10 samples
                }
    except Exception as e:
        logger.warning(f"Failed to collect metric samples: {e}")
        stats_data["error"] = str(e)

    return stats_data


# -------------------------------------------------------------------------
# Startup/Shutdown Events
# -------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize monitoring server on startup."""
    logger.info("Monitoring server starting up")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info("Monitoring server shutting down")


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info",
    )
