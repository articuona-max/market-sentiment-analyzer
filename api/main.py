"""
FastAPI Application — Market Sentiment Analyzer API Surface.

Endpoints:
  GET  /              — Serves the real-time dashboard.
  POST /analyze       — Accept alert JSON, run full pipeline, return classification.
  POST /ingest/rss    — Trigger a manual RSS polling cycle.
  POST /ingest/pdf    — Upload a PDF for processing and vector storage.
  GET  /health        — System health check (component status).
  GET  /api/dashboard/stats   — Aggregate pipeline statistics.
  GET  /api/dashboard/recent  — Recent analysis results.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings
from src.database.models import (
    PipelineResult,
    SentimentClassification,
    SentimentEnum,
)
from src.orchestrator.pipeline_runner import PipelineRunner
from src.ingestion.rss_worker import RSSWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models for the API
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    id: str = Field(..., description="Unique alert identifier.")
    title: str = Field(..., description="Alert headline.")
    summary: str = Field(..., description="Alert summary text.")
    source: str = Field(..., description="Feed source URL or name.")
    published_at: str = Field(..., description="ISO-8601 publication timestamp.")
    content: Optional[str] = Field(None, description="Full article content.")
    entity_tags: List[str] = Field(default_factory=list, description="Asset/entity tags.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    components: Dict[str, str]


class IngestPDFResponse(BaseModel):
    """Response body for the /ingest/pdf endpoint."""
    document_title: str
    chunks_stored: int


class IngestRSSResponse(BaseModel):
    """Response body for the /ingest/rss endpoint."""
    alerts_fetched: int
    alert_ids: List[str]


# ---------------------------------------------------------------------------
# Application lifecycle and state
# ---------------------------------------------------------------------------

_pipeline: Optional[PipelineRunner] = None
_rss_worker: Optional[RSSWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle handler.

    Initializes the pipeline runner and RSS worker on startup,
    and cleans up resources on shutdown.
    """
    global _pipeline, _rss_worker

    settings = get_settings()
    logger.info("Starting Market Sentiment Analyzer...")

    # Initialize the pipeline (includes Qdrant, cache, classifier)
    _pipeline = PipelineRunner(settings=settings)

    # Initialize RSS worker (without auto-start — triggered manually or via cron)
    _rss_worker = RSSWorker(
        feed_urls=settings.ingestion.rss_feeds,
        poll_interval_seconds=settings.ingestion.poll_interval_seconds,
    )

    logger.info("All components initialized. API is ready.")
    yield

    # Shutdown
    if _rss_worker:
        _rss_worker.stop()
    logger.info("Market Sentiment Analyzer shut down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Market Sentiment Analyzer",
    description=(
        "Production-grade financial sentiment analysis with temporal-aware "
        "caching, cross-source context fusion, and exponential decay analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for the dashboard
_dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if _dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_dashboard_dir)), name="static")


# ---------------------------------------------------------------------------
# Dashboard Route
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the main dashboard HTML page."""
    index_path = _dashboard_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dashboard Data API
# ---------------------------------------------------------------------------

@app.get(
    "/api/dashboard/stats",
    summary="Dashboard statistics",
    description="Returns aggregate pipeline statistics for the dashboard.",
)
async def dashboard_stats() -> Dict[str, Any]:
    """Returns aggregate stats for the dashboard."""
    if _pipeline is None:
        return {
            "total_analyzed": 0, "cache_hits": 0, "cache_hit_rate": 0.0,
            "sentiment_distribution": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
            "avg_exposure": 0.0, "avg_confidence": 0.0, "dominant_sentiment": "NEUTRAL",
        }
    return _pipeline.get_stats()


@app.get(
    "/api/dashboard/recent",
    summary="Recent analyses",
    description="Returns the most recent analysis results for the dashboard feed.",
)
async def dashboard_recent(
    limit: int = Query(default=20, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Returns recent analysis results."""
    if _pipeline is None:
        return []
    return _pipeline.get_recent_results(limit=limit)


# ---------------------------------------------------------------------------
# Core Pipeline Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=PipelineResult,
    summary="Analyze market sentiment",
    description=(
        "Accepts an RSS alert, runs the full pipeline "
        "(cache → fusion → LLM → decay → write-back), "
        "and returns the structured sentiment classification."
    ),
)
async def analyze_alert(request: AnalyzeRequest) -> PipelineResult:
    """Full pipeline analysis of a market alert."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        result = await asyncio.to_thread(
            _pipeline.analyze, request.model_dump()
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post(
    "/ingest/rss",
    response_model=IngestRSSResponse,
    summary="Trigger RSS polling",
    description="Manually triggers a single RSS polling cycle across all configured feeds.",
)
async def ingest_rss() -> IngestRSSResponse:
    """Manually trigger an RSS polling cycle."""
    if _rss_worker is None:
        raise HTTPException(status_code=503, detail="RSS worker not initialized.")

    try:
        alerts = await _rss_worker.poll_once()
        return IngestRSSResponse(
            alerts_fetched=len(alerts),
            alert_ids=[a.id for a in alerts],
        )
    except Exception as e:
        logger.error(f"RSS polling failed: {e}")
        raise HTTPException(status_code=500, detail=f"RSS polling failed: {e}")


@app.post(
    "/ingest/pdf",
    response_model=IngestPDFResponse,
    summary="Ingest a PDF document",
    description=(
        "Uploads a PDF file, extracts text in chunks, embeds them, "
        "and stores in the vector database for context fusion."
    ),
)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest."),
    document_title: str = Query(
        default="uploaded_document",
        description="Human-readable document title.",
    ),
    entity_tags: str = Query(
        default="",
        description="Comma-separated entity tags (e.g., 'AAPL,MSFT').",
    ),
) -> IngestPDFResponse:
    """Upload and ingest a PDF into the vector store."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted."
        )

    try:
        pdf_bytes = await file.read()
        tags = [t.strip() for t in entity_tags.split(",") if t.strip()]

        chunks_stored = await asyncio.to_thread(
            _pipeline.ingest_pdf,
            pdf_bytes=pdf_bytes,
            document_title=document_title,
            entity_tags=tags,
        )

        return IngestPDFResponse(
            document_title=document_title,
            chunks_stored=chunks_stored,
        )
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Reports the operational status of all system components.",
)
async def health_check() -> HealthResponse:
    """Check health of all system components."""
    components: Dict[str, str] = {}

    # Pipeline status
    components["pipeline"] = "ok" if _pipeline is not None else "not_initialized"

    # Redis status
    try:
        from src.cache.redis_client import RedisClient
        redis_client = RedisClient.get_client()
        if redis_client and redis_client.ping():
            components["redis"] = "ok"
        else:
            components["redis"] = "unavailable"
    except Exception:
        components["redis"] = "error"

    # Qdrant status
    try:
        if _pipeline and _pipeline.qdrant_store:
            collections = _pipeline.qdrant_store.client.get_collections()
            components["qdrant"] = f"ok ({len(collections.collections)} collections)"
        else:
            components["qdrant"] = "not_initialized"
    except Exception:
        components["qdrant"] = "error"

    # RSS worker status
    components["rss_worker"] = "ok" if _rss_worker is not None else "not_initialized"

    # Overall status
    critical = ["pipeline", "qdrant"]
    all_ok = all(components.get(c, "").startswith("ok") for c in critical)

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        components=components,
    )
