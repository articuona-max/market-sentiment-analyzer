"""
Pydantic v2 data schemas for the Market Sentiment Analyzer.

Defines the type-safe contracts for every data boundary in the system:
  - Ingestion → Fusion (RSSAlert, PDFContext)
  - Fusion → LLM (FusedPayload)
  - LLM → Orchestrator (SentimentClassification)
"""
from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Helper to get timezone-aware UTC datetime for Pydantic factories."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Ingestion Schemas
# ---------------------------------------------------------------------------

class RSSAlert(BaseModel):
    """Schema for incoming real-time RSS alerts."""
    id: str
    title: str
    summary: str
    source: str
    published_at: datetime
    content: Optional[str] = None
    entity_tags: List[str] = Field(default_factory=list)


class PDFContext(BaseModel):
    """Schema for historical PDF context retrieved from Vector DB."""
    id: str
    document_title: str
    extracted_text: str
    published_at: datetime
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    base_sentiment_score: Optional[float] = None
    entity_tags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fusion Schema
# ---------------------------------------------------------------------------

class FusedPayload(BaseModel):
    """
    Combined real-time alert with time-decayed historical context.
    This is the final prompt-ready payload passed to the LLM classifier.
    """
    rss_alert: RSSAlert
    historical_contexts: List[PDFContext]
    decayed_scores: List[float]
    fusion_timestamp: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# LLM Output Schema
# ---------------------------------------------------------------------------

class SentimentEnum(str, Enum):
    """Market sentiment direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SentimentClassification(BaseModel):
    """
    Structured output from the LLM classifier.
    
    This schema is used both as the Pydantic response_format for OpenAI's
    structured outputs and as the response_schema for Gemini's JSON mode.
    """
    raw_sentiment: SentimentEnum = Field(
        ...,
        description="The overall market sentiment: BULLISH, BEARISH, or NEUTRAL."
    )
    exposure_score: float = Field(
        ...,
        description=(
            "A score between 0.0 and 1.0 representing the degree of market "
            "exposure or impact relevance of the analyzed event."
        ),
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,
        description="Classification confidence between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        ...,
        description="Brief reasoning for the assigned sentiment and exposure score."
    )


# ---------------------------------------------------------------------------
# Pipeline Result (wraps classification with metadata)
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """
    Final output of the end-to-end pipeline, enriching the raw LLM
    classification with provenance and cache metadata.
    """
    classification: SentimentClassification
    alert_id: str
    source: str
    cache_hit: bool = False
    contexts_used: int = 0
    processed_at: datetime = Field(default_factory=utc_now)
