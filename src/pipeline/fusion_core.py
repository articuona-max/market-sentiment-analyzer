"""
Cross-Source Context Fusion Pipeline.

Orchestrates the merging of real-time RSS alerts with historical PDF
context from the vector database. Retrieves context, applies time-decay
to historical sentiments, and prepares an enriched payload for the
LLM classifier.

Performance Target: 15% accuracy gain by eliminating noise from raw RSS
streams through deterministic context enrichment.
"""
import logging
from typing import List, Dict, Any, Protocol, Optional
from datetime import datetime, timezone

from pydantic import ValidationError

from src.database.models import RSSAlert, PDFContext, FusedPayload
from src.services.decay_engine import SentimentDecayEngine

logger = logging.getLogger(__name__)


class VectorDBClientProtocol(Protocol):
    """
    Interface for the Vector Database client.

    Ensures loose coupling — allows easy swapping between Qdrant,
    PGVector, or mock implementations without changing fusion logic.
    """

    def search_similar_pdfs(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        ...


class FusionCore:
    """
    Deterministic context fusion engine.

    Pipeline:
      1. Validate incoming RSS alert via Pydantic.
      2. Query vector DB for matching historical PDF contexts.
      3. Apply exponential time-decay to historical sentiment scores.
      4. Assemble and return the validated FusedPayload.
    """

    def __init__(
        self,
        decay_engine: Optional[SentimentDecayEngine] = None,
        vector_db: Optional[VectorDBClientProtocol] = None,
    ):
        self.decay_engine = decay_engine or SentimentDecayEngine()
        self.vector_db = vector_db

    def _retrieve_context(
        self, text: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieves historical PDF contexts matching the alert text."""
        if not self.vector_db:
            logger.warning(
                "No VectorDB client configured. "
                "Proceeding without historical context."
            )
            return []
        try:
            return self.vector_db.search_similar_pdfs(text, limit=limit)
        except Exception as e:
            logger.error(f"Failed to retrieve context from VectorDB: {e}")
            return []

    def process_alert(self, rss_alert_data: Dict[str, Any]) -> FusedPayload:
        """
        Executes the full fusion pipeline.

        Args:
            rss_alert_data: Raw dict representing an RSS alert.

        Returns:
            FusedPayload ready for LLM classification.

        Raises:
            ValueError: If the RSS alert data fails Pydantic validation.
        """
        # 1. Validate incoming RSS Alert
        try:
            rss_alert = RSSAlert(**rss_alert_data)
        except ValidationError as e:
            logger.error(f"Invalid RSS alert data: {e}")
            raise ValueError(f"RSS Alert validation failed: {e}")

        # 2. Retrieve historical contexts via vector search
        query_text = rss_alert.summary if rss_alert.summary else rss_alert.title
        pdf_contexts_data = self._retrieve_context(query_text)

        pdf_contexts = []
        for ctx_data in pdf_contexts_data:
            try:
                pdf_contexts.append(PDFContext(**ctx_data))
            except ValidationError as e:
                logger.warning(f"Skipping malformed PDF context: {e}")

        # 3. Apply time decay to historical context scores
        decayed_scores = []
        now = datetime.now(timezone.utc)

        for ctx in pdf_contexts:
            if ctx.base_sentiment_score is not None:
                decayed_score = self.decay_engine.apply_decay(
                    original_score=ctx.base_sentiment_score,
                    timestamp=ctx.published_at,
                    reference_time=now,
                )
                decayed_scores.append(decayed_score)
            else:
                decayed_scores.append(0.0)

        # 4. Assemble fused payload
        payload = FusedPayload(
            rss_alert=rss_alert,
            historical_contexts=pdf_contexts,
            decayed_scores=decayed_scores,
        )

        logger.info(
            f"Fusion complete for alert '{rss_alert.id}'. "
            f"Integrated {len(pdf_contexts)} contexts."
        )
        return payload

    def generate_llm_prompt(self, payload: FusedPayload) -> str:
        """
        Constructs a deterministic, structured text prompt for the
        LLM classifier from the fused payload.
        """
        lines = [
            "Analyze the market sentiment of the following real-time news alert.",
            "",
            "--- REAL-TIME ALERT ---",
            f"Title: {payload.rss_alert.title}",
            f"Source: {payload.rss_alert.source}",
            f"Date: {payload.rss_alert.published_at.isoformat()}",
            f"Summary: {payload.rss_alert.summary}",
        ]

        if payload.rss_alert.content:
            lines.append(f"Content: {payload.rss_alert.content}")

        if payload.rss_alert.entity_tags:
            lines.append(f"Entities: {', '.join(payload.rss_alert.entity_tags)}")

        if payload.historical_contexts:
            lines.append("")
            lines.append(
                "--- HISTORICAL CONTEXT (Cross-Referenced from PDFs) ---"
            )
            for i, (ctx, decayed_score) in enumerate(
                zip(payload.historical_contexts, payload.decayed_scores)
            ):
                lines.append("")
                lines.append(
                    f"Context {i + 1} (Source: {ctx.document_title}, "
                    f"Date: {ctx.published_at.date()}):"
                )
                lines.append(f"Extract: {ctx.extracted_text}")
                if ctx.base_sentiment_score is not None:
                    lines.append(
                        f"Historical Sentiment Score (Time-Decayed): "
                        f"{decayed_score:.4f} "
                        f"(Original: {ctx.base_sentiment_score:.4f})"
                    )

        lines.append("")
        lines.append(
            "Based on the real-time alert and the historical context provided, "
            "determine the overall sentiment (BULLISH, BEARISH, or NEUTRAL), "
            "an exposure score (0.0–1.0 indicating market impact), "
            "and a confidence score (0.0–1.0)."
        )

        return "\n".join(lines)
