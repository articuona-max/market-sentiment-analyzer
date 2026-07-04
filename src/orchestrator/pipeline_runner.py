"""
End-to-End Pipeline Runner (Orchestrator).

Wires all system components into the complete data flow:

  1. Accept RSSAlert or raw dict.
  2. Cache Interception: embed → check TemporalVectorCache.
  3. Cache Hit: return immediately, bypassing DB and LLM.
  4. Cache Miss: FusionCore enriches with historical PDF contexts.
  5. LLM Inference: classify the fused payload.
  6. Decay Aggregation: apply time-decay to final score.
  7. Write-Back: cache result in TemporalVectorCache + persist to Qdrant.
  8. Return PipelineResult.
"""
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from src.config import Settings, get_settings
from src.database.models import (
    RSSAlert,
    FusedPayload,
    SentimentClassification,
    PipelineResult,
)
from src.cache.vector_cache import TemporalVectorCache
from src.services.decay_engine import SentimentDecayEngine
from src.services.LLM_classifier import LLMClassifier
from src.pipeline.fusion_core import FusionCore
from src.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Orchestrates the complete sentiment analysis pipeline.

    Manages the lifecycle of all sub-components and implements the
    cache-interception pattern for latency reduction.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        qdrant_store: Optional[QdrantStore] = None,
        cache: Optional[TemporalVectorCache] = None,
        classifier: Optional[LLMClassifier] = None,
    ):
        self.settings = settings or get_settings()

        # Storage engine — also provides embedding function
        self.qdrant_store = qdrant_store or QdrantStore(
            collection_name=self.settings.qdrant.collection_name,
            embedding_dim=self.settings.qdrant.embedding_dimension,
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
            use_in_memory=self.settings.qdrant.use_in_memory,
        )

        # Decay engines — separate instances for RSS (fast) and PDF (slow)
        self.rss_decay = SentimentDecayEngine(
            half_life_hours=self.settings.decay.rss_half_life_hours
        )
        self.pdf_decay = SentimentDecayEngine(
            half_life_hours=self.settings.decay.pdf_half_life_hours
        )

        # Fusion core — wired to Qdrant and PDF decay engine
        self.fusion_core = FusionCore(
            decay_engine=self.pdf_decay,
            vector_db=self.qdrant_store,
        )

        # Semantic cache
        self.cache = cache or TemporalVectorCache(
            similarity_threshold=self.settings.cache.similarity_threshold,
        )

        # LLM classifier
        self.classifier = classifier or LLMClassifier(
            provider=self.settings.llm.provider,
            model_name=(
                self.settings.llm.gemini_model
                if self.settings.llm.provider == "gemini"
                else self.settings.llm.openai_model
            ),
            api_key=(
                self.settings.gemini_api_key
                if self.settings.llm.provider == "gemini"
                else self.settings.openai_api_key
            ),
        )

        # In-memory result history for the dashboard (bounded deque)
        self._result_history: deque = deque(maxlen=50)
        self._total_analyzed: int = 0
        self._cache_hits: int = 0

        logger.info("PipelineRunner initialized with all components wired.")

    def analyze(self, rss_alert_data: Dict[str, Any]) -> PipelineResult:
        """
        Runs the full end-to-end sentiment analysis pipeline.

        Args:
            rss_alert_data: Raw dict representing an incoming RSS alert.

        Returns:
            PipelineResult with classification, provenance, and cache metadata.
        """
        # Step 1: Validate alert
        alert = RSSAlert(**rss_alert_data)
        query_text = alert.summary or alert.title

        logger.info(f"Pipeline started for alert '{alert.id}': {alert.title[:60]}...")

        # Step 2: Cache Interception — embed and check cache
        try:
            query_embedding = self.qdrant_store.embed_text(query_text)
            cached_result = self.cache.get(query_embedding)
        except Exception as e:
            logger.warning(f"Cache lookup failed, proceeding without cache: {e}")
            query_embedding = None
            cached_result = None

        # Step 3: Cache Hit — return immediately
        if cached_result is not None:
            logger.info(f"Cache HIT for alert '{alert.id}'. Returning cached result.")
            classification = SentimentClassification(**cached_result)
            result = PipelineResult(
                classification=classification,
                alert_id=alert.id,
                source=alert.source,
                cache_hit=True,
                contexts_used=0,
            )
            self._record_result(result, alert.title)
            return result

        # Step 4: Cache Miss — Context Enrichment via FusionCore
        logger.info(f"Cache MISS for alert '{alert.id}'. Running fusion pipeline.")
        payload = self.fusion_core.process_alert(rss_alert_data)

        # Step 5: LLM Inference
        classification = self.classifier.classify_payload(
            payload, self.fusion_core
        )

        # Step 6: Decay Aggregation — apply RSS decay to the raw exposure score
        # This adjusts the score based on how old the alert itself is
        rss_weight = self.rss_decay.calculate_weight(alert.published_at)
        decayed_exposure = classification.exposure_score * rss_weight

        # Build final classification with decayed exposure
        final_classification = SentimentClassification(
            raw_sentiment=classification.raw_sentiment,
            exposure_score=round(decayed_exposure, 4),
            confidence=classification.confidence,
            rationale=classification.rationale,
        )

        # Step 7: Write-Back — cache and persist
        if query_embedding is not None:
            try:
                self.cache.set(
                    query=query_text,
                    query_embedding=query_embedding,
                    result=final_classification.model_dump(),
                    ttl_seconds=self.settings.cache.default_ttl_seconds,
                )
                logger.info(f"Cached classification for alert '{alert.id}'.")
            except Exception as e:
                logger.warning(f"Cache write-back failed: {e}")

        # Step 8: Return final result
        result = PipelineResult(
            classification=final_classification,
            alert_id=alert.id,
            source=alert.source,
            cache_hit=False,
            contexts_used=len(payload.historical_contexts),
        )

        self._record_result(result, alert.title)

        logger.info(
            f"Pipeline complete for '{alert.id}': "
            f"sentiment={final_classification.raw_sentiment.value}, "
            f"exposure={final_classification.exposure_score:.4f}, "
            f"confidence={final_classification.confidence:.4f}, "
            f"contexts_used={result.contexts_used}"
        )

        return result

    def _record_result(self, result: PipelineResult, title: str = ""):
        """Records a result to the in-memory history for dashboard queries."""
        self._total_analyzed += 1
        if result.cache_hit:
            self._cache_hits += 1

        self._result_history.appendleft({
            "alert_id": result.alert_id,
            "title": title,
            "source": result.source,
            "sentiment": result.classification.raw_sentiment.value,
            "exposure_score": result.classification.exposure_score,
            "confidence": result.classification.confidence,
            "rationale": result.classification.rationale,
            "cache_hit": result.cache_hit,
            "contexts_used": result.contexts_used,
            "processed_at": result.processed_at.isoformat(),
        })

    def get_recent_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent analysis results for the dashboard."""
        return list(self._result_history)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Returns aggregate pipeline statistics for the dashboard."""
        results = list(self._result_history)
        sentiments = [r["sentiment"] for r in results]
        exposures = [r["exposure_score"] for r in results]
        confidences = [r["confidence"] for r in results]

        return {
            "total_analyzed": self._total_analyzed,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                round(self._cache_hits / self._total_analyzed, 4)
                if self._total_analyzed > 0 else 0.0
            ),
            "sentiment_distribution": {
                "BULLISH": sentiments.count("BULLISH"),
                "BEARISH": sentiments.count("BEARISH"),
                "NEUTRAL": sentiments.count("NEUTRAL"),
            },
            "avg_exposure": (
                round(sum(exposures) / len(exposures), 4)
                if exposures else 0.0
            ),
            "avg_confidence": (
                round(sum(confidences) / len(confidences), 4)
                if confidences else 0.0
            ),
            "dominant_sentiment": (
                max(set(sentiments), key=sentiments.count)
                if sentiments else "NEUTRAL"
            ),
        }

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        document_title: str = "uploaded_document",
        entity_tags: Optional[List[str]] = None,
    ) -> int:
        """
        Ingests a PDF document into the vector store.

        Args:
            pdf_bytes: Raw PDF file bytes.
            document_title: Title for provenance tracking.
            entity_tags: Entity tags for metadata filtering.

        Returns:
            Number of chunks stored.
        """
        from src.ingestion.pdf_worker import PDFWorker

        worker = PDFWorker(
            chunk_size=self.settings.ingestion.pdf_chunk_size,
            chunk_overlap=self.settings.ingestion.pdf_chunk_overlap,
        )
        contexts = worker.extract_from_bytes(
            pdf_bytes=pdf_bytes,
            document_title=document_title,
            entity_tags=entity_tags,
        )
        count = self.qdrant_store.upsert_pdf_contexts(contexts)
        logger.info(
            f"Ingested PDF '{document_title}': {count} chunks stored."
        )
        return count
