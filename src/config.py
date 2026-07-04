"""
Centralized application configuration.

Reads from environment variables with sensible defaults for local development.
All sensitive values (API keys) must be provided via environment variables.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection settings."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant vector store settings."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "pdf_contexts"
    embedding_dimension: int = 768
    use_in_memory: bool = True  # In-memory mode for local dev


@dataclass(frozen=True)
class CacheConfig:
    """Semantic cache tuning knobs."""
    similarity_threshold: float = 0.95
    default_ttl_seconds: int = 3600  # 1 hour
    key_prefix: str = "semcache:"


@dataclass(frozen=True)
class DecayConfig:
    """Exponential decay parameters per data source."""
    rss_half_life_hours: float = 6.0    # Fast decay for real-time streams
    pdf_half_life_hours: float = 168.0  # Slow decay for structural documents (1 week)


@dataclass(frozen=True)
class IngestionConfig:
    """RSS polling and PDF processing settings."""
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    ])
    poll_interval_seconds: int = 300  # 5 minutes
    pdf_chunk_size: int = 1000        # Characters per chunk
    pdf_chunk_overlap: int = 200      # Overlap between chunks


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "gemini"          # "gemini" or "openai"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"


class Settings:
    """
    Application-wide settings singleton.
    
    Reads configuration from environment variables, falling back to
    dataclass defaults for local development.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.redis = RedisConfig(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
        )
        self.qdrant = QdrantConfig(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            collection_name=os.environ.get("QDRANT_COLLECTION", "pdf_contexts"),
            embedding_dimension=int(os.environ.get("EMBEDDING_DIM", "768")),
            use_in_memory=os.environ.get("QDRANT_IN_MEMORY", "true").lower() == "true",
        )
        self.cache = CacheConfig(
            similarity_threshold=float(os.environ.get("CACHE_SIM_THRESHOLD", "0.95")),
            default_ttl_seconds=int(os.environ.get("CACHE_TTL", "3600")),
        )
        self.decay = DecayConfig(
            rss_half_life_hours=float(os.environ.get("RSS_HALF_LIFE_HOURS", "6.0")),
            pdf_half_life_hours=float(os.environ.get("PDF_HALF_LIFE_HOURS", "168.0")),
        )

        feed_urls_env = os.environ.get("RSS_FEED_URLS")
        feeds = feed_urls_env.split(",") if feed_urls_env else IngestionConfig().rss_feeds
        self.ingestion = IngestionConfig(
            rss_feeds=feeds,
            poll_interval_seconds=int(os.environ.get("RSS_POLL_INTERVAL", "300")),
            pdf_chunk_size=int(os.environ.get("PDF_CHUNK_SIZE", "1000")),
            pdf_chunk_overlap=int(os.environ.get("PDF_CHUNK_OVERLAP", "200")),
        )
        self.llm = LLMConfig(
            provider=os.environ.get("LLM_PROVIDER", "gemini"),
            gemini_model=os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            openai_model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        )

        # API keys — read lazily, never logged
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        logger.info(
            f"Settings loaded: provider={self.llm.provider}, "
            f"redis={self.redis.host}:{self.redis.port}, "
            f"qdrant={'in-memory' if self.qdrant.use_in_memory else f'{self.qdrant.host}:{self.qdrant.port}'}"
        )


def get_settings() -> Settings:
    """Factory function for retrieving the global settings singleton."""
    return Settings()
