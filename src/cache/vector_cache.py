"""
Temporal-Aware Semantic Vector Cache.

Sits in front of the LLM and Vector DB to intercept repeated or
near-duplicate queries. Uses cosine similarity over cached embeddings
and enforces a sliding TTL window to prevent stale sentiment from
being served for fast-moving financial data.

Performance Target: 40% latency reduction on repeated/similar queries.

Note: For highly scalable production environments (>100k cached items),
migrate to Redis Stack (RediSearch) for native vector similarity search.
This implementation uses scan + in-memory cosine similarity, suitable
for moderate cache sizes.
"""
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np

from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)


class TemporalVectorCache:
    """
    Semantic cache mapping query embeddings to sentiment results in Redis.
    
    Cache semantics:
      - GET: Scan all cached vectors, return best cosine-similarity match
        if it exceeds the threshold AND the key has not expired (TTL).
      - SET: Store query text + embedding + result with SETEX for automatic
        temporal expiry.
    """

    def __init__(self, similarity_threshold: float = 0.95, prefix: str = "semcache:"):
        self.redis_client = RedisClient()
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    def _generate_key(self, query: str) -> str:
        """Generates a consistent SHA-256 hash key for the query."""
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        return f"{self.prefix}{query_hash}"

    def get(self, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Searches the cache for an embedding matching the query_embedding
        with similarity >= similarity_threshold.

        Handles Redis connection drops gracefully by returning a cache miss.
        
        Returns:
            The cached sentiment result dict, or None on cache miss.
        """
        redis = self.redis_client.client
        if not redis:
            logger.warning("Redis client unavailable, falling back to cache miss.")
            return None

        try:
            best_match = None
            highest_sim = -1.0

            # Iterate over active keys matching the prefix.
            # TTL expiration is natively handled by Redis — expired keys
            # won't appear in scan results.
            for key in redis.scan_iter(f"{self.prefix}*"):
                data_str = redis.get(key)
                if not data_str:
                    continue

                try:
                    data = json.loads(data_str)
                    cached_embedding = data.get("embedding")

                    if not cached_embedding:
                        continue

                    sim = self._cosine_similarity(query_embedding, cached_embedding)

                    if sim >= self.similarity_threshold and sim > highest_sim:
                        highest_sim = sim
                        best_match = data
                except json.JSONDecodeError:
                    logger.error(f"Corrupted cache data for key {key}, skipping.")
                    continue

            if best_match:
                logger.info(f"Semantic cache HIT (similarity: {highest_sim:.4f})")
                return best_match.get("result")

            logger.info("Semantic cache MISS (no vector matched threshold)")
            return None

        except Exception as e:
            logger.error(f"Error during semantic cache retrieval: {e}")
            return None

    def set(
        self,
        query: str,
        query_embedding: List[float],
        result: Dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Stores the query, its embedding, and the result in Redis with
        a temporal validity window (TTL).

        Args:
            query: The original text query.
            query_embedding: The vector representation of the query.
            result: The computed sentiment or LLM response to cache.
            ttl_seconds: Temporal validity window in seconds.

        Returns:
            True if cached successfully, False otherwise.
        """
        redis = self.redis_client.client
        if not redis:
            logger.warning("Redis client unavailable, skipping cache set.")
            return False

        try:
            cache_key = self._generate_key(query)

            payload = {
                "query": query,
                "embedding": query_embedding,
                "result": result,
                "cached_at": time.time(),
                "ttl": ttl_seconds,
            }

            # setex handles both the SET and EXPIRE operations atomically
            redis.setex(
                name=cache_key,
                time=ttl_seconds,
                value=json.dumps(payload),
            )
            logger.debug(f"Cached semantic result for query with TTL {ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Error during semantic cache set: {e}")
            return False
