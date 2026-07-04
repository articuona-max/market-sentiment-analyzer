"""
Singleton Redis client manager.

Ensures only one connection pool is used across the application for
optimal resource utilization. Handles connection failures gracefully
so upstream callers can fall back to cache-miss behavior.
"""
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Singleton Redis client with eager connection validation.
    
    Thread-safe via Redis' internal connection pool. The singleton
    pattern prevents pool fragmentation across the application.
    """
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None

    def __new__(cls, host: str = 'localhost', port: int = 6379, db: int = 0, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialize(host, port, db, **kwargs)
        return cls._instance

    def _initialize(self, host: str, port: int, db: int, **kwargs):
        """Initializes the Redis connection pool and tests the connection."""
        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                **kwargs
            )
            self._client = redis.Redis(connection_pool=pool)
            # Ping to eagerly validate connection
            self._client.ping()
            logger.info(f"Successfully connected to Redis at {host}:{port}/{db}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis during initialization: {e}")
            self._client = None

    @property
    def client(self) -> Optional[redis.Redis]:
        """Returns the active Redis client or None if disconnected."""
        return self._client

    @classmethod
    def get_client(cls) -> Optional[redis.Redis]:
        """Class method to easily retrieve the active redis client."""
        if cls._instance is None:
            cls()
        return cls._instance.client if cls._instance else None

    @classmethod
    def reset(cls):
        """Reset the singleton for testing purposes."""
        cls._instance = None
        cls._client = None
