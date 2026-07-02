import hashlib
from datetime import datetime, timezone
import redis.asyncio as aioredis

class TemporalCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", window_minutes: int = 5):
        self.redis_url = redis_url
        self.window_minutes = window_minutes
        self.pool = None

    async def connect(self):
        self.pool = aioredis.from_url(self.redis_url, decode_responses=True)

    async def close(self):
        if self.pool:
            await self.pool.close()

    def generate_key(self, query: str) -> str:
        """Creates a deterministic, time-bucketed hash for O(1) lookups."""
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        current_minute = datetime.now(timezone.utc).minute
        bucket = current_minute // self.window_minutes
        return f"cache:vector:{query_hash}:bucket:{bucket}"

    async def get(self, query: str) -> str | None:
        key = self.generate_key(query)
        return await self.pool.get(key)

    async def set(self, query: str, value: str, ttl: int = 300):
        key = self.generate_key(query)
        await self.pool.setex(key, ttl, value)