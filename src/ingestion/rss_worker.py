"""
Asynchronous RSS Feed Worker.

Non-blocking cron poller that ingests live market feeds, normalizes
entries into the unified RSSAlert schema, and deduplicates using
an in-memory seen-ID set.

Uses aiohttp for async HTTP and feedparser for RSS/Atom parsing.
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Set, Callable, Awaitable, Optional

import aiohttp
import feedparser

from src.database.models import RSSAlert

logger = logging.getLogger(__name__)


class RSSWorker:
    """
    Async RSS polling worker.

    Polls a list of feed URLs at a configurable interval, parses entries
    into RSSAlert objects, deduplicates via entry ID/link hash, and
    dispatches new alerts to a registered callback.
    """

    def __init__(
        self,
        feed_urls: List[str],
        poll_interval_seconds: int = 300,
        on_alert: Optional[Callable[[RSSAlert], Awaitable[None]]] = None,
    ):
        """
        Args:
            feed_urls: List of RSS/Atom feed URLs to poll.
            poll_interval_seconds: Seconds between polling cycles.
            on_alert: Async callback invoked for each new RSSAlert.
        """
        self.feed_urls = feed_urls
        self.poll_interval_seconds = poll_interval_seconds
        self.on_alert = on_alert
        self._seen_ids: Set[str] = set()
        self._running = False

    @staticmethod
    def _generate_entry_id(entry: dict, feed_url: str) -> str:
        """
        Generates a deterministic unique ID for an RSS entry.
        Falls back to a hash of (feed_url + title) if no id/link is present.
        """
        raw_id = entry.get("id") or entry.get("link") or ""
        if not raw_id:
            raw_id = feed_url + entry.get("title", "")
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_entity_tags(entry: dict) -> List[str]:
        """
        Extracts entity tags from RSS entry metadata.
        Looks at 'tags' (common in Atom feeds) and title keywords.
        """
        tags = []
        for tag_obj in entry.get("tags", []):
            term = tag_obj.get("term", "").strip()
            if term:
                tags.append(term)
        return tags

    @staticmethod
    def _parse_published(entry: dict) -> datetime:
        """
        Extracts and normalizes the published datetime from an entry.
        Falls back to current UTC time if parsing fails.
        """
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            try:
                from time import mktime
                ts = mktime(published_parsed)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass
        return datetime.now(timezone.utc)

    def _entry_to_alert(self, entry: dict, feed_url: str) -> Optional[RSSAlert]:
        """Converts a single feedparser entry dict into an RSSAlert."""
        entry_id = self._generate_entry_id(entry, feed_url)

        # Deduplication gate
        if entry_id in self._seen_ids:
            return None
        self._seen_ids.add(entry_id)

        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()

        if not title and not summary:
            logger.debug(f"Skipping entry with no title or summary: {entry_id}")
            return None

        return RSSAlert(
            id=entry_id,
            title=title,
            summary=summary or title,
            source=feed_url,
            published_at=self._parse_published(entry),
            content=entry.get("content", [{}])[0].get("value") if entry.get("content") else None,
            entity_tags=self._extract_entity_tags(entry),
        )

    async def _fetch_feed(
        self, session: aiohttp.ClientSession, url: str
    ) -> List[RSSAlert]:
        """Fetches and parses a single RSS feed URL."""
        alerts = []
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.warning(f"Feed {url} returned status {resp.status}")
                    return alerts

                raw_text = await resp.text()
                feed = feedparser.parse(raw_text)

                if feed.bozo:
                    logger.warning(
                        f"Feed {url} has parsing issues: {feed.bozo_exception}"
                    )

                for entry in feed.entries:
                    alert = self._entry_to_alert(entry, url)
                    if alert:
                        alerts.append(alert)

                logger.info(
                    f"Fetched {len(alerts)} new alerts from {url} "
                    f"(total entries: {len(feed.entries)})"
                )
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching feed: {url}")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching feed {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching feed {url}: {e}")

        return alerts

    async def poll_once(self) -> List[RSSAlert]:
        """
        Executes a single polling cycle across all configured feeds.

        Returns:
            List of new (deduplicated) RSSAlert objects.
        """
        all_alerts: List[RSSAlert] = []

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_feed(session, url) for url in self.feed_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Feed polling task failed: {result}")
                    continue
                all_alerts.extend(result)

        # Dispatch to callback
        if self.on_alert and all_alerts:
            for alert in all_alerts:
                try:
                    await self.on_alert(alert)
                except Exception as e:
                    logger.error(
                        f"Alert callback failed for {alert.id}: {e}"
                    )

        logger.info(f"Poll cycle complete: {len(all_alerts)} new alerts total.")
        return all_alerts

    async def run(self):
        """
        Starts the continuous polling loop.
        Call stop() to gracefully terminate.
        """
        self._running = True
        logger.info(
            f"RSS Worker started. Polling {len(self.feed_urls)} feeds "
            f"every {self.poll_interval_seconds}s."
        )

        while self._running:
            try:
                await self.poll_once()
            except Exception as e:
                logger.error(f"Unhandled error in poll cycle: {e}")

            await asyncio.sleep(self.poll_interval_seconds)

    def stop(self):
        """Signals the polling loop to stop after the current cycle."""
        self._running = False
        logger.info("RSS Worker stop signal received.")
