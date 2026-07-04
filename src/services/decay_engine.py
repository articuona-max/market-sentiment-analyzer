"""
Exponential Sentiment Decay Engine.

Applies time-decay to historical sentiment scores so that older data
progressively loses influence on real-time volatility tracking:

    W(t) = e^(-λ · Δt)

Where Δt is the data age in hours and λ is a dynamic decay constant
configured per data source (fast for RSS, slow for structural PDFs).

Performance Target: 25% relevance gain by preventing stale sentiment
from polluting real-time readings.
"""
import math
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SentimentDecayEngine:
    """
    Configurable exponential time-decay for sentiment scores.

    Can be initialized with an explicit lambda (decay_rate) or via a
    human-friendly half-life duration. Supports per-source tuning by
    instantiating separate engines for RSS vs. PDF data.
    """

    def __init__(self, decay_rate: Optional[float] = None, half_life_hours: float = 24.0):
        """
        Args:
            decay_rate: Explicit lambda value for the decay formula.
            half_life_hours: If decay_rate is None, calculates lambda such
                             that the weight halves over this duration.
        """
        if decay_rate is not None:
            if decay_rate < 0:
                raise ValueError("decay_rate (lambda) must be non-negative.")
            self.decay_rate = decay_rate
        else:
            if half_life_hours <= 0:
                raise ValueError("half_life_hours must be strictly positive.")
            # e^(-lambda * half_life) = 0.5  =>  lambda = ln(2) / half_life
            self.decay_rate = math.log(2.0) / half_life_hours

        logger.info(
            f"Initialized SentimentDecayEngine with lambda (decay_rate): "
            f"{self.decay_rate:.6f}"
        )

    def calculate_weight(
        self, timestamp: datetime, reference_time: Optional[datetime] = None
    ) -> float:
        """
        Calculates the decay weight W(t) based on elapsed time.

        Args:
            timestamp: The original datetime of the sentiment/news.
            reference_time: The datetime to decay against (defaults to now UTC).

        Returns:
            The exponential weight in the range (0, 1].
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Robustly handle timezone-aware vs timezone-naive datetimes
        if timestamp.tzinfo is not None and reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo is None and reference_time.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        delta = reference_time - timestamp
        delta_hours = delta.total_seconds() / 3600.0

        if delta_hours < 0:
            logger.debug(
                f"Timestamp {timestamp} is in the future relative to "
                f"{reference_time}. Returning weight 1.0."
            )
            return 1.0

        return math.exp(-self.decay_rate * delta_hours)

    def apply_decay(
        self,
        original_score: float,
        timestamp: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Applies time decay to a given sentiment score.

        Args:
            original_score: The initial sentiment score (e.g., -1.0 to 1.0).
            timestamp: The datetime when the sentiment was recorded.
            reference_time: The current evaluation datetime (defaults to now).

        Returns:
            The time-decayed sentiment score.
        """
        weight = self.calculate_weight(timestamp, reference_time)
        decayed_score = original_score * weight

        logger.debug(
            f"Applied decay: Score {original_score:.4f} -> {decayed_score:.4f} "
            f"(Weight: {weight:.4f})"
        )
        return decayed_score
