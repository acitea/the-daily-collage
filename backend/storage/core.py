"""
Core data structures and interfaces for the caching system.

Defines:
- VibeHash: Deterministic cache key generation
- CacheMetadata: Metadata about cached visualizations
- StorageBackend: Abstract interface for storage implementations
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from backend.visualization.assets import Hitbox

logger = logging.getLogger(__name__)


class VibeHash:
    """
    Generates deterministic cache keys from vibe vector data.

    Cache key format: City_YYYY-MM-DD_TimeWindow_Hash
    where Hash is SHA256 of discretized scores.
    """

    # Time windows (6-hour windows as per Daily Collage spec)
    WINDOWS_PER_DAY = 4
    WINDOW_DURATION_HOURS = 24 // WINDOWS_PER_DAY  # 6 hours

    # Discretization: round scores to nearest multiple of this
    DISCRETIZATION_STEP = 0.1

    @classmethod
    def generate(
        cls,
        city: str,
        timestamp: datetime,
    ) -> str:
        """
        Generate deterministic cache key based on location and time.

        Args:
            city: Geographic location (e.g., 'stockholm', 'gothenburg')
            timestamp: Time to hash (used to determine time window)

        Returns:
            str: Cache key (e.g., 'stockholm_2025-12-11_00-06')
        """
        # Normalize city name
        city_normalized = city.lower().replace(" ", "_")

        # Get date and time window
        date_str = timestamp.strftime("%Y-%m-%d")
        window_index = timestamp.hour // cls.WINDOW_DURATION_HOURS
        window_str = f"{window_index * cls.WINDOW_DURATION_HOURS:02d}-{(window_index + 1) * cls.WINDOW_DURATION_HOURS:02d}"

        # Combine parts (no vibe vector hash needed)
        cache_key = f"{city_normalized}_{date_str}_{window_str}"

        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    @classmethod
    def extract_info(cls, cache_key: str) -> Optional[Dict]:
        """
        Parse information from a cache_key.

        Args:
            cache_key: Cache key string (format: city_YYYY-MM-DD_HH-HH)

        Returns:
            Dict with city, date, window or None if invalid
        """
        parts = cache_key.split("_")
        if len(parts) < 3:
            return None

        city = parts[0]
        date_str = parts[1]
        window_str = parts[2]

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return {
                "city": city,
                "date": date_obj,
                "window": window_str,
            }
        except ValueError:
            return None


class CacheMetadata:
    """Metadata about a cached visualization.
    
    Stores only essential data that cannot be retrieved elsewhere:
    - cache_key: Universal identifier (contains city/date/time)
    - hitboxes: Interactive regions (unique to this generation)
    
    Not stored (retrievable from other sources):
    - vibe_vector: Retrievable from Hopsworks feature store
    - source_articles: Retrievable from Hopsworks feature store
    - image_url: Not needed (retrieved via storage API)
    - city/timestamp: Already encoded in cache_key
    """

    def __init__(
        self,
        cache_key: str,
        hitboxes: List[Hitbox],
    ):
        self.cache_key = cache_key
        self.hitboxes = hitboxes
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/DB storage."""
        return {
            "cache_key": self.cache_key,
            "hitboxes": self.hitboxes,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict) -> "CacheMetadata":
        """Reconstruct from dictionary."""
        metadata = CacheMetadata(
            cache_key=data["cache_key"],
            hitboxes=[Hitbox(**hb) for hb in data["hitboxes"]],
        )
        metadata.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return metadata


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def get_image(self, cache_key: str) -> Optional[bytes]:
        """Retrieve image bytes for a cache_key."""
        pass

    @abstractmethod
    def put_image(self, cache_key: str, image_data: bytes) -> str:
        """Store image and return URL."""
        pass

    @abstractmethod
    def get_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata for a cache_key."""
        pass

    @abstractmethod
    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata."""
        pass

    @abstractmethod
    def exists(self, cache_key: str) -> bool:
        """Check if cache_key is cached."""
        pass
