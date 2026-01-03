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
    def extract_info(cls, vibe_hash: str) -> Optional[Dict]:
        """
        Parse information from a vibe hash.

        Args:
            vibe_hash: Hash string

        Returns:
            Dict with city, date, window or None if invalid
        """
        parts = vibe_hash.split("_")
        if len(parts) < 4:
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
    """Metadata about a cached visualization."""

    def __init__(
        self,
        vibe_hash: str,
        city: str,
        timestamp: datetime,
        vibe_vector: Dict[str, float],
        image_url: str,
        hitboxes: List[Dict],
        source_articles: List[Dict] = None,
    ):
        self.vibe_hash = vibe_hash
        self.city = city
        self.timestamp = timestamp.isoformat()
        self.vibe_vector = vibe_vector
        self.image_url = image_url
        self.hitboxes = hitboxes
        self.source_articles = source_articles or []
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/DB storage."""
        return {
            "vibe_hash": self.vibe_hash,
            "city": self.city,
            "timestamp": self.timestamp,
            "vibe_vector": self.vibe_vector,
            "image_url": self.image_url,
            "hitboxes": self.hitboxes,
            "source_articles": self.source_articles,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict) -> "CacheMetadata":
        """Reconstruct from dictionary."""
        metadata = CacheMetadata(
            vibe_hash=data["vibe_hash"],
            city=data["city"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            vibe_vector=data["vibe_vector"],
            image_url=data["image_url"],
            hitboxes=data["hitboxes"],
            source_articles=data.get("source_articles", []),
        )
        metadata.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return metadata


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def get_image(self, vibe_hash: str) -> Optional[bytes]:
        """Retrieve image bytes for a vibe hash."""
        pass

    @abstractmethod
    def put_image(self, vibe_hash: str, image_data: bytes) -> str:
        """Store image and return URL."""
        pass

    @abstractmethod
    def get_metadata(self, vibe_hash: str) -> Optional[CacheMetadata]:
        """Retrieve metadata for a vibe hash."""
        pass

    @abstractmethod
    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata."""
        pass

    @abstractmethod
    def exists(self, vibe_hash: str) -> bool:
        """Check if vibe hash is cached."""
        pass
