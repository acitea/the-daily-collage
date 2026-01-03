"""
High-level cache interface for visualization management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.visualization.caching.core import CacheMetadata, StorageBackend, VibeHash

logger = logging.getLogger(__name__)


class VibeCache:
    """
    High-level caching interface.

    Handles vibe hash generation, storage, and retrieval.
    """

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    def get(
        self,
        city: str,
        timestamp: datetime,
    ) -> Tuple[Optional[bytes], Optional[CacheMetadata]]:
        """
        Retrieve cached image and metadata.

        Args:
            city: Geographic location
            timestamp: Time window for vibe

        Returns:
            Tuple of (image_bytes, metadata) or (None, None) if not cached
        """
        vibe_hash = VibeHash.generate(city, timestamp)

        image_data = self.storage.get_image(vibe_hash)
        metadata = self.storage.get_metadata(vibe_hash)

        if image_data and metadata:
            logger.info(f"Cache hit: {vibe_hash}")
            return image_data, metadata
        elif image_data or metadata:
            logger.warning(f"Partial cache hit for {vibe_hash}")
            return image_data, metadata
        else:
            logger.info(f"Cache miss: {vibe_hash}")
            return None, None

    def set(
        self,
        city: str,
        timestamp: datetime,
        image_data: bytes,
        hitboxes: List[Dict],
        vibe_vector: Dict[str, float],
        source_articles: List[Dict] = None,
    ) -> Tuple[str, CacheMetadata]:
        """
        Cache image and metadata.

        Args:
            city: Geographic location
            timestamp: Time window
            image_data: PNG image bytes
            hitboxes: List of hitbox dicts
            vibe_vector: Signal scores (stored in metadata only)
            source_articles: Optional articles that contributed

        Returns:
            Tuple of (image_url, metadata)
        """
        vibe_hash = VibeHash.generate(city, timestamp)

        # Store image
        image_url = self.storage.put_image(vibe_hash, image_data)

        # Create and store metadata
        metadata = CacheMetadata(
            vibe_hash=vibe_hash,
            city=city,
            timestamp=timestamp,
            vibe_vector=vibe_vector,
            image_url=image_url,
            hitboxes=hitboxes,
            source_articles=source_articles,
        )
        self.storage.put_metadata(metadata)

        logger.info(f"Cached: {vibe_hash}")
        return image_url, metadata

    def exists(
        self,
        city: str,
        timestamp: datetime,
    ) -> bool:
        """Check if vibe is cached."""
        vibe_hash = VibeHash.generate(city, timestamp)
        return self.storage.exists(vibe_hash)
