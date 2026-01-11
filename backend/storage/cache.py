"""
High-level cache interface for visualization management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .core import CacheMetadata, StorageBackend, VibeHash

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
        cache_key = VibeHash.generate(city, timestamp)

        image_data = self.storage.get_image(cache_key)
        metadata = self.storage.get_metadata(cache_key)

        if image_data and metadata:
            logger.info(f"Cache hit: {cache_key}")
            return image_data, metadata
        elif image_data or metadata:
            logger.warning(f"Partial cache hit for {cache_key}")
            return image_data, metadata
        else:
            logger.info(f"Cache miss: {cache_key}")
            return None, None

    def set(
        self,
        city: str,
        timestamp: datetime,
        image_data: bytes,
        hitboxes: List[Dict],
    ) -> Tuple[str, CacheMetadata]:
        """
        Cache image and metadata.

        Args:
            city: Geographic location
            timestamp: Time window
            image_data: PNG image bytes
            hitboxes: List of hitbox dicts
            vibe_vector: (Deprecated) No longer stored - retrieve from feature store
            source_articles: (Deprecated) No longer stored - retrieve from feature store

        Returns:
            Tuple of (cache_key, metadata)
        """
        cache_key = VibeHash.generate(city, timestamp)

        # Store image
        image_url = self.storage.put_image(cache_key, image_data)

        # Create and store metadata (only hitboxes)
        metadata = CacheMetadata(
            cache_key=cache_key,
            hitboxes=hitboxes,
        )
        self.storage.put_metadata(cache_key, metadata)

        logger.info(f"Cached: {cache_key}")
        return cache_key, metadata

    def exists(
        self,
        city: str,
        timestamp: datetime,
    ) -> bool:
        """Check if vibe is cached."""
        cache_key = VibeHash.generate(city, timestamp)
        return self.storage.exists(cache_key)
