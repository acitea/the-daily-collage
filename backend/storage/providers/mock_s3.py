"""
Mock S3 storage backend for testing/development.
"""

import logging
from typing import Dict, Optional

from storage.core import CacheMetadata, StorageBackend

logger = logging.getLogger(__name__)


class MockS3StorageBackend(StorageBackend):
    """
    Mock S3 storage for testing/development.

    Stores in-memory, simulating S3 behavior.
    """

    def __init__(self, bucket_name: str = "vibe-images"):
        self.bucket_name = bucket_name
        self.images: Dict[str, bytes] = {}
        self.metadata: Dict[str, CacheMetadata] = {}

    def get_image(self, cache_key: str) -> Optional[bytes]:
        """Retrieve image."""
        return self.images.get(cache_key)

    def put_image(self, cache_key: str, image_data: bytes) -> str:
        """Store image and return URL."""
        self.images[cache_key] = image_data
        logger.info(f"Mock S3: Stored image {cache_key}")
        return f"s3://{self.bucket_name}/{cache_key}.png"

    def get_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata."""
        return self.metadata.get(cache_key)

    def put_metadata(self, cache_key: str, metadata: CacheMetadata) -> None:
        """Store metadata."""
        self.metadata[metadata.cache_key] = metadata
        logger.info(f"Mock S3: Stored metadata {metadata.cache_key}")

    def exists(self, cache_key: str) -> bool:
        """Check if cached."""
        return cache_key in self.images
