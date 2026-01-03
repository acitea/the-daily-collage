"""
Mock S3 storage backend for testing/development.
"""

import logging
from typing import Dict, Optional

from backend.visualization.caching.core import CacheMetadata, StorageBackend

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

    def get_image(self, vibe_hash: str) -> Optional[bytes]:
        """Retrieve image."""
        return self.images.get(vibe_hash)

    def put_image(self, vibe_hash: str, image_data: bytes) -> str:
        """Store image and return URL."""
        self.images[vibe_hash] = image_data
        logger.info(f"Mock S3: Stored image {vibe_hash}")
        return f"s3://{self.bucket_name}/{vibe_hash}.png"

    def get_metadata(self, vibe_hash: str) -> Optional[CacheMetadata]:
        """Retrieve metadata."""
        return self.metadata.get(vibe_hash)

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata."""
        self.metadata[metadata.vibe_hash] = metadata
        logger.info(f"Mock S3: Stored metadata {metadata.vibe_hash}")

    def exists(self, vibe_hash: str) -> bool:
        """Check if cached."""
        return vibe_hash in self.images
