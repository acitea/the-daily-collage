"""
Factory for creating storage backends based on configuration.
"""

import logging
from typing import Optional

from storage.core import StorageBackend
from storage.providers import (
    HopsworksStorageBackend,
    LocalStorageBackend,
    MockS3StorageBackend,
)

logger = logging.getLogger(__name__)


def create_storage_backend(
    backend_type: str,
    bucket_name: str = "vibe-images",
    local_storage_dir: str = "./storage/vibes",
    hopsworks_service=None,
    artifact_collection: str = "vibe_images",
) -> StorageBackend:
    """
    Factory function to create appropriate storage 

    Args:
        backend_type: Type of backend ('local', 's3', 'minio', 'hopsworks')
        bucket_name: S3/MinIO bucket name
        local_storage_dir: Local storage directory path
        hopsworks_service: HopsworksService instance (required for hopsworks backend)
        artifact_collection: Hopsworks artifact collection name

    Returns:
        StorageBackend instance

    Raises:
        ValueError: If backend_type is 'hopsworks' but no hopsworks_service provided
    """
    backend_type = backend_type.lower()

    if backend_type == "local":
        logger.info(f"Using LocalStorageBackend: {local_storage_dir}")
        return LocalStorageBackend(storage_dir=local_storage_dir)

    elif backend_type in ("s3", "minio"):
        logger.info(f"Using MockS3StorageBackend: {bucket_name}")
        return MockS3StorageBackend(bucket_name=bucket_name)

    elif backend_type == "hopsworks":
        if hopsworks_service is None:
            raise ValueError(
                "hopsworks_service is required when backend_type='hopsworks'"
            )
        logger.info(f"Using HopsworksStorageBackend: {artifact_collection}")
        return HopsworksStorageBackend(
            hopsworks_service=hopsworks_service,
            artifact_collection=artifact_collection,
        )

    else:
        logger.warning(
            f"Unknown storage backend: {backend_type}, defaulting to local"
        )
        return LocalStorageBackend(storage_dir=local_storage_dir)
