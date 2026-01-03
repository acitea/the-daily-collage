"""
Hopsworks dataset storage backend.

Delegates to HopsworksService for all Hopsworks operations.
Stores images and metadata in the Resources dataset under a specified directory.
"""

import io
import json
import logging
from typing import Optional

from backend.server.services.hopsworks import HopsworksService
from backend.storage.core import CacheMetadata, StorageBackend

logger = logging.getLogger(__name__)


class HopsworksStorageBackend(StorageBackend):
    """
    Hopsworks dataset storage backend.

    Stores images and metadata in Hopsworks Resources dataset.
    Delegates all Hopsworks operations to HopsworksService.
    """

    def __init__(self, hopsworks_service: HopsworksService, artifact_collection: str = "vibe_images"):
        """
        Initialize Hopsworks storage backend.

        Args:
            hopsworks_service: Instance of HopsworksService for dataset operations
            artifact_collection: Directory name in Resources dataset (e.g., "vibe_images")
        """
        self.hopsworks_service = hopsworks_service
        self.artifact_collection = artifact_collection

        # Ensure connection
        if not self.hopsworks_service._dataset_api:
            self.hopsworks_service.connect()

        # Get dataset API
        try:
            self.dataset_api = self.hopsworks_service._dataset_api
            logger.info(f"Initialized Hopsworks storage with dataset directory: Resources/{artifact_collection}")
        except Exception as e:
            logger.error(f"Failed to get dataset API: {e}")
            self.dataset_api = None

    def _is_connected(self) -> bool:
        """Check if connected to Hopsworks."""
        return self.dataset_api is not None

    def get_image(self, cache_key: str) -> Optional[bytes]:
        """Retrieve image from Hopsworks dataset."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks")
            return None

        try:
            import tempfile
            import os
            
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{cache_key}.png"
                dataset_path = f"Resources/{self.artifact_collection}/{filename}"
                local_path = os.path.join(tmpdir, filename)
                
                # Download from dataset
                self.dataset_api.download(dataset_path, local_path)
                
                # Read file
                with open(local_path, "rb") as f:
                    image_data = f.read()
                
                logger.debug(f"Retrieved image from Hopsworks dataset: {cache_key}")
                return image_data

        except Exception as e:
            logger.debug(f"Failed to retrieve image {cache_key}: {e}")
            return None

    def put_image(self, cache_key: str, image_data: bytes) -> str:
        """Store image in Hopsworks dataset."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks, cannot store image")
            return ""

        try:
            import tempfile
            import os
            
            # Create temporary directory and write file
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{cache_key}.png"
                local_path = os.path.join(tmpdir, filename)
                
                # Write image data to file
                with open(local_path, "wb") as f:
                    f.write(image_data)
                
                # Upload to Hopsworks dataset
                upload_path = f"Resources/{self.artifact_collection}"
                self.dataset_api.upload(local_path, upload_path, overwrite=True)
                
                logger.info(f"Stored image in Hopsworks dataset: {cache_key}")
                return f"hopsworks://Resources/{self.artifact_collection}/{filename}"

        except Exception as e:
            logger.error(f"Failed to store image in Hopsworks: {e}")
            return ""

    def get_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata from Hopsworks dataset."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks")
            return None

        try:
            import tempfile
            import os
            
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{cache_key}_metadata.json"
                dataset_path = f"Resources/{self.artifact_collection}/{filename}"
                local_path = os.path.join(tmpdir, filename)
                
                # Download from dataset
                self.dataset_api.download(dataset_path, local_path)
                
                # Read file
                with open(local_path, "r") as f:
                    metadata_dict = json.load(f)
                
                logger.debug(f"Retrieved metadata from Hopsworks dataset: {cache_key}")
                return CacheMetadata.from_dict(metadata_dict)

        except Exception as e:
            logger.debug(f"Failed to retrieve metadata {cache_key}: {e}")
            return None

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata in Hopsworks dataset as JSON."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks, cannot store metadata")
            return

        try:
            import tempfile
            import os
            
            # Create temporary directory and write file
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{metadata.cache_key}_metadata.json"
                local_path = os.path.join(tmpdir, filename)
                
                # Write metadata to file
                with open(local_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                # Upload to Hopsworks dataset
                upload_path = f"Resources/{self.artifact_collection}"
                self.dataset_api.upload(local_path, upload_path, overwrite=True)
                
                logger.info(f"Stored metadata in Hopsworks dataset: {metadata.cache_key}")

        except Exception as e:
            logger.error(f"Failed to store metadata in Hopsworks: {e}")

    def exists(self, cache_key: str) -> bool:
        """Check if image exists in Hopsworks dataset."""
        if not self._is_connected():
            return False

        try:
            # Try to list files in the dataset directory
            dataset_path = f"Resources/{self.artifact_collection}"
            filename = f"{cache_key}.png"
            
            # List directory contents
            return self.dataset_api.exists(f"{dataset_path}/{filename}")

        except Exception as e:
            logger.debug(f"Failed to check file existence {cache_key}: {e}")
            return False
