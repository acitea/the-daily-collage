"""
Hopsworks artifact storage backend.

Delegates to HopsworksService for all Hopsworks operations.
Stores images and metadata as artifacts in the same collection.
"""

import io
import json
import logging
from typing import Optional

from backend.storage.core import CacheMetadata, StorageBackend

logger = logging.getLogger(__name__)


class HopsworksStorageBackend(StorageBackend):
    """
    Hopsworks artifact store backend.

    Stores images and metadata as artifacts in Hopsworks.
    Delegates all Hopsworks operations to HopsworksService.
    """

    def __init__(self, hopsworks_service, artifact_collection: str = "vibe_images"):
        """
        Initialize Hopsworks storage backend.

        Args:
            hopsworks_service: Instance of HopsworksService for artifact operations
            artifact_collection: Name of artifact collection to store in
        """
        self.hopsworks_service = hopsworks_service
        self.artifact_collection = artifact_collection

        # Ensure connection
        if not self.hopsworks_service._project:
            self.hopsworks_service.connect()

        # Get artifact registry
        try:
            import hopsworks
            self.artifacts = self.hopsworks_service._project.get_artifacts_api()
            logger.info(f"Initialized Hopsworks storage with collection: {artifact_collection}")
        except Exception as e:
            logger.error(f"Failed to get artifacts API: {e}")
            self.artifacts = None

    def _is_connected(self) -> bool:
        """Check if connected to Hopsworks."""
        return self.artifacts is not None

    def get_image(self, cache_key: str) -> Optional[bytes]:
        """Retrieve image from Hopsworks artifact registry."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks")
            return None

        try:
            # Download artifact
            artifact_name = f"{cache_key}.png"
            artifact_path = self.artifacts.download(
                name=artifact_name,
                collection=self.artifact_collection,
            )

            # Read downloaded file
            if artifact_path:
                with open(artifact_path, "rb") as f:
                    image_data = f.read()
                logger.debug(f"Retrieved image from Hopsworks: {cache_key}")
                return image_data

            return None

        except Exception as e:
            logger.debug(f"Failed to retrieve artifact {cache_key}: {e}")
            return None

    def put_image(self, cache_key: str, image_data: bytes) -> str:
        """Store image in Hopsworks artifact registry."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks, cannot store image")
            return ""

        try:
            # Create in-memory file
            image_file = io.BytesIO(image_data)
            image_file.name = f"{cache_key}.png"

            # Upload to Hopsworks artifact registry
            self.artifacts.upload(
                artifact=image_file,
                name=f"{cache_key}.png",
                collection=self.artifact_collection,
                description=f"Vibe visualization for {cache_key}",
            )

            logger.info(f"Stored image in Hopsworks: {cache_key}")
            return f"hopsworks://{self.artifact_collection}/{cache_key}.png"

        except Exception as e:
            logger.error(f"Failed to store image in Hopsworks: {e}")
            return ""

    def get_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata from Hopsworks artifact registry."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks")
            return None

        try:
            # Download metadata artifact
            metadata_name = f"{cache_key}_metadata.json"
            metadata_path = self.artifacts.download(
                name=metadata_name,
                collection=self.artifact_collection,
            )

            if metadata_path:
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                logger.debug(f"Retrieved metadata from Hopsworks: {cache_key}")
                return CacheMetadata.from_dict(metadata_dict)

            return None

        except Exception as e:
            logger.debug(f"Failed to retrieve metadata {cache_key}: {e}")
            return None

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata in Hopsworks artifact registry as JSON."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks, cannot store metadata")
            return

        try:
            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata.to_dict(), indent=2)
            metadata_file = io.BytesIO(metadata_json.encode())
            metadata_file.name = f"{metadata.cache_key}_metadata.json"

            # Upload to Hopsworks artifact registry
            self.artifacts.upload(
                artifact=metadata_file,
                name=f"{metadata.cache_key}_metadata.json",
                collection=self.artifact_collection,
                description=f"Metadata for vibe visualization {metadata.cache_key}",
            )

            logger.info(f"Stored metadata in Hopsworks: {metadata.cache_key}")

        except Exception as e:
            logger.error(f"Failed to store metadata in Hopsworks: {e}")

    def exists(self, cache_key: str) -> bool:
        """Check if image exists in Hopsworks artifact registry."""
        if not self._is_connected():
            return False

        try:
            # Try to list artifacts with this name
            artifact_name = f"{cache_key}.png"
            artifacts = self.artifacts.list(collection=self.artifact_collection)

            # Check if our artifact exists in the list
            for artifact in artifacts:
                if artifact.name == artifact_name:
                    return True

            return False

        except Exception as e:
            logger.debug(f"Failed to check artifact existence {cache_key}: {e}")
            return False
