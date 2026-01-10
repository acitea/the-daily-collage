"""
Local file system storage 
"""

import json
import logging
from pathlib import Path
from typing import Optional

from storage.core import CacheMetadata, StorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Local file system storage 

    Stores images in {storage_dir}/images/ and metadata in {storage_dir}/metadata.json
    """

    def __init__(self, storage_dir: str = "./storage/vibes"):
        self.storage_dir = Path(storage_dir)
        self.images_dir = self.storage_dir / "images"
        self.metadata_file = self.storage_dir / "metadata.json"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata store
        self._load_metadata_store()

    def _load_metadata_store(self) -> None:
        """Load metadata store from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata_store = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata store: {e}")
                self.metadata_store = {}
        else:
            self.metadata_store = {}

    def _save_metadata_store(self) -> None:
        """Save metadata store to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata_store, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata store: {e}")

    def get_image(self, cache_key: str) -> Optional[bytes]:
        """Retrieve image from local storage."""
        image_path = self.images_dir / f"{cache_key}.png"
        if image_path.exists():
            try:
                with open(image_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read image {cache_key}: {e}")
        return None

    def put_image(self, cache_key: str, image_data: bytes) -> str:
        """Store image and return URL."""
        image_path = self.images_dir / f"{cache_key}.png"
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Stored image: {image_path}")
            return f"/api/cache/images/{cache_key}.png"
        except Exception as e:
            logger.error(f"Failed to write image {cache_key}: {e}")
            return ""

    def get_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata."""
        if cache_key in self.metadata_store:
            try:
                return CacheMetadata.from_dict(self.metadata_store[cache_key])
            except Exception as e:
                logger.error(f"Failed to parse metadata {cache_key}: {e}")
        return None

    def put_metadata(self, cache_key: str, metadata: CacheMetadata) -> None:
        """Store metadata."""
        self.metadata_store[metadata.cache_key] = metadata.to_dict()
        self._save_metadata_store()
        logger.info(f"Stored metadata: {metadata.cache_key}")

    def exists(self, cache_key: str) -> bool:
        """Check if cached."""
        return (self.images_dir / f"{cache_key}.png").exists()
