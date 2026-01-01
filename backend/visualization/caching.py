"""
Vibe-hash based caching system with persistent storage.

Implements deterministic cache keys based on city, date, and discretized scores.
Stores cached images and metadata in S3/MinIO/local storage/Hopsworks with DB lookups.
"""

import logging
import hashlib
import json
import io
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import asdict
from pathlib import Path

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
        vibe_vector: Dict[str, float],
    ) -> str:
        """
        Generate deterministic vibe hash.

        Args:
            city: Geographic location (e.g., 'stockholm', 'gothenburg')
            timestamp: Time to hash (used to determine time window)
            vibe_vector: Dict mapping signal categories to scores (-1.0 to 1.0)
                        e.g., {'traffic': 0.45, 'weather': -0.3, ...}

        Returns:
            str: Vibe hash (e.g., 'stockholm_2025-12-11_00-06_a3f4e2c1...')
        """
        # Normalize city name
        city_normalized = city.lower().replace(" ", "_")

        # Get date and time window
        date_str = timestamp.strftime("%Y-%m-%d")
        window_index = timestamp.hour // cls.WINDOW_DURATION_HOURS
        window_str = f"{window_index * cls.WINDOW_DURATION_HOURS:02d}-{(window_index + 1) * cls.WINDOW_DURATION_HOURS:02d}"

        # Discretize and sort vibe vector for deterministic hashing
        discretized = {}
        for category, score in sorted(vibe_vector.items()):
            # Discretize to nearest step
            discretized_score = round(score / cls.DISCRETIZATION_STEP) * cls.DISCRETIZATION_STEP
            discretized[category] = round(discretized_score, 1)  # Round to 1 decimal

        # Create deterministic string representation
        vibe_str = "|".join(
            f"{cat}:{score}" for cat, score in sorted(discretized.items())
        )

        # Hash the vibe vector
        vibe_hash = hashlib.sha256(vibe_str.encode()).hexdigest()[:8]

        # Combine parts
        full_hash = f"{city_normalized}_{date_str}_{window_str}_{vibe_hash}"

        logger.debug(f"Generated vibe hash: {full_hash}")
        return full_hash

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
                "city": city.replace("_", " "),
                "date": date_obj.date(),
                "window": window_str,
                "hash": vibe_hash,
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


class StorageBackend:
    """Abstract base class for storage backends."""

    def get_image(self, vibe_hash: str) -> Optional[bytes]:
        """Retrieve image bytes for a vibe hash."""
        raise NotImplementedError

    def put_image(self, vibe_hash: str, image_data: bytes) -> str:
        """Store image and return URL."""
        raise NotImplementedError

    def get_metadata(self, vibe_hash: str) -> Optional[CacheMetadata]:
        """Retrieve metadata for a vibe hash."""
        raise NotImplementedError

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata."""
        raise NotImplementedError

    def exists(self, vibe_hash: str) -> bool:
        """Check if vibe hash is cached."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """
    Local file system storage backend.

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

    def get_image(self, vibe_hash: str) -> Optional[bytes]:
        """Retrieve image from local storage."""
        image_path = self.images_dir / f"{vibe_hash}.png"
        if image_path.exists():
            try:
                with open(image_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read image {vibe_hash}: {e}")
        return None

    def put_image(self, vibe_hash: str, image_data: bytes) -> str:
        """Store image and return URL."""
        image_path = self.images_dir / f"{vibe_hash}.png"
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Stored image: {image_path}")
            return f"/api/cache/images/{vibe_hash}.png"
        except Exception as e:
            logger.error(f"Failed to write image {vibe_hash}: {e}")
            return ""

    def get_metadata(self, vibe_hash: str) -> Optional[CacheMetadata]:
        """Retrieve metadata."""
        if vibe_hash in self.metadata_store:
            try:
                return CacheMetadata.from_dict(self.metadata_store[vibe_hash])
            except Exception as e:
                logger.error(f"Failed to deserialize metadata {vibe_hash}: {e}")
        return None

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata."""
        self.metadata_store[metadata.vibe_hash] = metadata.to_dict()
        self._save_metadata_store()
        logger.info(f"Stored metadata: {metadata.vibe_hash}")

    def exists(self, vibe_hash: str) -> bool:
        """Check if cached."""
        return (self.images_dir / f"{vibe_hash}.png").exists()


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


class HopsworksStorageBackend(StorageBackend):
    """
    Hopsworks artifact store backend.

    Stores images as artifacts and metadata as feature group data.
    Requires: hopsworks package and valid Hopsworks credentials.
    """

    def __init__(
        self,
        api_key: str,
        project_name: str = "daily_collage",
        host: Optional[str] = None,
        region: str = "us",
        artifact_collection: str = "vibe_images",
    ):
        """
        Initialize Hopsworks backend.

        Args:
            api_key: Hopsworks API key
            project_name: Hopsworks project name
            host: Hopsworks host (e.g., c.app.hopsworks.ai)
            region: Hopsworks region
            artifact_collection: Name of artifact collection
        """
        self.api_key = api_key
        self.project_name = project_name
        self.host = host
        self.region = region
        self.artifact_collection = artifact_collection
        self.project = None
        self.artifacts = None
        self.metadata_store = {}

        try:
            import hopsworks
        except ImportError:
            logger.error(
                "hopsworks package not installed. Install with: pip install hopsworks"
            )
            return

        try:
            # Login to Hopsworks
            if host:
                self.project = hopsworks.login(
                    host=host,
                    api_key_value=api_key,
                    project=project_name,
                )
            else:
                self.project = hopsworks.login(
                    api_key_value=api_key,
                    project=project_name,
                )

            # Get artifact collection
            self.artifacts = self.project.get_artifacts()
            logger.info(f"Connected to Hopsworks project: {project_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            self.project = None

    def _is_connected(self) -> bool:
        """Check if connected to Hopsworks."""
        return self.project is not None

    def get_image(self, vibe_hash: str) -> Optional[bytes]:
        """Retrieve image from Hopsworks artifacts."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks")
            return None

        try:
            artifact = self.artifacts.get_artifact(
                name=f"{vibe_hash}.png",
                collection=self.artifact_collection,
            )
            if artifact:
                return artifact.content
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve artifact {vibe_hash}: {e}")
            return None

    def put_image(self, vibe_hash: str, image_data: bytes) -> str:
        """Store image in Hopsworks artifacts."""
        if not self._is_connected():
            logger.warning("Not connected to Hopsworks, cannot store image")
            return ""

        try:
            import io

            # Create in-memory file
            image_file = io.BytesIO(image_data)
            image_file.name = f"{vibe_hash}.png"

            # Upload to Hopsworks
            self.artifacts.save_artifact(
                artifact_path=image_file,
                name=f"{vibe_hash}.png",
                collection=self.artifact_collection,
                description=f"Vibe visualization for {vibe_hash}",
            )

            logger.info(f"Stored image in Hopsworks: {vibe_hash}")
            return f"hopsworks://{self.artifact_collection}/{vibe_hash}.png"

        except Exception as e:
            logger.error(f"Failed to store image in Hopsworks: {e}")
            return ""

    def get_metadata(self, vibe_hash: str) -> Optional[CacheMetadata]:
        """Retrieve metadata (currently from in-memory store)."""
        if vibe_hash in self.metadata_store:
            try:
                return CacheMetadata.from_dict(self.metadata_store[vibe_hash])
            except Exception as e:
                logger.error(f"Failed to deserialize metadata {vibe_hash}: {e}")
        return None

    def put_metadata(self, metadata: CacheMetadata) -> None:
        """Store metadata (currently in in-memory store)."""
        self.metadata_store[metadata.vibe_hash] = metadata.to_dict()
        logger.info(f"Stored metadata in Hopsworks backend: {metadata.vibe_hash}")

    def exists(self, vibe_hash: str) -> bool:
        """Check if image exists in artifacts."""
        if not self._is_connected():
            return False

        try:
            artifact = self.artifacts.get_artifact(
                name=f"{vibe_hash}.png",
                collection=self.artifact_collection,
            )
            return artifact is not None
        except Exception:
            return False


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
        vibe_vector: Dict[str, float],
    ) -> Tuple[Optional[bytes], Optional[CacheMetadata]]:
        """
        Retrieve cached image and metadata.

        Args:
            city: Geographic location
            timestamp: Time window for vibe
            vibe_vector: Signal scores

        Returns:
            Tuple of (image_bytes, metadata) or (None, None) if not cached
        """
        vibe_hash = VibeHash.generate(city, timestamp, vibe_vector)

        image_data = self.storage.get_image(vibe_hash)
        metadata = self.storage.get_metadata(vibe_hash)

        if image_data and metadata:
            logger.info(f"Cache hit: {vibe_hash}")
            return image_data, metadata
        elif image_data or metadata:
            logger.warning(f"Partial cache hit for {vibe_hash}")
            return None, None
        else:
            logger.info(f"Cache miss: {vibe_hash}")
            return None, None

    def set(
        self,
        city: str,
        timestamp: datetime,
        vibe_vector: Dict[str, float],
        image_data: bytes,
        hitboxes: List[Dict],
        source_articles: List[Dict] = None,
    ) -> Tuple[str, CacheMetadata]:
        """
        Cache image and metadata.

        Args:
            city: Geographic location
            timestamp: Time window
            vibe_vector: Signal scores
            image_data: PNG image bytes
            hitboxes: List of hitbox dicts
            source_articles: Optional articles that contributed

        Returns:
            Tuple of (image_url, metadata)
        """
        vibe_hash = VibeHash.generate(city, timestamp, vibe_vector)

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
        vibe_vector: Dict[str, float],
    ) -> bool:
        """Check if vibe is cached."""
        vibe_hash = VibeHash.generate(city, timestamp, vibe_vector)
        return self.storage.exists(vibe_hash)
