"""
Configuration settings for The Daily Collage backend.

Manages Stability AI credentials, storage buckets, cache TTLs, and other
configuration via environment variables with sensible defaults.
"""

import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class VibeHashSettings:
    """Settings for vibe-hash cache key generation."""

    # Cache key discretization: scores are binned to nearest multiple of this
    discretization_step: float = 0.1

    # Cache TTL in seconds (86400 = 24 hours)
    cache_ttl_seconds: int = 86400

    # Enable caching globally
    enable_cache: bool = True


@dataclass
class StabilityAISettings:
    """Settings for Stability AI Img2Img polishing."""

    # Stability AI API key
    api_key: str = os.getenv("STABILITY_API_KEY", "")

    # Stability AI API host
    api_host: str = os.getenv(
        "STABILITY_API_HOST", "https://api.stability.ai"
    )

    # Img2Img denoising strength (0-1, lower = more preserve layout)
    image_strength: float = 0.35

    # Model to use for polishing
    model_id: str = os.getenv(
        "STABILITY_MODEL_ID", "stable-diffusion-v1-6-768-768"
    )

    # Enable polishing (can be disabled for testing)
    enable_polish: bool = os.getenv(
        "STABILITY_ENABLE_POLISH", "true"
    ).lower() == "true"

    # Timeout for Stability API requests (seconds)
    timeout_seconds: int = 60


@dataclass
class StorageSettings:
    """
    Settings for persistent storage (S3/MinIO/Hopsworks and metadata DB).
    
    Supported backends:
    - 'local': Local file system storage
    - 's3': AWS S3 or S3-compatible storage (currently uses mock)
    - 'minio': MinIO storage (currently uses mock)
    - 'hopsworks': Hopsworks artifact registry
    """

    # Storage backend type: 'local', 's3', 'minio', or 'hopsworks'
    backend: str = os.getenv("STORAGE_BACKEND", "local")

    # S3/MinIO bucket for cached images
    bucket_name: str = os.getenv("STORAGE_BUCKET_NAME", "vibe-images")

    # S3 endpoint (for MinIO or S3-compatible services)
    s3_endpoint: Optional[str] = os.getenv("S3_ENDPOINT", None)

    # S3 access key
    s3_access_key: Optional[str] = os.getenv("S3_ACCESS_KEY", None)

    # S3 secret key
    s3_secret_key: Optional[str] = os.getenv("S3_SECRET_KEY", None)

    # AWS region
    s3_region: str = os.getenv("S3_REGION", "us-east-1")

    # Local storage directory (for local backend)
    local_storage_dir: str = os.getenv(
        "LOCAL_STORAGE_DIR", "./storage/vibes"
    )

    # Metadata DB connection string (PostgreSQL, SQLite, etc.)
    # Format: postgresql://user:password@host/dbname or sqlite:///path/to/db.sqlite
    metadata_db_url: str = os.getenv(
        "METADATA_DB_URL", "sqlite:///storage/metadata.db"
    )


@dataclass
class HopsworksSettings:
    """Settings for Hopsworks feature store and artifact store."""

    # Enable Hopsworks integration
    enabled: bool = os.getenv("HOPSWORKS_ENABLED", "false").lower() == "true"

    # Hopsworks API key
    api_key: Optional[str] = os.getenv("HOPSWORKS_API_KEY", None)

    # Hopsworks project name
    project_name: str = os.getenv("HOPSWORKS_PROJECT_NAME", "daily_collage")

    # Hopsworks host (e.g., c.app.hopsworks.ai)
    host: Optional[str] = os.getenv("HOPSWORKS_HOST", None)

    # Region (for managed Hopsworks)
    region: str = os.getenv("HOPSWORKS_REGION", "us")

    # Feature group name for vibe vectors
    vibe_feature_group: str = os.getenv(
        "HOPSWORKS_VIBE_FG", "vibe_vectors"
    )

    # Artifact collection name for cached images
    artifact_collection: str = os.getenv(
        "HOPSWORKS_ARTIFACT_COLLECTION", "vibe_images"
    )


@dataclass
class AssetSettings:
    """Settings for asset library (PNG stickers)."""

    # Directory containing asset PNGs
    assets_dir: str = os.getenv(
        "ASSETS_DIR", "./backend/assets"
    )

    # Enable fallback to generic category icons if tag-specific not found
    use_fallback_icons: bool = True

    # Default fallback icon if category not found
    default_icon: str = "generic.png"


@dataclass
class LayoutSettings:
    """Settings for layout composition."""

    # Image dimensions
    image_width: int = 1024
    image_height: int = 768

    # Canvas zones (as fractions of height)
    sky_zone_height: float = 0.25
    city_zone_height: float = 0.50
    street_zone_height: float = 0.25

    # Padding and spacing
    padding: int = 20
    element_spacing: int = 30

    # Scaling
    max_element_size: int = 150
    min_element_size: int = 30

    # Intensity thresholds for size mapping
    # intensity_to_size[0] = min, intensity_to_size[1] = max
    intensity_min: float = 0.0
    intensity_max: float = 1.0


@dataclass
class APISettings:
    """Settings for FastAPI server."""

    # Server host and port
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))

    # Enable CORS
    enable_cors: bool = os.getenv(
        "API_ENABLE_CORS", "true"
    ).lower() == "true"

    # Allowed CORS origins
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
    ])

    # API title and description
    title: str = "The Daily Collage API"
    description: str = "REST API for generating location-based news visualizations"
    version: str = "0.2.0"


class Settings:
    """
    Global settings container combining all configuration sections.

    Usage:
        from backend.settings import settings
        print(settings.stability_ai.api_key)
        print(settings.storage.backend)
    """

    def __init__(self):
        self.vibe_hash = VibeHashSettings()
        self.stability_ai = StabilityAISettings()
        self.storage = StorageSettings()
        self.hopsworks = HopsworksSettings()
        self.assets = AssetSettings()
        self.layout = LayoutSettings()
        self.api = APISettings()

    def validate(self) -> list[str]:
        """
        Validates configuration and returns list of warnings/errors.

        Returns:
            list[str]: List of configuration issues (empty if all valid)
        """
        issues = []

        if self.stability_ai.enable_polish and not self.stability_ai.api_key:
            issues.append(
                "WARNING: Stability AI polishing enabled but API key not set"
            )

        if self.storage.backend == "s3" and (
            not self.storage.s3_access_key
            or not self.storage.s3_secret_key
        ):
            issues.append(
                "ERROR: S3 storage selected but AWS credentials not configured"
            )

        if self.hopsworks.enabled and (
            not self.hopsworks.api_key or not self.hopsworks.host
        ):
            issues.append(
                "WARNING: Hopsworks enabled but API key or host not configured"
            )

        if self.storage.backend == "local":
            # Ensure local storage directory exists
            os.makedirs(self.storage.local_storage_dir, exist_ok=True)

        if not os.path.exists(self.assets.assets_dir):
            issues.append(
                f"WARNING: Assets directory not found: {self.assets.assets_dir}"
            )

        # Validate atmosphere strategy
        valid_strategies = [s.value for s in AtmosphereStrategy]
        if self.stability_ai.atmosphere_strategy not in valid_strategies:
            issues.append(
                f"ERROR: Invalid atmosphere_strategy '{self.stability_ai.atmosphere_strategy}'. Must be one of {valid_strategies}"
            )

        return issues


# Global settings instance
settings = Settings()
