"""
Configuration settings for The Daily Collage backend.

Manages Stability AI credentials, storage buckets, cache TTLs, and other
configuration via environment variables with sensible defaults.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv; load_dotenv()


@dataclass
class VibeHashSettings:
    """Settings for vibe-hash cache key generation."""

    # Enable caching globally
    enable_cache: bool = True


@dataclass
class PolishSettings:
    """Settings for image polishing provider selection."""

    # Which provider to use for polishing: 'stability' or 'replicate'
    provider: str = os.getenv("POLISH_PROVIDER", "stability").lower()

    # Enable polishing globally (can be disabled for testing)
    enable: bool = os.getenv("ENABLE_POLISH", "false").lower() == "true"


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
    # 0.75 allows AI to redraw scenery while preserving rough composition
    image_strength: float = 0.75

    # CFG Scale: How strongly to follow the prompt (7-15, higher = stricter adherence)
    cfg_scale: float = 12.0

    # Style preset for cartoonish vibes
    style_preset: str = "comic-book"

    # Sampler algorithm
    sampler: str = "K_DPMPP_2M"

    # Model to use for polishing
    model_id: str = os.getenv(
        "STABILITY_MODEL_ID", "stable-diffusion-xl-1024-v1-0"
    )

    # Enable polishing (can be disabled for testing)
    enable_polish: bool = os.getenv(
        "STABILITY_ENABLE_POLISH", "false"
    ).lower() == "true"

    # Timeout for Stability API requests (seconds)
    timeout_seconds: int = 60


@dataclass
class ReplicateAISettings:
    """Settings for Replicate API-based image polishing."""

    # Replicate API token
    api_token: str = os.getenv("REPLICATE_API_TOKEN", "")

    # Model version to use (default: Stability SDXL)
    model_id: str = os.getenv(
        "REPLICATE_MODEL_ID",
        "black-forest-labs/flux-2-pro"
    )

    # Number of output images
    num_outputs: int = 1

    # Enable polishing via Replicate (can be disabled for testing)
    enable_polish: bool = os.getenv(
        "REPLICATE_ENABLE_POLISH", "false"
    ).lower() == "true"



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


@dataclass
class HopsworksSettings:
    """Settings for Hopsworks feature store and artifact store."""

    # Hopsworks API key
    api_key: Optional[str] = os.getenv("HOPSWORKS_API_KEY", None)

    # Hopsworks project name
    project_name: str = os.getenv("HOPSWORKS_PROJECT", "daily_collage")

    # Hopsworks host (e.g., c.app.hopsworks.ai)
    host: Optional[str] = os.getenv("HOPSWORKS_HOST", None)

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
        "ASSETS_DIR", "./assets"
    )


@dataclass
class LayoutSettings:
    """Settings for layout composition."""

    # Image dimensions
    image_width: int = 1344
    image_height: int = 768

    # Canvas zones (as fractions of height)
    sky_zone_height: float = 0.50
    city_zone_height: float = 0.20
    street_zone_height: float = 0.30

@dataclass
class APISettings:
    """Settings for FastAPI server."""

    # Server host and port
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


class Settings:
    """
    Global settings container combining all configuration sections.

    Usage:
        from backend.settings import settings
        print(settings.stability_ai.api_key)
        print(settings.storage.backend)
    """

    def __init__(self):
        self.polish = PolishSettings()
        self.stability_ai = StabilityAISettings()
        self.replicate_ai = ReplicateAISettings()
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

        # Validate polish provider
        if self.polish.enable:
            if self.polish.provider == "stability":
                if not self.stability_ai.api_key:
                    issues.append(
                        "WARNING: Stability AI polishing enabled but API key not set"
                    )
            elif self.polish.provider == "replicate":
                if not self.replicate_ai.api_token:
                    issues.append(
                        "WARNING: Replicate AI polishing enabled but API token not set"
                    )
            else:
                issues.append(
                    f"WARNING: Unknown polish provider '{self.polish.provider}'. "
                    "Use 'stability' or 'replicate'"
                )

        if self.storage.backend == "s3" and (
            not self.storage.s3_access_key
            or not self.storage.s3_secret_key
        ):
            issues.append(
                "ERROR: S3 storage selected but AWS credentials not configured"
            )

        elif self.storage.backend == 'hopsworks':
             if (
                not self.hopsworks.api_key
                or not self.hopsworks.host
            ):
                issues.append(
                    "ERROR: Hopsworks storage selected but API key or host not configured"
                )

        if self.storage.backend == "local":
            # Ensure local storage directory exists
            os.makedirs(self.storage.local_storage_dir, exist_ok=True)

        if not os.path.exists(self.assets.assets_dir):
            issues.append(
                f"WARNING: Assets directory not found: {self.assets.assets_dir}"
            )

        return issues


# Global settings instance
settings = Settings()
