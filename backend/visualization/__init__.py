"""
The Daily Collage - Image Visualization Module

Handles image generation and composition for news visualizations.
"""

from .composition import (
    VisualizationService,
    HybridComposer,
    SignalIntensity,
)
from .caching import (
    VibeHash,
    VibeCache,
    CacheMetadata,
    StorageBackend,
    LocalStorageBackend,
    MockS3StorageBackend,
    HopsworksStorageBackend,
)
from .assets import (
    AssetLibrary,
    ZoneLayoutComposer,
    Hitbox,
)
from .polish import (
    StabilityAIPoller,
    MockStabilityAIPoller,
    create_poller,
)
from .atmosphere import (
    AtmosphereStrategy,
    AtmosphereDescriptor,
)

__all__ = [
    "VisualizationService",
    "HybridComposer",
    "VisualizationCache",
    "SignalIntensity",
    "VibeHash",
    "VibeCache",
    "CacheMetadata",
    "StorageBackend",
    "LocalStorageBackend",
    "MockS3StorageBackend",
    "AssetLibrary",
    "ZoneLayoutComposer",
    "Hitbox",
    "StabilityAIPoller",
    "MockStabilityAIPoller",
    "create_poller",
]
