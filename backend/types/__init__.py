"""
The Daily Collage - Common Type Definitions

Centralized type definitions used throughout the backend.
"""

from .signal import (
    SignalCategory,
    SignalTag,
    IntensityLevel,
    Signal,
    Hitbox,
)
from .api import (
    # HitboxData removed - use Hitbox directly
    VibeVectorRequest,
    VisualizationResponse,
    CacheStatusResponse,
    BackfillRequest,
    BackfillStatusResponse,
    SignalCategoriesResponse,
    HealthCheckResponse,
)

__all__ = [
    # Signal types
    "SignalCategory",
    "SignalTag",
    "IntensityLevel",
    "Signal",
    "Hitbox",
    # API types
    # "HitboxData",  # Removed - use Hitbox directly
    "VibeVectorRequest",
    "VisualizationResponse",
    "CacheStatusResponse",
    "BackfillRequest",
    "BackfillStatusResponse",
    "SignalCategoriesResponse",
    "HealthCheckResponse",
]
