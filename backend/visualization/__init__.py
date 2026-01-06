"""
The Daily Collage - Image Visualization Module

Handles image generation and composition for news visualizations.
"""

from .composition import (
    VisualizationService,
    HybridComposer,
    SignalIntensity,
)
from _types import (
    Signal,
    SignalCategory,
    SignalTag,
    IntensityLevel,
    Hitbox,
)
from .assets import (
    AssetLibrary,
    ZoneLayoutComposer,
)
from .polish import (
    create_poller,
)
from .atmosphere import (
    AtmosphereDescriptor,
)

__all__ = [
    "VisualizationService",
    "HybridComposer",
    "SignalIntensity",
    "AtmosphereDescriptor",
    "AssetLibrary",
    "ZoneLayoutComposer",
    "Hitbox",
    "Signal",
    "SignalCategory",
    "SignalTag",
    "IntensityLevel",
    "create_poller",
]
