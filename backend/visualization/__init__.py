"""
The Daily Collage - Image Visualization Module

Handles image generation and composition for news visualizations.
"""

from .composition import (
    VisualizationService,
    TemplateComposer,
    VisualizationCache,
    SignalIntensity,
)

__all__ = [
    "VisualizationService",
    "TemplateComposer",
    "VisualizationCache",
    "SignalIntensity",
]
