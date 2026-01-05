"""
Image polishing providers.

Each provider implements the ImagePoller interface and can be registered
with the factory function for dynamic provider selection.
"""

from backend.visualization.polish.providers.stability import StabilityAIPoller
from backend.visualization.polish.providers.replicate import ReplicateAIPoller
from backend.visualization.polish.providers.mock import MockImagePoller

__all__ = ["StabilityAIPoller", "ReplicateAIPoller", "MockImagePoller"]
