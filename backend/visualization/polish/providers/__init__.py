"""
Image polishing providers.

Each provider implements the ImagePoller interface and can be registered
with the factory function for dynamic provider selection.
"""

from visualization.polish.providers.stability import StabilityAIPoller
from visualization.polish.providers.replicate import ReplicateAIPoller
from visualization.polish.providers.mock import MockImagePoller

__all__ = ["StabilityAIPoller", "ReplicateAIPoller", "MockImagePoller"]
