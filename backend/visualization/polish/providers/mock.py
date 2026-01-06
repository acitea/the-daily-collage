"""
Mock polishing provider for testing and development.

Returns the input image unchanged, simulating successful polish.
"""

import logging
from typing import Optional

from backend.visualization.polish.base import ImagePoller

logger = logging.getLogger(__name__)


class MockImagePoller(ImagePoller):
    """
    Mock image poller for testing/development.

    Returns the input image unchanged, simulating successful polish.
    """

    def __init__(self, image_strength: float = 0.35, **kwargs):
        """
        Initialize mock poller.

        Args:
            image_strength: Denoising strength (tracked for testing)
            **kwargs: Ignored arguments (accepts any args for compatibility)
        """
        self.image_strength = image_strength

    def polish(
        self,
        image_data: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Mock polish: return input unchanged.

        Args:
            image_data: Input image bytes
            prompt: Logged but ignored
            negative_prompt: Logged but ignored

        Returns:
            bytes: Same image data
        """
        logger.info(
            f"Mock Polish: returning input unchanged (image_strength={self.image_strength})"
        )
        if prompt:
            logger.debug(f"Mock Polish prompt: {prompt}")
        
        if negative_prompt:
            logger.debug(f"Mock Polish negative prompt: {negative_prompt}")
        return image_data
