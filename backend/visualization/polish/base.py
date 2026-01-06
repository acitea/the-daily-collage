"""
Abstract base class for image polishing providers.

All polishing providers must inherit from this class and implement
the `polish()` method.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ImagePoller(ABC):
    """
    Abstract base class for image polishing providers.

    Polishing takes a layout image and applies subtle AI-based enhancements
    while preserving hitbox locations via low denoising strength.
    """

    @abstractmethod
    def polish(
        self,
        image_data: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Polish an image with AI enhancement.

        Args:
            image_data: PIL Image bytes (PNG or JPEG)
            prompt: Complete positive prompt for style enhancement
            negative_prompt: Complete negative prompt (what to avoid)

        Returns:
            bytes: Polished image data, or input image if polish failed
        """
        pass
