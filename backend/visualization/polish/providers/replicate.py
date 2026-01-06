"""
Replicate AI image-to-image polishing provider.

Uses the official Replicate Python SDK to enhance image style while
preserving layout. Supports async prediction with automatic polling.
"""

import logging
from typing import Optional

import replicate

from backend.visualization.polish.base import ImagePoller
from backend.visualization.polish.utils import prepare_image

logger = logging.getLogger(__name__)


class ReplicateAIPoller(ImagePoller):
    """
    Interfaces with Replicate API for image-to-image polishing.

    Takes layout PNG and applies subtle style improvements using
    Replicate's hosted models (e.g., Stability Diffusion XL).
    """

    def __init__(
        self,
        api_token: str,
        model_id: str = "black-forest-labs/flux-2-pro",
        guidance_scale: float = 12.0,
        style_preset: str = "comic-book",
        num_outputs: int = 1,
        num_inference_steps: int = 30,
        timeout: int = 300,
    ):
        """
        Initialize Replicate API client.

        Args:
            api_token: Replicate API token (automatically used by SDK)
            model_id: Model/version ID to use
            guidance_scale: Prompt adherence (7-15, higher = follow prompt more strictly)
            style_preset: Style preset (e.g., 'comic-book', 'digital-art')
            num_outputs: Number of output images
            num_inference_steps: Number of inference steps
            timeout: Request timeout in seconds
        """
        if not api_token:
            raise ValueError("Replicate api_token is required")

        self.api_token = api_token
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self.style_preset = style_preset
        self.num_outputs = num_outputs
        self.num_inference_steps = num_inference_steps
        self.timeout = timeout
        
        # Set API token for Replicate SDK
        import os
        os.environ["REPLICATE_API_TOKEN"] = api_token

    def polish(
        self,
        image_data: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Apply Replicate AI polish to an image.

        Uses Img2Img with low denoising strength to enhance style
        while preserving the layout. Includes polling for async results.

        Args:
            image_data: PIL Image bytes (PNG or JPEG)
            prompt: Complete positive prompt (including atmosphere if needed)
            negative_prompt: Complete negative prompt (including atmosphere if needed)

        Returns:
            bytes: Polished image PNG data, or input image if polish failed
        """
        if not self.api_token:
            logger.warning("Replicate API token not set, skipping polish")
            return image_data

        try:
            # Attempt to polish with async polling
            polished_data = self._call_replicate_api(
                image_data=image_data,
                prompt=prompt,
            )
            
            if polished_data:
                logger.info("Successfully polished image with Replicate AI")
                return polished_data
            else:
                logger.warning("Replicate API returned no data, using unpolished image")
                return image_data
                
        except Exception as e:
            logger.error(f"Replicate AI polish failed: {e}")
            logger.info("Falling back to unpolished image")
            return image_data

    def _call_replicate_api(
        self,
        image_data: bytes,
        prompt: str = None,
    ) -> Optional[bytes]:
        """
        Call Replicate API using the official SDK.
        
        The SDK handles async polling automatically, so this is much simpler
        than manual HTTP requests.
        
        Args:
            image_data: PIL Image bytes (PNG or JPEG)
            prompt: Complete positive prompt for style (includes atmosphere)
            negative_prompt: Complete negative prompt (includes atmosphere)
            
        Returns:
            bytes: Polished image PNG data or None if API returns error
        """
        # Prepare image using shared utility (resize + convert to JPEG)
        image_bytes = prepare_image(
            image_data,
            target_width=1344,
            target_height=768,
            output_format="JPEG",
            quality=95,
            convert_to_rgb=True,
        )

        logger.info(
            f"Calling Replicate API model {self.model_id}"
        )
        logger.debug(f"Prompt: {prompt}")

        try:
            # Call Replicate API using the official SDK
            # The SDK automatically handles polling for async predictions
            output = replicate.run(
                self.model_id,
                input={
                    "prompt": prompt,
                    "input_images": [image_bytes],
                    "aspect_ratio": "match_input_image",
                    "output_format": "jpg",
                    "output_quality": 80,
                    "resolution": "1 MP",
                    "safety_tolerance": 5
                },
                timeout=self.timeout,
            )
            
            # To access the file URL:
            logger.info(f"Generated Image URL: {output.url}")

            return output.read()
                
        except Exception as e:
            logger.error(f"Replicate API error: {e}")
            return None
