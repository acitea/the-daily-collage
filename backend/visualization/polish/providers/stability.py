"""
Stability AI image-to-image polishing provider.

Interfaces with Stability AI API to enhance image style while preserving layout.
"""

import logging
import io
import base64
from typing import Optional

import requests
from PIL import Image

from backend.visualization.polish.base import ImagePoller
from backend.visualization.polish.utils import prepare_image

logger = logging.getLogger(__name__)


class StabilityAIPoller(ImagePoller):
    """
    Interfaces with Stability AI Img2Img API.

    Takes layout PNG and applies subtle style improvements.
    """

    def __init__(
        self,
        api_key: str,
        api_host: str = "https://api.stability.ai",
        engine_id: str = "stable-diffusion-xl-1024-v1-0",
        image_strength: float = 0.35,
        cfg_scale: float = 12.0,
        style_preset: str = "comic-book",
        sampler: str = "K_DPMPP_2M",
        timeout: int = 60,
    ):
        """
        Initialize Stability AI client.

        Args:
            api_key: Stability AI API key
            api_host: API endpoint host
            engine_id: Model/engine ID to use
            image_strength: Denoising strength (0-1, lower = preserve more)
            cfg_scale: Prompt adherence (7-15, higher = follow prompt more strictly)
            style_preset: Style preset (e.g., 'comic-book', 'digital-art')
            sampler: Sampler algorithm (e.g., 'K_DPMPP_2M', 'K_EULER')
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_host = api_host
        self.engine_id = engine_id
        self.image_strength = image_strength
        self.cfg_scale = cfg_scale
        self.style_preset = style_preset
        self.sampler = sampler
        self.timeout = timeout

    def polish(
        self,
        image_data: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Apply Stability AI polish to an image.

        Uses Img2Img with low denoising strength to enhance style
        while preserving the layout.

        Args:
            image_data: PIL Image bytes (PNG)
            prompt: Complete positive prompt (including atmosphere if needed)
            negative_prompt: Complete negative prompt (including atmosphere if needed)

        Returns:
            bytes: Polished image PNG data, or input image if polish failed
        """
        if not self.api_key:
            logger.warning("Stability AI API key not set, skipping polish")
            return image_data

        try:
            # Attempt to polish
            polished_data = self._call_stability_api(
                image_data=image_data,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
            
            if polished_data:
                logger.info("Successfully polished image with Stability AI")
                return polished_data
            else:
                logger.warning("Stability AI returned no data, using unpolished image")
                return image_data
                
        except Exception as e:
            logger.error(f"Stability AI polish failed: {e}")
            logger.info("Falling back to unpolished image")
            return image_data

    def _call_stability_api(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Internal method to call Stability AI API.
        
        Args:
            image_data: PIL Image bytes (PNG)
            prompt: Complete positive prompt for style (includes atmosphere)
            negative_prompt: Complete negative prompt (includes atmosphere)
            
        Returns:
            bytes: Polished image PNG data or None if API returns error
        """
        # Prepare image using shared utility (resize + convert to JPEG)
        image_data = prepare_image(
            image_data,
            target_width=1344,
            target_height=768,
            output_format="JPEG",
            quality=95,
            convert_to_rgb=True,
        )
        
        # Use provided prompts or defaults
        if not prompt:
            prompt = (
                "a colorful sticker scrapbook collage, playful cartoon stickers, "
                "vibrant colors, whimsical illustration style, "
                "artistic arrangement, scrapbook aesthetic"
            )

        if not negative_prompt:
            negative_prompt = (
                "blurry, low quality, distorted, photorealistic, realistic, "
                "moved objects, 3d render, photograph"
            )

        # Prepare request
        url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "image_strength": self.image_strength,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1.0,
            "text_prompts[1][text]": negative_prompt,
            "text_prompts[1][weight]": -1.0,
            "style_preset": self.style_preset,
            "sampler": self.sampler,
            "steps": 30,
            "cfg_scale": self.cfg_scale,
            "samples": 1,
        }

        logger.info(
            f"Calling Stability AI with image_strength={self.image_strength}"
        )
        logger.debug(f"Prompt: {prompt}")

        # Make request
        response = requests.post(
            url,
            headers=headers,
            files={
                "init_image": image_data
            },
            data=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            logger.error(
                f"Stability AI error {response.status_code}: {response.text}"
            )
            return None

        result = response.json()

        # Extract image from response
        if "artifacts" in result and len(result["artifacts"]) > 0:
            image_b64_result = result["artifacts"][0]["base64"]
            polished_data = base64.b64decode(image_b64_result)
            return polished_data
        else:
            logger.error("No image artifacts in Stability AI response")
            return None
