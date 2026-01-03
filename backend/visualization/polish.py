"""
Stability AI image-to-image polishing integration.

Takes a layout image and applies subtle AI-based enhancements
while preserving hitbox locations.
"""

import logging
import io
import base64
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class StabilityAIPoller:
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
        timeout: int = 60,
    ):
        """
        Initialize Stability AI client.

        Args:
            api_key: Stability AI API key
            api_host: API endpoint host
            engine_id: Model/engine ID to use
            image_strength: Denoising strength (0-1, lower = preserve more)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_host = api_host
        self.engine_id = engine_id
        self.image_strength = image_strength
        self.timeout = timeout

    def polish(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        atmosphere_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Apply Stability AI polish to an image.

        Uses Img2Img with low denoising strength to enhance style
        while preserving the layout. Includes retry logic for transient failures.

        Args:
            image_data: PIL Image bytes (PNG)
            prompt: Optional positive prompt for style
            negative_prompt: Optional negative prompt
            atmosphere_prompt: Optional description of atmosphere to enhance
                              (e.g., "rainy weather", "celebration mood")

        Returns:
            bytes: Polished image PNG data, or input image if polish failed
        """
        if not self.api_key:
            logger.warning("Stability AI API key not set, skipping polish")
            return image_data

        try:
            # Attempt to polish with retry logic
            polished_data = self._call_stability_api(
                image_data=image_data,
                prompt=prompt,
                negative_prompt=negative_prompt,
                atmosphere_prompt=atmosphere_prompt,
            )
            
            if polished_data:
                logger.info("Successfully polished image with Stability AI")
                return polished_data
            else:
                logger.warning("Stability AI returned no data, using unpolished image")
                return image_data
                
        except Exception as e:
            logger.error(f"All retry attempts failed for Stability AI: {e}")
            logger.info("Falling back to unpolished image")
            return image_data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_stability_api(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        atmosphere_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Internal method to call Stability AI API with retry logic.
        
        Retries up to 3 times with exponential backoff for timeout/connection errors.
        
        Args:
            image_data: PIL Image bytes (PNG)
            prompt: Optional positive prompt for style
            negative_prompt: Optional negative prompt
            atmosphere_prompt: Optional atmosphere description
            
        Returns:
            bytes: Polished image PNG data or None if API returns error
            
        Raises:
            requests.Timeout: If request times out (will be retried)
            requests.ConnectionError: If connection fails (will be retried)
        """
        # Build prompts
        if not prompt:
            prompt = (
                "a colorful sticker scrapbook collage, playful cartoon stickers, "
                "vibrant colors, whimsical illustration style, "
                "artistic arrangement, scrapbook aesthetic"
            )

        # Incorporate atmosphere prompt if provided
        if atmosphere_prompt:
            prompt = f"{prompt}, {atmosphere_prompt}"

        if not negative_prompt:
            negative_prompt = (
                "blurry, low quality, distorted, photorealistic, realistic, "
                "moved objects, 3d render, photograph"
            )

        # Prepare request
        url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image"

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "image_strength": self.image_strength,
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": negative_prompt, "weight": -1.0},
            ],
            "style_preset": "digital-art",
            "steps": 30,
            "cfg_scale": 7.0,
        }

        logger.info(
            f"Calling Stability AI with image_strength={self.image_strength}"
        )
        logger.debug(f"Prompt: {prompt}")

        # Make request (will be retried on timeout/connection errors)
        response = requests.post(
            url,
            headers=headers,
            files={
                "init_image": image_data  # image size 1344x768
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


class MockStabilityAIPoller:
    """
    Mock Stability AI poller for testing/development.

    Returns the input image unchanged, simulating successful polish.
    """

    def __init__(self, **kwargs):
        """Initialize mock poller (accepts any args for compatibility)."""
        self.image_strength = kwargs.get("image_strength", 0.35)

    def polish(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        atmosphere_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Mock polish: return input unchanged.

        Args:
            image_data: Input image bytes
            prompt: Ignored
            negative_prompt: Ignored
            atmosphere_prompt: Ignored

        Returns:
            bytes: Same image data
        """
        logger.info(
            f"Mock Polish: returning input unchanged (image_strength={self.image_strength})"
        )
        if atmosphere_prompt:
            logger.debug(f"Mock Polish (would apply atmosphere): {atmosphere_prompt}")
        return image_data


def create_poller(
    enable_polish: bool = True,
    api_key: Optional[str] = None,
    api_host: str = "https://api.stability.ai",
    engine_id: str = "stable-diffusion-xl-1024-v1-0",
    image_strength: float = 0.35,
    timeout: int = 60,
):
    """
    Factory function to create appropriate poller.

    Args:
        enable_polish: If False, use mock poller
        api_key: Stability AI API key
        api_host: API host
        engine_id: Engine/model ID
        image_strength: Denoising strength
        timeout: Request timeout

    Returns:
        StabilityAIPoller or MockStabilityAIPoller
    """
    if not enable_polish or not api_key:
        logger.info("Using MockStabilityAIPoller")
        return MockStabilityAIPoller(image_strength=image_strength)

    logger.info("Using real StabilityAIPoller")
    return StabilityAIPoller(
        api_key=api_key,
        api_host=api_host,
        engine_id=engine_id,
        image_strength=image_strength,
        timeout=timeout,
    )
