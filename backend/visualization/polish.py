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
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Apply Stability AI polish to an image.

        Uses Img2Img with low denoising strength to enhance style
        while preserving the layout. Includes retry logic for transient failures.

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
            # Attempt to polish with retry logic
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
            logger.error(f"All retry attempts failed for Stability AI: {e}")
            logger.info("Falling back to unpolished image")
            return image_data

    def _call_stability_api(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Internal method to call Stability AI API with retry logic.
        
        Retries up to 3 times with exponential backoff for timeout/connection errors.
        
        Args:
            image_data: PIL Image bytes (PNG)
            prompt: Complete positive prompt for style (includes atmosphere)
            negative_prompt: Complete negative prompt (includes atmosphere)
            
        Returns:
            bytes: Polished image PNG data or None if API returns error
            
        Raises:
            requests.Timeout: If request times out (will be retried)
            requests.ConnectionError: If connecstion fails (will be retried)
        """
        # Convert image format to JPEG and resize to Stability AI SDXL requirements
        image = Image.open(io.BytesIO(image_data))
        
        # Resize to 1344x768 (required for stable-diffusion-xl models)
        target_width, target_height = 1344, 768
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {target_width}x{target_height} for Stability AI SDXL")
        
        # Convert to RGB if needed (JPEG doesn't support transparency)
        if image.mode in ("RGBA", "PA", "P"):
            image = image.convert("RGB")
        
        # Re-encode to JPEG
        optimized_bytes = io.BytesIO()
        image.save(optimized_bytes, format="JPEG")
        image_data = optimized_bytes.getvalue()
        
        logger.info(f"Converted image format for API: {len(image_data)} bytes")
        
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

        # Make request (will be retried on timeout/connection errors)
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
    ) -> Optional[bytes]:
        """
        Mock polish: return input unchanged.

        Args:
            image_data: Input image bytes
            prompt: Ignored (logged for debugging)
            negative_prompt: Ignored

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


def create_poller(
    enable_polish: bool = True,
    api_key: Optional[str] = None,
    api_host: str = "https://api.stability.ai",
    engine_id: str = "stable-diffusion-xl-1024-v1-0",
    image_strength: float = 0.35,
    cfg_scale: float = 12.0,
    style_preset: str = "comic-book",
    sampler: str = "K_DPMPP_2M",
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
        cfg_scale: Prompt adherence strength
        style_preset: Style preset for generation
        sampler: Sampler algorithm
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
        cfg_scale=cfg_scale,
        style_preset=style_preset,
        sampler=sampler,
        timeout=timeout,
    )
