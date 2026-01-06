"""
Image polishing module.

Provides factory function to create polishing providers and registry
for extending with custom providers.

Usage:
    from backend.visualization.polish import create_poller

    poller = create_poller(
        provider="stability",
        enable_polish=True,
        api_key="your_key"
    )

    polished_bytes = poller.polish(image_data, prompt, negative_prompt)
"""

import logging
from typing import Optional, Dict, Type

from backend.visualization.polish.base import ImagePoller
from backend.visualization.polish.providers import (
    StabilityAIPoller,
    ReplicateAIPoller,
    MockImagePoller,
)
from backend.visualization.polish.utils import (
    prepare_image,
    image_to_base64,
    get_image_dimensions,
    validate_image,
)

logger = logging.getLogger(__name__)

# Registry of available providers
PROVIDERS: Dict[str, Type[ImagePoller]] = {
    "stability": StabilityAIPoller,
    "replicate": ReplicateAIPoller,
    "mock": MockImagePoller,
}


def register_provider(name: str, provider_class: Type[ImagePoller]) -> None:
    """
    Register a new polishing provider.

    This allows you to add custom providers at runtime without modifying
    the core code. Custom providers must inherit from ImagePoller.

    Args:
        name: Unique name for the provider (e.g., "openai", "huggingface")
        provider_class: Class that implements ImagePoller interface

    Example:
        class CustomPoller(ImagePoller):
            def polish(self, image_data, prompt, negative_prompt):
                # Your custom implementation
                return polished_data

        register_provider("custom", CustomPoller)
    """
    if name in PROVIDERS:
        logger.warning(f"Provider '{name}' already registered, overwriting")
    PROVIDERS[name] = provider_class
    logger.info(f"Registered provider: {name}")


def get_provider_class(name: str) -> Optional[Type[ImagePoller]]:
    """
    Get a registered provider class by name.

    Args:
        name: Provider name (e.g., "stability", "replicate", "custom")

    Returns:
        Provider class or None if not found
    """
    return PROVIDERS.get(name.lower())


def create_poller(
    provider: str = "stability",
    enable_polish: bool = True,
    api_key: Optional[str] = None,
    api_token: Optional[str] = None,
    api_host: str = "https://api.stability.ai",
    engine_id: str = "stable-diffusion-xl-1024-v1-0",
    replicate_model_id: str = "black-forest-labs/flux-2-pro",
    image_strength: float = 0.35,
    cfg_scale: float = 12.0,
    guidance_scale: float = 12.0,
    style_preset: str = "comic-book",
    sampler: str = "K_DPMPP_2M",
    timeout: int = 60,
    **kwargs,
) -> ImagePoller:
    """
    Factory function to create a polishing provider.

    Automatically selects and initializes the correct provider based on the
    provider name. Additional keyword arguments are passed to the provider
    constructor for custom configuration.

    Args:
        provider: Which provider to use ('stability', 'replicate', 'mock', or custom)
        enable_polish: If False, use mock poller regardless
        api_key: Stability AI API key
        api_token: Replicate API token
        api_host: API host for Stability AI
        engine_id: Engine/model ID for Stability AI
        replicate_model_id: Model version ID for Replicate
        image_strength: Denoising strength (0-1, lower = preserve more)
        cfg_scale: Prompt adherence strength for Stability AI
        guidance_scale: Prompt adherence strength for Replicate
        style_preset: Style preset for generation
        sampler: Sampler algorithm (Stability AI only)
        timeout: Request timeout
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Initialized provider instance (ImagePoller subclass)

    Raises:
        ValueError: If provider name not found in registry
    """
    def _mock_with_reason(reason: str) -> ImagePoller:
        logger.warning(f"{reason}, using MockImagePoller")
        return MockImagePoller(image_strength=image_strength)

    if not enable_polish:
        logger.info("Polishing disabled, using MockImagePoller")
        return MockImagePoller(image_strength=image_strength)

    provider_name = (provider or "stability").lower()
    provider_class = get_provider_class(provider_name)

    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        return _mock_with_reason(
            f"Unknown polish provider '{provider_name}' (available: {available})"
        )

    logger.info(f"Using {provider_name.capitalize()} polishing provider")

    try:
        if provider_name == "stability":
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

        if provider_name == "replicate":
            return ReplicateAIPoller(
                api_token=api_token,
                model_id=replicate_model_id,
                guidance_scale=guidance_scale,
                style_preset=style_preset,
                timeout=timeout,
            )

        if provider_name == "mock":
            logger.info("Using mock polishing provider")
            return MockImagePoller(image_strength=image_strength)

        # Custom provider - pass all kwargs through
        return provider_class(
            image_strength=image_strength,
            **kwargs,
        )

    except Exception as exc:
        logger.error(
            f"Failed to initialize '{provider_name}' polishing provider: {exc}"
        )
        return _mock_with_reason("Initialization failed")


__all__ = [
    "create_poller",
    "register_provider",
    "get_provider_class",
    "ImagePoller",
    "PROVIDERS",
    # Image utilities
    "prepare_image",
    "image_to_base64",
    "get_image_dimensions",
    "validate_image",
]
