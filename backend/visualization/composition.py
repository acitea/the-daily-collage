"""
Image generation and template composition module.

Generates cartoonish visualizations based on detected signals
using hybrid layout + polish approach:
1. Asset-based layout with zone placement and hitbox tracking
2. AI polish (Stability AI or Replicate) with low denoise to preserve layout
"""

import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import io

from backend.settings import settings
from backend.types import Signal, SignalCategory, SignalTag
from backend.visualization.assets import AssetLibrary, ZoneLayoutComposer
from backend.visualization.polish import create_poller
from backend.storage import VibeCache, create_storage_backend
from backend.visualization.atmosphere import AtmosphereDescriptor

logger = logging.getLogger(__name__)


@dataclass
class SignalIntensity:
    """Represents a signal with its intensity level."""

    signal_name: str
    intensity: float  # 0.0-1.0 scale


class HybridComposer:
    """
    Hybrid image composition pipeline.

    Step 1: Asset-based layout with zone placement and hitbox tracking
    Step 2: Stability AI polish to enhance style while preserving layout
    """

    def __init__(self):
        """Initialize composer with layout and polish engines."""
        logger.info("Initializing HybridComposer")

        # Layout engine
        self.layout_composer = ZoneLayoutComposer(
            image_width=settings.layout.image_width,
            image_height=settings.layout.image_height,
            assets_dir=settings.assets.assets_dir,
            bg_color=(245, 245, 250),
            sky_zone_height=settings.layout.sky_zone_height,
            city_zone_height=settings.layout.city_zone_height,
            street_zone_height=settings.layout.street_zone_height,
        )

        # Polish engine - choose provider based on settings
        if settings.polish.provider == "replicate":
            logger.info("Polish provider: Replicate AI")
            self.poller = create_poller(
                provider="replicate",
                enable_polish=settings.polish.enable,
                api_token=settings.replicate_ai.api_token,
                replicate_model_id=settings.replicate_ai.model_id,
            )
        else:
            # Default to Stability AI
            logger.info("Polish provider: Stability AI")
            self.poller = create_poller(
                provider="stability",
                enable_polish=settings.polish.enable,
                api_key=settings.stability_ai.api_key,
                api_host=settings.stability_ai.api_host,
                engine_id=settings.stability_ai.model_id,
                image_strength=settings.stability_ai.image_strength,
                cfg_scale=settings.stability_ai.cfg_scale,
                style_preset=settings.stability_ai.style_preset,
                sampler=settings.stability_ai.sampler,
                timeout=settings.stability_ai.timeout_seconds,
            )

    def compose(
        self,
        vibe_vector: Dict[str, float],
        location: str = "Unknown",
    ) -> Tuple[bytes, List[Dict]]:
        """
        Compose visualization from vibe vector.

        Uses prompt-based atmosphere enhancement to add weather mood and
        ambience without overlaying assets that could obscure core elements.

        Args:
            vibe_vector: Dict mapping signal categories to scores
                        e.g., {'transportation': 0.45, 'crime': -0.3, ...}
                        Score range: -1.0 to 1.0
            location: Geographic location for context

        Returns:
            Tuple[image_bytes, hitboxes]: PNG data and hitbox list
        """
        logger.info(
            f"Composing visualization for {location} with {len(vibe_vector)} signals"
        )

        # Step 1: Layout - place assets and track hitboxes
        expanded = self._expand_vibe_vector(vibe_vector)
        signals = [
            Signal(category=category, tag=tag, score=score)
            for category, (tag, score) in expanded.items()
        ]

        # Layout without atmosphere assets (prompt-only strategy)
        layout_image, hitboxes = self.layout_composer.compose(
            signals
        )

        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        layout_image.save(img_bytes, format="PNG")
        layout_data = img_bytes.getvalue()

        # Step 2: Polish - enhance style while preserving layout
        polish_provider = settings.polish.provider.capitalize()

        logger.info(f"Polishing image with {polish_provider}")

        # Generate weather-based atmosphere prompts
        atmosphere_positive, atmosphere_negative = AtmosphereDescriptor.generate_atmosphere_prompt(
            signals
        )

        # Build base prompt: instruct AI to reimagine the scene incorporating concepts
        tag_strings = [signal.tag.value for signal in signals if abs(signal.score) > 0.1]
        base_prompt = f"Transform into {location}'s cityscape as a vibrant, whimsical cartoon illustration, where the narratives shown by sections that look like pasted stickers are woven organically into the urban landscape. Render the entire scene in a unified cartoon aesthetic with cel-shaded depth, bold outlines, warm saturated colors, and playful geometric architecture. Each narrative element flows seamlessly as an integral part of the composition, not as overlaid additions—the cityscape itself tells these interconnected success stories through visual metaphor and scene design rather than literal symbols. Whimsical, charming, optimistic mood throughout, in the style of Pixar's conceptual art, Ghibli's environmental storytelling, and European comic illustration tradition."

        # Combine with atmosphere
        final_prompt = base_prompt
        if atmosphere_positive:
            final_prompt = f"{base_prompt} {atmosphere_positive}"
            logger.debug(f"Using atmosphere: {atmosphere_positive}")

        # Build negative prompt (avoid collage aesthetic, want integrated scene)
        base_negative = (
            "blurry, low quality, distorted, photorealistic, realistic, 3d render, photograph, "
            "stickers, cutouts, collage, pasted elements, separated layers, paper scraps"
        )
        final_negative = base_negative
        if atmosphere_negative:
            final_negative = f"{base_negative}, {atmosphere_negative}"

        polished_data = self.poller.polish(
            layout_data,
            prompt=final_prompt,
            negative_prompt=final_negative,
        )

        final_data = polished_data if polished_data else layout_data

        return final_data, hitboxes

    @staticmethod
    def _expand_vibe_vector(
        vibe_vector: Dict[str, float],
    ) -> Dict[SignalCategory, Tuple[SignalTag, float]]:
        """
        Expand vibe vector to (tag, score) tuples by category.
        
        Derives tags as POSITIVE or NEGATIVE based on score sign.
        Filters out insignificant signals with |score| < 0.1.

        Args:
            vibe_vector: Dict mapping category string to score
                        e.g., {'transportation': 0.75, 'crime': -0.45}
                        Score range: -1.0 to 1.0

        Returns:
            Dict mapping SignalCategory enum to (SignalTag, score) tuples
            Tag is SignalTag.POSITIVE or SignalTag.NEGATIVE
        """
        result = {}
        for category_str, score in vibe_vector.items():
            # Intensity is absolute value of score (0-1)
            intensity = abs(score)
            
            # Filter out insignificant signals
            if intensity < 0.1:
                continue
            
            # Convert category string to enum
            try:
                category = SignalCategory(category_str)
            except ValueError:
                logger.warning(f"Unknown category: {category_str}, skipping")
                continue
            
            # Tag is simply positive or negative based on score sign
            tag = SignalTag.POSITIVE if score >= 0 else SignalTag.NEGATIVE
            
            result[category] = (tag, score)

        return result


class VisualizationService:
    """
    Orchestrates the full visualization generation pipeline.

    Handles vibe-hash caching and hybrid composition:
    1. Check cache by vibe hash
    2. If miss: generate layout + polish
    3. Store in cache
    4. Return image URL + hitboxes
    """

    hopsworks_service = None 

    def __init__(self, use_hopsworks=False):
        """
        Initialize service with composer and cache.
        
        Args:
            hopsworks_service: Optional HopsworksService instance for hopsworks backend
        """
        logger.info("Initializing VisualizationService")

        self.composer = HybridComposer()

        if use_hopsworks:
            logger.info("Using Hopsworks storage backend")
            from backend.server.services.hopsworks import get_or_create_hopsworks_service

            self.hopsworks_service = get_or_create_hopsworks_service(
                api_key=settings.hopsworks.api_key,
                project_name=settings.hopsworks.project_name,
                host=settings.hopsworks.host,
            )

        # Initialize storage backend using factory
        storage = create_storage_backend(
            backend_type=settings.storage.backend,
            bucket_name=settings.storage.bucket_name,
            local_storage_dir=settings.storage.local_storage_dir,
            hopsworks_service=self.hopsworks_service,
            artifact_collection=settings.hopsworks.artifact_collection,
        )

        self.cache = VibeCache(storage)

    def generate_or_get(
        self,
        city: str,
        vibe_vector: Dict[str, float],
        timestamp: Optional[datetime] = None,
        force_regenerate: bool = False,
        source_articles: Optional[List[Dict]] = None,
    ) -> Tuple[bytes, Dict]:
        """
        Get or generate visualization for a city and vibe vector.

        Args:
            city: Geographic location
            vibe_vector: Dict mapping categories to scores
                        e.g., {'transportation': 0.45, 'crime': -0.3}
                        Score range: -1.0 to 1.0
            timestamp: Time window (defaults to now)
            force_regenerate: Skip cache and regenerate
            source_articles: Articles that contributed to vibe

        Returns:
            Tuple[image_bytes, response_dict]: Image data and metadata
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        logger.info(
            f"Getting visualization for {city} with {len(vibe_vector)} signals"
        )

        # Check cache
        if not force_regenerate:
            image_data, metadata = self.cache.get(city, timestamp)
            if image_data and metadata:
                logger.info(f"Cache hit for {city}")
                return image_data, {
                    "cached": True,
                    "cache_key": metadata.cache_key,
                    "hitboxes": metadata.hitboxes,
                }

        # Cache miss or force regenerate
        logger.info(f"Generating visualization for {city}")
        image_data, hitboxes = self.composer.compose(vibe_vector, city)

        # Cache result (only hitboxes are stored in metadata now)
        cache_key, metadata = self.cache.set(
            city=city,
            timestamp=timestamp,
            image_data=image_data,
            hitboxes=hitboxes,
            vibe_vector=vibe_vector,  # Not stored but kept for API compatibility
            source_articles=source_articles or [],  # Not stored but kept for API compatibility
        )

        return image_data, {
            "cached": False,
            "cache_key": cache_key,
            "hitboxes": hitboxes,
        }

