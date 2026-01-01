"""
Atmosphere generation for visual and textual enhancements.

Supports two strategies:
1. Asset-based: PNG overlays applied to entire image
2. Prompt-based: Text descriptions incorporated into img2img prompts
"""

from typing import List, Tuple, Dict, Optional
from enum import Enum


class AtmosphereStrategy(Enum):
    """Strategy for applying atmospheric effects."""

    ASSET = "asset"  # Use PNG overlays
    PROMPT = "prompt"  # Use text prompts


class AtmosphereDescriptor:
    """
    Generates atmospheric descriptions and moods based on signal composition.

    Maps vibe vector signals to evocative atmospheric language for use in
    img2img prompts.
    """

    # Mapping of (category, tag) -> atmosphere descriptions
    ATMOSPHERE_DESCRIPTIONS = {
        ("weather_wet", "rain"): [
            "rainy and gloomy",
            "wet streets reflecting light",
            "overcast atmosphere",
        ],
        ("weather_wet", "snow"): [
            "snowy and peaceful",
            "winter wonderland",
            "cold and crisp",
        ],
        ("weather_wet", "flood"): [
            "chaotic and wet",
            "water-logged environment",
            "disaster atmosphere",
        ],
        ("weather_temp", "hot"): [
            "hot and vibrant",
            "heat haze shimmer",
            "bright and warm",
        ],
        ("weather_temp", "cold"): [
            "cold and stark",
            "frosty atmosphere",
            "crisp and icy",
        ],
        ("emergencies", "fire"): [
            "flames and danger",
            "fiery and intense",
            "dramatic emergency",
        ],
        ("emergencies", "earthquake"): [
            "chaotic and unstable",
            "trembling ground",
            "disaster atmosphere",
        ],
        ("festivals", "celebration"): [
            "festive and joyful",
            "celebration mood",
            "colorful and lively",
        ],
        ("festivals", "crowd"): [
            "crowded and energetic",
            "bustling activity",
            "mass gathering",
        ],
        ("politics", "protest"): [
            "tense and confrontational",
            "protest mood",
            "charged atmosphere",
        ],
        ("crime", "police"): [
            "law enforcement presence",
            "official atmosphere",
            "security focus",
        ],
        ("sports", "victory"): [
            "celebratory and triumphant",
            "victory atmosphere",
            "excitement and joy",
        ],
        ("economics", "market"): [
            "busy commercial activity",
            "trading floor energy",
            "business atmosphere",
        ],
    }

    @classmethod
    def generate_atmosphere_prompt(
        cls,
        signals: List[Tuple[str, str, float, float]],
        max_descriptions: int = 3,
    ) -> Optional[str]:
        """
        Generate atmospheric prompt from dominant signals.

        Selects strongest signals by intensity and combines their
        atmospheric descriptions into a cohesive prompt.

        Args:
            signals: List of (category, tag, intensity, score) tuples
            max_descriptions: Maximum number of descriptions to combine

        Returns:
            str: Atmospheric prompt or None if no matching signals
        """
        # Collect atmosphere descriptions for strong signals
        descriptions = []

        for category, tag, intensity, score in signals:
            if intensity > 0.3:  # Only strong signals
                key = (category, tag)
                if key in cls.ATMOSPHERE_DESCRIPTIONS:
                    # Pick description based on intensity
                    descriptions.append(
                        (intensity, cls.ATMOSPHERE_DESCRIPTIONS[key])
                    )

        if not descriptions:
            return None

        # Sort by intensity and take top signals
        descriptions.sort(key=lambda x: x[0], reverse=True)
        descriptions = descriptions[:max_descriptions]

        # Combine descriptions
        selected = []
        for intensity, desc_list in descriptions:
            # Pick first description (could also randomize)
            selected.append(desc_list[0])

        # Create prompt
        prompt = ", ".join(selected)
        return prompt

    @classmethod
    def get_mood(
        cls,
        signals: List[Tuple[str, str, float, float]],
    ) -> Optional[str]:
        """
        Determine overall mood from signal composition.

        Args:
            signals: List of (category, tag, intensity, score) tuples

        Returns:
            str: Mood descriptor (e.g., "festive", "chaotic", "peaceful")
        """
        # Count positive/negative/neutral signals
        positive = 0
        negative = 0
        intensity_total = 0

        for category, tag, intensity, score in signals:
            if intensity > 0.3:
                intensity_total += intensity
                if score > 0.5:
                    positive += 1
                elif score < -0.5:
                    negative += 1

        if intensity_total == 0:
            return "calm"

        # Determine mood based on composition
        if negative > positive:
            return "chaotic"
        elif positive > negative:
            return "festive"
        else:
            return "neutral"
