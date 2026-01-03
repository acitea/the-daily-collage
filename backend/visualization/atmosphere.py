"""
Atmosphere generation through text prompting.

Generates high-quality atmospheric prompts based on weather conditions
and signal composition for use in img2img enhancement.
"""

from typing import List, Tuple, Dict, Optional


class AtmosphereDescriptor:
    """
    Generates atmospheric prompts based on weather and signal composition.

    Uses high-quality positive and negative prompts to create mood and
    atmosphere without overlaying assets that could obscure core elements.
    """

    # Weather-specific atmosphere prompts: (category, tag) -> {positive, negative}
    WEATHER_ATMOSPHERE_PROMPTS = {
        # Rain
        ("weather_wet", "rain"): {
            "positive": (
                "heavy rain, storm, wet ground, puddles, reflections on streets, "
                "overcast sky, gloomy atmosphere, cinematic lighting, raindrops, "
                "mist, high contrast, cool color tone, splashing water"
            ),
            "negative": (
                "bright sun, dry ground, clear sky, blue sky, dust, warm lighting, "
                "happy, vivid colors, flat lighting"
            ),
        },
        # Snow
        ("weather_wet", "snow"): {
            "positive": (
                "heavy snow, winter storm, snow covered ground, frost, ice, "
                "snowflakes in air, white atmosphere, frozen, cold, blizzard, "
                "soft blue ambient light, winter"
            ),
            "negative": (
                "summer, green grass, flowers, sun, heat, warm colors, rain, "
                "water, asphalt, leaves"
            ),
        },
        # Flood (use rain as base + intensity)
        ("weather_wet", "flood"): {
            "positive": (
                "heavy rain, storm, thunderstorm, flooding, water everywhere, "
                "puddles, reflections, overcast sky, dramatic weather, "
                "wet surfaces, water splashing"
            ),
            "negative": (
                "bright sun, dry ground, clear sky, dust, warm lighting, happy"
            ),
        },
        # Hot/Dry
        ("weather_temp", "hot"): {
            "positive": (
                "desert heat, heat wave, heat haze, shimmering air, intense sun, "
                "drought, dry cracked ground, dusty, sepia tone, overexposed, "
                "harsh sunlight, arid, warm color palette, yellow tint"
            ),
            "negative": (
                "water, rain, clouds, cold, blue tones, lush vegetation, wet, "
                "snow, puddles, soft lighting, cool colors"
            ),
        },
        # Cold (but not snowing)
        ("weather_temp", "cold"): {
            "positive": (
                "cold atmosphere, frost, crisp air, winter, cool tones, "
                "blue ambient light, freezing, icy, sharp details"
            ),
            "negative": (
                "summer, heat, warm colors, sun, flowers, green, tropical"
            ),
        },
    }

    # Sunny/bright weather (default for neutral weather_temp)
    SUNNY_WEATHER_PROMPT = {
        "positive": (
            "bright sunny day, clear blue sky, harsh sunlight, lens flare, "
            "hard shadows, vibrant colors, golden hour, sunbeams, high exposure, "
            "summer vibe, crisp detail, glistening"
        ),
        "negative": (
            "clouds, rain, fog, overcast, gloom, grey sky, wet ground, "
            "low contrast, night, dark, depression"
        ),
    }

    # Neutral weather (overcast/mild)
    NEUTRAL_WEATHER_PROMPT = {
        "positive": (
            "overcast, soft lighting, diffused light, flat lighting, neutral colors, "
            "cloudy sky, ambient occlusion, realistic, everyday atmosphere, "
            "balanced exposure, photorealistic"
        ),
        "negative": (
            "direct sunlight, hard shadows, high contrast, rain, snow, "
            "sunset, sunrise, neon, extreme weather, overexposed, underexposed"
        ),
    }

    @classmethod
    def generate_atmosphere_prompt(
        cls,
        signals: List[Tuple[str, str, float, float]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate atmospheric prompts from weather signals.

        Analyzes weather_wet and weather_temp signals to determine the
        dominant weather condition and returns appropriate positive/negative prompts.

        Args:
            signals: List of (category, tag, intensity, score) tuples

        Returns:
            Tuple[positive_prompt, negative_prompt]: Weather-specific prompts
                                                     or (None, None) if no weather signals
        """
        # Extract weather signals
        weather_wet_signal = None
        weather_temp_signal = None
        max_wet_intensity = 0.0
        max_temp_intensity = 0.0

        for category, tag, intensity, score in signals:
            if category == "weather_wet" and intensity > max_wet_intensity:
                weather_wet_signal = (category, tag, intensity)
                max_wet_intensity = intensity
            elif category == "weather_temp" and intensity > max_temp_intensity:
                weather_temp_signal = (category, tag, intensity)
                max_temp_intensity = intensity

        # Priority: weather_wet (rain/snow) overrides weather_temp
        if weather_wet_signal and max_wet_intensity > 0.3:
            category, tag, intensity = weather_wet_signal
            key = (category, tag)
            if key in cls.WEATHER_ATMOSPHERE_PROMPTS:
                prompts = cls.WEATHER_ATMOSPHERE_PROMPTS[key]
                return prompts["positive"], prompts["negative"]

        # If no precipitation, check temperature
        if weather_temp_signal and max_temp_intensity > 0.3:
            category, tag, intensity = weather_temp_signal
            key = (category, tag)
            if key in cls.WEATHER_ATMOSPHERE_PROMPTS:
                prompts = cls.WEATHER_ATMOSPHERE_PROMPTS[key]
                return prompts["positive"], prompts["negative"]
            # Default sunny for unrecognized temp tags
            return cls.SUNNY_WEATHER_PROMPT["positive"], cls.SUNNY_WEATHER_PROMPT["negative"]

        # No significant weather signals - use neutral
        return cls.NEUTRAL_WEATHER_PROMPT["positive"], cls.NEUTRAL_WEATHER_PROMPT["negative"]


