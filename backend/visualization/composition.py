"""
Image generation and template composition module.

Generates cartoonish visualizations based on detected signals
using a template-based composition approach.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import io

logger = logging.getLogger(__name__)


@dataclass
class SignalIntensity:
    """Represents a signal with its intensity level."""

    signal_name: str
    intensity: float  # 0-100 scale


class VisualizationCache:
    """
    In-memory cache for generated visualizations.

    In production, this would interface with MinIO, S3, or PostgreSQL.
    """

    def __init__(self):
        self.cache: Dict[str, bytes] = {}
        self.metadata: Dict[str, Dict] = {}

    def generate_cache_key(
        self, signal_intensities: List[SignalIntensity]
    ) -> str:
        """
        Generates a unique cache key from signal combination and intensities.

        The key is deterministic so identical signal profiles always map to same key.

        Args:
            signal_intensities: List of signals with intensities

        Returns:
            str: SHA256 cache key
        """
        # Sort by signal name for consistency
        sorted_signals = sorted(signal_intensities, key=lambda s: s.signal_name)

        # Create key string: "signal1:45.5,signal2:82.3,..."
        # We discretize intensities to bins (0, 10, 20, ..., 100) to prevent cache fragmentation
        key_parts = []
        for signal in sorted_signals:
            discretized = int(signal.intensity / 10) * 10
            key_parts.append(f"{signal.signal_name}:{discretized}")

        key_string = "|".join(key_parts)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()

        logger.debug(f"Generated cache key for signals: {key_string}")
        return cache_key

    def get(self, signal_intensities: List[SignalIntensity]) -> Optional[bytes]:
        """
        Retrieves a cached visualization if it exists.

        Args:
            signal_intensities: List of signals

        Returns:
            bytes: Image data if cached, None otherwise
        """
        cache_key = self.generate_cache_key(signal_intensities)
        image_data = self.cache.get(cache_key)

        if image_data:
            logger.info(f"Cache hit for key: {cache_key}")
        else:
            logger.info(f"Cache miss for key: {cache_key}")

        return image_data

    def set(
        self,
        signal_intensities: List[SignalIntensity],
        image_data: bytes,
        metadata: Dict = None,
    ) -> str:
        """
        Caches a generated visualization.

        Args:
            signal_intensities: List of signals
            image_data: Generated image bytes
            metadata: Optional metadata (source articles, generation timestamp, etc.)

        Returns:
            str: Cache key used
        """
        cache_key = self.generate_cache_key(signal_intensities)
        self.cache[cache_key] = image_data

        if metadata:
            self.metadata[cache_key] = metadata

        logger.info(f"Cached visualization with key: {cache_key}")
        return cache_key

    def get_metadata(self, cache_key: str) -> Optional[Dict]:
        """Retrieves metadata for a cached visualization."""
        return self.metadata.get(cache_key)

    def get_stats(self) -> Dict[str, int]:
        """Returns cache statistics."""
        return {
            "cached_visualizations": len(self.cache),
            "cache_size_estimates": sum(len(data) for data in self.cache.values()),
        }


class TemplateComposer:
    """
    Composes cartoonish visualizations from signal templates.

    Generates PNG images with signal elements positioned and scaled
    based on their intensity levels.
    """

    def __init__(self):
        logger.info("Initializing TemplateComposer")

        # Signal templates with visual properties
        self.templates = {
            "traffic": {"emoji": "🚗", "color": "#FF6B6B", "label": "Traffic"},
            "weather": {"emoji": "🌧️", "color": "#4ECDC4", "label": "Weather"},
            "crime": {"emoji": "🚨", "color": "#45B7D1", "label": "Crime"},
            "festivals": {"emoji": "🎉", "color": "#FFA07A", "label": "Festivals"},
            "politics": {"emoji": "🏛️", "color": "#98D8C8", "label": "Politics"},
            "sports": {"emoji": "⚽", "color": "#F7DC6F", "label": "Sports"},
            "accidents": {"emoji": "🔥", "color": "#EB6B56", "label": "Accidents"},
            "economic": {"emoji": "💼", "color": "#BB8FCE", "label": "Economic"},
        }
        
        # Image generation parameters
        self.width = 1024
        self.height = 768
        self.bg_color = (245, 245, 250)  # Light lavender-ish
        self.text_color = (60, 60, 80)

    def compose(
        self, signal_intensities: List[SignalIntensity], location: str = "Unknown"
    ) -> bytes:
        """
        Composes a visualization from signals and renders to PNG.

        Creates a cartoonish image with signal elements positioned
        based on their intensities.

        Args:
            signal_intensities: List of signals with intensities
            location: Geographic location for visualization

        Returns:
            bytes: PNG image data
        """
        logger.info(
            f"Composing visualization for {location} with {len(signal_intensities)} signals"
        )

        # Create image with gradient-like background
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add a subtle gradient by drawing rectangles of slightly different colors
        # (PIL doesn't have built-in gradients, so we approximate)
        for y in range(self.height):
            ratio = y / self.height
            r = int(245 + (102 - 245) * ratio * 0.15)  # Slight blue tint towards bottom
            g = int(245 + (155 - 245) * ratio * 0.15)
            b = int(250 + (201 - 250) * ratio * 0.15)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        # Draw decorative header background
        header_bg_color = (100, 120, 180)  # Dark blue
        draw.rectangle(
            [(0, 0), (self.width, 80)],
            fill=header_bg_color,
        )

        # Draw location title
        header_text = f"The Daily Collage • {location.title()}"
        title_color = (255, 255, 255)
        try:
            draw.text(
                (self.width // 2, 40),
                header_text,
                fill=title_color,
                anchor="mm",
            )
        except:
            # Fallback if font rendering fails
            draw.text(
                (self.width // 2, 40),
                header_text,
                fill=title_color,
                anchor="mm",
            )

        # Sort signals by intensity (descending) for better visual hierarchy
        sorted_signals = sorted(
            signal_intensities, key=lambda s: s.intensity, reverse=True
        )

        # Layout signals in a grid-like pattern
        cols = 4
        cell_width = self.width // cols
        cell_height = (self.height - 120) // 2

        for idx, signal in enumerate(sorted_signals):
            col = idx % cols
            row = idx // cols

            x = col * cell_width + cell_width // 2
            y = 80 + 20 + row * cell_height + cell_height // 2

            self._draw_signal_element(
                draw, signal, (x, y), cell_width - 20
            )

        # Add footer with timestamp info
        footer_y = self.height - 30
        footer_text = "Updated every 6 hours • Powered by GDELT"
        footer_color = (120, 120, 140)
        draw.text(
            (self.width // 2, footer_y),
            footer_text,
            fill=footer_color,
            anchor="mm",
        )

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return img_bytes.getvalue()

    def _draw_signal_element(
        self,
        draw: ImageDraw.ImageDraw,
        signal: SignalIntensity,
        center: Tuple[int, int],
        max_size: int,
    ) -> None:
        """
        Draws a single signal element with size scaled by intensity.

        Args:
            draw: PIL ImageDraw context
            signal: Signal with intensity
            center: (x, y) center coordinates
            max_size: Maximum element size
        """
        template = self.templates.get(signal.signal_name, {})
        color_str = template.get("color", "#999999")
        label = template.get("label", signal.signal_name)

        # Scale circle size by intensity
        radius = int((signal.intensity / 100) * (max_size / 2))
        x, y = center

        # Draw circle background
        color_rgb = self._hex_to_rgb(color_str)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color_rgb,
            outline=self.text_color,
            width=2,
        )

        # Draw label
        label_y = y + radius + 15
        draw.text(
            (x, label_y),
            label,
            fill=self.text_color,
            anchor="mm",
        )

        # Draw intensity percentage
        intensity_text = f"{signal.intensity:.0f}%"
        intensity_y = label_y + 20
        draw.text(
            (x, intensity_y),
            intensity_text,
            fill=(100, 100, 100),
            anchor="mm",
        )

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        Converts hex color string to RGB tuple.

        Args:
            hex_color: Color in format '#RRGGBB'

        Returns:
            Tuple[int, int, int]: RGB tuple
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def get_supported_signals(self) -> List[str]:
        """Returns list of signal types this composer supports."""
        return list(self.templates.keys())


class VisualizationService:
    """
    Orchestrates the full visualization generation pipeline.

    Coordinates caching and composition.
    """

    def __init__(self):
        self.cache = VisualizationCache()
        self.composer = TemplateComposer()

    def generate_or_get(
        self,
        signal_intensities: List[SignalIntensity],
        location: str = "Unknown",
        force_regenerate: bool = False,
    ) -> Tuple[bytes, Dict]:
        """
        Generates a visualization, using cache if available.

        Args:
            signal_intensities: List of signals with intensities
            location: Geographic location
            force_regenerate: If True, skip cache and regenerate

        Returns:
            Tuple[bytes, Dict]: Image data and metadata dict
        """
        # Check cache first
        if not force_regenerate:
            cached_image = self.cache.get(signal_intensities)
            if cached_image:
                metadata = self.cache.get_metadata(
                    self.cache.generate_cache_key(signal_intensities)
                )
                return cached_image, metadata or {}

        # Generate new visualization
        logger.info(f"Generating new visualization for {location}")
        image_data = self.composer.compose(signal_intensities, location)

        # Build metadata
        metadata = {
            "location": location,
            "signal_count": len(signal_intensities),
            "signals": [
                {"name": s.signal_name, "intensity": s.intensity}
                for s in signal_intensities
            ],
        }

        # Cache result
        self.cache.set(signal_intensities, image_data, metadata)

        return image_data, metadata
