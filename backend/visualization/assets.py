"""
Asset mapping and zone-based layout composition.

Maps signal categories + tags + intensity to PNG assets, places them
in zones (sky/city/street), and records hitbox metadata.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Hitbox:
    """Represents a clickable region in the composed image."""

    x: int
    y: int
    width: int
    height: int
    signal_category: str
    signal_tag: str
    signal_intensity: float
    signal_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "signal_category": self.signal_category,
            "signal_tag": self.signal_tag,
            "signal_intensity": self.signal_intensity,
            "signal_score": self.signal_score,
        }


class AssetLibrary:
    """
    Manages asset loading and mapping.

    Maps signal category + tag + intensity to appropriate PNG files.
    Provides fallback icons for missing tags.
    Supports both zone-based and atmosphere assets.
    """

    # Mapping of (category, tag) -> asset filename (zone-based assets)
    ASSET_MAP = {
        # Transportation
        ("transportation", "traffic"): "transportation_traffic.png",
        ("transportation", "congestion"): "transportation_congestion.png",
        ("transportation", "accident"): "transportation_accident.png",
        ("transportation", "default"): "transportation_default.png",
        # Weather - Temperature
        ("weather_temp", "hot"): "weather_hot.png",
        ("weather_temp", "cold"): "weather_cold.png",
        ("weather_temp", "default"): "weather_temp_default.png",
        # Weather - Precipitation
        ("weather_wet", "rain"): "weather_rain.png",
        ("weather_wet", "snow"): "weather_snow.png",
        ("weather_wet", "flood"): "weather_flood.png",
        ("weather_wet", "default"): "weather_wet_default.png",
        # Crime
        ("crime", "theft"): "crime_theft.png",
        ("crime", "assault"): "crime_assault.png",
        ("crime", "police"): "crime_police.png",
        ("crime", "default"): "crime_default.png",
        # Festivals
        ("festivals", "concert"): "festivals_concert.png",
        ("festivals", "celebration"): "festivals_celebration.png",
        ("festivals", "crowd"): "festivals_crowd.png",
        ("festivals", "default"): "festivals_default.png",
        # Sports
        ("sports", "football"): "sports_football.png",
        ("sports", "hockey"): "sports_hockey.png",
        ("sports", "victory"): "sports_victory.png",
        ("sports", "default"): "sports_default.png",
        # Economics
        ("economics", "market"): "economics_market.png",
        ("economics", "business"): "economics_business.png",
        ("economics", "trade"): "economics_trade.png",
        ("economics", "default"): "economics_default.png",
        # Politics
        ("politics", "protest"): "politics_protest.png",
        ("politics", "election"): "politics_election.png",
        ("politics", "government"): "politics_government.png",
        ("politics", "default"): "politics_default.png",
        # Emergencies
        ("emergencies", "fire"): "emergencies_fire.png",
        ("emergencies", "earthquake"): "emergencies_earthquake.png",
        ("emergencies", "evacuation"): "emergencies_evacuation.png",
        ("emergencies", "default"): "emergencies_default.png",
    }

    # Atmosphere assets: apply to entire image, not in zones
    # Mapped by category -> tag -> filename
    ATMOSPHERE_MAP = {
        "weather_wet": {
            "rain": "atmosphere_rain.png",
            "snow": "atmosphere_snow.png",
            "flood": "atmosphere_flood.png",
        },
        "weather_temp": {
            "hot": "atmosphere_heat.png",
            "cold": "atmosphere_cold.png",
        },
        "festivals": {
            "celebration": "atmosphere_celebration.png",
            "crowd": "atmosphere_festive.png",
        },
        "politics": {
            "protest": "atmosphere_tension.png",
        },
        "emergencies": {
            "fire": "atmosphere_fire_glow.png",
            "earthquake": "atmosphere_tremor.png",
        },
    }

    # Category-level fallback icons
    CATEGORY_FALLBACK = {
        "transportation": "transportation_default.png",
        "weather_temp": "weather_temp_default.png",
        "weather_wet": "weather_wet_default.png",
        "crime": "crime_default.png",
        "festivals": "festivals_default.png",
        "sports": "sports_default.png",
        "economics": "economics_default.png",
        "politics": "politics_default.png",
        "emergencies": "emergencies_default.png",
    }

    # Ultimate fallback
    ULTIMATE_FALLBACK = "generic_default.png"

    def __init__(self, assets_dir: str):
        """
        Initialize asset library.

        Args:
            assets_dir: Path to directory containing asset PNGs
        """
        self.assets_dir = Path(assets_dir)
        self.loaded_assets: Dict[str, Optional[Image.Image]] = {}

        if not self.assets_dir.exists():
            logger.warning(f"Assets directory not found: {self.assets_dir}")

    def get_asset(
        self,
        category: str,
        tag: str,
        use_fallback: bool = True,
    ) -> Optional[Image.Image]:
        """
        Get asset image for a category + tag combination.

        Falls back through: exact match → category default → ultimate fallback.

        Args:
            category: Signal category (e.g., 'traffic', 'weather')
            tag: Signal tag (e.g., 'congestion', 'rain')
            use_fallback: Whether to use fallback assets if specific not found

        Returns:
            PIL Image or None if not found and fallback disabled
        """
        # Try exact match first
        asset_key = (category, tag)
        filename = self.ASSET_MAP.get(asset_key)

        if not filename and use_fallback:
            # Fall back to category default
            filename = self.CATEGORY_FALLBACK.get(
                category, self.ULTIMATE_FALLBACK
            )

        if not filename:
            return None

        # Check cache
        if filename in self.loaded_assets:
            return self.loaded_assets[filename]

        # Try to load from disk
        asset_path = self.assets_dir / filename
        if asset_path.exists():
            try:
                img = Image.open(asset_path).convert("RGBA")
                self.loaded_assets[filename] = img
                return img
            except Exception as e:
                logger.error(f"Failed to load asset {filename}: {e}")
                self.loaded_assets[filename] = None
                return None
        else:
            logger.debug(f"Asset file not found: {asset_path}")
            self.loaded_assets[filename] = None
            return None

    def get_asset_size(
        self, category: str, tag: str
    ) -> Optional[Tuple[int, int]]:
        """
        Get (width, height) of asset image.

        Args:
            category: Signal category
            tag: Signal tag

        Returns:
            (width, height) tuple or None if asset not found
        """
        asset = self.get_asset(category, tag)
        return asset.size if asset else None

    def get_atmosphere_asset(
        self,
        category: str,
        tag: str,
    ) -> Optional[Image.Image]:
        """
        Get atmosphere asset for a category + tag combination.

        Atmosphere assets overlay the entire image rather than being
        placed in a specific zone. Used to inject overall mood.

        Args:
            category: Signal category (e.g., 'weather_wet')
            tag: Signal tag (e.g., 'rain')

        Returns:
            PIL Image or None if not found
        """
        # Check if category has atmosphere assets
        if category not in self.ATMOSPHERE_MAP:
            return None

        # Try exact tag match
        filename = self.ATMOSPHERE_MAP[category].get(tag)

        if not filename:
            return None

        # Check cache
        if filename in self.loaded_assets:
            return self.loaded_assets[filename]

        # Try to load from disk
        asset_path = self.assets_dir / filename
        if asset_path.exists():
            try:
                img = Image.open(asset_path).convert("RGBA")
                self.loaded_assets[filename] = img
                return img
            except Exception as e:
                logger.error(f"Failed to load atmosphere asset {filename}: {e}")
                self.loaded_assets[filename] = None
                return None
        else:
            logger.debug(f"Atmosphere asset file not found: {asset_path}")
            self.loaded_assets[filename] = None
            return None

    def has_atmosphere_asset(self, category: str, tag: str) -> bool:
        """
        Check if an atmosphere asset exists for category + tag.

        Args:
            category: Signal category
            tag: Signal tag

        Returns:
            bool: True if atmosphere asset available
        """
        return (
            category in self.ATMOSPHERE_MAP
            and tag in self.ATMOSPHERE_MAP[category]
        )


class ZoneLayoutComposer:
    """
    Composes layout by placing assets in vertical zones (sky/city/street).

    Records hitbox metadata for each placed element.
    """

    # Zones define vertical regions
    ZONES = {
        "sky": {"start_ratio": 0.0, "height_ratio": 0.25, "name": "Sky"},
        "city": {"start_ratio": 0.25, "height_ratio": 0.50, "name": "City"},
        "street": {
            "start_ratio": 0.75,
            "height_ratio": 0.25,
            "name": "Street",
        },
    }

    def __init__(
        self,
        image_width: int,
        image_height: int,
        assets_dir: str,
        bg_color: Tuple[int, int, int] = (245, 245, 250),
    ):
        """
        Initialize composer.

        Args:
            image_width: Canvas width
            image_height: Canvas height
            assets_dir: Path to assets directory
            bg_color: Background color (RGB)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.bg_color = bg_color
        self.asset_library = AssetLibrary(assets_dir)
        self.hitboxes: List[Hitbox] = []

    def compose(
        self,
        signals: List[Tuple[str, str, float, float]],
        apply_atmosphere_assets: bool = True,
    ) -> Tuple[Image.Image, List[Hitbox]]:
        """
        Compose image with signals placed in zones, optionally with atmosphere overlay.

        Args:
            signals: List of (category, tag, intensity, score) tuples
                     intensity: 0.0-1.0 (used for sizing)
                     score: original -1.0 to 1.0 score from model
            apply_atmosphere_assets: If True, apply atmosphere overlays from signals

        Returns:
            Tuple[Image, hitboxes]: Composed PIL Image and hitbox list
        """
        # Create blank canvas
        image = Image.new("RGB", (self.image_width, self.image_height), self.bg_color)

        # Reset hitboxes
        self.hitboxes = []

        # Assign signals to zones based on category
        zone_assignments = self._assign_signals_to_zones(signals)

        # Place assets in each zone
        for zone_name, zone_signals in zone_assignments.items():
            self._place_signals_in_zone(image, zone_name, zone_signals)

        # Apply atmosphere overlays on top
        if apply_atmosphere_assets:
            self._apply_atmosphere_layers(image, signals)

        return image, self.hitboxes

    def _assign_signals_to_zones(
        self, signals: List[Tuple[str, str, float, float]]
    ) -> Dict[str, List[Tuple[str, str, float, float]]]:
        """
        Assign signals to zones based on category.

        Args:
            signals: List of (category, tag, intensity, score) tuples

        Returns:
            Dict mapping zone name to list of signals
        """
        zones = {"sky": [], "city": [], "street": []}

        for category, tag, intensity, score in signals:
            # Route by category
            if category in ("weather_temp", "weather_wet"):
                zones["sky"].append((category, tag, intensity, score))
            elif category in ("crime", "emergencies", "transportation"):
                zones["street"].append((category, tag, intensity, score))
            else:  # festivals, sports, economics, politics
                zones["city"].append((category, tag, intensity, score))

        return zones

    def _place_signals_in_zone(
        self,
        image: Image.Image,
        zone_name: str,
        signals: List[Tuple[str, str, float, float]],
    ) -> None:
        """
        Place signals within a zone.

        Args:
            image: PIL Image to draw on
            zone_name: Zone name (sky/city/street)
            signals: Signals to place in this zone
        """
        if not signals:
            return

        zone = self.ZONES[zone_name]
        zone_y_start = int(self.image_height * zone["start_ratio"])
        zone_height = int(self.image_height * zone["height_ratio"])
        zone_y_end = zone_y_start + zone_height

        # Simple grid-based placement
        cols = max(1, len(signals))  # Place signals horizontally
        col_width = self.image_width // cols

        for idx, (category, tag, intensity, score) in enumerate(signals):
            col = idx % cols

            # Center in column
            x = col * col_width + col_width // 2
            # Center in zone vertically
            y = zone_y_start + zone_height // 2

            # Get asset
            asset = self.asset_library.get_asset(category, tag)
            if not asset:
                logger.debug(f"No asset for {category}/{tag}, skipping")
                continue

            # Scale based on intensity
            scale_factor = 0.3 + (intensity * 0.4)  # Scale 0.3-0.7 based on intensity
            new_size = (
                int(asset.width * scale_factor),
                int(asset.height * scale_factor),
            )
            asset_scaled = asset.resize(new_size, Image.Resampling.LANCZOS)

            # Place on canvas (centered at x, y)
            paste_x = max(0, min(x - new_size[0] // 2, self.image_width - new_size[0]))
            paste_y = max(zone_y_start, min(y - new_size[1] // 2, zone_y_end - new_size[1]))

            image.paste(asset_scaled, (paste_x, paste_y), asset_scaled)

            # Record hitbox
            hitbox = Hitbox(
                x=paste_x,
                y=paste_y,
                width=new_size[0],
                height=new_size[1],
                signal_category=category,
                signal_tag=tag,
                signal_intensity=intensity,
                signal_score=score,
            )
            self.hitboxes.append(hitbox)

    def get_hitboxes(self) -> List[Dict]:
        """
        Get hitboxes as dictionaries.

        Returns:
            List of hitbox dicts
        """
        return [h.to_dict() for h in self.hitboxes]
    def _apply_atmosphere_layers(
        self,
        image: Image.Image,
        signals: List[Tuple[str, str, float, float]],
    ) -> None:
        """
        Apply full-image atmosphere overlays based on dominant signals.

        Atmosphere assets (e.g., rain effect, heat haze) are overlaid
        on top of zone-based elements to enhance mood.

        Args:
            image: PIL Image to apply effects to (modified in-place)
            signals: List of (category, tag, intensity, score) tuples
        """
        # Collect atmosphere assets by intensity (apply strongest last for prominence)
        atmospheres = []

        for category, tag, intensity, score in signals:
            if self.asset_library.has_atmosphere_asset(category, tag):
                atmospheres.append((intensity, category, tag))

        # Sort by intensity (weakest first, so strongest overlays on top)
        atmospheres.sort(key=lambda x: x[0])

        # Apply each atmosphere asset
        for intensity, category, tag in atmospheres:
            atmosphere = self.asset_library.get_atmosphere_asset(category, tag)
            if atmosphere:
                # Scale atmosphere to canvas size
                atmosphere_scaled = atmosphere.resize(
                    (self.image_width, self.image_height),
                    Image.Resampling.LANCZOS
                )

                # Apply with opacity based on intensity
                # Higher intensity = more opaque atmosphere
                opacity = int(255 * min(intensity, 1.0))
                atmosphere_scaled.putalpha(opacity)

                # Composite over current image
                image.paste(
                    atmosphere_scaled,
                    (0, 0),
                    atmosphere_scaled,
                )

                logger.debug(
                    f"Applied atmosphere: {category}/{tag} at opacity {opacity}"
                )