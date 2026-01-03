"""
Asset mapping and zone-based layout composition.

Maps signal categories + tags + intensity to PNG assets, places them
in zones (sky/city/street), and records hitbox metadata.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
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

    # Mapping of (category, tag, intensity_level) -> asset filename
    # intensity_level: "low" (0.0-0.33), "med" (0.33-0.66), "high" (0.66-1.0)
    ASSET_MAP = {
        # Transportation
        ("transportation", "traffic", "low"): "transportation_traffic_low.png",
        ("transportation", "traffic", "med"): "transportation_traffic_med.png",
        ("transportation", "traffic", "high"): "transportation_traffic_high.png",
        ("transportation", "congestion", "low"): "transportation_congestion_low.png",
        ("transportation", "congestion", "med"): "transportation_congestion_med.png",
        ("transportation", "congestion", "high"): "transportation_congestion_high.png",
        ("transportation", "accident", "low"): "transportation_accident_low.png",
        ("transportation", "accident", "med"): "transportation_accident_med.png",
        ("transportation", "accident", "high"): "transportation_accident_high.png",
        # Crime
        ("crime", "theft", "low"): "crime_theft_low.png",
        ("crime", "theft", "med"): "crime_theft_med.png",
        ("crime", "theft", "high"): "crime_theft_high.png",
        ("crime", "assault", "low"): "crime_assault_low.png",
        ("crime", "assault", "med"): "crime_assault_med.png",
        ("crime", "assault", "high"): "crime_assault_high.png",
        ("crime", "police", "low"): "crime_police_low.png",
        ("crime", "police", "med"): "crime_police_med.png",
        ("crime", "police", "high"): "crime_police_high.png",
        # Festivals
        ("festivals", "concert", "low"): "festivals_concert_low.png",
        ("festivals", "concert", "med"): "festivals_concert_med.png",
        ("festivals", "concert", "high"): "festivals_concert_high.png",
        ("festivals", "celebration", "low"): "festivals_celebration_low.png",
        ("festivals", "celebration", "med"): "festivals_celebration_med.png",
        ("festivals", "celebration", "high"): "festivals_celebration_high.png",
        ("festivals", "crowd", "low"): "festivals_crowd_low.png",
        ("festivals", "crowd", "med"): "festivals_crowd_med.png",
        ("festivals", "crowd", "high"): "festivals_crowd_high.png",
        # Sports
        ("sports", "football", "low"): "sports_football_low.png",
        ("sports", "football", "med"): "sports_football_med.png",
        ("sports", "football", "high"): "sports_football_high.png",
        ("sports", "hockey", "low"): "sports_hockey_low.png",
        ("sports", "hockey", "med"): "sports_hockey_med.png",
        ("sports", "hockey", "high"): "sports_hockey_high.png",
        ("sports", "victory", "low"): "sports_victory_low.png",
        ("sports", "victory", "med"): "sports_victory_med.png",
        ("sports", "victory", "high"): "sports_victory_high.png",
        # Economics
        ("economics", "market", "low"): "economics_market_low.png",
        ("economics", "market", "med"): "economics_market_med.png",
        ("economics", "market", "high"): "economics_market_high.png",
        ("economics", "business", "low"): "economics_business_low.png",
        ("economics", "business", "med"): "economics_business_med.png",
        ("economics", "business", "high"): "economics_business_high.png",
        ("economics", "trade", "low"): "economics_trade_low.png",
        ("economics", "trade", "med"): "economics_trade_med.png",
        ("economics", "trade", "high"): "economics_trade_high.png",
        # Politics
        ("politics", "protest", "low"): "politics_protest_low.png",
        ("politics", "protest", "med"): "politics_protest_med.png",
        ("politics", "protest", "high"): "politics_protest_high.png",
        ("politics", "election", "low"): "politics_election_low.png",
        ("politics", "election", "med"): "politics_election_med.png",
        ("politics", "election", "high"): "politics_election_high.png",
        ("politics", "government", "low"): "politics_government_low.png",
        ("politics", "government", "med"): "politics_government_med.png",
        ("politics", "government", "high"): "politics_government_high.png",
        # Emergencies
        ("emergencies", "fire", "low"): "emergencies_fire_low.png",
        ("emergencies", "fire", "med"): "emergencies_fire_med.png",
        ("emergencies", "fire", "high"): "emergencies_fire_high.png",
        ("emergencies", "earthquake", "low"): "emergencies_earthquake_low.png",
        ("emergencies", "earthquake", "med"): "emergencies_earthquake_med.png",
        ("emergencies", "earthquake", "high"): "emergencies_earthquake_high.png",
        ("emergencies", "evacuation", "low"): "emergencies_evacuation_low.png",
        ("emergencies", "evacuation", "med"): "emergencies_evacuation_med.png",
        ("emergencies", "evacuation", "high"): "emergencies_evacuation_high.png",
    }
    
    # Generic fallbacks by category and intensity
    CATEGORY_INTENSITY_FALLBACK = {
        ("transportation", "low"): "transportation_generic_low.png",
        ("transportation", "med"): "transportation_generic_med.png",
        ("transportation", "high"): "transportation_generic_high.png",
        ("crime", "low"): "crime_generic_low.png",
        ("crime", "med"): "crime_generic_med.png",
        ("crime", "high"): "crime_generic_high.png",
        ("festivals", "low"): "festivals_generic_low.png",
        ("festivals", "med"): "festivals_generic_med.png",
        ("festivals", "high"): "festivals_generic_high.png",
        ("sports", "low"): "sports_generic_low.png",
        ("sports", "med"): "sports_generic_med.png",
        ("sports", "high"): "sports_generic_high.png",
        ("economics", "low"): "economics_generic_low.png",
        ("economics", "med"): "economics_generic_med.png",
        ("economics", "high"): "economics_generic_high.png",
        ("politics", "low"): "politics_generic_low.png",
        ("politics", "med"): "politics_generic_med.png",
        ("politics", "high"): "politics_generic_high.png",
        ("emergencies", "low"): "emergencies_generic_low.png",
        ("emergencies", "med"): "emergencies_generic_med.png",
        ("emergencies", "high"): "emergencies_generic_high.png",
    }

    # Ultimate fallback
    ULTIMATE_FALLBACK = "generic_default.png"
    
    @staticmethod
    def get_intensity_level(intensity: float) -> str:
        """
        Discretize intensity into fixed levels: low, med, high.
        
        Args:
            intensity: Float in range [0.0, 1.0]
            
        Returns:
            "low", "med", or "high"
        """
        if intensity < 0.33:
            return "low"
        elif intensity < 0.66:
            return "med"
        else:
            return "high"

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
        intensity: float,
        use_fallback: bool = True,
    ) -> Optional[Image.Image]:
        """
        Get asset image for a category + tag + intensity combination.

        Falls back through: exact match → category+intensity → ultimate fallback.

        Args:
            category: Signal category (e.g., 'traffic', 'weather')
            tag: Signal tag (e.g., 'congestion', 'rain')
            intensity: Intensity value [0.0, 1.0]
            use_fallback: Whether to use fallback assets if specific not found

        Returns:
            PIL Image or None if not found and fallback disabled
        """
        intensity_level = self.get_intensity_level(intensity)
        
        # Try exact match first (category, tag, intensity)
        asset_key = (category, tag, intensity_level)
        filename = self.ASSET_MAP.get(asset_key)

        if not filename and use_fallback:
            # Fall back to category + intensity generic
            fallback_key = (category, intensity_level)
            filename = self.CATEGORY_INTENSITY_FALLBACK.get(fallback_key)
            
        if not filename and use_fallback:
            # Ultimate fallback
            filename = self.ULTIMATE_FALLBACK

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
        self, category: str, tag: str, intensity: float
    ) -> Optional[Tuple[int, int]]:
        """
        Get (width, height) of asset image.

        Args:
            category: Signal category
            tag: Signal tag
            intensity: Intensity value [0.0, 1.0]

        Returns:
            (width, height) tuple or None if asset not found
        """
        asset = self.get_asset(category, tag, intensity)
        return asset.size if asset else None


class ZoneLayoutComposer:
    """
    Composes layout by placing assets in vertical zones (sky/city/street).
    
    Uses grid-based randomized placement to avoid overlaps.
    Records hitbox metadata for each placed element.
    """
    
    # Grid configuration for placement
    GRID_COLS = 8  # More columns for flexibility
    GRID_ROWS_PER_ZONE = 3  # Rows per zone for vertical distribution

    def __init__(
        self,
        image_width: int,
        image_height: int,
        assets_dir: str,
        bg_color: Tuple[int, int, int] = (245, 245, 250),
        sky_zone_height: float = 0.25,
        city_zone_height: float = 0.50,
        street_zone_height: float = 0.25,
    ):
        """
        Initialize composer.

        Args:
            image_width: Canvas width
            image_height: Canvas height
            assets_dir: Path to assets directory
            bg_color: Background color (RGB)
            sky_zone_height: Height of sky zone as fraction of total height
            city_zone_height: Height of city zone as fraction of total height
            street_zone_height: Height of street zone as fraction of total height
        """
        self.image_width = image_width
        self.image_height = image_height
        self.bg_color = bg_color
        self.asset_library = AssetLibrary(assets_dir)
        self.hitboxes: List[Hitbox] = []
        
        # Compute zones dynamically from settings
        self.zones = {
            "sky": {
                "start_ratio": 0.0,
                "height_ratio": sky_zone_height,
                "name": "Sky"
            },
            "city": {
                "start_ratio": sky_zone_height,
                "height_ratio": city_zone_height,
                "name": "City"
            },
            "street": {
                "start_ratio": sky_zone_height + city_zone_height,
                "height_ratio": street_zone_height,
                "name": "Street"
            },
        }

    def compose(
        self,
        signals: List[Tuple[str, str, float, float]],
    ) -> Tuple[Image.Image, List[Hitbox]]:
        """
        Compose image with signals placed in zones.

        Note: apply_atmosphere_assets parameter is deprecated and ignored.
        Atmosphere is now handled via prompting only.

        Args:
            signals: List of (category, tag, intensity, score) tuples
                     intensity: 0.0-1.0 (used for sizing)
                     score: original -1.0 to 1.0 score from model
            apply_atmosphere_assets: Deprecated, ignored (kept for API compatibility)

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
        Place signals within a zone using randomized grid-based placement.

        Args:
            image: PIL Image to draw on
            zone_name: Zone name (sky/city/street)
            signals: Signals to place in this zone
        """
        if not signals:
            return

        zone = self.zones[zone_name]
        zone_y_start = int(self.image_height * zone["start_ratio"])
        zone_height = int(self.image_height * zone["height_ratio"])
        zone_y_end = zone_y_start + zone_height

        # Create grid cells for placement
        cell_width = self.image_width // self.GRID_COLS
        cell_height = zone_height // self.GRID_ROWS_PER_ZONE
        
        # Generate all available cells in this zone
        available_cells = [
            (col, row) 
            for row in range(self.GRID_ROWS_PER_ZONE) 
            for col in range(self.GRID_COLS)
        ]
        
        # Shuffle for randomness
        random.shuffle(available_cells)
        
        # Ensure we have enough cells
        if len(signals) > len(available_cells):
            logger.warning(
                f"Zone {zone_name} has {len(signals)} signals but only {len(available_cells)} cells. "
                f"Some signals may overlap."
            )

        for idx, (category, tag, intensity, score) in enumerate(signals):
            if idx >= len(available_cells):
                logger.warning(f"Ran out of cells in zone {zone_name}, skipping signal {category}/{tag}")
                break
                
            col, row = available_cells[idx]

            # Calculate cell center
            cell_center_x = col * cell_width + cell_width // 2
            cell_center_y = zone_y_start + row * cell_height + cell_height // 2
            
            # Add random offset within cell (±25% of cell size)
            offset_x = random.randint(-cell_width // 4, cell_width // 4)
            offset_y = random.randint(-cell_height // 4, cell_height // 4)
            
            x = max(0, min(cell_center_x + offset_x, self.image_width))
            y = max(zone_y_start, min(cell_center_y + offset_y, zone_y_end))

            # Get asset
            asset = self.asset_library.get_asset(category, tag, intensity)
            if not asset:
                logger.debug(f"No asset for {category}/{tag}, skipping")
                continue

            # No additional scaling - asset files are pre-sized for low/med/high
            # Just use the asset as-is
            asset_scaled = asset

            # Place on canvas (centered at x, y)
            paste_x = max(0, min(x - asset_scaled.width // 2, self.image_width - asset_scaled.width))
            paste_y = max(zone_y_start, min(y - asset_scaled.height // 2, zone_y_end - asset_scaled.height))

            image.paste(asset_scaled, (paste_x, paste_y), asset_scaled)

            # Record hitbox
            hitbox = Hitbox(
                x=paste_x,
                y=paste_y,
                width=asset_scaled.width,
                height=asset_scaled.height,
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