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

    Maps signal category + tag + intensity level to appropriate PNG files.
    Tags are "positive" or "negative" based on score sign.
    Intensity levels are "low", "med", or "high".
    
    Asset filename pattern: <category>_<tag>_<level>.png
    Example: transportation_positive_high.png
    """

    # Mapping of (category, tag, intensity_level) -> asset filename
    # tag: "positive" or "negative"
    # intensity_level: "low" (0.0-0.33), "med" (0.33-0.66), "high" (0.66-1.0)
    ASSET_MAP = {
        # Transportation
        ("transportation", "positive", "low"): "transportation_positive_low.png",
        ("transportation", "positive", "med"): "transportation_positive_med.png",
        ("transportation", "positive", "high"): "transportation_positive_high.png",
        ("transportation", "negative", "low"): "transportation_negative_low.png",
        ("transportation", "negative", "med"): "transportation_negative_med.png",
        ("transportation", "negative", "high"): "transportation_negative_high.png",
        # Crime
        ("crime", "positive", "low"): "crime_positive_low.png",
        ("crime", "positive", "med"): "crime_positive_med.png",
        ("crime", "positive", "high"): "crime_positive_high.png",
        ("crime", "negative", "low"): "crime_negative_low.png",
        ("crime", "negative", "med"): "crime_negative_med.png",
        ("crime", "negative", "high"): "crime_negative_high.png",
        # Festivals
        ("festivals", "positive", "low"): "festivals_positive_low.png",
        ("festivals", "positive", "med"): "festivals_positive_med.png",
        ("festivals", "positive", "high"): "festivals_positive_high.png",
        ("festivals", "negative", "low"): "festivals_negative_low.png",
        ("festivals", "negative", "med"): "festivals_negative_med.png",
        ("festivals", "negative", "high"): "festivals_negative_high.png",
        # Sports
        ("sports", "positive", "low"): "sports_positive_low.png",
        ("sports", "positive", "med"): "sports_positive_med.png",
        ("sports", "positive", "high"): "sports_positive_high.png",
        ("sports", "negative", "low"): "sports_negative_low.png",
        ("sports", "negative", "med"): "sports_negative_med.png",
        ("sports", "negative", "high"): "sports_negative_high.png",
        # Economics
        ("economics", "positive", "low"): "economics_positive_low.png",
        ("economics", "positive", "med"): "economics_positive_med.png",
        ("economics", "positive", "high"): "economics_positive_high.png",
        ("economics", "negative", "low"): "economics_negative_low.png",
        ("economics", "negative", "med"): "economics_negative_med.png",
        ("economics", "negative", "high"): "economics_negative_high.png",
        # Politics
        ("politics", "positive", "low"): "politics_positive_low.png",
        ("politics", "positive", "med"): "politics_positive_med.png",
        ("politics", "positive", "high"): "politics_positive_high.png",
        ("politics", "negative", "low"): "politics_negative_low.png",
        ("politics", "negative", "med"): "politics_negative_med.png",
        ("politics", "negative", "high"): "politics_negative_high.png",
        # Emergencies
        ("emergencies", "positive", "low"): "emergencies_positive_low.png",
        ("emergencies", "positive", "med"): "emergencies_positive_med.png",
        ("emergencies", "positive", "high"): "emergencies_positive_high.png",
        ("emergencies", "negative", "low"): "emergencies_negative_low.png",
        ("emergencies", "negative", "med"): "emergencies_negative_med.png",
        ("emergencies", "negative", "high"): "emergencies_negative_high.png",
    }
    
    # Generic fallbacks by category and intensity level
    # Uses "generic" tag with the same intensity level
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
        intensity: str,
        use_fallback: bool = True,
    ) -> Optional[Image.Image]:
        """
        Get asset image for a category + tag + intensity combination.

        Tag should be "positive" or "negative".
        Falls back through: exact match → category+tag → ultimate fallback.

        Args:
            category: Signal category (e.g., 'transportation', 'crime')
            tag: Signal tag ("positive" or "negative")
            intensity: Intensity level ("low", "med", "high")
            use_fallback: Whether to use fallback assets if specific not found

        Returns:
            PIL Image or None if not found and fallback disabled
        """

        # Try exact match first (category, tag, intensity_level)
        asset_key = (category, tag, intensity)
        filename = self.ASSET_MAP.get(asset_key)

        if not filename and use_fallback:
            # Fall back to generic tag with same intensity level
            fallback_key = (category, intensity)
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
        self, category: str, tag: str, intensity: str
    ) -> Optional[Tuple[int, int]]:
        """
        Get (width, height) of asset image.

        Args:
            category: Signal category
            tag: Signal tag
            intensity: Intensity level ("low", "med", "high")

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
            image_width: Default canvas width (used if base_scenery.png not found)
            image_height: Default canvas height (used if base_scenery.png not found)
            assets_dir: Path to assets directory
            bg_color: Background color (RGB) - used as fallback if base_scenery.png not found
            sky_zone_height: Height of sky zone as fraction of total height
            city_zone_height: Height of city zone as fraction of total height
            street_zone_height: Height of street zone as fraction of total height
        """
        self.assets_dir = Path(assets_dir)
        self.bg_color = bg_color
        self.asset_library = AssetLibrary(assets_dir)
        self.hitboxes: List[Hitbox] = []
        
        # Load base scenery image and get dimensions
        base_scenery_path = self.assets_dir / "base_scenery.png"
        if base_scenery_path.exists():
            try:
                self.base_scenery = Image.open(base_scenery_path).convert("RGBA")
                self.image_width, self.image_height = self.base_scenery.size
                logger.info(f"Loaded base scenery: {self.image_width}x{self.image_height}")
            except Exception as e:
                logger.error(f"Failed to load base_scenery.png: {e}")
                logger.info(f"Using default dimensions: {image_width}x{image_height}")
                self.base_scenery = None
                self.image_width = image_width
                self.image_height = image_height
        else:
            logger.warning(f"base_scenery.png not found at {base_scenery_path}")
            logger.info(f"Using default dimensions: {image_width}x{image_height}")
            self.base_scenery = None
            self.image_width = image_width
            self.image_height = image_height
        
        # Compute zones dynamically from settings and actual image dimensions
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
        signals: List[Tuple[str, str, str, float]],
    ) -> Tuple[Image.Image, List[Hitbox]]:
        """
        Compose image with signals placed in zones.

        Note: Atmosphere is handled via prompting only (no asset overlays).

        Args:
            signals: List of (category, tag, intensity, score) tuples
                     category: Signal category (e.g., 'transportation', 'crime')
                     tag: Specific event tag (e.g., 'traffic', 'theft')
                     intensity: Intensity level ("low", "med", "high") for asset selection
                     score: Original -1.0 to 1.0 score from model

        Returns:
            Tuple[Image, hitboxes]: Composed PIL Image and hitbox list
        """
        # Create canvas - use base scenery if available, otherwise blank
        if self.base_scenery:
            image = self.base_scenery.copy()
        else:
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
            # Skip weather categories - they are handled via prompts only
            if category in ("weather_temp", "weather_wet"):
                continue
            # Route by category
            elif category in ("crime", "emergencies", "transportation"):
                zones["street"].append((category, tag, intensity, score))
            else:  # festivals, sports, economics, politics
                zones["city"].append((category, tag, intensity, score))

        return zones

    def _place_signals_in_zone(
        self,
        image: Image.Image,
        zone_name: str,
        signals: List[Tuple[str, str, str, float]],
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

            # Get asset (pre-sized based on intensity level)
            asset = self.asset_library.get_asset(category, tag, intensity)
            if not asset:
                logger.debug(f"No asset for {category}/{tag}, skipping")
                continue

            # Place on canvas (centered at x, y)
            paste_x = max(0, min(x - asset.width // 2, self.image_width - asset.width))
            paste_y = max(zone_y_start, min(y - asset.height // 2, zone_y_end - asset.height))

            image.paste(asset, (paste_x, paste_y), asset)

            # Record hitbox
            hitbox = Hitbox(
                x=paste_x,
                y=paste_y,
                width=asset.width,
                height=asset.height,
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