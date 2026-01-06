"""
Signal and related type definitions.

Defines the core data structures for representing news signals
throughout the visualization pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Any
from pydantic import BaseModel, Field

class SignalCategory(str, Enum):
    """Signal categories tracked by the system."""
    
    TRANSPORTATION = "transportation"
    CRIME = "crime"
    FESTIVALS = "festivals"
    SPORTS = "sports"
    ECONOMICS = "economics"
    POLITICS = "politics"
    EMERGENCIES = "emergencies"
    WEATHER_TEMP = "weather_temp"  # Prompt-only (no assets)
    WEATHER_WET = "weather_wet"    # Prompt-only (no assets)


class SignalTag(str, Enum):
    """Signal sentiment/direction tags."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"


class IntensityLevel(str, Enum):
    """Intensity levels for asset selection."""
    
    LOW = "low"    # Intensity: 0.0 - 0.33
    MED = "med"    # Intensity: 0.33 - 0.66
    HIGH = "high"  # Intensity: 0.66 - 1.0

    @classmethod
    def from_intensity(cls, intensity: float) -> "IntensityLevel":
        """
        Convert intensity value to level.
        
        Args:
            intensity: Intensity value [0.0, 1.0]
            
        Returns:
            IntensityLevel enum (LOW, MED, or HIGH)
        """
        if intensity < 0.33:
            return cls.LOW
        elif intensity < 0.66:
            return cls.MED
        else:
            return cls.HIGH


@dataclass
class Signal:
    """Represents a detected signal with its metadata."""
    
    category: SignalCategory   # Signal category (enum)
    tag: SignalTag            # Positive or negative sentiment
    score: float             # Original model score [-1.0, 1.0]
    intensity: IntensityLevel = None  # Intensity level (derived if not provided)
    
    def __post_init__(self):
        """Derive intensity from score if not provided."""
        if self.intensity is None:
            intensity_float = abs(self.score)
            self.intensity = IntensityLevel.from_intensity(intensity_float)
    
    def to_tuple(self) -> Tuple[str, str, str, float]:
        """
        Convert to tuple for backward compatibility.
        
        Returns:
            Tuple of (category, tag, intensity_level, score) as strings/float
        """
        return (
            self.category.value,
            self.tag.value,
            self.intensity.value,
            self.score
        )
    
    @classmethod
    def from_tuple(cls, data: Tuple[str, str, str, float]) -> "Signal":
        """
        Create Signal from tuple representation.
        
        Args:
            data: Tuple of (category, tag, intensity_level, score)
            
        Returns:
            Signal instance
        """
        category_str, tag_str, intensity_str, score = data
        return cls(
            category=SignalCategory(category_str),
            tag=SignalTag(tag_str),
            score=score,
            intensity=IntensityLevel(intensity_str),
        )
    
    @classmethod
    def from_score(
        cls,
        category: SignalCategory,
        score: float
    ) -> "Signal":
        """
        Create Signal from category and score.
        
        Automatically derives tag from score sign and intensity from absolute value.
        
        Args:
            category: SignalCategory enum
            score: Model score [-1.0, 1.0]
            
        Returns:
            Signal instance
        """
        tag = SignalTag.POSITIVE if score >= 0 else SignalTag.NEGATIVE
        
        return cls(
            category=category,
            tag=tag,
            score=score,
        )


class Hitbox(BaseModel):
    """Represents a clickable region in the visualization."""
    
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")
    signal_category: SignalCategory = Field(..., description="Associated signal category")
    signal_tag: SignalTag = Field(..., description="Associated signal tag")
    signal_intensity: IntensityLevel = Field(..., description="Associated intensity level")
    signal_score: float = Field(..., description="Original model score")
    
    class Config:
        """Pydantic config for enum serialization."""
        use_enum_values = True  # Automatically serialize enums to their values
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Note: This method is kept for backward compatibility.
        Pydantic's .dict() or .model_dump() should be preferred.
        
        Returns:
            Dictionary with all hitbox data
        """
        return self.model_dump()
