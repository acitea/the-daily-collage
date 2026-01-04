"""
API request/response models for FastAPI endpoints.

Pydantic models for HTTP request/response serialization.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel
from .signal import Hitbox  # Import Hitbox directly (HitboxData removed)


class VibeVectorRequest(BaseModel):
    """Request body for explicit vibe vector visualization."""

    city: str
    vibe_vector: Dict[str, float]  # category -> score (-1.0 to 1.0)
    timestamp: Optional[str] = None  # ISO format datetime
    source_articles: Optional[List[Dict]] = None  # Optional article metadata


class VisualizationResponse(BaseModel):
    """Response for visualization request."""

    city: str
    cache_key: str
    image_url: str
    hitboxes: List[Hitbox]  # Now using Hitbox directly
    vibe_vector: Dict[str, float]
    cached: bool
    generated_at: str


class CacheStatusResponse(BaseModel):
    """Response for cache status check."""

    cache_key: str
    cached: bool
    timestamp: str


class BackfillRequest(BaseModel):
    """Request body for triggering backfill ingestion."""

    city: str
    start_date: str  # ISO format date
    end_date: str    # ISO format date
    lookback_hours: Optional[int] = 24


class BackfillStatusResponse(BaseModel):
    """Response for backfill status check."""

    city: str
    status: str  # "running", "completed", "failed", "not_found"
    message: str


class SignalCategoriesResponse(BaseModel):
    """Response listing all available signal categories."""

    categories: List[str]
    count: int


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    service: str
