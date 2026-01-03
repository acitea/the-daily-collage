"""
FastAPI backend server for The Daily Collage.

Provides REST API endpoints for requesting visualizations and retrieving
underlying news articles.
"""

import logging
import sys
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on the Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialize FastAPI app
app = FastAPI(
    title="The Daily Collage API",
    description="REST API for generating location-based news visualizations",
    version="0.2.0",
)

# Import modules
try:
    from backend.visualization.composition import VisualizationService
    from backend.server.services.hopsworks import create_hopsworks_service
    from backend.settings import settings
except ImportError as e:
    logger.error(f"Import error: {e}")


# Pydantic models for API requests/responses
class HitboxData(BaseModel):
    """Hitbox metadata for interactive elements."""

    x: int
    y: int
    width: int
    height: int
    signal_category: str
    signal_tag: str
    signal_intensity: float
    signal_score: float


class VibeVectorRequest(BaseModel):
    """Request body for explicit vibe vector visualization."""

    city: str
    vibe_vector: Dict[str, float]  # category -> score (-1.0 to 1.0)
    timestamp: Optional[str] = None  # ISO format datetime
    source_articles: Optional[List[Dict]] = None  # Optional article metadata


class VisualizationResponse(BaseModel):
    """Response for visualization request."""

    city: str
    vibe_hash: str
    image_url: str
    hitboxes: List[HitboxData]
    vibe_vector: Dict[str, float]
    cached: bool
    generated_at: str


class CacheStatusResponse(BaseModel):
    """Response for cache status."""

    vibe_hash: str
    cached: bool
    timestamp: str


# Global visualization service
viz_service = None


def init_visualization_service():
    """Initialize visualization service on startup."""
    global viz_service
    try:
        # Initialize HopsworksService if using hopsworks backend
        hopsworks_service = None
        if settings.storage.backend == "hopsworks":
            logger.info("Initializing HopsworksService for storage backend")
            hopsworks_service = create_hopsworks_service(
                enabled=settings.hopsworks.enabled,
                api_key=settings.hopsworks.api_key,
                project_name=settings.hopsworks.project_name,
                host=settings.hopsworks.host,
            )
        
        viz_service = VisualizationService(hopsworks_service=hopsworks_service)
        logger.info("Visualization service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize visualization service: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event handler."""
    logger.info("Starting The Daily Collage API server")
    init_visualization_service()

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event handler."""
    logger.info("Shutting down The Daily Collage API server")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "name": "The Daily Collage API",
        "status": "operational",
        "version": "0.2.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "visualization" if viz_service else "initializing",
    }


@app.post("/api/visualization", tags=["Visualization"], response_model=VisualizationResponse)
async def create_visualization(request: VibeVectorRequest):
    """
    Generate visualization from explicit vibe vector.

    Takes real signal scores from ML pipeline and generates layout + polish.

    Args:
        request: VibeVectorRequest with city, vibe_vector, optional timestamp

    Returns:
        VisualizationResponse with image URL and hitboxes
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Parse timestamp if provided
        timestamp = None
        if request.timestamp:
            try:
                timestamp = datetime.fromisoformat(request.timestamp)
            except ValueError:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        # Generate or fetch from cache
        image_data, metadata = viz_service.generate_or_get(
            city=request.city,
            vibe_vector=request.vibe_vector,
            timestamp=timestamp,
            source_articles=request.source_articles,
        )

        # Build response with hitbox objects
        hitboxes = [HitboxData(**hb) for hb in metadata.get("hitboxes", [])]

        return VisualizationResponse(
            city=request.city,
            vibe_hash=metadata["vibe_hash"],
            image_url=metadata["image_url"],
            hitboxes=hitboxes,
            vibe_vector=request.vibe_vector,
            cached=metadata["cached"],
            generated_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/{vibe_hash}/image", tags=["Visualization"])
async def get_visualization_image(vibe_hash: str):
    """
    Get the actual image for a cached visualization.

    Args:
        vibe_hash: Hash from previous /api/visualization request

    Returns:
        PNG image data
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Try to retrieve from storage
        storage = viz_service.cache.storage
        image_data = storage.get_image(vibe_hash)

        if not image_data:
            raise HTTPException(
                status_code=404,
                detail=f"Image not found for vibe hash: {vibe_hash}",
            )

        return Response(content=image_data, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/status", tags=["Cache"])
async def check_cache_status(
    city: str = Query(..., description="City name"),
    vibe_vector: Optional[str] = Query(None, description="JSON-encoded vibe vector"),
) -> CacheStatusResponse:
    """
    Check if a vibe is cached.

    Args:
        city: Geographic location
        vibe_vector: JSON-encoded vibe vector dict

    Returns:
        Cache status with vibe hash
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        import json

        if not vibe_vector:
            raise HTTPException(status_code=400, detail="vibe_vector required")

        vector = json.loads(vibe_vector)
        timestamp = datetime.utcnow()

        cached = viz_service.cache.exists(city, timestamp)

        from backend.visualization.caching import VibeHash

        vibe_hash = VibeHash.generate(city, timestamp)

        return CacheStatusResponse(
            vibe_hash=vibe_hash,
            cached=cached,
            timestamp=timestamp.isoformat(),
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid vibe_vector JSON")
    except Exception as e:
        logger.error(f"Error checking cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hitboxes/{vibe_hash}", tags=["Metadata"])
async def get_hitboxes(vibe_hash: str):
    """
    Get hitbox metadata for a cached visualization.

    These are the clickable regions in the image.

    Args:
        vibe_hash: Hash from /api/visualization request

    Returns:
        List of hitboxes with signal info
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        storage = viz_service.cache.storage
        metadata = storage.get_metadata(vibe_hash)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for vibe hash: {vibe_hash}",
            )

        return JSONResponse(content={"hitboxes": metadata.hitboxes})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching hitboxes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/{vibe_hash}", tags=["Articles"])
async def get_articles_for_vibe(vibe_hash: str):
    """
    Get source articles that contributed to a visualization.

    Args:
        vibe_hash: Hash from /api/visualization request

    Returns:
        List of articles with metadata and signal associations
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        storage = viz_service.cache.storage
        metadata = storage.get_metadata(vibe_hash)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for vibe hash: {vibe_hash}",
            )

        return JSONResponse(
            content={
                "vibe_hash": vibe_hash,
                "city": metadata.city,
                "articles": metadata.source_articles,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/supported-locations", tags=["Metadata"])
async def get_supported_locations():
    """Get list of currently supported geographic locations."""
    supported = [
        {"code": "se", "name": "Sweden", "type": "country"},
        {"code": "stockholm", "name": "Stockholm", "type": "city"},
        {"code": "gothenburg", "name": "Gothenburg", "type": "city"},
        {"code": "malmo", "name": "Malmö", "type": "city"},
    ]
    return JSONResponse(content={"locations": supported})


@app.get("/api/signal-categories", tags=["Metadata"])
async def get_signal_categories():
    """Get list of all signal categories used for classification."""
    categories = [
        {
            "id": "transportation",
            "name": "Transportation",
            "description": "Traffic, congestion, accidents",
        },
        {
            "id": "weather_temp",
            "name": "Weather - Temperature",
            "description": "Hot, cold, extreme temperatures",
        },
        {
            "id": "weather_wet",
            "name": "Weather - Precipitation",
            "description": "Rain, snow, flooding",
        },
        {
            "id": "crime",
            "name": "Crime & Safety",
            "description": "Theft, assault, police activity",
        },
        {
            "id": "festivals",
            "name": "Festivals & Events",
            "description": "Concerts, celebrations, gatherings",
        },
        {
            "id": "sports",
            "name": "Sports",
            "description": "Games, victories, sporting events",
        },
        {
            "id": "emergencies",
            "name": "Emergencies",
            "description": "Fires, earthquakes, evacuations",
        },
        {
            "id": "economics",
            "name": "Economics",
            "description": "Market news, business developments",
        },
        {
            "id": "politics",
            "name": "Politics",
            "description": "Elections, protests, government",
        },
    ]
    return JSONResponse(content={"categories": categories})


@app.get("/api/cache-stats", tags=["Monitoring"])
async def get_cache_stats():
    """Get visualization cache statistics."""
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Storage backend stats (if available)
        backend_type = settings.storage.backend
        stats = {
            "storage_backend": backend_type,
            "deprecated_cache_stats": viz_service.deprecated_cache.get_stats(),
        }
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error fetching cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server")
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
    )

