"""
FastAPI backend server for The Daily Collage.

Provides REST API endpoints for requesting visualizations and retrieving
underlying news articles.
"""

import logging
import sys
from typing import Optional, List, Dict, Set
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, Response, FileResponse

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
    from backend.server.services.hopsworks import get_or_create_hopsworks_service
    from backend.server.services.backfill import trigger_backfill_ingestion
    from backend.settings import settings
    from backend.types import (
        Hitbox,  # Use Hitbox directly (HitboxData removed)
        VibeVectorRequest,
        VisualizationResponse,
        CacheStatusResponse,
        SignalCategory,
    )
except ImportError as e:
    logger.error(f"Import error: {e}")


# Global services
viz_service = None
hopsworks_service = None
active_backfills: Set[str] = set()  # Track cities currently being backfilled


def init_visualization_service():
    """Initialize visualization service on startup."""
    global viz_service, hopsworks_service
    
    # Initialize HopsworksService if enabled (graceful degradation)
    if settings.storage.backend == 'hopsworks':
        try:
            logger.info("Initializing HopsworksService")
            hopsworks_service = get_or_create_hopsworks_service(
                api_key=settings.hopsworks.api_key,
                project_name=settings.hopsworks.project_name,
                host=settings.hopsworks.host,
            )
            logger.info("HopsworksService initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to initialize HopsworksService: {str(e)}. "
                "Endpoints requiring Hopsworks will return 503."
            )
            hopsworks_service = None
    
    # Initialize VisualizationService
    try:
        viz_service = VisualizationService(use_hopsworks=settings.storage.backend == 'hopsworks')
        logger.info("Visualization service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize visualization service: {str(e)}")
        viz_service = None


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
async def create_visualization(
    request: VibeVectorRequest,
    regenerate: bool = Query(False, description="Skip cache and force regeneration"),
):
    """
    Generate visualization from explicit vibe vector.

    Takes real signal scores from ML pipeline and generates layout + polish.

    Args:
        request: VibeVectorRequest with city, vibe_vector, optional timestamp
        regenerate: Force regeneration (ignores cache)

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
            force_regenerate=regenerate,
        )

        # Build response with hitbox objects
        # Hitbox is now a Pydantic model, can be used directly or reconstructed
        hitboxes = metadata.get("hitboxes", [])
        if hitboxes and not isinstance(hitboxes[0], Hitbox):
            hitboxes = [Hitbox(**hb) if isinstance(hb, dict) else hb for hb in hitboxes]

        return VisualizationResponse(
            city=request.city,
            cache_key=metadata["cache_key"],
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


@app.get("/api/vibe/{cache_key}", tags=["Vibe"], response_model=VisualizationResponse)
async def get_vibe(
    cache_key: str,
    background_tasks: BackgroundTasks,
    regenerate: bool = Query(False, description="Force regenerate from data"),
):
    """
    Get vibe visualization for any location/date/time using cache_key.
    
    This is the PRIMARY endpoint for the frontend. The cache_key encodes city/date/time.
    It orchestrates the full data flow:
    1. Try to get from cache using cache_key
    2. If not in cache -> Parse cache_key to get city/timestamp
    3. Query Hopsworks for vibe vector at that timestamp
    4. If None -> Trigger backfill in background, return 503 with Retry-After header
    5. If found -> Fetch headlines, generate visualization, return comprehensive response
    
    Cache key format: {city}_{YYYY-MM-DD}_{HH-HH}
    Example: stockholm_2026-01-03_12-18 (Stockholm, Jan 3 2026, 12pm-6pm window)
    
    Args:
        cache_key: Cache key encoding location/date/time (e.g., "stockholm_2026-01-03_12-18")
        background_tasks: FastAPI background tasks for async backfill
        regenerate: Force regeneration (ignores cache)
    
    Returns:
        VisualizationResponse with image URL, hitboxes, vibe scores, and cache_key
        
    Raises:
        400: Invalid cache_key format
        503: Hopsworks service unavailable or data being backfilled (with Retry-After header)
        404: City not supported
        500: Internal server error
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )
    
    if not hopsworks_service:
        raise HTTPException(
            status_code=503,
            detail="Hopsworks service not available. Cannot retrieve vibe data.",
        )
    
    try:
        # Import VibeHash for parsing cache_key
        from backend.storage.core import VibeHash
        
        # Parse cache_key to extract city and timestamp
        cache_info = VibeHash.extract_info(cache_key)
        if not cache_info:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache_key format: {cache_key}. Expected format: city_YYYY-MM-DD_HH-HH",
            )
        
        city = cache_info["city"].title()  # Normalize to title case
        date = cache_info["date"]
        window = cache_info["window"]
        
        # Reconstruct timestamp from window (use start of window)
        window_start_hour = int(window.split("-")[0])
        timestamp = date.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)
        
        logger.info(f"Parsed cache_key: city={city}, timestamp={timestamp}, window={window}")
        
        # Check if we already have this in cache (unless regenerating)
        if not regenerate:
            storage = viz_service.cache.storage
            metadata = storage.get_metadata(cache_key)
            image_data = storage.get_image(cache_key)
            
            if metadata and image_data:
                logger.info(f"Cache hit for {cache_key}")
                hitboxes = metadata.hitboxes
                if hitboxes and not isinstance(hitboxes[0], Hitbox):
                    hitboxes = [Hitbox(**hb) if isinstance(hb, dict) else hb for hb in hitboxes]
                
                # Retrieve vibe_vector from Hopsworks (not stored in metadata)
                vibe_data = hopsworks_service.get_vibe_vector_at_time(city=city, timestamp=timestamp)
                vibe_vector = {
                    category: score
                    for category, (score, tag, count) in vibe_data.items()
                } if vibe_data else {}
                
                # Construct image URL from cache_key
                image_url = f"/api/visualization/{cache_key}/image"
                
                return VisualizationResponse(
                    city=city,
                    cache_key=cache_key,
                    image_url=image_url,
                    hitboxes=hitboxes,
                    vibe_vector=vibe_vector,
                    cached=True,
                    generated_at=datetime.utcnow().isoformat(),
                )
        
        # Not in cache or regenerating - query Hopsworks for vibe data at this timestamp
        logger.info(f"Querying Hopsworks for vibe vector: {city} at {timestamp}")
        vibe_data = hopsworks_service.get_vibe_vector_at_time(city=city, timestamp=timestamp)
        
        if vibe_data is None:
            # No data available - trigger backfill
            logger.warning(f"No vibe data for {city} at {timestamp}, triggering backfill")
            
            # Map city to country code (hardcoded for supported cities)
            country_map = {
                "stockholm": "sweden",
                "sweden": "sweden",
                "gothenburg": "sweden",
                "malmo": "sweden",
                "malmö": "sweden",
            }
            country = country_map.get(city.lower())
            
            if not country:
                raise HTTPException(
                    status_code=404,
                    detail=f"City '{city}' not supported. Supported cities: {list(country_map.keys())}",
                )
            
            # Check if already backfilling (use cache_key for tracking)
            if cache_key in active_backfills:
                logger.info(f"Backfill already in progress for {cache_key}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Data backfill in progress for {cache_key}. Please retry in 5 minutes.",
                    headers={"Retry-After": "300"},
                )
            
            # Mark as active and queue backfill task
            active_backfills.add(cache_key)
            logger.info(f"Added {cache_key} to active backfills")
            
            background_tasks.add_task(
                trigger_backfill_ingestion,
                city=city,
                country=country,
                hopsworks_service=hopsworks_service,
                active_backfills=active_backfills,
                max_articles=250,
            )
            
            # Return 503 with Retry-After header
            raise HTTPException(
                status_code=503,
                detail=f"No data available for {cache_key}. Backfill triggered. Please retry in 5 minutes.",
                headers={"Retry-After": "300"},
            )
        
        # Convert Hopsworks vibe data format to API format
        # Hopsworks format: {"emergencies": (0.8, "fire", 5), ...}
        # API format: {"emergencies": 0.8, ...}
        vibe_vector = {
            category: score
            for category, (score, tag, count) in vibe_data.items()
        }
        
        logger.info(f"Retrieved vibe vector for {city} at {timestamp}: {len(vibe_vector)} categories")
        
        # Get source articles from Hopsworks (optional, may be empty)
        source_articles = []
        
        try:
            # Attempt to get headlines, but don't fail if unavailable
            source_articles = hopsworks_service.get_headlines_for_city(
                city=city,
                timestamp=timestamp,
            )
            logger.info(f"Retrieved {len(source_articles)} source articles")
        except Exception as e:
            logger.warning(f"Could not retrieve source articles: {e}")
        
        # Generate or get visualization from cache
        image_data, metadata = viz_service.generate_or_get(
            city=city,
            vibe_vector=vibe_vector,
            timestamp=timestamp,
            source_articles=source_articles,
            force_regenerate=regenerate,
        )
        
        # Build response with hitbox objects
        hitboxes = metadata.get("hitboxes", [])
        if hitboxes and not isinstance(hitboxes[0], Hitbox):
            hitboxes = [Hitbox(**hb) if isinstance(hb, dict) else hb for hb in hitboxes]
        
        # Construct image URL from cache_key
        image_url = f"/api/visualization/{metadata['cache_key']}/image"
        
        return VisualizationResponse(
            city=city,
            cache_key=metadata["cache_key"],
            image_url=image_url,
            hitboxes=hitboxes,
            vibe_vector=vibe_vector,
            cached=metadata.get("cached", False),
            generated_at=datetime.utcnow().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vibe for {cache_key}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache-key/{city}", tags=["Utility"])
async def generate_cache_key(
    city: str,
    timestamp: Optional[str] = Query(None, description="ISO format timestamp (defaults to current time)"),
):
    """
    Generate a cache_key for a given city and timestamp.
    
    This is a utility endpoint to help frontends construct valid cache_keys.
    
    Args:
        city: City name (e.g., "stockholm")
        timestamp: Optional ISO format timestamp (defaults to current time)
        
    Returns:
        Dictionary with cache_key and parsed info
        
    Example:
        GET /api/cache-key/stockholm?timestamp=2026-01-03T15:30:00
        Returns: {"cache_key": "stockholm_2026-01-03_12-18", "city": "stockholm", ...}
    """
    try:
        from backend.storage.core import VibeHash
        
        # Parse timestamp or use current
        if timestamp:
            try:
                ts = datetime.fromisoformat(timestamp)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid timestamp format: {timestamp}. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                )
        else:
            ts = datetime.utcnow()
        
        # Generate cache_key
        cache_key = VibeHash.generate(city=city, timestamp=ts)
        
        # Parse it back for validation
        info = VibeHash.extract_info(cache_key)
        
        return JSONResponse(content={
            "cache_key": cache_key,
            "city": info["city"],
            "date": info["date"].strftime("%Y-%m-%d"),
            "window": info["window"],
            "timestamp": ts.isoformat(),
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating cache key: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/{cache_key}/image", tags=["Visualization"])
async def get_visualization_image(
    cache_key: str,
    regenerate: bool = Query(False, description="Force regeneration if not in cache"),
):
    """
    Get the actual image for a cached visualization.
    
    If the image is not found and regenerate=True, attempts to regenerate
    from metadata stored with the cache_key.

    Args:
        cache_key: Key from previous /api/visualization or /api/vibe request
        regenerate: If True and image not found, attempt regeneration

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
        image_data = storage.get_image(cache_key)

        if image_data:
            return Response(content=image_data, media_type="image/png")
        
        # Image not found
        if not regenerate:
            raise HTTPException(
                status_code=404,
                detail=f"Image not found for cache key: {cache_key}. Use ?regenerate=true to generate.",
            )
        
        # Attempt on-demand regeneration
        logger.info(f"Image not found for {cache_key}, attempting regeneration")
        
        # Parse cache_key to get city and timestamp
        from backend.storage.core import VibeHash
        cache_info = VibeHash.extract_info(cache_key)
        
        if not cache_info:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache_key format: {cache_key}",
            )
        
        city = cache_info["city"].title()
        date = cache_info["date"]
        window = cache_info["window"]
        window_start_hour = int(window.split("-")[0])
        timestamp = date.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)
        
        # Get vibe_vector from Hopsworks
        if not hopsworks_service:
            raise HTTPException(
                status_code=503,
                detail="Hopsworks service not available. Cannot retrieve vibe data for regeneration.",
            )
        
        vibe_data = hopsworks_service.get_vibe_vector_at_time(city=city, timestamp=timestamp)
        if not vibe_data:
            raise HTTPException(
                status_code=404,
                detail=f"No vibe data found in Hopsworks for {cache_key}. Cannot regenerate.",
            )
        
        vibe_vector = {
            category: score
            for category, (score, tag, count) in vibe_data.items()
        }
        
        # Get source articles
        source_articles = []
        try:
            source_articles = hopsworks_service.get_headlines_for_city(city=city, timestamp=timestamp)
        except Exception as e:
            logger.warning(f"Could not retrieve source articles: {e}")
        
        # Regenerate visualization
        image_data, new_metadata = viz_service.generate_or_get(
            city=city,
            vibe_vector=vibe_vector,
            timestamp=timestamp,
            source_articles=source_articles,
            force_regenerate=True,
        )
        
        logger.info(f"Successfully regenerated image for {cache_key}")
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

        from backend.storage import VibeHash
        cache_key = VibeHash.generate(city, timestamp)

        return CacheStatusResponse(
            cache_key=cache_key,
            cached=cached,
            timestamp=timestamp.isoformat(),
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid vibe_vector JSON")
    except Exception as e:
        logger.error(f"Error checking cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hitboxes/{cache_key}", tags=["Metadata"])
async def get_hitboxes(cache_key: str):
    """
    Get hitbox metadata for a cached visualization.

    These are the clickable regions in the image.

    Args:
        cache_key: Key from /api/visualization request

    Returns:
        List of hitboxes with signal info
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        storage = viz_service.cache.storage
        metadata = storage.get_metadata(cache_key)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for cache key: {cache_key}",
            )

        return JSONResponse(content={"hitboxes": metadata.hitboxes})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching hitboxes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/{cache_key}", tags=["Articles"])
async def get_articles_for_vibe(cache_key: str):
    """
    Get source articles that contributed to a visualization.
    
    Articles are retrieved from Hopsworks feature store, not from cached metadata.

    Args:
        cache_key: Cache key for the visualization

    Returns:
        List of articles with metadata and signal associations
    """
    if not hopsworks_service:
        raise HTTPException(
            status_code=503,
            detail="Hopsworks service not available. Cannot retrieve articles.",
        )

    try:
        # Parse cache_key to extract city and timestamp
        from backend.storage.core import VibeHash
        cache_info = VibeHash.extract_info(cache_key)
        
        if not cache_info:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache_key format: {cache_key}",
            )
        
        city = cache_info["city"].title()
        date = cache_info["date"]
        window = cache_info["window"]
        window_start_hour = int(window.split("-")[0])
        timestamp = date.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)
        
        # Retrieve headlines from Hopsworks
        articles = hopsworks_service.get_headlines_for_city(
            city=city,
            timestamp=timestamp,
        )
        
        return JSONResponse(
            content={
                "cache_key": cache_key,
                "city": city,
                "timestamp": timestamp.isoformat(),
                "articles": articles,
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
    # Use centralized enum to ensure consistency
    category_metadata = {
        SignalCategory.TRANSPORTATION: {
            "name": "Transportation",
            "description": "Traffic, congestion, accidents",
        },
        SignalCategory.WEATHER_TEMP: {
            "name": "Weather - Temperature",
            "description": "Hot, cold, extreme temperatures",
        },
        SignalCategory.WEATHER_WET: {
            "name": "Weather - Precipitation",
            "description": "Rain, snow, flooding",
        },
        SignalCategory.CRIME: {
            "name": "Crime & Safety",
            "description": "Theft, assault, police activity",
        },
        SignalCategory.FESTIVALS: {
            "name": "Festivals & Events",
            "description": "Concerts, celebrations, gatherings",
        },
        SignalCategory.SPORTS: {
            "name": "Sports",
            "description": "Games, victories, sporting events",
        },
        SignalCategory.EMERGENCIES: {
            "name": "Emergencies",
            "description": "Fires, earthquakes, evacuations",
        },
        SignalCategory.ECONOMICS: {
            "name": "Economics",
            "description": "Market news, business developments",
        },
        SignalCategory.POLITICS: {
            "name": "Politics",
            "description": "Elections, protests, government",
        },
    }
    
    categories = [
        {
            "id": category.value,
            "name": metadata["name"],
            "description": metadata["description"],
        }
        for category, metadata in category_metadata.items()
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

