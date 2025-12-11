"""
FastAPI backend server for The Daily Collage.

Provides REST API endpoints for requesting visualizations and retrieving
underlying news articles.
"""

import logging
import sys
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    version="0.1.0",
)

# Import modules (these imports will work once dependencies are installed)
try:
    from visualization.composition import (
        VisualizationService,
        SignalIntensity,
    )
    from utils.classification import classify_articles
    from ingestion.script import get_news_for_location, normalize_country_input
except ImportError:
    logger.warning(
        "visualization module not available - running in stub mode"
    )


# Pydantic models for API requests/responses
class SignalData(BaseModel):
    """Signal data in API response."""

    name: str
    intensity: float


class VisualizationResponse(BaseModel):
    """Response for visualization request."""

    location: str
    generated_at: str
    signal_count: int
    signals: List[SignalData]
    image_url: str
    cached: bool


class ArticleMetadata(BaseModel):
    """Metadata for a source article."""

    title: str
    url: str
    source: str
    date: Optional[str] = None


class LocationStatusResponse(BaseModel):
    """Response for location status request."""

    location: str
    status: str
    last_updated: Optional[str] = None
    message: str


# Global visualization service
viz_service = None


def init_visualization_service():
    """Initialize visualization service on startup."""
    global viz_service
    try:
        viz_service = VisualizationService()
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
        "version": "0.1.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/visualization", tags=["Visualization"])
async def get_visualization(
    location: str = Query(
        "sweden", description="Location name (e.g., 'sweden', 'stockholm')"
    ),
    force_regenerate: bool = Query(
        False,
        description="If true, skip cache and regenerate visualization",
    ),
) -> VisualizationResponse:
    """
    Get a visualization for a location.

    Args:
        location: Geographic location name
        force_regenerate: Force regeneration instead of using cache

    Returns:
        VisualizationResponse: Generated visualization with metadata
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Placeholder: In production, would fetch actual signal data from ingestion + classification
        # For now, return mock data to demonstrate API structure
        mock_signals = [
            SignalIntensity("weather", 65.0),
            SignalIntensity("traffic", 45.0),
            SignalIntensity("politics", 75.0),
        ]

        image_data, metadata = viz_service.generate_or_get(
            mock_signals,
            location=location,
            force_regenerate=force_regenerate,
        )

        return VisualizationResponse(
            location=location,
            generated_at=datetime.utcnow().isoformat(),
            signal_count=len(mock_signals),
            signals=[SignalData(name=s.signal_name, intensity=s.intensity) for s in mock_signals],
            image_url=f"/api/visualization/{location}/image",
            cached=not force_regenerate,
        )

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/{location}/image", tags=["Visualization"])
async def get_visualization_image(location: str):
    """
    Get the actual image data for a visualization.

    Args:
        location: Geographic location

    Returns:
        PNG image data
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Generate mock signals (in production, would fetch from ingestion)
        signals = [
            SignalIntensity("weather", 65),
            SignalIntensity("traffic", 45),
            SignalIntensity("politics", 75),
        ]

        # Get or generate visualization
        image_data, _ = viz_service.generate_or_get(signals, location)

        # Return PNG image
        from fastapi.responses import Response
        return Response(content=image_data, media_type="image/png")

    except Exception as e:
        logger.error(f"Error fetching image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles", tags=["Articles"])
async def get_articles_for_location(
    location: str = Query("sweden", description="Location name"),
    signal: Optional[str] = Query(
        None, description="Filter articles by signal category"
    ),
) -> JSONResponse:
    """
    Get source articles that contributed to a visualization.

    Args:
        location: Geographic location
        signal: Optional signal category filter

    Returns:
        List of articles with metadata
    """
    # Placeholder: In production, would fetch actual articles from database
    mock_articles = [
        {
            "title": "Heavy traffic on Stockholm ring road",
            "url": "https://example.com/article1",
            "source": "SVT Nyheter",
            "date": "2025-12-11T14:30:00Z",
            "signal": "traffic",
        },
        {
            "title": "Rainstorm expected in southern Sweden",
            "url": "https://example.com/article2",
            "source": "SMHI",
            "date": "2025-12-11T10:15:00Z",
            "signal": "weather",
        },
    ]

    if signal:
        mock_articles = [a for a in mock_articles if a["signal"] == signal]

    return JSONResponse(content={"location": location, "articles": mock_articles})


@app.get("/api/visualization/gdelt/{location}", tags=["Visualization"])
async def get_visualization_from_gdelt(
    location: str = "sweden",
    force_regenerate: bool = Query(
        False, description="Force regeneration instead of using cache"
    ),
):
    """
    Get a visualization based on real GDELT news data.

    This endpoint fetches actual news articles from GDELT, classifies them,
    and generates a visualization based on real data.

    Args:
        location: Geographic location (e.g., 'sweden', 'stockholm')
        force_regenerate: Force regeneration instead of using cache

    Returns:
        JSON with visualization metadata and image URL
    """
    if not viz_service:
        raise HTTPException(
            status_code=503, detail="Visualization service not initialized"
        )

    try:
        # Normalize location to country code
        try:
            country_code = normalize_country_input(location)
        except ValueError:
            # Try as-is if not a known country
            country_code = location.upper()[:2]

        # Fetch real news from GDELT
        logger.info(f"Fetching GDELT news for {location} ({country_code})")
        articles_df = get_news_for_location(country_code, max_articles=100)

        if len(articles_df) == 0:
            logger.warning(f"No GDELT articles found for {location}")
            raise HTTPException(
                status_code=404,
                detail=f"No news found for location: {location}",
            )

        # Classify articles into signals
        logger.info(f"Classifying {len(articles_df)} articles")
        classified = classify_articles(articles_df)

        # Aggregate signals
        signal_dict = {}
        for article in classified:
            if article.primary_signal:
                signal = article.primary_signal.value
                intensity = article.signals[0].intensity if article.signals else 0
                if signal not in signal_dict:
                    signal_dict[signal] = []
                signal_dict[signal].append(intensity)

        # Convert to SignalIntensity objects
        signals_for_viz = []
        for signal_name in sorted(
            signal_dict.keys(),
            key=lambda s: sum(signal_dict[s]) / len(signal_dict[s]),
            reverse=True,
        ):
            intensities = signal_dict[signal_name]
            avg_intensity = sum(intensities) / len(intensities)
            signals_for_viz.append(SignalIntensity(signal_name, avg_intensity))

        # Generate visualization
        image_data, metadata = viz_service.generate_or_get(
            signals_for_viz,
            location=location,
            force_regenerate=force_regenerate,
        )

        return VisualizationResponse(
            location=location,
            generated_at=datetime.utcnow().isoformat(),
            signal_count=len(signals_for_viz),
            signals=[
                SignalData(name=s.signal_name, intensity=s.intensity)
                for s in signals_for_viz
            ],
            image_url=f"/api/visualization/{location}/image",
            cached=not force_regenerate,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching GDELT data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing GDELT data: {str(e)}",
        )


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
            "id": "traffic",
            "name": "Traffic & Transportation",
            "description": "Road congestion, public transit disruptions",
            "icon": "🚗",
        },
        {
            "id": "weather",
            "name": "Weather Events",
            "description": "Storms, heatwaves, snow, flooding",
            "icon": "🌧️",
        },
        {
            "id": "crime",
            "name": "Crime & Safety",
            "description": "Incidents, police activity, emergency services",
            "icon": "🚨",
        },
        {
            "id": "festivals",
            "name": "Festivals & Events",
            "description": "Cultural celebrations, concerts, public gatherings",
            "icon": "🎉",
        },
        {
            "id": "politics",
            "name": "Politics",
            "description": "Elections, protests, government announcements",
            "icon": "🏛️",
        },
        {
            "id": "sports",
            "name": "Sports",
            "description": "Major games, victories, sporting events",
            "icon": "⚽",
        },
        {
            "id": "accidents",
            "name": "Accidents & Emergencies",
            "description": "Fires, industrial accidents, medical emergencies",
            "icon": "🔥",
        },
        {
            "id": "economic",
            "name": "Economic",
            "description": "Market news, business developments, employment",
            "icon": "💼",
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
        stats = viz_service.cache.get_stats()
        return JSONResponse(content={"cache": stats})
    except Exception as e:
        logger.error(f"Error fetching cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

