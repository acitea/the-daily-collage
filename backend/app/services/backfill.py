"""
Backfill service for triggering ingestion when Hopsworks has no data.

Uses FastAPI BackgroundTasks for async execution. Reusable functions
can be called from CRON jobs or API endpoints.
"""

import logging
from typing import Set

logger = logging.getLogger(__name__)


def trigger_backfill_ingestion(
    city: str,
    country: str,
    active_backfills: Set[str],
    max_articles: int = 250,
):
    """
    Background task to run ingestion pipeline when data is missing.
    
    This function is meant to run in a BackgroundTask or CRON job. It will:
    1. Fetch news from GDELT
    2. Classify articles using the ML model
    3. Store results to Hopsworks Feature Store
    
    Args:
        city: City name for logging and tracking
        country: Country code for GDELT (e.g., "sweden")
        hopsworks_service: HopsworksService instance
        active_backfills: Set to track in-flight backfills (prevents duplicates)
        max_articles: Max articles to fetch from GDELT
    """
    logger.info(f"🔄 Starting backfill for {city} (country={country})")
    
    try:
        # Import here to avoid circular dependencies
        from ml.ingestion.hopsworks_pipeline import run_ingestion_pipeline
        
        # Run the full ingestion pipeline
        # This will fetch GDELT articles, classify them, and store in Hopsworks
        vibe_vector = run_ingestion_pipeline(
            country=country,
            max_articles=max_articles,
            store_in_hopsworks=True,
        )
        
        if vibe_vector:
            logger.info(f"✅ Backfill completed for {city}: {len(vibe_vector)} signals detected")
        else:
            logger.warning(f"⚠️ Backfill completed for {city} but no vibe vector returned")
        
    except Exception as e:
        logger.error(f"❌ Backfill failed for {city}: {str(e)}", exc_info=True)
    
    finally:
        # Remove from active set regardless of success/failure
        if city in active_backfills:
            active_backfills.remove(city)
            logger.debug(f"Removed {city} from active backfills")
