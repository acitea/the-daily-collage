"""
Hopsworks-integrated ingestion pipeline for The Daily Collage.

This pipeline:
1. Fetches news from GDELT
2. Classifies into signal categories
3. Aggregates into vibe vectors (max-pooling)
4. Pushes to Hopsworks Feature Store
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from pathlib import Path

import polars as pl

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news
from backend.server.services.hopsworks import create_hopsworks_service
from backend.settings import settings

# Try to import fine-tuned classifier (optional)
try:
    from ml.models.inference import get_fine_tuned_classifier
    HAS_FINE_TUNED_MODEL = True
except ImportError:
    HAS_FINE_TUNED_MODEL = False

# Try to import embedding-based labeling (optional)
try:
    from ml.utils.embedding_labeling import classify_article_embedding
    HAS_EMBEDDING_LABELING = True
except ImportError:
    HAS_EMBEDDING_LABELING = False

_fine_tuned_classifier = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Signal categories (9 primary categories as per spec)
SIGNAL_CATEGORIES = [
    "emergencies",
    "crime",
    "festivals",
    "transportation",
    "weather_temp",
    "weather_wet",
    "sports",
    "economics",
    "politics",
]

# Tag vocabulary for each category (used by ML model)
TAG_VOCAB = {
    "emergencies": ["", "fire", "earthquake", "explosion", "evacuation", "accident"],
    "crime": ["", "theft", "assault", "robbery", "police", "vandalism"],
    "festivals": ["", "concert", "celebration", "parade", "crowd", "event"],
    "transportation": ["", "traffic", "accident", "congestion", "delay", "closure"],
    "weather_temp": ["", "hot", "cold", "heatwave", "freeze"],
    "weather_wet": ["", "rain", "snow", "flood", "storm", "drought"],
    "sports": ["", "football", "hockey", "victory", "championship", "game"],
    "economics": ["", "market", "business", "trade", "employment", "inflation"],
    "politics": ["", "election", "protest", "government", "policy", "vote"],
}


def classify_article(
    title: str,
    description: str = "",
    method: str = "auto"
) -> Dict[str, Tuple[float, str]]:
    """
    Classify a single article into signal categories.
    
    Method priority (auto mode):
    1. Fine-tuned BERT model (if available)
    2. Embedding-based classification (if available)
    3. Keyword-based fallback (always available)
    
    Args:
        title: Article title
        description: Article description/body
        method: Classification method:
            - "auto": Try fine-tuned model, then embedding, then keywords
            - "embedding": Use embedding-based (semantic similarity)
            - "keywords": Use keyword matching
            - "ml": Use fine-tuned model only, fall back to keywords if unavailable
        
    Returns:
        Dict mapping category to (score, tag) tuple
        e.g., {"emergencies": (0.8, "fire"), "crime": (0.0, "")}
    """
    # Try fine-tuned model first (if method allows)
    if method in ["auto", "ml"]:
        global _fine_tuned_classifier
        
        if HAS_FINE_TUNED_MODEL:
            try:
                if _fine_tuned_classifier is None:
                    _fine_tuned_classifier = get_fine_tuned_classifier()
                
                result = _fine_tuned_classifier.classify(title, description)
                if result:
                    logger.debug(f"ML Classification: {result}")
                    return result
            except Exception as e:
                logger.warning(f"Fine-tuned model failed: {e}")
                if method == "ml":
                    # If ML method requested but failed, fall back to keywords
                    pass
    
    # Try embedding-based classification (if method allows and available)
    if method in ["auto", "embedding"]:
        if HAS_EMBEDDING_LABELING:
            try:
                result = classify_article_embedding(title, description)
                if result:
                    logger.debug(f"Embedding Classification: {result}")
                    return result
            except Exception as e:
                logger.warning(f"Embedding classification failed: {e}")
    
    # Fallback to keyword-based classification
    text = (title + " " + description).lower()
    signals = {}
    
    # Emergencies
    if any(kw in text for kw in ["fire", "explosion", "earthquake", "evacuation", "brand", "jordbävning"]):
        if "fire" in text or "brand" in text:
            signals["emergencies"] = (0.8, "fire")
        elif "earthquake" in text or "jordbävning" in text:
            signals["emergencies"] = (0.7, "earthquake")
        else:
            signals["emergencies"] = (0.6, "evacuation")
    
    # Crime
    if any(kw in text for kw in ["crime", "theft", "robbery", "assault", "police", "brott", "stöld", "polis"]):
        if "theft" in text or "robbery" in text or "stöld" in text:
            signals["crime"] = (0.6, "theft")
        elif "assault" in text:
            signals["crime"] = (0.7, "assault")
        else:
            signals["crime"] = (0.5, "police")
    
    # Festivals
    if any(kw in text for kw in ["festival", "concert", "celebration", "event", "konsert", "fest", "firande"]):
        if "concert" in text or "konsert" in text:
            signals["festivals"] = (0.7, "concert")
        elif "celebration" in text or "firande" in text:
            signals["festivals"] = (0.8, "celebration")
        else:
            signals["festivals"] = (0.6, "crowd")
    
    # Transportation
    if any(kw in text for kw in ["traffic", "congestion", "accident", "crash", "trafik", "olycka", "kö"]):
        if "traffic" in text or "congestion" in text or "trafik" in text or "kö" in text:
            signals["transportation"] = (0.6, "traffic")
        elif "accident" in text or "crash" in text or "olycka" in text:
            signals["transportation"] = (0.8, "accident")
        else:
            signals["transportation"] = (0.5, "congestion")
    
    # Weather - Temperature
    if any(kw in text for kw in ["hot", "heat", "cold", "freeze", "temperature", "varmt", "kallt", "temperatur"]):
        if "hot" in text or "heat" in text or "varmt" in text:
            signals["weather_temp"] = (0.7, "hot")
        else:
            signals["weather_temp"] = (0.6, "cold")
    
    # Weather - Precipitation
    if any(kw in text for kw in ["rain", "snow", "flood", "storm", "regn", "snö", "översvämning"]):
        if "rain" in text or "regn" in text:
            signals["weather_wet"] = (0.6, "rain")
        elif "snow" in text or "snö" in text:
            signals["weather_wet"] = (0.5, "snow")
        elif "flood" in text or "översvämning" in text:
            signals["weather_wet"] = (0.9, "flood")
        else:
            signals["weather_wet"] = (0.5, "rain")
    
    # Sports
    if any(kw in text for kw in ["sports", "game", "match", "victory", "football", "hockey", "sport", "match", "seger"]):
        if "football" in text or "fotboll" in text:
            signals["sports"] = (0.7, "football")
        elif "hockey" in text:
            signals["sports"] = (0.7, "hockey")
        elif "victory" in text or "seger" in text:
            signals["sports"] = (0.8, "victory")
        else:
            signals["sports"] = (0.6, "football")
    
    # Economics
    if any(kw in text for kw in ["market", "business", "trade", "economy", "economic", "marknad", "företag", "ekonomi"]):
        if "market" in text or "marknad" in text:
            signals["economics"] = (0.6, "market")
        elif "business" in text or "företag" in text:
            signals["economics"] = (0.5, "business")
        else:
            signals["economics"] = (0.5, "trade")
    
    # Politics
    if any(kw in text for kw in ["politics", "election", "government", "protest", "politik", "val", "regering", "protest"]):
        if "protest" in text:
            signals["politics"] = (0.7, "protest")
        elif "election" in text or "val" in text:
            signals["politics"] = (0.6, "election")
        else:
            signals["politics"] = (0.5, "government")
    
    return signals


def aggregate_vibe_vector(
    articles_df: pl.DataFrame,
) -> Dict[str, Tuple[float, str, int]]:
    """
    Aggregate article classifications into a single vibe vector using weighted aggregation.
    
    Instead of max-pooling, this uses a weighted approach where:
    - More articles of a category increase the intensity
    - Higher individual scores boost the aggregated score
    - Formula: weighted_score = (sum of scores * frequency_weight) / num_articles
    
    This ensures that:
    - Many low-intensity signals accumulate to medium intensity
    - Many medium-intensity signals accumulate to high intensity
    - The most common tag is selected
    
    Args:
        articles_df: DataFrame with title and description columns
        
    Returns:
        Dict mapping category to (aggregated_score, dominant_tag, count)
    """
    # Classify all articles
    all_signals: Dict[str, List[Tuple[float, str]]] = {cat: [] for cat in SIGNAL_CATEGORIES}
    
    for row in articles_df.iter_rows(named=True):
        title = row.get("title", "")
        description = row.get("description", "")
        
        article_signals = classify_article(title, description)
        
        for category, (score, tag) in article_signals.items():
            all_signals[category].append((score, tag))
    
    # Aggregate with weighted scoring
    vibe_vector = {}
    
    for category in SIGNAL_CATEGORIES:
        if not all_signals[category]:
            continue
            
        signals = all_signals[category]
        count = len(signals)
        
        # Calculate base score (average of individual scores)
        scores = [s[0] for s in signals]
        base_score = sum(scores) / len(scores)
        
        # Apply frequency multiplier
        # More articles = higher multiplier (caps at 2.0x)
        # Formula: 1 + log10(count) / 2
        # Examples: 1 article = 1.0x, 5 articles = 1.35x, 10 articles = 1.5x, 50 articles = 1.85x
        import math
        frequency_multiplier = min(1.0 + math.log10(count) / 2, 2.0)
        
        # Calculate final weighted score
        weighted_score = min(base_score * frequency_multiplier, 1.0)
        
        # Find dominant tag (most common)
        tags = [s[1] for s in signals]
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        dominant_tag = max(tag_counts.items(), key=lambda x: x[1])[0]
        
        # Only include if weighted score > 0.1 (filter noise)
        if weighted_score > 0.1:
            vibe_vector[category] = (weighted_score, dominant_tag, count)
            
            logger.debug(
                f"{category}: base={base_score:.2f}, count={count}, "
                f"multiplier={frequency_multiplier:.2f}, weighted={weighted_score:.2f}, tag={dominant_tag}"
            )
    
    return vibe_vector


def run_ingestion_pipeline(
    country: str = "sweden",
    max_articles: int = 250,
    store_in_hopsworks: bool = True,
):
    """
    Run the full ingestion pipeline.
    
    1. Fetch news from GDELT
    2. Classify articles individually
    3. Store individual classifications in headline_classifications feature group
    4. Aggregate into vibe vector with weighted scoring
    5. Store aggregated vibe in vibe_vectors feature group
    
    Args:
        country: Country to fetch news from
        max_articles: Max articles to fetch
        store_in_hopsworks: Whether to store in Hopsworks Feature Store
    """
    logger.info(f"Starting ingestion pipeline for {country}")
    
    # Step 1: Fetch news
    logger.info("Fetching news from GDELT...")
    articles_df = fetch_news(country=country, max_articles=max_articles)
    
    if articles_df.is_empty():
        logger.warning("No articles fetched, exiting")
        return
    
    logger.info(f"Fetched {len(articles_df)} articles")
    
    # Step 2: Classify each article individually
    logger.info("Classifying articles individually...")
    headlines = []
    
    for idx, row in enumerate(articles_df.iter_rows(named=True)):
        title = row.get("title", "")
        description = row.get("description", "")
        url = row.get("url", "")
        source = row.get("source", row.get("domain", ""))
        
        # Generate unique article ID
        import hashlib
        article_id = hashlib.sha256(f"{url}{title}".encode()).hexdigest()[:16]
        
        # Classify
        classifications = classify_article(title, description)
        
        headlines.append({
            "article_id": article_id,
            "title": title,
            "url": url,
            "source": source,
            "classifications": classifications,
        })
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Classified {idx + 1}/{len(articles_df)} articles")
    
    logger.info(f"Classified all {len(headlines)} articles")
    
    # Step 3: Aggregate into vibe vector
    logger.info("Aggregating into vibe vector with weighted scoring...")
    vibe_vector = aggregate_vibe_vector(articles_df)
    
    logger.info(f"Generated vibe vector with {len(vibe_vector)} active signals:")
    for category, (score, tag, count) in vibe_vector.items():
        logger.info(f"  {category}: score={score:.2f}, tag={tag}, count={count}")
    
    # Step 4: Store in Hopsworks
    if store_in_hopsworks:
        logger.info("Storing data in Hopsworks...")
        
        hopsworks_service = create_hopsworks_service(
            enabled=settings.hopsworks.enabled,
            api_key=settings.hopsworks.api_key,
            project_name=settings.hopsworks.project_name,
            host=settings.hopsworks.host,
        )
        
        hopsworks_service.connect()
        
        timestamp = datetime.utcnow()
        city = country.title()  # Simple mapping for now
        
        # Store individual headline classifications
        logger.info(f"Storing {len(headlines)} headline classifications...")
        hopsworks_service.store_headline_classifications(
            headlines=headlines,
            city=city,
            timestamp=timestamp,
            fg_name="headline_classifications",
        )
        
        # Store aggregated vibe vector
        logger.info("Storing aggregated vibe vector...")
        hopsworks_service.store_vibe_vector(
            city=city,
            timestamp=timestamp,
            vibe_vector=vibe_vector,
            fg_name=settings.hopsworks.vibe_feature_group,
        )
        
        logger.info(f"Stored all data for {city} at {timestamp}")
    
    logger.info("Ingestion pipeline completed successfully")
    
    return vibe_vector


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run The Daily Collage ingestion pipeline")
    parser.add_argument("--country", type=str, default="sweden", help="Country to fetch news from")
    parser.add_argument("--max-articles", type=int, default=250, help="Max articles to fetch")
    parser.add_argument("--no-hopsworks", action="store_true", help="Skip Hopsworks storage")
    
    args = parser.parse_args()
    
    try:
        vibe_vector = run_ingestion_pipeline(
            country=args.country,
            max_articles=args.max_articles,
            store_in_hopsworks=not args.no_hopsworks,
        )
        
        print("\n=== Vibe Vector ===")
        for category, (score, tag, count) in vibe_vector.items():
            print(f"{category:20s}: {score:.2f} ({tag}) - {count} articles")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        sys.exit(1)
