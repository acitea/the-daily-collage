"""
Fetch recent headlines from GDELT and store them in the Hopsworks headline feature group.

Intended to run as the first stage of the 6-hour pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import polars as pl
from gdeltdoc import GdeltDoc, Filters

CURRENT_DIR = Path(__file__).resolve()
BACKEND_ROOT = CURRENT_DIR.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from storage.core import VibeHash
from jobs.utils import (
    ensure_backend_path,
    make_article_id,
    parse_window_start,
    build_window_datetimes,
)

# Ensure local imports resolve when run as a script
ensure_backend_path()

from app.services.hopsworks import get_or_create_hopsworks_service
from settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


SUPPORTED_COUNTRIES = {
    "sweden": "SW",
    "se": "SW",
    "united_states": "US",
    "us": "US",
}


def normalize_country_input(country_input: str) -> str:
    """Normalize country input to a FIPS code."""
    normalized = country_input.lower().strip()
    if normalized not in SUPPORTED_COUNTRIES:
        supported = ", ".join(SUPPORTED_COUNTRIES.keys())
        raise ValueError(f"Unsupported country: {country_input}. Supported: {supported}")
    return SUPPORTED_COUNTRIES[normalized]


def fetch_recent_articles_range(country: str, start_dt, end_dt, max_articles: int = 250) -> pl.DataFrame:
    """Fetch recent news for a country using GDELT within a start/end datetime range (UTC)."""
    fips_code = normalize_country_input(country)
    gd = GdeltDoc()
    # Use explicit start/end to align with VibeHash window
    filters = Filters(country=fips_code, num_records=max_articles, start_date=start_dt, end_date=end_dt)
    articles_pd = gd.article_search(filters)
    if articles_pd is None or articles_pd.empty:
        return pl.DataFrame()
    return pl.from_pandas(articles_pd)


def to_headline_rows(df: pl.DataFrame) -> List[Dict]:
    """Convert raw articles into headline records expected by Hopsworks storage."""
    rows: List[Dict] = []
    for article in df.iter_rows(named=True):
        title = (article.get("title") or "").strip()
        url = (article.get("url") or "").strip()
        description = (article.get("description") or article.get("excerpt") or "").strip()
        source = (article.get("source") or article.get("domain") or "").strip()
        published_at = article.get("date")
        try:
            # Try to parse to datetime if it's a string
            if isinstance(published_at, str):
                from datetime import datetime as _dt
                published_at = _dt.fromisoformat(published_at.replace("Z", ""))
        except Exception:
            published_at = None

        rows.append(
            {
                "article_id": make_article_id(url, title),
                "title": title,
                "description": description,
                "url": url,
                "source": source,
                "published_at": published_at,
                "classifications": {},  # Filled in later stages
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch headlines and store them in Hopsworks")
    parser.add_argument("--country", type=str, default="sweden", help="Country name or FIPS code")
    parser.add_argument("--city", type=str, default="stockholm", help="City/region label for storage")
    parser.add_argument("--max-articles", type=int, default=250, help="Maximum articles to fetch")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD (UTC)")
    parser.add_argument("--window", type=str, default=None, help="6h window string in 'HH-HH' format (UTC)")
    parser.add_argument("--window-start", type=str, default=None, help="Legacy: ISO timestamp for window start (UTC)")
    args = parser.parse_args()

    # Prefer date+window, fallback to explicit window_start
    if args.date and args.window:
        start_dt, end_dt = build_window_datetimes(args.date, args.window)
    else:
        start_dt = parse_window_start(args.window_start)
        end_dt = start_dt.replace(hour=start_dt.hour + VibeHash.WINDOW_DURATION_HOURS)
    logger.info(f"Fetching articles for range: {start_dt} to {end_dt}")

    try:
        articles = fetch_recent_articles_range(args.country, start_dt, end_dt, args.max_articles)
    except Exception as exc:
        logger.exception("Failed to fetch articles")
        raise SystemExit(1) from exc

    if articles.is_empty():
        logger.warning("No articles returned from GDELT; skipping storage")
        return

    headlines = to_headline_rows(articles)
    logger.info(f"Prepared {len(headlines)} headlines for storage")

    service = get_or_create_hopsworks_service(
        api_key=settings.hopsworks.api_key,
        project_name=settings.hopsworks.project_name,
        host=settings.hopsworks.host,
    )
    if service is None:
        logger.error("Hopsworks service is not configured; aborting")
        raise SystemExit(1)

    try:
        service.store_headline_classifications(
            headlines=headlines,
            city=args.city,
            timestamp=start_dt,
        )
        logger.info("Stored headlines in feature group")
    except Exception as exc:
        logger.exception("Failed to store headlines")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
