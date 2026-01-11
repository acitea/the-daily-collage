
"""
News ingestion module for The Daily Collage.

Fetches news articles from GDELT API and converts them to a unified format
for downstream processing (sentiment classification, visualization generation).
"""

import logging
import sys
from typing import Optional
from datetime import datetime

import polars as pl
from gdeltdoc import GdeltDoc, Filters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FIPS country codes for supported locations
SUPPORTED_COUNTRIES = {
    "sweden": "SW",
    "se": "SW",
    "united_states": "US",
    "us": "US",
}


def get_news_for_location(country_code: str, max_articles: int = 250) -> pl.DataFrame:
    """
    Fetches recent news articles from a specific country using the GDELT API.

    Args:
        country_code: FIPS country code (e.g., 'SW' for Sweden, 'US' for USA)
        max_articles: Maximum number of articles to retrieve (default: 250)

    Returns:
        pl.DataFrame: A polars DataFrame containing fetched articles with columns:
            - date: Publication date
            - title: Article headline
            - url: Source URL
            - source: News source name
            - tone: Sentiment tone score (from GDELT)

    Raises:
        ValueError: If invalid country code is provided
        Exception: If GDELT API call fails after retries
    """
    logger.info(f"Fetching news for country code: {country_code}")

    try:
        # Initialize GDELT client
        gd = GdeltDoc()

        # Define filters for the GDELT API query
        # Get news from the last 1 week (timespan format: e.g., "1w" for 1 week)
        filters = Filters(
            country=country_code,
            num_records=max_articles,
            timespan="1w",  # Get news from the last week
        )

        # Search for articles (returns pandas DataFrame)
        logger.debug(f"Querying GDELT API with filters: {filters}")
        articles_pd = gd.article_search(filters)

        if articles_pd is None or articles_pd.empty:
            logger.warning(f"No articles found for country code: {country_code}")
            return pl.DataFrame()

        # Convert pandas DataFrame to polars for efficiency
        articles_pl = pl.from_pandas(articles_pd)

        logger.info(
            f"Successfully fetched {len(articles_pl)} articles for {country_code}"
        )

        return articles_pl

    except Exception as e:
        logger.error(f"Failed to fetch news for {country_code}: {str(e)}")
        raise


def normalize_country_input(country_input: str) -> str:
    """
    Normalizes country input to FIPS code.

    Args:
        country_input: Country name or FIPS code (case-insensitive)

    Returns:
        str: FIPS country code

    Raises:
        ValueError: If country is not in supported list
    """
    normalized = country_input.lower().strip()
    if normalized not in SUPPORTED_COUNTRIES:
        supported = ", ".join(SUPPORTED_COUNTRIES.keys())
        raise ValueError(
            f"Unsupported country: {country_input}. Supported: {supported}"
        )
    return SUPPORTED_COUNTRIES[normalized]


def fetch_news(country: str = "sweden", max_articles: int = 250) -> pl.DataFrame:
    """
    Main function to fetch and process news articles.

    Args:
        country: Country name or FIPS code (default: 'sweden')
        max_articles: Maximum articles to retrieve

    Returns:
        pl.DataFrame: Processed news articles
    """
    try:
        fips_code = normalize_country_input(country)
        articles = get_news_for_location(fips_code, max_articles)
        return articles
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during news fetching: {str(e)}")
        raise


def fetch_news_batched(
    country: str = "sweden",
    total_articles: int = 500,
    batch_size: int = 250,
    days_lookback: int = 30,
    batch_delay: float = 0.5
) -> pl.DataFrame:
    """
    Fetch articles in batches to overcome GDELT API's 250-article-per-request limit.
    Makes multiple requests with different date ranges to accumulate articles.

    Args:
        country: Country name or FIPS code
        total_articles: Total articles to collect across all batches
        batch_size: Articles per single GDELT request (max 250)
        days_lookback: How many days back to search
        batch_delay: Delay in seconds between batches to avoid rate limiting

    Returns:
        pl.DataFrame: Combined articles from all batches, deduplicated by URL
    """
    import time
    
    fips_code = normalize_country_input(country)
    batch_size = min(batch_size, 250)  # GDELT hard limit
    num_batches = (total_articles + batch_size - 1) // batch_size
    
    logger.info(
        f"Fetching {total_articles} articles from {country} in {num_batches} batches"
    )
    
    all_articles = []
    seen_urls = set()
    
    for batch_num in range(num_batches):
        # Calculate date range for this batch
        # Spread requests across the lookback period
        offset_days = int((batch_num / num_batches) * days_lookback)
        timespan = f"{days_lookback - offset_days}d"  # e.g., "30d", "20d", "10d"
        
        logger.info(f"Batch {batch_num + 1}/{num_batches}: timespan={timespan}, batch_size={batch_size}")
        
        try:
            gd = GdeltDoc()
            filters = Filters(
                country=fips_code,
                num_records=batch_size,
                timespan=timespan,
            )
            
            articles_pd = gd.article_search(filters)
            
            if articles_pd is not None and not articles_pd.empty:
                articles_pl = pl.from_pandas(articles_pd)
                
                # Deduplicate by URL
                for article in articles_pl.iter_rows(named=True):
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(article)
                
                logger.info(
                    f"Batch {batch_num + 1}: Got {len(articles_pl)} articles, "
                    f"unique so far: {len(all_articles)}"
                )
            else:
                logger.warning(f"Batch {batch_num + 1}: No articles returned")
        
        except Exception as e:
            logger.error(f"Batch {batch_num + 1} failed: {str(e)}")
        
        # Add delay between requests to avoid rate limiting
        if batch_num < num_batches - 1:
            time.sleep(batch_delay)
    
    logger.info(f"Total unique articles collected: {len(all_articles)}")
    
    if not all_articles:
        logger.warning("No articles collected in batched fetch")
        return pl.DataFrame()
    
    # Convert list of dicts back to DataFrame
    return pl.DataFrame(all_articles)


if __name__ == "__main__":
    try:
        logger.info("Starting news ingestion...")
        swedish_news = fetch_news(country="sweden", max_articles=250)

        if not swedish_news.is_empty():
            logger.info(f"Fetched {len(swedish_news)} articles")
            print(swedish_news.head(10))
            print(f"\nTotal articles: {len(swedish_news)}")
        else:
            logger.warning("No articles returned from GDELT")

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)