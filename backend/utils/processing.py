"""
Data processing utilities for The Daily Collage.

Handles text cleaning, deduplication, and data validation.
"""

import logging
import hashlib
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalizes text for comparison and deduplication.

    Args:
        text: Raw text to normalize

    Returns:
        str: Normalized text (lowercase, stripped whitespace)
    """
    return text.lower().strip()


def generate_article_hash(title: str, source: str = "") -> str:
    """
    Generates a hash for an article to detect duplicates.

    Args:
        title: Article title
        source: News source

    Returns:
        str: SHA256 hash of normalized article identifier
    """
    identifier = f"{normalize_text(title)}:{normalize_text(source)}"
    return hashlib.sha256(identifier.encode()).hexdigest()


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """
    Removes duplicate articles based on title and source.

    Args:
        articles: List of article dictionaries with 'title' and 'source' keys

    Returns:
        List[Dict]: Deduplicated list of articles
    """
    seen_hashes = set()
    deduplicated = []

    for article in articles:
        article_hash = generate_article_hash(
            article.get("title", ""), article.get("source", "")
        )

        if article_hash not in seen_hashes:
            seen_hashes.add(article_hash)
            deduplicated.append(article)
        else:
            logger.debug(f"Skipping duplicate article: {article.get('title', '')}")

    logger.info(
        f"Removed {len(articles) - len(deduplicated)} duplicate articles"
    )
    return deduplicated


def validate_article(article: Dict) -> bool:
    """
    Validates that an article has required fields.

    Args:
        article: Article dictionary

    Returns:
        bool: True if article is valid, False otherwise
    """
    required_fields = ["title", "url", "source"]
    return all(field in article and article[field] for field in required_fields)


def filter_articles(articles: List[Dict]) -> List[Dict]:
    """
    Filters out invalid articles.

    Args:
        articles: List of article dictionaries

    Returns:
        List[Dict]: Filtered list of valid articles
    """
    valid_articles = [a for a in articles if validate_article(a)]

    if len(valid_articles) < len(articles):
        logger.warning(
            f"Filtered out {len(articles) - len(valid_articles)} invalid articles"
        )

    return valid_articles


class ArticleProcessor:
    """Handles end-to-end article processing pipeline."""

    def __init__(self):
        self.processed_count = 0
        self.duplicate_count = 0
        self.invalid_count = 0

    def process(self, articles: List[Dict]) -> List[Dict]:
        """
        Processes articles through the full pipeline.

        Args:
            articles: Raw articles from ingestion

        Returns:
            List[Dict]: Cleaned and validated articles
        """
        # Step 1: Validate
        before_validation = len(articles)
        articles = filter_articles(articles)
        self.invalid_count = before_validation - len(articles)

        # Step 2: Deduplicate
        before_dedup = len(articles)
        articles = deduplicate_articles(articles)
        self.duplicate_count = before_dedup - len(articles)

        self.processed_count = len(articles)

        logger.info(
            f"Processing complete: {self.processed_count} articles "
            f"({self.invalid_count} invalid, {self.duplicate_count} duplicates removed)"
        )

        return articles

    def get_stats(self) -> Dict[str, int]:
        """Returns processing statistics."""
        return {
            "processed": self.processed_count,
            "duplicates_removed": self.duplicate_count,
            "invalid_removed": self.invalid_count,
        }
