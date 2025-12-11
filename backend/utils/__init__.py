"""
The Daily Collage - Utilities Module

Contains shared utilities for text processing, classification, and data handling.
"""

from .classification import (
    SignalCategory,
    SignalScore,
    ClassifiedArticle,
    classify_articles,
    aggregate_signals,
)
from .processing import (
    ArticleProcessor,
    deduplicate_articles,
    filter_articles,
    normalize_text,
)

__all__ = [
    "SignalCategory",
    "SignalScore",
    "ClassifiedArticle",
    "classify_articles",
    "aggregate_signals",
    "ArticleProcessor",
    "deduplicate_articles",
    "filter_articles",
    "normalize_text",
]
