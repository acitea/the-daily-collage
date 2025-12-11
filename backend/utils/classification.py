"""
Sentiment classification and signal categorization module.

Classifies news articles into predefined signal categories and extracts
sentiment/intensity information for visualization generation.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)

# Signal categories as defined in project spec
class SignalCategory(Enum):
    """Predefined signal categories for news articles."""

    TRAFFIC = "traffic"
    WEATHER = "weather"
    CRIME = "crime"
    FESTIVALS = "festivals"
    POLITICS = "politics"
    SPORTS = "sports"
    ACCIDENTS = "accidents"
    ECONOMIC = "economic"


# Keywords for signal classification (can be expanded or replaced with ML model)
SIGNAL_KEYWORDS = {
    SignalCategory.TRAFFIC: [
        "traffic",
        "congestion",
        "trafik",
        "kö",
        "väg",
        "vägar",
        "tåg",
        "buss",
        "transport",
        "accident",
        "olycka",
    ],
    SignalCategory.WEATHER: [
        "weather",
        "storm",
        "rain",
        "snow",
        "heatwave",
        "flood",
        "väder",
        "regn",
        "snö",
        "värmebölga",
        "översvämning",
        "temperatur",
    ],
    SignalCategory.CRIME: [
        "crime",
        "police",
        "robbery",
        "theft",
        "murder",
        "brott",
        "polis",
        "stöld",
        "mord",
        "bråk",
        "våld",
    ],
    SignalCategory.FESTIVALS: [
        "festival",
        "concert",
        "event",
        "celebration",
        "party",
        "festival",
        "konsert",
        "fest",
        "firande",
    ],
    SignalCategory.POLITICS: [
        "election",
        "protest",
        "government",
        "parliament",
        "minister",
        "val",
        "protest",
        "regering",
        "riksdag",
        "politiker",
    ],
    SignalCategory.SPORTS: [
        "sport",
        "game",
        "match",
        "football",
        "hockey",
        "victory",
        "fotboll",
        "hockey",
        "seger",
        "match",
        "lag",
    ],
    SignalCategory.ACCIDENTS: [
        "accident",
        "emergency",
        "fire",
        "explosion",
        "rescue",
        "olycka",
        "nödsituation",
        "brand",
        "explosion",
        "räddning",
    ],
    SignalCategory.ECONOMIC: [
        "economy",
        "market",
        "business",
        "employment",
        "stock",
        "ekonomi",
        "marknad",
        "företag",
        "sysselsättning",
        "aktie",
    ],
}


@dataclass
class SignalScore:
    """Represents a detected signal with its intensity score."""

    category: SignalCategory
    intensity: float  # 0-100 scale
    confidence: float  # 0-1 scale (how confident in this classification)
    keywords_matched: List[str]  # Which keywords triggered this classification


@dataclass
class ClassifiedArticle:
    """An article with classification results."""

    title: str
    url: str
    source: str
    date: str
    tone: Optional[float]  # GDELT tone score
    signals: List[SignalScore]
    primary_signal: Optional[SignalCategory]  # Most confident signal


def classify_article_by_keywords(
    title: str, source: str = "", content: str = ""
) -> List[SignalScore]:
    """
    Classifies an article based on keyword matching.

    This is a simple baseline approach. In production, this would be replaced
    with a fine-tuned transformer model (e.g., BERT-based) for Swedish news.

    Args:
        title: Article headline
        source: News source name
        content: Full article content (optional)

    Returns:
        List[SignalScore]: Detected signals with intensity scores
    """
    combined_text = f"{title} {source} {content}".lower()
    detected_signals: Dict[SignalCategory, SignalScore] = {}

    for category, keywords in SIGNAL_KEYWORDS.items():
        matched_keywords = [kw for kw in keywords if kw in combined_text]

        if matched_keywords:
            # Calculate intensity based on keyword frequency and match count
            intensity = min(100.0, len(matched_keywords) * 15.0)
            confidence = min(0.95, len(matched_keywords) * 0.15)

            detected_signals[category] = SignalScore(
                category=category,
                intensity=intensity,
                confidence=confidence,
                keywords_matched=matched_keywords,
            )

    return list(detected_signals.values())


def classify_articles(articles_df: pl.DataFrame) -> List[ClassifiedArticle]:
    """
    Classifies a batch of articles.

    Args:
        articles_df: Polars DataFrame with columns: title, url, source, date, tone

    Returns:
        List[ClassifiedArticle]: Classified articles with detected signals
    """
    classified: List[ClassifiedArticle] = []

    for row in articles_df.iter_rows(named=True):
        signals = classify_article_by_keywords(
            title=row.get("title", ""),
            source=row.get("source", ""),
            content=row.get("content", ""),
        )

        primary_signal = (
            max(signals, key=lambda s: s.confidence).category if signals else None
        )

        classified.append(
            ClassifiedArticle(
                title=row.get("title", ""),
                url=row.get("url", ""),
                source=row.get("source", ""),
                date=row.get("date", ""),
                tone=row.get("tone"),
                signals=signals,
                primary_signal=primary_signal,
            )
        )

    return classified


def aggregate_signals(
    classified_articles: List[ClassifiedArticle],
) -> Dict[SignalCategory, float]:
    """
    Aggregates signal intensities across all articles.

    Returns the overall intensity for each detected signal category.

    Args:
        classified_articles: List of classified articles

    Returns:
        Dict[SignalCategory, float]: Signal categories with aggregated intensities
    """
    signal_totals: Dict[SignalCategory, List[float]] = {
        cat: [] for cat in SignalCategory
    }

    for article in classified_articles:
        for signal in article.signals:
            signal_totals[signal.category].append(signal.intensity)

    # Calculate average intensity for each signal
    aggregated = {}
    for category, intensities in signal_totals.items():
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            aggregated[category] = avg_intensity

    return aggregated
