"""
Classify stored headlines via external API and aggregate a vibe vector.

Intended to run after fetch_headlines.py.
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

CURRENT_DIR = Path(__file__).resolve()
BACKEND_ROOT = CURRENT_DIR.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jobs.utils import ensure_backend_path, parse_window_start, build_window_datetimes

# Ensure local imports resolve when run as a script
ensure_backend_path()

from app.services.hopsworks import get_or_create_hopsworks_service
from settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_ENDPOINT = "https://terahidro2003--daily-collage-classifier-api-predict.modal.run"


def row_to_headline(row, categories: List[str]) -> Dict:
    """Convert a feature group row into a headline payload."""
    classifications = {}
    for category in categories:
        score = float(row.get(f"{category}_score", 0.0))
        tag = row.get(f"{category}_tag", "")
        if score != 0.0 or tag:
            classifications[category] = (score, tag)

    return {
        "article_id": row.get("article_id", ""),
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "url": row.get("url", ""),
        "source": row.get("source", ""),
        "classifications": classifications,
    }


def already_classified(headline: Dict) -> bool:
    return bool(headline.get("classifications"))


def classify_article(client: httpx.Client, endpoint: str, title: str, description: str) -> Tuple[str, float, str]:
    payload = {"title": title or "", "description": description or ""}
    resp = client.post(endpoint, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if "data" in data:
        data = data["data"]
    elif "predictions" in data and data["predictions"]:
        data = data["predictions"][0]

    category = data.get("category")
    score = float(data.get("score", 0.0) or 0.0)
    tag = data.get("tag", "")
    return category, score, tag


def aggregate_weighted(headlines: List[Dict], categories: List[str]) -> Dict[str, Tuple[float, str, int]]:
    """Aggregate classifications using the weighted strategy from ml/ingestion/hopsworks_pipeline.py."""
    buckets = {cat: [] for cat in categories}
    for headline in headlines:
        for cat, (score, tag) in headline.get("classifications", {}).items():
            # Ignore zeros to reduce noise
            if abs(score) <= 0.0:
                continue
            buckets.setdefault(cat, []).append((score, tag))

    vibe_vector: Dict[str, Tuple[float, str, int]] = {}

    for cat, signals in buckets.items():
        if not signals:
            continue

        count = len(signals)
        scores = [s for s, _ in signals]
        base_score = sum(scores) / len(scores)

        frequency_multiplier = min(1.0 + math.log10(count) / 2, 2.0)
        weighted_score = min(base_score * frequency_multiplier, 1.0)

        tags = [t for _, t in signals]
        tag_counts: Dict[str, int] = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        dominant_tag = max(tag_counts.items(), key=lambda x: x[1])[0]

        if weighted_score > 0.1:
            vibe_vector[cat] = (weighted_score, dominant_tag, count)

    return vibe_vector


def main():
    parser = argparse.ArgumentParser(description="Classify headlines via API and aggregate vibe vector")
    parser.add_argument("--city", type=str, default="stockholm", help="City/region label used in storage")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD (UTC)")
    parser.add_argument("--window", type=str, default=None, help="6h window string in 'HH-HH' format (UTC)")
    parser.add_argument("--window-start", type=str, default=None, help="Legacy: ISO timestamp for window start (UTC)")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Classifier API endpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify even if scores already present",
    )
    args = parser.parse_args()

    if args.date and args.window:
        window_start, window_end = build_window_datetimes(args.date, args.window)
    else:
        window_start = parse_window_start(args.window_start)
        from datetime import timedelta
        window_end = window_start + timedelta(hours=6)
    logger.info(f"Using window: {window_start} to {window_end}")

    service = get_or_create_hopsworks_service(
        api_key=settings.hopsworks.api_key,
        project_name=settings.hopsworks.project_name,
        host=settings.hopsworks.host,
    )
    if service is None:
        logger.error("Hopsworks service is not configured; aborting")
        raise SystemExit(1)

    fg = service.get_or_create_headline_feature_group()
    df = fg.select_all().filter((fg.city == args.city) & (fg.timestamp >= window_start) & (fg.timestamp < window_end)).read()

    if df.empty:
        logger.warning("No headlines found to classify")
        return

    categories = service.SIGNAL_CATEGORIES
    headlines: List[Dict] = [row_to_headline(row, categories) for _, row in df.iterrows()]

    classified: List[Dict] = []
    aggregator_input: List[Dict] = []

    with httpx.Client(timeout=15.0) as client:
        for headline in headlines:
            if already_classified(headline) and not args.force:
                aggregator_input.append(headline)
                continue

            try:
                category, score, tag = classify_article(
                    client,
                    args.endpoint,
                    headline.get("title", ""),
                    headline.get("description", ""),
                )
            except Exception as exc:
                logger.error(f"Classification failed for article {headline.get('article_id')}: {exc}")
                continue

            if category and category in categories:
                headline["classifications"] = {category: (score, tag)}
                logger.info(
                    f"Classified article {headline.get('article_id')} -> {category} ({score:.3f}, {tag})"
                )
            else:
                logger.warning(
                    f"Classifier returned unknown category '{category}' for article {headline.get('article_id')}"
                )
                headline["classifications"] = {}

            classified.append(headline)
            aggregator_input.append(headline)

    if classified:
        try:
            service.store_headline_classifications(
                headlines=classified,
                city=args.city,
                timestamp=window_start,
            )
            logger.info(f"Stored {len(classified)} classified headlines")
        except Exception as exc:
            logger.exception("Failed to store headline classifications")
            raise SystemExit(1) from exc
    else:
        logger.info("No new classifications to store")

    vibe_vector = aggregate_weighted(aggregator_input, categories)
    if not vibe_vector:
        logger.warning("No classifications available to aggregate a vibe vector")
        return

    try:
        service.store_vibe_vector(
            city=args.city,
            timestamp=window_start,
            vibe_vector=vibe_vector,
        )
        logger.info("Stored aggregated vibe vector")
    except Exception as exc:
        logger.exception("Failed to store vibe vector")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
