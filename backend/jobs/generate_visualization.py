"""
Generate or fetch the visualization for the current 6-hour vibe vector.

Intended to run after classify_headlines.py.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

CURRENT_DIR = Path(__file__).resolve()
BACKEND_ROOT = CURRENT_DIR.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jobs.utils import ensure_backend_path, build_window_datetimes

# Ensure local imports resolve when run as a script
ensure_backend_path()

from app.services.hopsworks import get_or_create_hopsworks_service
from settings import settings
from visualization.composition import VisualizationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def to_score_map(vibe_vector: Dict[str, tuple]) -> Dict[str, float]:
    return {category: score for category, (score, _, _) in vibe_vector.items()}


def main():
    parser = argparse.ArgumentParser(description="Generate visualization for a vibe vector")
    parser.add_argument("--city", type=str, default="stockholm", help="City/region label used in storage")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD (UTC)")
    parser.add_argument("--window", type=str, required=True, help="6h window string in 'HH-HH' format (UTC)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if cached",
    )
    args = parser.parse_args()

    if args.date and args.window:
        window_start, _ = build_window_datetimes(args.date, args.window)

    hopsworks_service = get_or_create_hopsworks_service(
        api_key=settings.hopsworks.api_key,
        project_name=settings.hopsworks.project_name,
        host=settings.hopsworks.host,
    )
    if hopsworks_service is None:
        logger.error("Hopsworks service is not configured; aborting")
        raise SystemExit(1)

    vibe_vector = hopsworks_service.get_vibe_vector_at_time(
        city=args.city,
        timestamp=window_start,
    )
    if not vibe_vector:
        logger.error("No vibe vector found for requested window")
        raise SystemExit(1)

    score_map = to_score_map(vibe_vector)
    use_hopsworks_storage = settings.storage.backend == "hopsworks"

    viz_service = VisualizationService(use_hopsworks=use_hopsworks_storage)
    image_data, metadata = viz_service.generate_or_get(
        city=args.city,
        vibe_vector=score_map,
        timestamp=window_start,
        force_regenerate=args.force,
    )

    logger.info(
        "Visualization ready | cached=%s cache_key=%s hitboxes=%s",
        metadata.get("cached"),
        metadata.get("cache_key"),
        len(metadata.get("hitboxes", [])),
    )
    logger.info("Generated image bytes: %s bytes", len(image_data))


if __name__ == "__main__":
    main()
