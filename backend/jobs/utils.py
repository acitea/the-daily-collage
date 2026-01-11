"""
Shared helpers for backend job scripts.
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

for path in (BACKEND_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from storage.core import VibeHash


def ensure_backend_path() -> None:
    """Ensure backend and repo roots are on sys.path for script execution."""
    for path in (BACKEND_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def current_window_start(now: Optional[datetime] = None) -> datetime:
    """Return the UTC window start time aligned to the configured 6-hour buckets."""
    now = now or datetime.utcnow()
    window_hours = VibeHash.WINDOW_DURATION_HOURS
    start_hour = (now.hour // window_hours) * window_hours
    return datetime(now.year, now.month, now.day, start_hour, tzinfo=None)


def parse_window_start(ts: Optional[str]) -> datetime:
    """Parse an ISO timestamp or fall back to the current window start."""
    if not ts:
        return current_window_start()
    try:
        return datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError("Invalid timestamp format. Use ISO 8601, e.g., 2025-12-11T00:00:00") from exc


def make_article_id(url: str, title: str) -> str:
    """Create a deterministic article id from URL/title."""
    seed = (url or "") + "|" + (title or "")
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

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