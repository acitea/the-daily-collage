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


def parse_window_str(window: str) -> tuple[int, int]:
    """Parse a window string like '00-06' into (start_hour, end_hour)."""
    try:
        parts = window.split("-")
        if len(parts) != 2:
            raise ValueError
        start = int(parts[0])
        end = int(parts[1])
        if start % VibeHash.WINDOW_DURATION_HOURS != 0 or end - start != VibeHash.WINDOW_DURATION_HOURS:
            raise ValueError
        if not (0 <= start < 24 and 0 < end <= 24):
            raise ValueError
        return start, end
    except Exception as exc:
        raise ValueError("Window must be in 'HH-HH' 6-hour format, e.g., '00-06'.") from exc


def build_window_datetimes(date_str: str, window_str: str) -> tuple[datetime, datetime]:
    """Construct start and end datetimes (UTC) from date and window strings."""
    try:
        d = datetime.fromisoformat(f"{date_str}T00:00:00").date()
    except Exception as exc:
        raise ValueError("Date must be in 'YYYY-MM-DD' format.") from exc
    start_hour, end_hour = parse_window_str(window_str)
    start = datetime(d.year, d.month, d.day, start_hour)
    
    # If end_hour is 24, set to midnight of next day
    if end_hour == 24:
        from datetime import timedelta
        end = start + timedelta(days=1)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        end = datetime(d.year, d.month, d.day, end_hour)
    
    return start, end
