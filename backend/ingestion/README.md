# News Ingestion Module

Fetches news articles from the GDELT Project API and converts them to a standardized format for downstream processing.

## Overview

The ingestion module serves as the entry point of The Daily Collage data pipeline. It:

1. **Connects to GDELT API** for real-time global news data
2. **Filters articles by geography** using FIPS country codes
3. **Converts data format** from pandas to polars for efficiency
4. **Implements error handling** and logging for reliability
5. **Supports multiple locations** through a standardized interface

## Quick Start

```bash
cd backend/ingestion
uv sync
uv run python script.py
```

This fetches news for Sweden and prints results.

## Usage

### Fetch News for a Specific Country

```python
from backend.ingestion.script import fetch_news

# Fetch Swedish news
articles = fetch_news(country="sweden", max_articles=250)
print(f"Fetched {len(articles)} articles")

# Fetch US news
articles = fetch_news(country="united_states", max_articles=500)
```

### Fetch News Using FIPS Code Directly

```python
from backend.ingestion.script import get_news_for_location

# SW = Sweden (FIPS code)
articles = get_news_for_location(country_code="SW", max_articles=250)
```

### Handle Errors Gracefully

```python
from backend.ingestion.script import fetch_news
import logging

logging.basicConfig(level=logging.INFO)

try:
    articles = fetch_news(country="norway")
except ValueError as e:
    print(f"Unsupported country: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Module Functions

### `fetch_news(country: str = "sweden", max_articles: int = 250) -> pl.DataFrame`

Main entry point for news fetching.

**Parameters:**
- `country` (str): Country name or FIPS code (case-insensitive)
  - Examples: "sweden", "se", "united_states", "us"
- `max_articles` (int): Maximum number of articles to retrieve

**Returns:**
- `pl.DataFrame`: Polars DataFrame with columns:
  - `date`: Publication date
  - `title`: Article headline
  - `url`: Source URL
  - `source`: News organization
  - `tone`: GDELT sentiment score

**Raises:**
- `ValueError`: If country is not supported
- `Exception`: If GDELT API call fails

**Example:**
```python
articles = fetch_news(country="sweden")
print(articles.select(["title", "source", "tone"]))
```

### `get_news_for_location(country_code: str, max_articles: int = 250) -> pl.DataFrame`

Low-level function for fetching by FIPS code.

**Parameters:**
- `country_code` (str): FIPS country code (e.g., "SW", "US", "UK")
- `max_articles` (int): Maximum articles to retrieve

**Returns:**
- `pl.DataFrame`: Articles data

**Example:**
```python
# Fetch German news
articles = get_news_for_location("GM", max_articles=500)
```

### `normalize_country_input(country_input: str) -> str`

Converts country name to FIPS code.

**Parameters:**
- `country_input` (str): Country name or FIPS code

**Returns:**
- `str`: FIPS code

**Raises:**
- `ValueError`: If country not supported

**Example:**
```python
fips = normalize_country_input("sweden")  # Returns "SW"
fips = normalize_country_input("SE")      # Returns "SW"
```

## Supported Countries

Current supported countries:

| Country | Name Aliases | FIPS Code |
|---------|-------------|-----------|
| Sweden | sweden, se | SW |
| United States | united_states, us | US |

### Adding New Countries

To add support for more countries:

1. Find the FIPS country code at: https://en.wikipedia.org/wiki/FIPS_10-4
2. Add to `SUPPORTED_COUNTRIES` dict in `script.py`:

```python
SUPPORTED_COUNTRIES = {
    "sweden": "SW",
    "united_states": "US",
    "norway": "NO",           # Add this
    "denmark": "DA",          # Add this
    "germany": "GM",          # Add this
}
```

3. Test with:
```python
articles = fetch_news(country="norway")
```

## Output Format

Articles are returned as a Polars DataFrame:

```
shape: (10, 5)
┌────────────────┬──────────────────────┬──────────────────┬─────────────┬────────┐
│ date           ┆ title                ┆ url              ┆ source      ┆ tone   │
│ ---            ┆ ---                  ┆ ---              ┆ ---         ┆ ---    │
│ str            ┆ str                  ┆ str              ┆ str         ┆ f64    │
╞════════════════╪══════════════════════╪══════════════════╪═════════════╪════════╡
│ 2025-12-11 ... ┆ Heavy traffic on ... ┆ https://example… ┆ SVT Nyheter ┆ -2.5   │
│ 2025-12-11 ... ┆ New weather warning  ┆ https://example… ┆ SMHI        ┆ 3.2    │
└────────────────┴──────────────────────┴──────────────────┴─────────────┴────────┘
```

**Column Descriptions:**
- `date`: ISO 8601 formatted publication date
- `title`: Article headline
- `url`: Direct link to article
- `source`: News organization/publication
- `tone`: GDELT tone score (-100 to +100, where negative is negative sentiment)

## Integration with Processing Pipeline

The ingestion module is the first step in the pipeline:

```
Ingestion (script.py)
    ↓ Outputs: pl.DataFrame
Processing (backend/utils/processing.py)
    ↓ Deduplicates & validates
Classification (backend/utils/classification.py)
    ↓ Extracts signals
Visualization (backend/visualization/composition.py)
    ↓ Generates images
API (backend/server/main.py)
    ↓ Serves visualizations
```

### Complete Example

```python
from backend.ingestion.script import fetch_news
from backend.utils.processing import ArticleProcessor
from backend.utils.classification import classify_articles, aggregate_signals

# Step 1: Fetch articles
articles_df = fetch_news(country="sweden", max_articles=250)

# Step 2: Convert to dict for processing
articles_list = articles_df.to_dicts()

# Step 3: Clean and validate
processor = ArticleProcessor()
cleaned = processor.process(articles_list)

# Step 4: Classify to signals
classified = classify_articles_from_dicts(cleaned)
signals = aggregate_signals(classified)

# Step 5: Generate visualization
# (Would use backend/visualization/composition.py)
```

## GDELT API Information

The ingestion module uses the GDELT 2.0 DOC API:

- **Endpoint**: https://api.gdeltproject.org/api/v2/doc/doc
- **Data**: Real-time global news headlines
- **Update Frequency**: Near real-time (seconds to minutes)
- **Geographic Filtering**: By country (FIPS code)
- **Documentation**: https://www.gdeltproject.org/data.html

### Rate Limiting

Current implementation has no explicit rate limiting, but GDELT recommends:
- Maximum 1 request per second for public API
- Batch multiple queries if needed
- Use exponential backoff for retries

To add rate limiting:

```python
import time
from functools import wraps

def rate_limit(min_interval=1.0):
    """Decorator to rate limit function calls."""
    def decorator(func):
        last_called = [0.0]
        
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(min_interval=1.0)
def fetch_news_rate_limited(country="sweden"):
    return fetch_news(country)
```

## Error Handling

The module includes comprehensive error handling:

```python
# Invalid country
try:
    fetch_news(country="atlantis")
except ValueError as e:
    # Handle unsupported country
    print(f"Error: {e}")

# API failure
try:
    fetch_news(country="sweden")
except Exception as e:
    # Handle network/API errors
    print(f"API error: {e}")

# Empty results
articles = fetch_news(country="sweden")
if articles.is_empty():
    print("No articles found")
else:
    print(f"Found {len(articles)} articles")
```

## Logging

The module uses Python's standard logging:

```python
import logging

# Enable debug logging to see API requests
logging.basicConfig(level=logging.DEBUG)
articles = fetch_news(country="sweden")
```

Expected log output:

```
2025-12-11 14:30:00 - __main__ - INFO - Fetching news for country code: SW
2025-12-11 14:30:00 - __main__ - DEBUG - Querying GDELT API with filters: ...
2025-12-11 14:30:02 - __main__ - INFO - Successfully fetched 247 articles for SW
```

## Performance

Typical performance metrics:

| Operation | Time | Notes |
|-----------|------|-------|
| API request | 2-5 seconds | Depends on network |
| DataFrame conversion | <100ms | Pandas to polars |
| Query GDELT | 2-5 seconds | Near real-time data |
| Total fetch_news() | 2-5 seconds | Typical |

### Optimization Tips

1. **Cache results**: Use article hashes to skip reprocessing
2. **Batch queries**: Group multiple location requests
3. **Use polars directly**: Avoid pandas conversion in loops
4. **Async requests**: Use aiohttp for concurrent requests

## Dependencies

Required packages (from `pyproject.toml`):

```toml
dependencies = [
    "gdeltdoc",     # GDELT API client
    "polars",       # DataFrame operations
    "pandas",       # Required by gdeltdoc
]
```

Install with:
```bash
pip install gdeltdoc polars pandas
# or
uv sync
```

## Testing

```python
# Manual testing
from backend.ingestion.script import fetch_news

# Test Sweden
articles = fetch_news(country="sweden")
assert not articles.is_empty(), "Should fetch articles"
assert "title" in articles.columns, "Should have title column"

# Test US
articles = fetch_news(country="us")
assert not articles.is_empty(), "Should fetch US articles"

# Test invalid country
try:
    fetch_news(country="fake")
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected
```

## Scheduling

For production use with 6-hour updates:

### Using APScheduler

```python
from apscheduler.schedulers.background import BackgroundScheduler
from backend.ingestion.script import fetch_news

scheduler = BackgroundScheduler()

def scheduled_fetch():
    try:
        articles = fetch_news(country="sweden")
        # Store articles in database
        print(f"Fetched {len(articles)} articles")
    except Exception as e:
        print(f"Fetch failed: {e}")

# Run every 6 hours
scheduler.add_job(scheduled_fetch, 'interval', hours=6)
scheduler.start()
```

### Using Celery

```python
from celery import Celery
from backend.ingestion.script import fetch_news

app = Celery('daily_collage')

@app.task
def fetch_news_task(country="sweden"):
    articles = fetch_news(country=country)
    # Process articles
    return len(articles)

# Schedule with crontab
from celery.schedules import crontab

app.conf.beat_schedule = {
    'fetch-news-every-6-hours': {
        'task': 'tasks.fetch_news_task',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
    },
}
```

## Future Enhancements

1. **Multi-location concurrent fetch**: Async requests for multiple countries
2. **Weather integration**: Fetch weather data alongside news
3. **Source verification**: Filter by article credibility scores
4. **Historical data**: Archive articles for trend analysis
5. **Custom filters**: Time ranges, specific topics, sentiment ranges
