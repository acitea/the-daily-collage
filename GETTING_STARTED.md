# Getting Started - The Daily Collage

This guide will help you set up and run The Daily Collage system locally.

## System Overview

The Daily Collage is a Python-based system that generates cartoonish visualizations of news "vibes" for geographic locations. The architecture consists of:

1. **Backend**: Python FastAPI server providing REST API endpoints
2. **Ingestion**: GDELT API integration for fetching news
3. **Classification**: Sentiment analysis and signal categorization
4. **Frontend**: Web interface for viewing visualizations and source articles
5. **Caching**: In-memory visualization cache (extensible to MinIO/S3)

## Prerequisites

- Python 3.13+
- UV package manager (https://github.com/astral-sh/uv)
- Internet connection (for GDELT API)
- Modern web browser

## Installation

### 1. Clone or Navigate to Repository

```bash
cd /path/to/the-daily-collage
```

### 2. Install Backend Dependencies

The project uses UV package manager with Python 3.13+. First, install UV if not already installed:

```bash
# Install UV (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using brew on macOS
brew install uv
```

Then sync dependencies for each component:

```bash
# Install ingestion service dependencies
cd backend/ingestion
uv sync

# Install server dependencies
cd ../server
uv sync
```

## Project Structure

```
the-daily-collage/
├── backend/
│   ├── ingestion/
│   │   ├── script.py           # GDELT API data fetching
│   │   └── pyproject.toml      # Dependencies
│   ├── server/
│   │   ├── main.py             # FastAPI application
│   │   ├── pyproject.toml      # Dependencies
│   │   └── README.md           # Server documentation
│   ├── utils/
│   │   ├── classification.py   # Sentiment classification & signal detection
│   │   └── processing.py       # Data cleaning & deduplication
│   └── visualization/
│       └── composition.py      # Image generation & caching
├── frontend/
│   └── index.html              # Web UI
├── README.md                   # Full technical specification
├── ingestion.Dockerfile        # Docker configuration
└── [data/, notebooks/, docs/]  # Supporting directories
```

## Running the System

### 1. Start the Backend Server

```bash
cd backend/server

# Sync environment first
uv sync

# Run the server
uv run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### 2. Access the Frontend

Open your browser and navigate to:

```
frontend/index.html
```

Or simply open the file directly with your browser.

### 3. Test the API (Optional)

The API will be available at `http://localhost:8000` with interactive documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Running the Ingestion Service

### To fetch actual news data from GDELT:

```bash
cd backend/ingestion

# Sync environment first
uv sync

# Run the ingestion script
uv run python script.py
```

This will:
1. Query GDELT API for Swedish news
2. Convert results to polars DataFrame
3. Print summary of fetched articles

### Supported Countries (Configurable)

By default, the ingestion script supports:
- `sweden` or `se` (FIPS code: SW)
- `united_states` or `us` (FIPS code: US)

To fetch news from other countries, modify the `SUPPORTED_COUNTRIES` dict in `script.py` with appropriate FIPS codes.

## Core Modules

### Classification Module (`backend/utils/classification.py`)

Classifies news articles into signal categories:

```python
from backend.utils.classification import classify_articles, aggregate_signals

# Classify articles (currently using keyword-based approach)
classified = classify_articles(articles_df)

# Aggregate signals across all articles
signal_intensities = aggregate_signals(classified)
```

**Signal Categories**:
- Traffic & Transportation
- Weather Events
- Crime & Safety
- Festivals & Events
- Politics
- Sports
- Accidents & Emergencies
- Economic

### Visualization Module (`backend/visualization/composition.py`)

Generates visualizations from signal data:

```python
from backend.visualization.composition import VisualizationService, SignalIntensity

service = VisualizationService()

# Create signals
signals = [
    SignalIntensity("weather", 65.0),
    SignalIntensity("traffic", 45.0),
]

# Generate or retrieve from cache
image_data, metadata = service.generate_or_get(signals, location="Stockholm")
```

### Data Processing (`backend/utils/processing.py`)

Handles cleaning and validation:

```python
from backend.utils.processing import ArticleProcessor

processor = ArticleProcessor()
cleaned_articles = processor.process(raw_articles)
stats = processor.get_stats()
```

## API Endpoints

### Visualization Endpoints

```
GET /api/visualization
  Query params: location (string), force_regenerate (bool)
  Returns: Visualization metadata with signal breakdown

GET /api/visualization/{location}/image
  Returns: Image data for visualization

GET /api/articles
  Query params: location (string), signal (optional string)
  Returns: Source articles for location
```

### Metadata Endpoints

```
GET /api/supported-locations
  Returns: List of currently supported locations

GET /api/signal-categories
  Returns: List of all signal categories with descriptions

GET /api/cache-stats
  Returns: Visualization cache statistics
```

### Health Endpoints

```
GET /
  Returns: API health and version info

GET /health
  Returns: Service health status
```

## Configuration

### Environment Variables (Optional)

Currently, no environment variables are required. In production, you may want to add:

- `GDELT_API_KEY`: API key for GDELT (if rate limiting applies)
- `CACHE_BACKEND`: Cache storage backend (inmemory, redis, s3, etc.)
- `WEATHER_API_KEY`: API key for weather data integration
- `DATABASE_URL`: PostgreSQL connection for metadata storage

### Adding Custom Locations

Edit `backend/ingestion/script.py` and add to `SUPPORTED_COUNTRIES`:

```python
SUPPORTED_COUNTRIES = {
    "sweden": "SW",
    "norway": "NO",  # Add this
    "denmark": "DA",  # Add this
}
```

## Development Workflow

### Running Tests

```bash
# From backend/server (after uv sync)
uv run pytest tests/

# With coverage
uv run pytest --cov=. tests/
```

### Code Quality

```bash
# Format with ruff
uv run ruff format backend/

# Lint
uv run ruff check backend/
```

### 3. Adding New Signal Categories

1. Add to `SignalCategory` enum in `backend/utils/classification.py`
2. Add keywords to `SIGNAL_KEYWORDS` dict
3. Create visual template (in full implementation)
4. Update API documentation

### 4. Extending Classification

Current implementation uses keyword matching. For better accuracy, replace with ML:

```python
# In backend/utils/classification.py
def classify_article_by_model(title: str) -> List[SignalScore]:
    # Use pre-trained transformer model
    model = load_model("bert-swedish-classifier")
    logits = model(title)
    # Convert to SignalScore objects
    ...
```

## Docker Deployment

### Build and Run Ingestion Container

```bash
docker build -f ingestion.Dockerfile -t daily-collage-ingestion .

docker run daily-collage-ingestion
```

### Build Server Container

Create `server.Dockerfile`:

```dockerfile
FROM ghcr.io/astral-sh/uv:0.4.30-python3.13-slim

WORKDIR /app

COPY backend/server/ /app/
COPY backend/utils/ /app/utils/
COPY backend/visualization/ /app/visualization/

RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -f server.Dockerfile -t daily-collage-server .

docker run -p 8000:8000 daily-collage-server
```

## Production Deployment

### Recommendations

1. **Use persistent cache**: Replace in-memory cache with MinIO or PostgreSQL
2. **Add API authentication**: Implement rate limiting and API keys
3. **Use production ASGI server**: Replace uvicorn with gunicorn + uvicorn workers
4. **Set up monitoring**: Add Prometheus metrics and structured logging
5. **Database**: Add PostgreSQL for storing articles, signal history, user preferences
6. **Queue system**: Implement Celery or similar for async ingestion tasks
7. **Frontend hosting**: Serve frontend from CDN or static hosting

### Production Startup

Example production startup:

```bash
uv run gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### \"ImportError: No module named 'gdeltdoc'\"

Make sure dependencies are synced:

```bash
uv sync
uv run python script.py
```

### \"Connection error to GDELT API\"

- Check internet connection
- Verify GDELT service is operational: https://www.gdeltproject.org/
- Try increasing timeout in `backend/ingestion/script.py`
- Run with: `uv run python backend/ingestion/script.py --verbose`

### Frontend shows "API error"

- Verify backend server is running on `http://localhost:8000`
- Check browser console for CORS issues (may need to enable CORS in development)
- Check backend logs for errors

### Cache not working as expected

Current implementation uses in-memory cache (lost on restart). For persistent caching:

```python
# In backend/visualization/composition.py
# Replace VisualizationCache with database-backed version
```

## Next Steps

1. **Fine-tune sentiment model**: Train on Swedish news dataset
2. **Create visual templates**: Design cartoon-style stickers for each signal
3. **Add weather integration**: Fetch real weather data to adjust visualization mood
4. **Implement user preferences**: Allow users to customize which signals they see
5. **Add historical trends**: Show how location sentiment changes over time
6. **Mobile responsive**: Optimize frontend for mobile devices

## Resources

- **GDELT Project**: https://www.gdeltproject.org/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Polars Documentation**: https://docs.pola-rs.com/
- **Project README**: See `README.md` for full technical specifications

## Support

For issues or questions:

1. Check the README.md for architectural details
2. Review module docstrings in Python code
3. Check API documentation at `http://localhost:8000/docs` when server is running

## License

[To be defined by project team]
