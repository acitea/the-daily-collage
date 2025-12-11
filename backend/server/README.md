# The Daily Collage - Backend Server

FastAPI-based REST API server for The Daily Collage visualization system.

## Quick Start

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run server
uv run python main.py
```

Server will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

### Core Components

1. **VisualizationService** (`backend/visualization/composition.py`)
   - Orchestrates visualization generation
   - Manages caching
   - Coordinates with TemplateComposer

2. **Classification** (`backend/utils/classification.py`)
   - Converts news articles to signal categories
   - Calculates signal intensities
   - Aggregates signals across articles

3. **Data Processing** (`backend/utils/processing.py`)
   - Validates articles
   - Removes duplicates
   - Cleans text

### Request Flow

```
User Request
    ↓
FastAPI Endpoint
    ↓
Check Cache
    ├─ Hit: Return cached visualization
    └─ Miss: Generate new visualization
        ├─ Fetch articles (or use provided)
        ├─ Classify to signals
        ├─ Compose visualization
        └─ Cache result
    ↓
Return Response
```

## API Endpoints Reference

### GET /
Root health check endpoint.

**Response:**
```json
{
  "name": "The Daily Collage API",
  "status": "operational",
  "version": "0.1.0"
}
```

### GET /health
Service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T14:30:00.000Z"
}
```

### GET /api/visualization

Generates or retrieves a cached visualization for a location.

**Query Parameters:**
- `location` (string): Location name (default: "sweden")
- `force_regenerate` (boolean): Skip cache and regenerate (default: false)

**Response:**
```json
{
  "location": "sweden",
  "generated_at": "2025-12-11T14:30:00.000Z",
  "signal_count": 3,
  "signals": [
    {"name": "weather", "intensity": 65.0},
    {"name": "traffic", "intensity": 45.0},
    {"name": "politics", "intensity": 75.0}
  ],
  "image_url": "/api/visualization/sweden/image",
  "cached": false
}
```

**Example:**
```bash
curl "http://localhost:8000/api/visualization?location=stockholm"
```

### GET /api/visualization/{location}/image

Retrieves the actual image data for a visualization.

**Path Parameters:**
- `location` (string): Location name

**Response:** Image data (PNG/WebP in full implementation)

**Example:**
```bash
curl "http://localhost:8000/api/visualization/stockholm/image" > stockholm.png
```

### GET /api/articles

Retrieves source articles that contributed to a visualization.

**Query Parameters:**
- `location` (string): Location name (default: "sweden")
- `signal` (string, optional): Filter articles by signal category

**Response:**
```json
{
  "location": "sweden",
  "articles": [
    {
      "title": "Heavy traffic on Stockholm ring road",
      "url": "https://example.com/article1",
      "source": "SVT Nyheter",
      "date": "2025-12-11T14:30:00Z",
      "signal": "traffic"
    }
  ]
}
```

**Examples:**
```bash
# All articles for a location
curl "http://localhost:8000/api/articles?location=stockholm"

# Filter by signal
curl "http://localhost:8000/api/articles?location=stockholm&signal=weather"
```

### GET /api/supported-locations

Returns list of currently supported geographic locations.

**Response:**
```json
{
  "locations": [
    {
      "code": "se",
      "name": "Sweden",
      "type": "country"
    },
    {
      "code": "stockholm",
      "name": "Stockholm",
      "type": "city"
    }
  ]
}
```

### GET /api/signal-categories

Returns all signal categories with descriptions.

**Response:**
```json
{
  "categories": [
    {
      "id": "traffic",
      "name": "Traffic & Transportation",
      "description": "Road congestion, public transit disruptions",
      "icon": "🚗"
    }
  ]
}
```

### GET /api/cache-stats

Returns cache statistics and performance metrics.

**Response:**
```json
{
  "cache": {
    "cached_visualizations": 12,
    "cache_size_estimates": 2048576
  }
}
```

## Configuration

### Current Implementation

- **Cache**: In-memory dictionary (lost on restart)
- **Image Generation**: Placeholder (returns text description)
- **Classification**: Keyword-based (not ML model)

### Production Recommendations

1. **Persistent Cache**
   - Use Redis for distributed caching
   - Or MinIO/S3 for artifact storage
   - Or PostgreSQL with blob storage

2. **Better Classification**
   - Fine-tune BERT model on Swedish news
   - Use zero-shot classification
   - Implement multi-label prediction

3. **Image Generation**
   - Replace placeholder with actual image rendering
   - Use PIL/Pillow for template composition
   - Or integrate generative AI (Stable Diffusion, Flux)

4. **Database**
   - PostgreSQL for articles, signals, metadata
   - Track historical data for trends

## Development

### Running with Auto-Reload

```bash
uv run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Unit tests (to be added)
uv run pytest tests/

# Manual API testing
curl http://localhost:8000/api/visualization?location=sweden | jq
```

### Adding New Endpoints

1. Import necessary modules at top
2. Define Pydantic request/response models
3. Create route function with `@app.get()` or `@app.post()` decorator
4. Add docstring for documentation
5. Return JSONResponse or Pydantic model

Example:

```python
@app.get("/api/custom-endpoint", tags=["Custom"])
async def custom_endpoint(param: str = Query(...)):
    """
    Custom endpoint description.
    
    Args:
        param: Parameter description
    
    Returns:
        Response description
    """
    # Implementation
    return JSONResponse(content={"result": "value"})
```

## Error Handling

All endpoints implement proper error handling:

- **400 Bad Request**: Invalid query parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Unexpected server error
- **503 Service Unavailable**: Service not initialized

Example error response:

```json
{
  "detail": "Visualization service not initialized"
}
```

## Performance Considerations

1. **Caching Strategy**
   - Signal combinations discretized to bins (0, 10, 20, ..., 100)
   - Prevents cache fragmentation
   - Improves hit rate

2. **Async Operations**
   - All endpoints are async for concurrent request handling
   - Database I/O should use async driver

3. **Response Compression**
   - FastAPI automatically compresses responses with gzip
   - Enable with `app.add_middleware(GZIPMiddleware, minimum_size=1000)`

## CORS Configuration

For frontend integration, CORS may need to be enabled:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Monitoring & Logging

The server includes structured logging with timestamps and log levels. For production:

1. Use structured logging library (e.g., python-json-logger)
2. Send logs to centralized logging system (ELK, Datadog, etc.)
3. Add Prometheus metrics for monitoring
4. Set up health check monitoring

## Deployment

### Docker

```dockerfile
FROM ghcr.io/astral-sh/uv:0.4.30-python3.13-slim

WORKDIR /app

COPY backend/server/ /app/
COPY backend/utils/ /app/utils/
COPY backend/visualization/ /app/visualization/

RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0"]
```

### Production Server

Use gunicorn with uvicorn workers:

```bash
uv run gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Related Modules

- **Classification**: `backend/utils/classification.py`
- **Image Composition**: `backend/visualization/composition.py`
- **Data Processing**: `backend/utils/processing.py`
- **News Ingestion**: `backend/ingestion/script.py`

See `GETTING_STARTED.md` for complete system documentation.
