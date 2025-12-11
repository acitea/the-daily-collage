# AI Coding Agent Instructions - The Daily Collage

## Project Overview

**The Daily Collage** is a proof-of-concept system that transforms news headlines from geographic locations (primarily Sweden) into cartoonish visualizations capturing the collective "vibe" of current events. The system updates every 6 hours.

**Core Value**: Users glance at one generated image to understand location sentiment and major events, then drill down to source articles if interested.

## Architecture Overview

The codebase is organized into modular Python components following a data pipeline pattern:

```
backend/
├── ingestion/        # GDELT API polling → news fetching
├── server/           # FastAPI REST API for visualization requests
├── utils/            # Shared utilities (TBD expansion)
└── models/           # ML models (under development)
```

### Data Flow (Critical Mental Model)

1. **Ingestion** (every 6 hours): `backend/ingestion/script.py` queries GDELT API for location-filtered news
2. **Processing**: Raw articles → sentiment classification → signal category extraction (Traffic, Weather, Crime, Sports, etc.)
3. **Generation**: Signal profiles checked against cache; cache miss triggers visualization composition
4. **API Response**: FastAPI serves cached/generated visualization + metadata (signal breakdown, source articles)

Key insight: **Caching is essential** - signal combinations are deduplicated to avoid redundant generation.

## Signal Categories & Visual Mapping

Eight primary signal categories define the visualization vocabulary:
- **Traffic & Transportation**: Road congestion, public transit disruptions
- **Weather Events**: Storms, heatwaves, snow, flooding
- **Crime & Safety**: Incidents, police activity
- **Festivals & Events**: Cultural celebrations, concerts
- **Politics**: Elections, protests, government announcements
- **Sports**: Major games, victories
- **Accidents & Emergencies**: Fires, industrial accidents
- **Economic**: Market news, business developments

Each category has a corresponding visual "sticker" that scales in size/intensity (0-100) based on signal strength. This is the foundation of template-based composition (currently favored over generative AI for predictability).

## Technical Stack & Key Dependencies

**Python 3.13+** across all modules.

**Core Libraries**:
- `gdeltdoc`: GDELT API client for news ingestion (see `backend/ingestion/script.py` for example usage)
- `polars`: DataFrame processing (preferred over pandas for performance; conversion from pandas necessary for GDELT output)
- `pandas`: Required by gdeltdoc; convert to polars for internal pipelines
- `fastapi[standard]`: REST API framework (backend/server)

**Containerization**: Docker (single container or microservice approach; not Docker Compose)

**Database**: Artifact store (MinIO, S3-compatible, or PostgreSQL blob storage) for caching generated visualizations

## Conventions & Patterns

### Data Processing Workflow
- **Ingestion output**: Use `gdeltdoc.GdeltDoc().article_search(filters)` returning pandas DataFrame
- **Internal format**: Convert to `polars.DataFrame` immediately (see `backend/ingestion/script.py` line ~11-12)
- **Geographic filtering**: Use FIPS country codes (e.g., "SW" for Sweden) in `Filters(source_country=...)`

### API Design (FastAPI)
- Query parameters: `location` (city/country), optional date/time range
- Response payload includes generated image + metadata dict with signal breakdown
- Endpoints should be idempotent for caching compatibility

### Image Generation (Template-Based)
- Cached at composite level: key = sorted signal names + intensity levels
- Cache misses trigger composition (not regeneration of full image)
- Weather data integration adjusts visual mood (color palette, background opacity)

## Extending the System

### Adding New Signal Categories
1. Define category in signal list (8 primary, conceptually extensible)
2. Create template asset (vector graphics or layered image)
3. Update sentiment classification model to detect new category
4. Register in visualization composition service

### Adding Geographic Regions
- Extend ingestion script `source_country` parameter with new FIPS codes
- Ensure sentiment classification model handles new language/dialect if needed
- Update frontend location selector

## Critical Gotchas & Notes

1. **GDELT Data Format**: Always convert pandas output from `gdeltdoc` to polars for consistency (pandas not used internally)
2. **Cache Key Design**: Signal intensity levels must be discretized consistently (prevents cache fragmentation)
3. **Sentiment Model Scope**: Currently focused on Swedish news; non-Swedish expansion requires model retraining
4. **Image Style Consistency**: Template approach essential for POC; avoid mixing generative AI until architecture matures
5. **6-Hour Update Cadence**: Ingestion scheduling is critical for "hot news" relevance; design for asynchronous queue processing

## Key Files for Reference

- `README.md`: Full technical spec including image caching strategy details
- `backend/ingestion/script.py`: GDELT integration pattern (reference for new data sources)
- `backend/server/pyproject.toml`: Dependency versioning (Python 3.13+ requirement)
- `ingestion.Dockerfile`: Container setup for ingestion service

## Success Criteria (for PRs/features)

- [ ] News ingestion works for target location via GDELT
- [ ] Classification accurately maps headlines → signal categories
- [ ] Visualizations represent signal combinations meaningfully
- [ ] Frontend drill-down to source articles functions
- [ ] Caching reduces redundant generation for similar profiles

