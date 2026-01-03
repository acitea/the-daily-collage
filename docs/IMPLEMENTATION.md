# Implementation Guide - The Daily Collage

## Overview

This document describes the implemented system architecture for The Daily Collage, following the hybrid composition model with Hopsworks Feature Store integration.

## System Architecture

### Data Flow

```
GDELT API
    ↓
News Ingestion (ml/ingestion/hopsworks_pipeline.py)
    ↓
ML Classification (keyword-based, ready for ML model)
    ↓
Vibe Vector Aggregation (max-pooling)
    ↓
Hopsworks Feature Store (vibe_vectors feature group)
    ↓
Visualization Request (backend/server/main.py)
    ↓
Check Hopsworks Artifact Store for cached image
    ↓
[Cache Miss] → Generate Layout (backend/visualization/composition.py)
    ↓
Grid-Based Asset Placement (backend/visualization/assets.py)
    ↓
Stability AI Polish (backend/visualization/polish.py)
    ↓
Store in Hopsworks Artifact Store
    ↓
Return Image + Hitboxes
```

## Key Components

### 1. Asset Management (backend/visualization/assets.py)

**Features Implemented:**
- ✅ Asset naming with intensity levels: `{category}_{tag}_{intensity}.png`
- ✅ Three intensity levels: low (0.0-0.33), med (0.33-0.66), high (0.66-1.0)
- ✅ Fallback hierarchy: exact → category+intensity → generic
- ✅ Grid-based randomized placement (8 cols × 3 rows per zone)
- ✅ No additional scaling - assets are pre-sized

**Asset Naming Examples:**
```
emergencies_fire_high.png
crime_theft_low.png
festivals_concert_med.png
transportation_traffic_high.png
```

**Fallback Logic:**
1. Try exact match: `(category, tag, intensity_level)`
2. Fall back to: `(category, intensity_level)_generic.png`
3. Ultimate fallback: `generic_default.png`

### 2. Grid-Based Placement

**Configuration:**
- **Grid**: 8 columns × 3 rows per zone
- **Zones**: Sky (25%), City (50%), Street (25%)
- **Randomization**: ±25% offset within each cell
- **Total Capacity**: 24 cells per zone = 72 cells total

**Placement Algorithm:**
1. Assign signals to zones by category
2. Generate all grid cells in zone
3. Shuffle cells randomly
4. For each signal:
   - Pick next available cell
   - Calculate cell center
   - Add random offset (±25% of cell size)
   - Place asset at final position
   - Record hitbox

**Benefits:**
- Prevents overlaps without complex collision detection
- Natural randomization for varied layouts
- Scalable to many signals (72 total slots)

### 3. Stability AI Polish (backend/visualization/polish.py)

**Prompt Strategy:**
```
Positive: "A colorful sticker scrapbook collage, playful cartoon stickers, 
          overlapping elements, vibrant colors, whimsical illustration style"

Negative: "blurry, low quality, distorted, moved objects, photorealistic, 
          realistic, 3d render, photograph"
```

**Configuration:**
- **Image Strength**: 0.35 (preserves layout)
- **Steps**: 30
- **CFG Scale**: 7.0
- **Style Goal**: Sticker scrapbook aesthetic

### 4. Hopsworks Integration (backend/server/services/hopsworks.py)

**Feature Store Schema:**
```python
vibe_vectors (feature group):
    - city: string (primary key)
    - timestamp: timestamp (primary key, event_time)
    - emergencies_score: float
    - emergencies_tag: string
    - crime_score: float
    - crime_tag: string
    - festivals_score: float
    - festivals_tag: string
    - transportation_score: float
    - transportation_tag: string
    - weather_temp_score: float
    - weather_temp_tag: string
    - weather_wet_score: float
    - weather_wet_tag: string
    - sports_score: float
    - sports_tag: string
    - economics_score: float
    - economics_tag: string
    - politics_score: float
    - politics_tag: string
```

**Artifact Store Structure:**
```
Resources/vibe_images/
    ├── {vibe_hash}.png
    └── {vibe_hash}_metadata.json
```

**Services Provided:**
- `store_vibe_vector()`: Push vibe vector to feature store
- `get_latest_vibe_vector()`: Retrieve latest vibe for a city
- `store_visualization()`: Store generated image + metadata
- `get_visualization()`: Retrieve cached visualization
- `visualization_exists()`: Quick existence check

### 5. Ingestion Pipeline (ml/ingestion/hopsworks_pipeline.py)

**Pipeline Steps:**
1. **Fetch News**: GDELT API → Polars DataFrame
2. **Classify**: Keyword-based (ready for ML model replacement)
3. **Aggregate**: Max-pooling across articles per category
4. **Store**: Push to Hopsworks Feature Store

**Max-Pooling Logic:**
```python
For each category:
    scores = [article1_score, article2_score, ...]
    vibe_vector[category] = max(scores)  # NOT average!
```

**Why Max-Pooling?**
- Captures hot/breaking news
- One major fire (0.9) dominates over many minor articles (0.2)
- Reflects "what's most intense right now"

### 6. Simplified Caching

**Strategy:**
- Check Hopsworks Artifact Store for existing visualization
- If exists: Return immediately (zero generation cost)
- If missing: Generate → Store → Return

**No Complex Cache:**
- No vibe-hash discretization
- No in-memory cache layers
- Simple file existence check in Hopsworks

## Configuration (backend/settings.py)

### Hopsworks Settings
```python
HOPSWORKS_ENABLED=true
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT_NAME=daily_collage
HOPSWORKS_HOST=c.app.hopsworks.ai
HOPSWORKS_REGION=us
HOPSWORKS_VIBE_FG=vibe_vectors
HOPSWORKS_ARTIFACT_COLLECTION=vibe_images
```

### Stability AI Settings
```python
STABILITY_API_KEY=your_api_key
STABILITY_ENABLE_POLISH=true
STABILITY_ATMOSPHERE_STRATEGY=prompt
STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true
```

### Layout Settings
```python
# Image size
image_width = 1024
image_height = 768

# Grid configuration (in code)
GRID_COLS = 8
GRID_ROWS_PER_ZONE = 3
```

## Usage Examples

### 1. Run Ingestion Pipeline

```bash
# Fetch news, classify, and store in Hopsworks
cd ml/ingestion
python hopsworks_pipeline.py --country sweden --max-articles 250

# Skip Hopsworks (for testing)
python hopsworks_pipeline.py --country sweden --no-hopsworks
```

### 2. Generate Visualization via API

```bash
# Start server
cd backend/server
uvicorn main:app --reload

# Request visualization
curl -X POST http://localhost:8000/api/visualization \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Sweden",
    "vibe_vector": {
      "emergencies": 0.8,
      "crime": 0.3,
      "festivals": 0.6
    }
  }'
```

### 3. Check Hopsworks Storage

```python
from backend.server.services.hopsworks import create_hopsworks_service
from backend.settings import settings

service = create_hopsworks_service(
    enabled=settings.hopsworks.enabled,
    api_key=settings.hopsworks.api_key,
    project_name=settings.hopsworks.project_name,
)

service.connect()

# Get latest vibe
vibe = service.get_latest_vibe_vector("Sweden")
print(vibe)

# Check if visualization exists
exists = service.visualization_exists("some_vibe_hash")
print(f"Cached: {exists}")
```

## ML Model Integration (Future)

The system is ready for ML model integration. Replace the keyword-based classification in `ml/ingestion/hopsworks_pipeline.py`:

```python
def classify_article(title: str, description: str) -> Dict[str, Tuple[float, str]]:
    """
    Current: Keyword-based classification
    Future: ML model inference
    """
    # TODO: Replace with:
    # model = load_model_from_hopsworks()
    # return model.predict(title, description)
    pass
```

**Model Requirements:**
- **Input**: Article title + description (text)
- **Output**: Dict with 9 categories
  - Each category: (score: float, tag: string)
  - Score range: -1.0 to 1.0
  - Tag: one of predefined tags for category

**Model Architecture Suggestion:**
- Multi-head transformer (BERT-based)
- 9 regression heads (for scores)
- 9 classification heads (for tags)
- Fine-tuned on Swedish news corpus

## Asset Creation Guide

### Quick Start Assets

For rapid prototyping, create simple placeholder assets:

```python
from PIL import Image, ImageDraw

def create_placeholder_asset(category, tag, intensity, size):
    """Create simple colored circle placeholder."""
    colors = {
        "emergencies": (255, 87, 51),  # Red-orange
        "crime": (138, 43, 226),       # Purple
        "festivals": (255, 215, 0),    # Gold
        # ... add more
    }
    
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    color = colors.get(category, (128, 128, 128))
    draw.ellipse([0, 0, size, size], fill=color + (230,))
    
    img.save(f"backend/assets/{category}_{tag}_{intensity}.png")

# Generate placeholders
for intensity, size in [("low", 80), ("med", 140), ("high", 200)]:
    create_placeholder_asset("emergencies", "fire", intensity, size)
```

### Professional Assets

For production, use:
1. **Vector Tools**: Figma, Adobe Illustrator
2. **AI Generation**: DALL-E, Midjourney with "sticker style" prompt
3. **Icon Libraries**: Flaticon, Icons8 (check licenses)

**Design Checklist:**
- [ ] Transparent background (PNG with alpha)
- [ ] Cartoon/sticker style (not realistic)
- [ ] High contrast for readability
- [ ] Consistent style across category
- [ ] Size appropriate for intensity level

## Testing

### Test Asset Loading
```bash
cd backend/server
python -c "
from backend.visualization.assets import AssetLibrary
lib = AssetLibrary('../assets')
asset = lib.get_asset('emergencies', 'fire', 0.8)
print(f'Asset size: {asset.size if asset else "Not found"}')
"
```

### Test Layout Generation
```bash
python -c "
from backend.visualization.composition import HybridComposer
composer = HybridComposer()
vibe = {'emergencies': 0.8, 'crime': 0.5}
image_bytes, hitboxes = composer.compose(vibe, 'TestCity')
print(f'Generated {len(image_bytes)} bytes, {len(hitboxes)} hitboxes')
"
```

### Test Full Pipeline
```bash
cd ml/ingestion
python hopsworks_pipeline.py --country sweden --max-articles 50
```

## Troubleshooting

### Issue: Assets not loading
**Solution**: 
1. Check `ASSETS_DIR` in settings
2. Verify PNG files exist with correct naming
3. Check file permissions

### Issue: Hopsworks connection fails
**Solution**:
1. Verify API key in `.env`
2. Check host configuration (use "c.app.hopsworks.ai" not full URL)
3. Ensure project name matches exactly

### Issue: Grid cells exhausted
**Solution**:
1. Increase `GRID_COLS` or `GRID_ROWS_PER_ZONE` in assets.py
2. Filter low-intensity signals before placement
3. Limit max signals per zone

### Issue: Stability AI timeout
**Solution**:
1. Increase timeout in settings (default: 60s)
2. Check API key validity
3. Use mock poller for testing (`STABILITY_ENABLE_POLISH=false`)

## Performance Metrics

**Expected Performance:**
- News fetch (250 articles): ~3-5 seconds
- Classification (250 articles): ~100-200ms (keyword) / ~2-5s (ML model)
- Layout generation: ~20-50ms
- Stability AI polish: ~10-20 seconds
- Hopsworks storage: ~500ms-1s
- **Total cold start**: ~15-30 seconds
- **Cached retrieval**: ~100-500ms

## Security Considerations

1. **API Keys**: Never commit to git, use `.env` files
2. **Hopsworks Access**: Use project-scoped API keys
3. **Rate Limiting**: Implement for public API endpoints
4. **Input Validation**: Sanitize city names and vibe vectors
5. **Image Storage**: Implement access controls on artifact store

## Next Steps

1. **Create Asset Library**: Design and add PNG stickers
2. **Train ML Model**: Replace keyword classification
3. **Schedule Ingestion**: Set up 6-hour cron job
4. **Deploy Backend**: Containerize and deploy to cloud
5. **Build Frontend**: Create React/Vue interface
6. **Add Analytics**: Track usage and cache hit rates
7. **Performance Tuning**: Optimize Hopsworks queries

## Support & Contribution

For questions or contributions:
1. Check existing documentation
2. Review code comments
3. Test changes locally
4. Submit pull requests with clear descriptions

---

**Implementation Status**: ✅ Core system complete, ready for assets and ML model integration
**Last Updated**: 2026-01-02
