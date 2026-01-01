# Phase 2 Quick Reference

## Architecture at a Glance

```
POST /api/visualization
├─ Input: { city, vibe_vector }
└─ Output: { vibe_hash, image_url, hitboxes }

Internally:
1. Hash = VibeHash.generate(city, timestamp, vibe_vector)
2. Check VibeCache.get(hash)
3. If miss: HybridComposer.compose() → [Layout → Polish → Store]
4. Return image_url + hitboxes
```

## Key Classes

### VibeHash
- Generates deterministic cache key
- Format: `city_YYYY-MM-DD_HH-HH_hash`
- Scores discretized to 0.1 steps
- Time windows: 6 hours

### HybridComposer
- Two-step pipeline: Layout → Polish
- Layout: Zone-based asset placement
- Polish: Stability AI Img2Img (0.35 denoise)

### VibeCache
- High-level caching interface
- Check: `cache.get(city, timestamp, vibe_vector)`
- Store: `cache.set(city, timestamp, vibe_vector, image_data, hitboxes)`

### VisualizationService
- Orchestrates full pipeline
- `generate_or_get(city, vibe_vector)`
- Returns: (image_bytes, metadata_dict)

## API Endpoints

### POST /api/visualization
```json
Request: {
  "city": "stockholm",
  "vibe_vector": {
    "traffic": 0.45,
    "weather_wet": 0.3
  },
  "source_articles": [...]
}

Response: {
  "city": "stockholm",
  "vibe_hash": "stockholm_2025-12-11_00-06_a3f4e2c1",
  "image_url": "/api/cache/images/stockholm_2025-12-11_00-06_a3f4e2c1.png",
  "hitboxes": [{...}],
  "vibe_vector": {...},
  "cached": false
}
```

### GET /api/visualization/{vibe_hash}/image
Returns PNG image data

### GET /api/cache/status?city=stockholm&vibe_vector={"traffic":0.5}
Check if cached

### GET /api/hitboxes/{vibe_hash}
Get clickable regions

### GET /api/articles/{vibe_hash}
Get source articles

## Configuration

```bash
export STABILITY_API_KEY="your-key"
export STABILITY_ENABLE_POLISH="true"
export STORAGE_BACKEND="local"
export ASSETS_DIR="./backend/assets"
```

## Testing

```bash
# All tests
pytest tests/ -v

# Quick verify
python test_phase2.py

# Specific test
pytest tests/test_cache_and_hitbox.py::TestVibeHashDeterminism -v
```

## CLI Tools

```bash
# Generate layout
python backend/utils/generate_layout.py --sample active

# Custom vibe
python backend/utils/generate_layout.py --vibe '{"traffic": 0.5}'

# Save output
python backend/utils/generate_layout.py --sample crisis --output layout.json
```

## Data Model

### Vibe Vector
```python
{
  "transportation": 0.45,      # -1.0 (severe) to 1.0 (high)
  "weather_temp": -0.2,
  "weather_wet": 0.3,
  "crime": -0.8,
  "festivals": 0.7,
  "sports": 0.5,
  "emergencies": -0.9,
  "economics": 0.1,
  "politics": 0.2
}
```

### Hitbox
```python
{
  "x": 100,
  "y": 200,
  "width": 80,
  "height": 80,
  "signal_category": "transportation",
  "signal_tag": "traffic",
  "signal_intensity": 0.45,
  "signal_score": 0.45
}
```

## Performance

- **Cache hit**: <100ms
- **Cache miss**: ~5-30s (includes Stability API call)
- **Image size**: ~4KB (PNG)
- **Hitboxes per image**: 0-9 (depends on signals)

## Debugging

### Check settings
```python
from backend.settings import settings
print(settings.storage.backend)
print(settings.stability_ai.image_strength)
```

### Test vibe hash
```python
from backend.visualization.caching import VibeHash
from datetime import datetime

hash = VibeHash.generate(
    "stockholm",
    datetime.now(),
    {"traffic": 0.5}
)
print(hash)
```

### Generate layout
```python
from backend.visualization.composition import HybridComposer

composer = HybridComposer()
image, hitboxes = composer.compose(
    {"traffic": 0.5, "weather_wet": 0.3},
    "stockholm"
)
print(len(image), "bytes,", len(hitboxes), "hitboxes")
```

## Common Issues

### Empty hitboxes
- Assets directory may not exist
- Set `ASSETS_DIR` environment variable
- Create sample PNG files in assets folder

### Cache not hitting
- Check vibe hash format
- Verify city name matches exactly
- Use `/api/cache/status` to debug

### Stability API errors
- Verify `STABILITY_API_KEY` set
- Check network connectivity
- Use `STABILITY_ENABLE_POLISH=false` to skip polish

### Wrong coordinates
- Verify canvas dimensions (should be 1024x768)
- Check hitbox math in `ZoneLayoutComposer`
- Enable debug logging

## Files Overview

| File | Purpose | Key Classes |
|------|---------|-------------|
| `settings.py` | Configuration | Settings, StabilityAISettings, StorageSettings |
| `assets.py` | Asset library | AssetLibrary, ZoneLayoutComposer, Hitbox |
| `polish.py` | Stability integration | StabilityAIPoller, MockStabilityAIPoller |
| `caching.py` | Cache management | VibeHash, VibeCache, StorageBackend |
| `composition.py` | Main pipeline | HybridComposer, VisualizationService |
| `server/main.py` | FastAPI server | FastAPI app, endpoints |

## Environment Variables

### Stability AI
- `STABILITY_API_KEY` - API key (required for real polish)
- `STABILITY_ENABLE_POLISH` - true/false (default: true)
- `STABILITY_MODEL_ID` - Model to use (default: stable-diffusion-v1-6-768-768)

### Storage
- `STORAGE_BACKEND` - local, s3, minio (default: local)
- `LOCAL_STORAGE_DIR` - Path (default: ./storage/vibes)
- `STORAGE_BUCKET_NAME` - S3 bucket (default: vibe-images)

### Assets & Layout
- `ASSETS_DIR` - Asset path (default: ./backend/assets)
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)

## Response Schema

```json
{
  "city": "string",
  "vibe_hash": "string (format: city_YYYY-MM-DD_HH-HH_hash)",
  "image_url": "string (URL to PNG)",
  "hitboxes": [
    {
      "x": "integer",
      "y": "integer",
      "width": "integer",
      "height": "integer",
      "signal_category": "string",
      "signal_tag": "string",
      "signal_intensity": "float 0-1",
      "signal_score": "float -1 to 1"
    }
  ],
  "vibe_vector": "object (category -> score)",
  "cached": "boolean",
  "generated_at": "ISO datetime string"
}
```

## Caching Strategy

1. **Deterministic Key**: Same input → same hash
2. **Discretization**: Reduce fragmentation with 0.1 step
3. **Time Windows**: 6 hours per window (4 per day)
4. **Persistence**: Store image + metadata + hitboxes
5. **Validation**: All bounds checked, coordinates integers

## For Frontend Developers

1. Use `/api/visualization` endpoint with vibe vector
2. Get back `image_url` and `hitboxes`
3. Render image
4. Layer hitbox divs for interactivity
5. On click, fetch `/api/articles/{vibe_hash}` for news

Example:
```javascript
const response = await fetch('/api/visualization', {
  method: 'POST',
  body: JSON.stringify({
    city: 'stockholm',
    vibe_vector: {traffic: 0.5, weather_wet: 0.3}
  })
});
const {image_url, hitboxes} = await response.json();
// Render image and create overlays
```

## Status Emoji

- ✅ Working
- 🔧 In development  
- 📝 Documented
- ✨ Tested
- 🚀 Ready for production

**Overall: ✅✅✅✅✅**
