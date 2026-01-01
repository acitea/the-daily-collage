# Phase 2: Hybrid Generation + Backend - Implementation Guide

Complete implementation of asset-based layout composition, Stability AI polishing, vibe-hash caching, and FastAPI endpoints.

## What's Been Implemented

### 1. **Settings & Configuration** (`backend/settings.py`)
- Centralized configuration for all backend components
- Stability AI credentials and settings (API key, model, denoise strength)
- Storage backend selection (local, S3, MinIO)
- Asset library and layout parameters
- FastAPI server settings
- All configurable via environment variables with safe defaults

**Key Classes:**
- `Settings`: Global configuration container
- `StabilityAISettings`: Polish engine configuration
- `StorageSettings`: Storage backend configuration
- `LayoutSettings`: Canvas and zone parameters
- `AssetSettings`: Asset library configuration

### 2. **Asset Library & Zone-Based Layout** (`backend/visualization/assets.py`)
- Maps signal categories + tags + intensity to PNG assets
- Fallback system for missing assets
- Zone-based vertical composition (sky/city/street)
- Hitbox tracking for interactive regions

**Key Classes:**
- `AssetLibrary`: Manages asset loading and mapping
- `ZoneLayoutComposer`: Places assets in zones and tracks hitboxes
- `Hitbox`: Represents clickable regions with metadata

**Features:**
- Automatic asset scaling based on signal intensity
- Zone assignment by category (weather→sky, crime→street, etc.)
- Comprehensive hitbox metadata (position, size, signal info)

### 3. **Stability AI Image Polish** (`backend/visualization/polish.py`)
- Real and mock implementations of image-to-image polishing
- Configurable denoising strength (~0.35 by default)
- Graceful fallback when API key not available

**Key Classes:**
- `StabilityAIPoller`: Real API integration
- `MockStabilityAIPoller`: Mock for testing/development
- `create_poller()`: Factory function

**Features:**
- Low denoising strength (0.35) preserves layout integrity
- Automatic prompt generation for cartoon/artistic style
- Timeout handling and error recovery

### 4. **Vibe-Hash Caching** (`backend/visualization/caching.py`)
- Deterministic cache keys from city + date + time window + discretized scores
- Persistent storage backends (local, S3, mock)
- Metadata storage with article associations
- Cache verification and management

**Key Classes:**
- `VibeHash`: Generates deterministic hashes
- `CacheMetadata`: Stores visualization metadata
- `StorageBackend`: Abstract storage interface
- `LocalStorageBackend`: File-based storage
- `MockS3StorageBackend`: In-memory S3-like storage
- `VibeCache`: High-level caching interface

**Features:**
- Hash format: `city_YYYY-MM-DD_HH-HH_hash`
- Discretization prevents cache fragmentation
- Hitbox preservation across requests
- Source article tracking

### 5. **Hybrid Composition Engine** (`backend/visualization/composition.py`)
- Replaced emoji templates with asset-based layout
- Step 1: Zone-based asset placement with hitbox tracking
- Step 2: Stability AI polish for artistic enhancement
- Complete vibe-vector to image pipeline

**Key Classes:**
- `HybridComposer`: Orchestrates layout + polish
- `VisualizationService`: High-level service interface
- `SignalIntensity`: Signal metadata

**Workflow:**
```
Vibe Vector (category→score) 
    ↓
[Layout Engine] Place assets in zones, record hitboxes
    ↓
[Polish Engine] Stability AI enhancement (0.35 denoise)
    ↓
[Cache] Store image + metadata + hitboxes
    ↓
Return: Image bytes + hitboxes + vibe hash
```

### 6. **FastAPI Endpoints** (`backend/server/main.py`)
Complete REST API for visualization generation and retrieval (no mock data).

**Key Endpoints:**

#### `POST /api/visualization`
Generate visualization from explicit vibe vector.
```json
Request:
{
  "city": "stockholm",
  "vibe_vector": {
    "traffic": 0.45,
    "weather_wet": 0.3,
    "crime": -0.2
  },
  "timestamp": "2025-12-11T03:30:00Z",
  "source_articles": [
    {"title": "...", "url": "...", "source": "..."}
  ]
}

Response:
{
  "city": "stockholm",
  "vibe_hash": "stockholm_2025-12-11_00-06_a3f4e2c1",
  "image_url": "/api/cache/images/stockholm_2025-12-11_00-06_a3f4e2c1.png",
  "hitboxes": [
    {
      "x": 100, "y": 200, "width": 80, "height": 80,
      "signal_category": "transportation",
      "signal_tag": "traffic",
      "signal_intensity": 0.45,
      "signal_score": 0.45
    }
  ],
  "vibe_vector": {...},
  "cached": false,
  "generated_at": "2025-12-11T..."
}
```

#### `GET /api/visualization/{vibe_hash}/image`
Retrieve cached image PNG.

#### `GET /api/cache/status`
Check if vibe is cached.

#### `GET /api/hitboxes/{vibe_hash}`
Get hitbox metadata for a visualization.

#### `GET /api/articles/{vibe_hash}`
Get source articles that contributed to visualization.

#### `GET /api/signal-categories`
List available signal categories.

#### `GET /api/supported-locations`
List supported geographic locations.

### 7. **Comprehensive Tests**

#### `tests/test_cache_and_hitbox.py`
Tests for cache determinism and hitbox stability:
- `TestVibeHashDeterminism`: Hash consistency, discretization, different cities/times
- `TestCacheDeterminism`: Cache hits, misses, and invalidation
- `TestHitboxStability`: Coordinate ranges, scaling, bounds checking
- `TestCacheMetadata`: Serialization and article preservation

#### `tests/test_integration.py`
End-to-end integration tests:
- `TestHybridComposerPipeline`: Full composition pipeline
- `TestVisualizationServiceIntegration`: Service-level functionality
- `TestAPIContractCompliance`: Response format validation
- `TestCacheStorageConsistency`: Metadata persistence
- `TestErrorHandling`: Graceful error handling

**Running Tests:**
```bash
pytest tests/test_cache_and_hitbox.py -v
pytest tests/test_integration.py -v
```

### 8. **CLI Utility for Frontend Development** (`backend/utils/generate_layout.py`)
Command-line tool to preview layouts before frontend integration.

**Usage:**
```bash
# Generate layout for sample vibe
python backend/utils/generate_layout.py --city stockholm --sample active

# Use custom vibe vector
python backend/utils/generate_layout.py --city stockholm --vibe '{"traffic": 0.5, "weather_wet": 0.3}'

# Load from JSON file
python backend/utils/generate_layout.py --city stockholm --config my_vibe.json

# List available samples
python backend/utils/generate_layout.py --list-samples

# Save to file
python backend/utils/generate_layout.py --sample crisis --output layout.json
```

**Output:**
```json
{
  "metadata": {
    "city": "stockholm",
    "vibe_hash": "stockholm_2025-12-11_00-06_a3f4e2c1",
    "timestamp": "2025-12-11T03:30:00Z",
    "vibe_vector": {...}
  },
  "layout": {
    "image_width": 1024,
    "image_height": 768
  },
  "hitboxes": [...]
}
```

## Architecture Diagram

```
FastAPI Request (Vibe Vector)
    ↓
[VisualizationService]
    ├─ Check VibeCache by hash
    ├─ Cache Miss → [HybridComposer]
    │   ├─ [ZoneLayoutComposer]
    │   │   └─ Place assets, track hitboxes
    │   │
    │   └─ [StabilityAIPoller]
    │       └─ Polish image (0.35 denoise)
    │
    ├─ Store in [StorageBackend]
    │   ├─ Save PNG to object storage
    │   └─ Save metadata to DB
    │
    └─ Return: Image URL + Hitboxes
         ↓
    FastAPI Response (VisualizationResponse)
```

## Data Flow

### Vibe Vector → Visualization Pipeline

1. **Input:** Vibe vector with signal scores (-1.0 to 1.0)
2. **Hashing:** Generate deterministic vibe hash from city + timestamp + discretized scores
3. **Cache Check:** Look up in VibeCache by hash
4. **Layout Generation:**
   - Expand vibe vector to (category, tag, intensity) tuples
   - Route signals to zones based on category
   - Place PNG assets, scale by intensity
   - Record hitbox coordinates and metadata
5. **Polish:** Apply Stability AI Img2Img (0.35 denoise) for style enhancement
6. **Caching:** Store image + metadata + hitboxes using vibe hash as key
7. **Response:** Return image URL + hitboxes for frontend rendering

### Frontend Integration

```javascript
// Frontend pseudocode
const response = await fetch('/api/visualization', {
  method: 'POST',
  body: JSON.stringify({
    city: 'stockholm',
    vibe_vector: { traffic: 0.5, weather_wet: 0.3 }
  })
});

const { image_url, hitboxes, vibe_hash } = await response.json();

// Render image
<img src={image_url} />

// Create interactive overlays from hitboxes
hitboxes.forEach(hb => {
  const overlay = <div 
    onClick={() => showArticles(hb.signal_category)}
    style={{
      position: 'absolute',
      left: hb.x, top: hb.y,
      width: hb.width, height: hb.height
    }}
  />
});
```

## Configuration via Environment Variables

```bash
# Stability AI
export STABILITY_API_KEY="your-key-here"
export STABILITY_API_HOST="https://api.stability.ai"
export STABILITY_MODEL_ID="stable-diffusion-v1-6-768-768"
export STABILITY_ENABLE_POLISH="true"

# Storage
export STORAGE_BACKEND="local"  # or "s3", "minio"
export STORAGE_BUCKET_NAME="vibe-images"
export LOCAL_STORAGE_DIR="./storage/vibes"
export METADATA_DB_URL="sqlite:///storage/metadata.db"

# Assets
export ASSETS_DIR="./backend/assets"

# API
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## Performance Characteristics

### Cache Efficiency
- **Deterministic hashing:** Same vibe vector always produces same key
- **Discretization:** Scores rounded to 0.1 prevent fragmentation
- **Time windows:** 6-hour windows reduce cache proliferation

### Image Generation
- **Layout:** ~100-200ms (asset loading + placement)
- **Polish:** ~5-30s (Stability API call with network latency)
- **Total:** First request ~5-35s, cached requests <100ms

### Storage
- **Local backend:** Filesystem-based, suitable for single machine
- **S3 backend:** Production-ready, supports distributed deployments
- **Metadata:** JSON + SQLite/PostgreSQL for flexible queries

## Testing Strategy

### Cache Determinism
- Same input → same hash every time
- Discretization prevents false cache misses
- Different cities/times → different hashes

### Hitbox Stability
- Coordinates always integers
- Within canvas bounds
- Scale proportional to intensity
- Preserved across requests

### Integration Tests
- Full pipeline: vibe → layout → polish → cache → response
- API contracts: response format, data ranges
- Error handling: missing assets, zero vectors, API failures

## Migration from Phase 1

### Removed
- Emoji-based templates
- Mock signal data (SignalIntensity list)
- In-memory only caching

### Added
- Asset PNG library integration
- Real vibe vector scoring (-1.0 to 1.0)
- Vibe-hash based caching
- Stability AI polish
- Storage backends (local/S3)
- Comprehensive hitbox metadata
- Proper error handling

### Backwards Compatibility
- `SignalIntensity` class kept for compatibility
- Deprecated cache still available for fallback
- Old endpoints removed but new ones fully featured

## Next Steps (Phase 3)

1. **Frontend Integration:**
   - React component for visualization display
   - Interactive hitbox overlays
   - Article modal integration

2. **ML Pipeline Integration:**
   - Real vibe vectors from classification model
   - Hopsworks feature store connection
   - GDELT news ingestion

3. **Production Deployment:**
   - Docker containerization
   - Database setup (PostgreSQL)
   - S3/MinIO configuration
   - Stability AI API key management
   - Monitoring and observability

## Troubleshooting

### Images not generating
- Check assets directory exists: `ls backend/assets/`
- Verify PIL/Pillow installed: `pip install pillow`
- Check logs for asset loading errors

### Stability API timeouts
- Verify API key: `echo $STABILITY_API_KEY`
- Check network connectivity
- Try MockStabilityAIPoller for testing

### Cache misses on similar vibes
- Remember discretization: vibe vectors rounded to 0.1
- Use `/api/cache/status` to check if vibe is cached
- Verify city name matches exactly (case-insensitive)

### Hitbox coordinates out of bounds
- Check composite image dimensions (should be 1024x768)
- Verify asset sizes are reasonable
- Enable logging for placement debug info

## References

- [Phase 2 Requirements](phases/phase2_generation_backend.md)
- [Stability AI API Docs](https://platform.stability.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pillow Image Library](https://pillow.readthedocs.io/)
