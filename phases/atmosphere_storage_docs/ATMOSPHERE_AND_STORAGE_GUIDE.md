# Atmosphere & Storage Enhancement Guide

## Overview

This guide explains the new atmosphere and storage backend features added to The Daily Collage backend.

## Atmosphere Effects: Two Strategies

The backend now supports **two approaches** for adding atmospheric vibes to visualizations. You can easily switch between them or try both.

### Strategy 1: Atmosphere Assets (Asset-Based)

**What it does:** Applies PNG overlay images to the entire canvas based on signal composition.

**Examples:**
- Rain effect overlay for weather signals
- Heat haze for temperature extremes
- Festive particle effects for celebrations
- Fire glow for emergencies

**Strengths:**
- ✅ Fully controllable visually
- ✅ Deterministic and predictable
- ✅ Works offline (no API calls)
- ✅ Easy to adjust opacity per signal intensity

**Weaknesses:**
- ❌ Requires pre-made PNG assets for each atmosphere
- ❌ Limited to predefined effects
- ❌ May not blend naturally with layout

**How to Use:**

```bash
# Set environment variable
export STABILITY_ATMOSPHERE_STRATEGY=asset

# Or in .env file
STABILITY_ATMOSPHERE_STRATEGY=asset
```

**Asset Mapping:**

The atmosphere assets are defined in `AssetLibrary.ATMOSPHERE_MAP`:

```python
ATMOSPHERE_MAP = {
    "weather_wet": {
        "rain": "atmosphere_rain.png",
        "snow": "atmosphere_snow.png",
        "flood": "atmosphere_flood.png",
    },
    "weather_temp": {
        "hot": "atmosphere_heat.png",
        "cold": "atmosphere_cold.png",
    },
    # ... more mappings
}
```

**Creating New Atmosphere Assets:**

1. Create PNG files with transparent backgrounds
2. Same dimensions as layout (1024x768)
3. Place in `backend/assets/` directory
4. Add mapping to `AssetLibrary.ATMOSPHERE_MAP`
5. The overlay will be applied at opacity proportional to signal intensity

**File Example:**
- `backend/assets/atmosphere_rain.png` - Rainy overlay (semi-transparent water droplets)
- `backend/assets/atmosphere_snow.png` - Snowy overlay (snowflakes)

### Strategy 2: Prompt-Based Atmosphere

**What it does:** Incorporates atmospheric descriptions into the Stability AI img2img prompt.

**Examples:**
- "rainy and gloomy" for rain signals
- "hot and vibrant" for heat signals
- "festive and joyful" for celebrations
- "chaotic and dangerous" for emergencies

**Strengths:**
- ✅ AI-generated effects (creative & natural)
- ✅ Blends seamlessly with layout
- ✅ Works with any signal (no asset files needed)
- ✅ Flexible and adaptable

**Weaknesses:**
- ❌ Requires Stability AI API access
- ❌ Less deterministic (API variations)
- ❌ Higher cost (additional API calls)
- ❌ May alter layout slightly

**How to Use:**

```bash
# Set environment variable
export STABILITY_ATMOSPHERE_STRATEGY=prompt

# Enable atmosphere in prompts (default: true)
export STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true

# Or in .env file
STABILITY_ATMOSPHERE_STRATEGY=prompt
STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true
```

**Prompt Generation:**

The `AtmosphereDescriptor` class automatically generates prompts:

```python
from backend.visualization.atmosphere import AtmosphereDescriptor

signals = [
    ("weather_wet", "rain", 0.8, 0.8),
    ("emergencies", "fire", 0.6, 0.6),
]

prompt = AtmosphereDescriptor.generate_atmosphere_prompt(signals)
# Result: "rainy and gloomy, fiery and intense"
```

**Customizing Descriptions:**

Edit `ATMOSPHERE_DESCRIPTIONS` in `atmosphere.py`:

```python
ATMOSPHERE_DESCRIPTIONS = {
    ("weather_wet", "rain"): [
        "rainy and gloomy",        # Default
        "wet streets reflecting light",
        "overcast atmosphere",
    ],
    # Add your own:
    ("custom_category", "custom_tag"): [
        "your custom description",
        "alternative description",
    ],
}
```

### Comparing the Strategies

| Aspect | Asset | Prompt |
|--------|-------|--------|
| **Setup** | Add PNG files | Configure API |
| **Cost** | None (local) | Per API call |
| **Speed** | Fast (local overlay) | Slower (API call) |
| **Flexibility** | Fixed effects | AI-generated |
| **Determinism** | 100% | ~85% (API variations) |
| **Blending** | Good with tuning | Excellent (AI) |
| **Visual Control** | Full | Partial |
| **Internet Required** | No | Yes (with real API) |

## Storage Backends

The backend now supports three storage options:

### 1. Local Storage (Default)

Stores images and metadata on the local filesystem.

**Configuration:**

```bash
export STORAGE_BACKEND=local
export LOCAL_STORAGE_DIR=./storage/vibes
```

**Storage Structure:**

```
storage/vibes/
├── images/
│   ├── stockholm_2025-01-01_00-06_a3f4e2c1.png
│   ├── stockholm_2025-01-01_06-12_b4e5f3d2.png
│   └── ...
└── metadata.json
```

**Pros:**
- ✅ Zero dependencies
- ✅ Fast access
- ✅ Good for development/testing
- ✅ No external service required

**Cons:**
- ❌ Not suitable for multi-instance deployments
- ❌ Limited scalability
- ❌ No cloud backup

### 2. S3 / MinIO Storage

Store images in S3-compatible object storage.

**Configuration:**

```bash
export STORAGE_BACKEND=s3
export STORAGE_BUCKET_NAME=vibe-images
export S3_ENDPOINT=https://s3.amazonaws.com  # or MinIO URL
export S3_ACCESS_KEY=your_access_key
export S3_SECRET_KEY=your_secret_key
export S3_REGION=us-east-1
```

**Pros:**
- ✅ Highly scalable
- ✅ Multi-instance ready
- ✅ Cloud-native
- ✅ Built-in redundancy (AWS)

**Cons:**
- ❌ Requires AWS account or MinIO setup
- ❌ Monthly costs
- ❌ Network latency

### 3. Hopsworks Artifact Store (NEW!)

Store images as artifacts in Hopsworks feature store.

**Installation:**

```bash
pip install hopsworks
```

**Configuration:**

```bash
export HOPSWORKS_ENABLED=true
export HOPSWORKS_API_KEY=your_hopsworks_api_key
export HOPSWORKS_HOST=c.app.hopsworks.ai  # or your instance
export HOPSWORKS_PROJECT_NAME=daily_collage
export HOPSWORKS_REGION=us
export HOPSWORKS_ARTIFACT_COLLECTION=vibe_images
```

**Usage in Code:**

```python
from backend.visualization.caching import HopsworksStorageBackend, VibeCache

# Create backend
backend = HopsworksStorageBackend(
    api_key="your_api_key",
    project_name="daily_collage",
    host="c.app.hopsworks.ai",
    artifact_collection="vibe_images",
)

# Use with VibeCache
cache = VibeCache(backend)
image_url, metadata = cache.set(
    city="stockholm",
    timestamp=datetime.now(),
    vibe_vector={"traffic": 0.5, "weather": 0.3},
    image_data=image_bytes,
    hitboxes=[...],
)
```

**Pros:**
- ✅ Native integration with Hopsworks ML platform
- ✅ Artifact versioning built-in
- ✅ Feature store alignment
- ✅ ML ops ready

**Cons:**
- ❌ Requires Hopsworks subscription
- ❌ New feature (less battle-tested)
- ❌ Metadata still in local store (current limitation)

## Switching Strategies

All settings are environment-based and can be changed at runtime:

### Quick Test Script

```python
from backend.visualization.composition import HybridComposer
from backend.settings import settings

# Test Asset-based atmosphere
settings.stability_ai.atmosphere_strategy = "asset"
composer = HybridComposer()
image_bytes, hitboxes = composer.compose(
    {"weather_wet": 0.8, "crime": 0.3},
    location="stockholm"
)

# Test Prompt-based atmosphere
settings.stability_ai.atmosphere_strategy = "prompt"
composer = HybridComposer()
image_bytes, hitboxes = composer.compose(
    {"weather_wet": 0.8, "crime": 0.3},
    location="stockholm"
)
```

### CLI Tool

```bash
# Generate with asset atmosphere
python backend/utils/generate_layout.py \
    --sample rainy \
    --output rainy_asset.json

# Generate with prompt atmosphere (via composition)
python backend/utils/generate_layout.py \
    --sample rainy \
    --output rainy_prompt.json
```

## Environment Variables Reference

### Atmosphere Settings

```bash
# Strategy: "asset" or "prompt"
STABILITY_ATMOSPHERE_STRATEGY=prompt

# Include atmosphere description in prompt (if using PROMPT strategy)
STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true
```

### Hopsworks Settings

```bash
# Enable/disable Hopsworks integration
HOPSWORKS_ENABLED=false

# API key for Hopsworks
HOPSWORKS_API_KEY=<your_key>

# Hopsworks project details
HOPSWORKS_PROJECT_NAME=daily_collage
HOPSWORKS_HOST=c.app.hopsworks.ai
HOPSWORKS_REGION=us

# Artifact and feature group names
HOPSWORKS_ARTIFACT_COLLECTION=vibe_images
HOPSWORKS_VIBE_FG=vibe_vectors
```

## Example: Trying Both Approaches

Create a test script:

```python
#!/usr/bin/env python3
"""
Compare asset vs. prompt atmosphere strategies.
"""

import os
import io
from PIL import Image
from backend.visualization.composition import HybridComposer
from backend.settings import settings

def test_atmosphere_strategy(strategy: str, vibe_vector: dict):
    """Test a single atmosphere strategy."""
    print(f"\n{'='*60}")
    print(f"Testing {strategy.upper()} atmosphere strategy")
    print(f"{'='*60}")
    
    # Update setting
    settings.stability_ai.atmosphere_strategy = strategy
    
    # Create fresh composer (reads settings)
    composer = HybridComposer()
    
    # Generate image
    image_bytes, hitboxes = composer.compose(
        vibe_vector,
        location="stockholm"
    )
    
    # Save output
    filename = f"test_atmosphere_{strategy}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes)
    
    print(f"✓ Generated: {filename}")
    print(f"  Size: {len(image_bytes)} bytes")
    print(f"  Hitboxes: {len(hitboxes)}")

# Test data
vibe_vector = {
    "weather_wet": 0.8,      # Rain - strong
    "festivals": 0.5,         # Celebration - moderate
    "emergencies": -0.4,      # Fire danger - slight
}

# Test both strategies
test_atmosphere_strategy("asset", vibe_vector)
test_atmosphere_strategy("prompt", vibe_vector)

print("\n✓ Comparison complete. Check test_atmosphere_*.png files")
```

## Troubleshooting

### "Atmosphere strategy not found"

**Issue:** `ERROR: Invalid atmosphere_strategy 'xyz'`

**Solution:**
```bash
# Must be 'asset' or 'prompt'
export STABILITY_ATMOSPHERE_STRATEGY=prompt
```

### "Hopsworks connection failed"

**Issue:** `Failed to connect to Hopsworks`

**Solution:**
```bash
# Check API key and host
export HOPSWORKS_API_KEY=your_correct_key
export HOPSWORKS_HOST=c.app.hopsworks.ai
# Install hopsworks package
pip install hopsworks
```

### "Atmosphere asset not found"

**Issue:** Atmosphere overlay not applying

**Solution:**
```bash
# Check asset file exists
ls backend/assets/atmosphere_*.png

# Verify mapping in assets.py
# If adding new atmosphere, ensure it's in ATMOSPHERE_MAP
```

### "Prompt not being applied"

**Issue:** Stability AI prompt isn't influencing output

**Solution:**
```bash
# Ensure PROMPT strategy is selected
export STABILITY_ATMOSPHERE_STRATEGY=prompt

# Ensure atmosphere is enabled in prompt
export STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true

# Check logs for generated prompt
# Add debug logging to see what prompt is sent
```

## Best Practices

1. **Start with Assets**: Use asset-based atmosphere for testing (no API calls)
2. **Iterate with Prompts**: Once happy with layout, try prompt-based for refinement
3. **Monitor Costs**: Prompt-based requires API calls; set budget alerts
4. **Version Assets**: Keep atmosphere PNGs in version control
5. **Test Both**: Run A/B tests to see which resonates with users
6. **Fallback Strategy**: Asset mode continues working even if API is down
7. **Use Hopsworks**: If on ML ops track, integrate artifacts with feature store

## Next Steps

1. Create atmosphere assets for primary signals
2. Configure environment variables for your preferred strategy
3. Test with sample vibe vectors
4. Integrate with real ML pipeline from Hopsworks
5. Monitor output quality and adjust settings

---

**Questions?** See [PHASE2_IMPLEMENTATION.md](PHASE2_IMPLEMENTATION.md) or check main README.
