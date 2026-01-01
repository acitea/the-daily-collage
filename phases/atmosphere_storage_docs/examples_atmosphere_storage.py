#!/usr/bin/env python3
"""
Practical examples for Atmosphere and Storage features.

Run these examples to see the new features in action.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║          ATMOSPHERE & STORAGE FEATURES - PRACTICAL EXAMPLES                ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# EXAMPLE 1: Comparing Atmosphere Strategies
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 1: Comparing Atmosphere Strategies Side-by-Side")
print("="*80)

example1_code = """
from backend.visualization.composition import HybridComposer
from backend.settings import settings
import io

# Test vibe vector (rainy, festive, slight crime)
vibe_vector = {
    "weather_wet": 0.8,    # Strong rain
    "festivals": 0.6,      # Moderate celebration
    "crime": -0.3,         # Slight crime
}

print("\\n1. ASSET STRATEGY (PNG Overlays)")
print("-" * 40)
settings.stability_ai.atmosphere_strategy = "asset"
composer_asset = HybridComposer()
image_bytes, hitboxes = composer_asset.compose(
    vibe_vector, 
    location="stockholm"
)
print(f"✓ Image generated: {len(image_bytes)} bytes")
print(f"✓ Atmosphere: PNG overlay applied (rain effect + celebration effect)")
print(f"✓ Deterministic: Yes (same input = same output)")
print(f"✓ API calls: 0")

print("\\n2. PROMPT STRATEGY (AI Enhancement)")
print("-" * 40)
settings.stability_ai.atmosphere_strategy = "prompt"
composer_prompt = HybridComposer()
image_bytes, hitboxes = composer_prompt.compose(
    vibe_vector,
    location="stockholm"
)
print(f"✓ Image generated: {len(image_bytes)} bytes")
print(f"✓ Atmosphere: 'rainy and gloomy, festive and joyful' added to prompt")
print(f"✓ Deterministic: ~85% (API variations)")
print(f"✓ API calls: 1 (Stability AI img2img)")

print("\\n3. WHICH TO USE?")
print("-" * 40)
print("Asset:  Fast, offline, deterministic, requires PNG files")
print("Prompt: AI-enhanced, natural blending, requires API access")
"""

print(example1_code)

# ============================================================================
# EXAMPLE 2: Creating Custom Atmosphere Assets
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Creating & Using Custom Atmosphere Assets")
print("="*80)

example2_code = """
from backend.visualization.assets import AssetLibrary, ZoneLayoutComposer

# Step 1: Create atmosphere asset (already in backend/assets/)
# File: backend/assets/atmosphere_custom_mood.png
# Dimensions: 1024x768 (same as canvas)
# Format: PNG with transparency

# Step 2: Register in AssetLibrary
# Edit backend/visualization/assets.py:
#
# ATMOSPHERE_MAP = {
#     ...existing...
#     "custom_category": {
#         "custom_mood": "atmosphere_custom_mood.png",
#     },
# }

# Step 3: Use it
library = AssetLibrary("./backend/assets")

# Check if atmosphere asset exists
has_custom = library.has_atmosphere_asset("weather_temp", "hot")
print(f"Has 'hot' atmosphere: {has_custom}")

# Get the atmosphere asset
atmosphere = library.get_atmosphere_asset("weather_temp", "hot")
if atmosphere:
    print(f"✓ Loaded atmosphere: {atmosphere.size}")

# Step 4: It's automatically applied during composition
signals = [("weather_temp", "hot", 0.8, 0.8)]
composer = ZoneLayoutComposer(
    image_width=1024,
    image_height=768,
    assets_dir="./backend/assets"
)
image, hitboxes = composer.compose(
    signals,
    apply_atmosphere_assets=True
)
print(f"✓ Atmosphere applied with opacity: {int(255 * 0.8)}")
"""

print(example2_code)

# ============================================================================
# EXAMPLE 3: Generating Atmosphere Prompts
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Generating Atmosphere Prompts for img2img")
print("="*80)

example3_code = """
from backend.visualization.atmosphere import AtmosphereDescriptor

# Complex signal composition
signals = [
    ("weather_wet", "rain", 0.9, 0.9),       # Heavy rain
    ("emergencies", "fire", 0.7, -0.7),      # Fire danger
    ("festivals", "celebration", 0.5, 0.5),  # Some celebration
    ("politics", "protest", 0.4, -0.4),      # Protest
]

# Generate atmosphere prompt
prompt = AtmosphereDescriptor.generate_atmosphere_prompt(
    signals, 
    max_descriptions=3
)
print(f"Generated prompt: '{prompt}'")
# Output: "rainy and gloomy, flames and danger, protest mood"

# Get overall mood
mood = AtmosphereDescriptor.get_mood(signals)
print(f"Detected mood: {mood}")
# Output: "chaotic"

# The prompt is automatically incorporated into Stability AI request:
# Full prompt becomes:
# "A colorful, artistic, cartoonish illustration of stockholm, news visualization,
#  rainy and gloomy, flames and danger, protest mood"
"""

print(example3_code)

# ============================================================================
# EXAMPLE 4: Using Hopsworks Storage
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Using Hopsworks Artifact Store")
print("="*80)

example4_code = """
from backend.visualization.caching import HopsworksStorageBackend, VibeCache, VibeHash
from datetime import datetime

# Initialize Hopsworks backend
backend = HopsworksStorageBackend(
    api_key="your_hopsworks_api_key",
    project_name="daily_collage",
    host="c.app.hopsworks.ai",
    artifact_collection="vibe_images"
)

# Use with VibeCache
cache = VibeCache(backend)

# Store a visualization
vibe_vector = {"traffic": 0.6, "weather": 0.3}
image_bytes = b"...png data..."
hitboxes = [{"x": 100, "y": 200, "width": 50, "height": 50, ...}]

image_url, metadata = cache.set(
    city="stockholm",
    timestamp=datetime.now(),
    vibe_vector=vibe_vector,
    image_data=image_bytes,
    hitboxes=hitboxes,
    source_articles=[
        {"title": "Traffic jam on E4", "url": "..."},
    ]
)

print(f"✓ Stored in Hopsworks: {image_url}")
print(f"✓ Vibe hash: {metadata.vibe_hash}")

# Retrieve later (with cache hit)
image_data, retrieved_metadata = cache.get(
    city="stockholm",
    timestamp=datetime.now(),
    vibe_vector=vibe_vector
)

if image_data:
    print(f"✓ Cache hit! Retrieved {len(image_data)} bytes from Hopsworks")
    print(f"✓ Articles: {len(retrieved_metadata.source_articles)}")
"""

print(example4_code)

# ============================================================================
# EXAMPLE 5: Environment-Based Configuration
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: Environment Variable Configuration")
print("="*80)

example5_code = """
# .env file or export commands

# ATMOSPHERE SETTINGS
export STABILITY_ATMOSPHERE_STRATEGY=prompt
export STABILITY_INCLUDE_ATMOSPHERE_IN_PROMPT=true

# STORAGE SETTINGS - Choose one:

# Option A: Local Storage (Development)
export STORAGE_BACKEND=local
export LOCAL_STORAGE_DIR=./storage/vibes

# Option B: S3/MinIO (Production)
export STORAGE_BACKEND=s3
export STORAGE_BUCKET_NAME=vibe-images
export S3_ENDPOINT=https://s3.amazonaws.com
export S3_ACCESS_KEY=your_key
export S3_SECRET_KEY=your_secret

# Option C: Hopsworks (ML Ops)
export HOPSWORKS_ENABLED=true
export HOPSWORKS_API_KEY=your_api_key
export HOPSWORKS_HOST=c.app.hopsworks.ai
export HOPSWORKS_PROJECT_NAME=daily_collage

# Run backend
python backend/server/main.py
# Settings automatically loaded from environment!
"""

print(example5_code)

# ============================================================================
# EXAMPLE 6: A/B Testing Both Strategies
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: A/B Testing Atmosphere Strategies")
print("="*80)

example6_code = """
import random
from datetime import datetime
from backend.visualization.composition import HybridComposer
from backend.settings import settings

def generate_visualization_with_random_strategy(vibe_vector, location):
    \"\"\"Randomly select atmosphere strategy for A/B test.\"\"\"
    
    # 50/50 split
    strategy = random.choice(["asset", "prompt"])
    settings.stability_ai.atmosphere_strategy = strategy
    
    # Create fresh composer
    composer = HybridComposer()
    image_bytes, hitboxes = composer.compose(vibe_vector, location)
    
    return {
        "image": image_bytes,
        "hitboxes": hitboxes,
        "strategy": strategy,
        "timestamp": datetime.now(),
    }

# Usage
vibe = {"weather_wet": 0.7, "festivals": 0.5}

# Generate 10 visualizations (5 with each strategy)
results = [
    generate_visualization_with_random_strategy(vibe, "stockholm")
    for _ in range(10)
]

asset_count = sum(1 for r in results if r["strategy"] == "asset")
prompt_count = sum(1 for r in results if r["strategy"] == "prompt")

print(f"Asset strategy: {asset_count}")
print(f"Prompt strategy: {prompt_count}")

# Later: Collect user feedback and compare strategies
# Update to use strategy with higher engagement/satisfaction
"""

print(example6_code)

# ============================================================================
# EXAMPLE 7: Error Handling & Fallbacks
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 7: Graceful Fallbacks & Error Handling")
print("="*80)

example7_code = """
from backend.visualization.caching import HopsworksStorageBackend, LocalStorageBackend
from backend.settings import settings

# Graceful fallback: Try Hopsworks, fall back to Local
def get_storage_backend():
    if settings.hopsworks.enabled and settings.hopsworks.api_key:
        try:
            backend = HopsworksStorageBackend(
                api_key=settings.hopsworks.api_key,
                host=settings.hopsworks.host,
                project_name=settings.hopsworks.project_name,
            )
            if backend._is_connected():
                print("✓ Using Hopsworks backend")
                return backend
        except Exception as e:
            print(f"⚠ Hopsworks unavailable: {e}")
    
    # Fallback to local
    print("✓ Using Local storage backend")
    return LocalStorageBackend(settings.storage.local_storage_dir)

# Missing atmosphere assets: Graceful fallback
from backend.visualization.assets import AssetLibrary

library = AssetLibrary("./backend/assets")

# Try to get specific atmosphere
atmosphere = library.get_atmosphere_asset("weather_wet", "rain")

if not atmosphere:
    print("⚠ Rain atmosphere asset not found, skipping")
    # Composition continues without that effect
else:
    print("✓ Rain atmosphere loaded")

# Invalid atmosphere strategy: Validation catches it
settings.stability_ai.atmosphere_strategy = "invalid"
issues = settings.validate()

if issues:
    for issue in issues:
        print(f"⚠ {issue}")
    # Fix: Set to valid value
    settings.stability_ai.atmosphere_strategy = "prompt"
"""

print(example7_code)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("QUICK REFERENCE")
print("="*80)

reference = """
ATMOSPHERE STRATEGIES:
  Asset:  STABILITY_ATMOSPHERE_STRATEGY=asset
  Prompt: STABILITY_ATMOSPHERE_STRATEGY=prompt

STORAGE BACKENDS:
  Local:  STORAGE_BACKEND=local
  S3:     STORAGE_BACKEND=s3 + S3_* variables
  HW:     HOPSWORKS_ENABLED=true + HW_* variables

ADDING NEW ATMOSPHERE ASSETS:
  1. Create PNG: backend/assets/atmosphere_*.png
  2. Map it: ATMOSPHERE_MAP["category"]["tag"] = "atmosphere_*.png"
  3. Auto-applied during composition

ADDING NEW ATMOSPHERE DESCRIPTIONS:
  1. Edit: backend/visualization/atmosphere.py
  2. Add: ATMOSPHERE_DESCRIPTIONS[("cat", "tag")] = ["desc1", ...]
  3. Auto-used in prompt generation

TESTING:
  python test_atmosphere_features.py

DOCUMENTATION:
  ATMOSPHERE_AND_STORAGE_GUIDE.md - Complete guide
  ATMOSPHERE_STORAGE_SUMMARY.md   - Feature summary
  examples_atmosphere_storage.py   - This file

Get Started:
  1. export STABILITY_ATMOSPHERE_STRATEGY=asset
  2. Add atmosphere PNGs to backend/assets/
  3. Generate visualizations and compare with prompt strategy
  4. Choose preferred approach based on quality/cost/speed
"""

print(reference)

print("\n" + "="*80)
print("Run any example above by copying code into a Python script!")
print("="*80 + "\n")
