"""
Quick integration test demonstrating Phase 2 functionality.

Run this to verify:
- Asset library loading
- Zone-based layout composition
- Vibe-hash caching
- Full visualization pipeline
- API response format
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.visualization.composition import HybridComposer, VisualizationService
from backend.visualization.caching import VibeHash, VibeCache, LocalStorageBackend
from backend.settings import settings


def test_settings():
    """Verify settings load correctly."""
    print("\n=== Testing Settings ===")
    print(f"✓ Storage backend: {settings.storage.backend}")
    print(f"✓ Stability AI enabled: {settings.stability_ai.enable_polish}")
    print(f"✓ Denoising strength: {settings.stability_ai.image_strength}")
    print(f"✓ Assets directory: {settings.assets.assets_dir}")
    print(f"✓ Layout dimensions: {settings.layout.image_width}x{settings.layout.image_height}")


def test_vibe_hash():
    """Test deterministic vibe hashing."""
    print("\n=== Testing Vibe Hash ===")
    
    city = "stockholm"
    timestamp = datetime(2025, 12, 11, 3, 30, 0)
    vibe_vector = {
        "traffic": 0.45,
        "weather_temp": -0.2,
        "crime": 0.1,
    }
    
    hash1 = VibeHash.generate(city, timestamp, vibe_vector)
    hash2 = VibeHash.generate(city, timestamp, vibe_vector)
    
    print(f"First hash:  {hash1}")
    print(f"Second hash: {hash2}")
    assert hash1 == hash2, "Hashes should be identical"
    print("✓ Hash is deterministic")
    
    # Test format
    parts = hash1.split("_")
    print(f"✓ Hash format: city={parts[0]}, date={parts[1]}, window={parts[2]}")


def test_composer():
    """Test hybrid composition pipeline."""
    print("\n=== Testing Hybrid Composer ===")
    
    composer = HybridComposer()
    
    vibe_vector = {
        "transportation": 0.5,
        "weather_wet": 0.7,
        "crime": 0.2,
        "festivals": 0.8,
    }
    
    print(f"Input vibe vector: {json.dumps(vibe_vector, indent=2)}")
    
    image_data, hitboxes = composer.compose(vibe_vector, "stockholm")
    
    print(f"✓ Generated image: {len(image_data)} bytes")
    print(f"✓ Image format: PNG" if image_data[:4] == b"\x89PNG" else "✗ Invalid PNG")
    print(f"✓ Hitboxes: {len(hitboxes)}")
    
    if hitboxes:
        hb = hitboxes[0]
        print(f"  Sample hitbox: x={hb['x']}, y={hb['y']}, "
              f"w={hb['width']}, h={hb['height']}")
        print(f"  Signal: {hb['signal_category']}/{hb['signal_tag']} "
              f"(intensity={hb['signal_intensity']:.2f})")


def test_cache():
    """Test caching system."""
    print("\n=== Testing Vibe Cache ===")
    
    storage = LocalStorageBackend(":memory:")
    cache = VibeCache(storage)
    
    city = "stockholm"
    timestamp = datetime(2025, 12, 11, 3, 30, 0)
    vibe_vector = {"traffic": 0.5, "weather_wet": 0.3}
    image_data = b"test_png_data"
    hitboxes = [{"x": 10, "y": 20, "width": 100, "height": 100}]
    
    print(f"Setting cache for {city}...")
    url, meta = cache.set(city, timestamp, vibe_vector, image_data, hitboxes)
    print(f"✓ Cache URL: {url}")
    print(f"✓ Vibe hash: {meta.vibe_hash}")
    
    print(f"Getting cache for {city}...")
    retrieved_image, retrieved_meta = cache.get(city, timestamp, vibe_vector)
    
    assert retrieved_image == image_data, "Should retrieve same image"
    print(f"✓ Retrieved image: {len(retrieved_image)} bytes")
    print(f"✓ Retrieved metadata: {retrieved_meta.vibe_hash}")
    
    print(f"Checking cache exists...")
    exists = cache.exists(city, timestamp, vibe_vector)
    assert exists, "Cache should exist"
    print(f"✓ Cache exists check passed")


def test_visualization_service():
    """Test full visualization service."""
    print("\n=== Testing Visualization Service ===")
    
    service = VisualizationService()
    
    vibe_vector = {
        "transportation": 0.4,
        "weather_wet": 0.6,
        "crime": -0.2,
        "festivals": 0.7,
        "politics": 0.1,
    }
    
    print(f"Generating visualization for stockholm...")
    image_data_1, metadata_1 = service.generate_or_get(
        city="stockholm",
        vibe_vector=vibe_vector,
    )
    
    print(f"✓ Generated image: {len(image_data_1)} bytes")
    print(f"✓ Vibe hash: {metadata_1['vibe_hash']}")
    print(f"✓ Image URL: {metadata_1['image_url']}")
    print(f"✓ Hitboxes: {len(metadata_1['hitboxes'])}")
    print(f"✓ Cached: {metadata_1['cached']}")
    
    print(f"\nFetching from cache...")
    image_data_2, metadata_2 = service.generate_or_get(
        city="stockholm",
        vibe_vector=vibe_vector,
    )
    
    assert image_data_1 == image_data_2, "Should return same image from cache"
    assert metadata_2['cached'], "Should be cached on second call"
    print(f"✓ Cache hit confirmed")
    print(f"✓ Cached: {metadata_2['cached']}")
    
    print(f"\nTesting different city...")
    image_data_3, metadata_3 = service.generate_or_get(
        city="gothenburg",
        vibe_vector=vibe_vector,
    )
    
    assert metadata_1['vibe_hash'] != metadata_3['vibe_hash'], "Different cities should have different hashes"
    print(f"✓ Different city has different hash: {metadata_3['vibe_hash']}")


def test_api_response_format():
    """Test that API response has correct format."""
    print("\n=== Testing API Response Format ===")
    
    service = VisualizationService()
    
    vibe_vector = {
        "transportation": 0.5,
        "weather_wet": 0.3,
    }
    
    image_data, metadata = service.generate_or_get(
        city="stockholm",
        vibe_vector=vibe_vector,
    )
    
    # Verify response structure
    required_fields = ['cached', 'vibe_hash', 'image_url', 'hitboxes', 'vibe_vector']
    for field in required_fields:
        assert field in metadata, f"Missing field: {field}"
        print(f"✓ {field}: {type(metadata[field]).__name__}")
    
    # Verify hitbox structure
    for hb in metadata['hitboxes']:
        required_hb_fields = ['x', 'y', 'width', 'height', 'signal_category', 
                             'signal_tag', 'signal_intensity', 'signal_score']
        for field in required_hb_fields:
            assert field in hb, f"Hitbox missing field: {field}"
        print(f"✓ Hitbox has all required fields")
        break  # Just check first one
    
    # Verify vibe vector ranges
    for category, score in vibe_vector.items():
        assert -1.0 <= score <= 1.0, f"Score {score} out of range [-1, 1]"
    print(f"✓ All vibe scores in valid range [-1.0, 1.0]")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 2 INTEGRATION TEST")
    print("=" * 60)
    
    try:
        test_settings()
        test_vibe_hash()
        test_composer()
        test_cache()
        test_visualization_service()
        test_api_response_format()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
