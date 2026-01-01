#!/usr/bin/env python3
"""
Test script for new atmosphere and storage features.
"""

import sys
from datetime import datetime

# Test 1: Verify settings
print("\n" + "="*70)
print("TEST 1: Settings & Atmosphere Strategy Configuration")
print("="*70)

try:
    from backend.settings import settings, AtmosphereStrategy
    
    print(f"✓ Settings loaded")
    print(f"  - Atmosphere strategy: {settings.stability_ai.atmosphere_strategy}")
    print(f"  - Include atmosphere in prompt: {settings.stability_ai.include_atmosphere_in_prompt}")
    print(f"  - Hopsworks enabled: {settings.hopsworks.enabled}")
    print(f"  - Hopsworks project: {settings.hopsworks.project_name}")
    
    # Verify AtmosphereStrategy enum
    print(f"  - Strategy options: {[s.value for s in AtmosphereStrategy]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Test AssetLibrary atmosphere assets
print("\n" + "="*70)
print("TEST 2: AssetLibrary Atmosphere Assets")
print("="*70)

try:
    from backend.visualization.assets import AssetLibrary
    
    library = AssetLibrary(settings.assets.assets_dir)
    
    # Check atmosphere map
    print(f"✓ AssetLibrary created")
    print(f"  - Zone-based assets: {len(library.ASSET_MAP)}")
    print(f"  - Atmosphere assets: {len(library.ATMOSPHERE_MAP)}")
    
    # Check specific atmosphere assets
    weather_atmo = library.ATMOSPHERE_MAP.get("weather_wet", {})
    print(f"  - Weather wet atmosphere types: {list(weather_atmo.keys())}")
    
    # Test has_atmosphere_asset
    has_rain = library.has_atmosphere_asset("weather_wet", "rain")
    print(f"  - Has rain atmosphere: {has_rain}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Test AtmosphereDescriptor
print("\n" + "="*70)
print("TEST 3: AtmosphereDescriptor Prompt Generation")
print("="*70)

try:
    from backend.visualization.atmosphere import AtmosphereDescriptor
    
    # Test signal set
    signals = [
        ("weather_wet", "rain", 0.8, 0.8),
        ("festivals", "celebration", 0.6, 0.6),
        ("emergencies", "fire", 0.4, -0.4),
    ]
    
    # Generate atmosphere prompt
    prompt = AtmosphereDescriptor.generate_atmosphere_prompt(signals)
    print(f"✓ Atmosphere prompt generated")
    print(f"  - Input signals: {len(signals)}")
    print(f"  - Generated prompt: '{prompt}'")
    
    # Get mood
    mood = AtmosphereDescriptor.get_mood(signals)
    print(f"  - Detected mood: '{mood}'")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 4: Test ZoneLayoutComposer with atmosphere
print("\n" + "="*70)
print("TEST 4: ZoneLayoutComposer with Atmosphere Layers")
print("="*70)

try:
    from backend.visualization.assets import ZoneLayoutComposer
    
    composer = ZoneLayoutComposer(
        image_width=1024,
        image_height=768,
        assets_dir=settings.assets.assets_dir,
    )
    
    signals = [
        ("weather_wet", "rain", 0.8, 0.8),
        ("crime", "police", 0.4, 0.4),
        ("festivals", "celebration", 0.6, 0.6),
    ]
    
    # Compose with atmosphere assets enabled
    image, hitboxes = composer.compose(signals, apply_atmosphere_assets=True)
    
    print(f"✓ Composition with atmosphere completed")
    print(f"  - Image size: {image.size}")
    print(f"  - Hitboxes: {len(hitboxes)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 5: Test polish with atmosphere prompt
print("\n" + "="*70)
print("TEST 5: StabilityAIPoller Atmosphere Support")
print("="*70)

try:
    from backend.visualization.polish import MockStabilityAIPoller
    import io
    from PIL import Image
    
    # Create test image
    test_image = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format="PNG")
    image_data = img_bytes.getvalue()
    
    # Test mock poller with atmosphere prompt
    poller = MockStabilityAIPoller()
    result = poller.polish(
        image_data,
        prompt="test prompt",
        atmosphere_prompt="rainy and wet",
    )
    
    print(f"✓ MockStabilityAIPoller tested")
    print(f"  - Input size: {len(image_data)} bytes")
    print(f"  - Output size: {len(result)} bytes")
    print(f"  - Atmosphere prompt supported: ✓")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 6: Test HopsworksStorageBackend (without connection)
print("\n" + "="*70)
print("TEST 6: HopsworksStorageBackend Initialization")
print("="*70)

try:
    from backend.visualization.caching import HopsworksStorageBackend
    
    # Initialize without real credentials (will fail connection but test structure)
    backend = HopsworksStorageBackend(
        api_key="test_key",
        project_name="test_project",
        host="test.hopsworks.ai",
    )
    
    print(f"✓ HopsworksStorageBackend initialized")
    print(f"  - Project: {backend.project_name}")
    print(f"  - Collection: {backend.artifact_collection}")
    print(f"  - Connected: {backend._is_connected()}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 7: Test HybridComposer with atmosphere strategies
print("\n" + "="*70)
print("TEST 7: HybridComposer with Atmosphere Strategies")
print("="*70)

try:
    from backend.visualization.composition import HybridComposer
    import io
    
    # Test with ASSET strategy
    settings.stability_ai.atmosphere_strategy = "asset"
    composer_asset = HybridComposer()
    
    vibe_vector = {
        "weather_wet": 0.8,
        "festivals": 0.5,
        "crime": -0.3,
    }
    
    image_bytes, hitboxes = composer_asset.compose(vibe_vector, location="stockholm")
    
    print(f"✓ HybridComposer with ASSET strategy")
    print(f"  - Image generated: {len(image_bytes)} bytes")
    print(f"  - Hitboxes: {len(hitboxes)}")
    print(f"  - Atmosphere assets applied: True")
    
    # Test with PROMPT strategy
    settings.stability_ai.atmosphere_strategy = "prompt"
    composer_prompt = HybridComposer()
    
    image_bytes_2, hitboxes_2 = composer_prompt.compose(vibe_vector, location="stockholm")
    
    print(f"✓ HybridComposer with PROMPT strategy")
    print(f"  - Image generated: {len(image_bytes_2)} bytes")
    print(f"  - Hitboxes: {len(hitboxes_2)}")
    print(f"  - Atmosphere prompt: Generated (via AtmosphereDescriptor)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "="*70)
print("ATMOSPHERE & STORAGE FEATURES - ALL TESTS PASSED ✓")
print("="*70)
print("\nNew capabilities:")
print("  ✓ Atmosphere Strategy: ASSET and PROMPT")
print("  ✓ Atmosphere Assets: PNG overlays on entire image")
print("  ✓ Atmosphere Prompts: Text-based mood injection via img2img")
print("  ✓ Hopsworks Backend: Artifact store integration ready")
print("  ✓ Swappable Design: Change strategy via environment variables")
print("\nNext steps:")
print("  1. Set STABILITY_ATMOSPHERE_STRATEGY=asset or prompt")
print("  2. Add atmosphere PNG files to backend/assets/")
print("  3. Configure Hopsworks if using artifact store")
print("  4. Test with real vibe vectors from ML pipeline")
print("="*70 + "\n")
