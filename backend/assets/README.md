# Assets Directory

This directory contains PNG sticker assets for The Daily Collage visualizations.

## Base Scenery

- `base_scenery.png`: The base canvas/background for all visualizations. Dimensions are automatically detected from this file.

## Naming Convention

Assets follow the pattern: `{category}_{tag}_{level}.png`

### Categories (7 primary)
- `emergencies`
- `crime`
- `festivals`
- `transportation`
- `sports`
- `economics`
- `politics`

**Note**: Weather categories (`weather_temp`, `weather_wet`) are handled via AI prompts and do not have physical assets.

### Tags (sentiment)
Tags represent the sentiment or direction of the signal:
- **positive**: Score ≥ 0 (e.g., more traffic, more events, higher activity)
- **negative**: Score < 0 (e.g., less traffic, cancelled events, lower activity)

The interpretation of positive/negative varies by category:
- **transportation**: positive = congestion/traffic, negative = clear roads
- **crime**: positive = incidents present, negative = safe/peaceful
- **festivals**: positive = events happening, negative = cancelled/quiet
- **emergencies**: positive = active emergencies, negative = safe conditions
- **sports**: positive = games/victories, negative = cancelled/losses
- **economics**: positive = growth/activity, negative = downturn
- **politics**: positive = protests/activity, negative = calm

### Intensity Levels (3 levels)
- `low`: 0.0 - 0.33 intensity (subtle presence)
- `med`: 0.33 - 0.66 intensity (moderate presence)
- `high`: 0.66 - 1.0 intensity (strong presence)

## Example Filenames

```
emergencies_positive_low.png    # Minor emergency
emergencies_positive_high.png   # Major emergency
crime_positive_med.png          # Moderate crime activity
crime_negative_low.png          # Very safe conditions
festivals_positive_high.png     # Major festival/event
transportation_positive_high.png # Heavy traffic
sports_positive_high.png        # Major sports event
economics_negative_high.png     # Severe economic downturn
politics_positive_med.png       # Moderate political activity
```

## Signal Filtering

Signals with insignificant scores (|score| < 0.1) are automatically filtered out to focus on more prominent categories.

## Fallback Assets

If a specific `{category}_{tag}_{level}.png` is not found, the system falls back to:

1. **Category + Intensity Generic**: `{category}_generic_{level}.png`
   - Keeps the same intensity level but uses generic tag
   - Example: If `transportation_positive_high.png` is missing → `transportation_generic_high.png`

2. **Ultimate Fallback**: `generic_default.png`

## Generic Fallback Examples

Required generic assets for each category at all three levels:
```
transportation_generic_low.png
transportation_generic_med.png
transportation_generic_high.png
crime_generic_low.png
crime_generic_med.png
crime_generic_high.png
festivals_generic_low.png
festivals_generic_med.png
festivals_generic_high.png
emergencies_generic_low.png
emergencies_generic_med.png
emergencies_generic_high.png
sports_generic_low.png
sports_generic_med.png
sports_generic_high.png
economics_generic_low.png
economics_generic_med.png
economics_generic_high.png
politics_generic_low.png
politics_generic_med.png
politics_generic_high.png
```

## Asset Count

- **Specific assets**: 7 categories × 2 tags × 3 levels = **42 assets**
- **Generic fallbacks**: 7 categories × 3 levels = **21 assets**
- **Base scenery**: 1 asset
- **Total**: **64 assets**

## Asset Specifications

- **Format**: PNG with transparency (RGBA)
- **Recommended Size**: 
  - Low intensity: ~80-120px
  - Med intensity: ~120-180px
  - High intensity: ~180-250px
- **Style**: Cartoon/sticker style, colorful, playful
- **Background**: Transparent (alpha channel)

## Design Guidelines

1. **Visual Clarity**: Each sticker should be immediately recognizable
2. **Size Variation**: Different intensity levels should have noticeably different sizes
3. **Color Consistency**: Use consistent color palettes within categories
4. **Overlap-Friendly**: Design with potential overlapping in mind (scrapbook style)
5. **Cultural Sensitivity**: Avoid culturally insensitive imagery

## Creating New Assets

When adding new tags or categories:

1. Create all 3 intensity levels (low, med, high)
2. Create generic fallbacks for the category
3. Update `backend/visualization/assets.py` ASSET_MAP
4. Test with the visualization service

## Current Status

⚠️ **Placeholder Status**: This directory currently contains only `.gitkeep`. 
Actual asset PNGs need to be created and added.

To get started quickly, you can:
1. Create simple colored circles/shapes as placeholders
2. Use tools like Figma, Canva, or DALL-E to generate sticker-style images
3. Source from free icon libraries (e.g., Flaticon, Icons8) with proper licensing

## Testing Assets

Test your assets with:

```bash
cd backend/server
python -c "
from backend.visualization.assets import AssetLibrary
lib = AssetLibrary('../assets')
img = lib.get_asset('emergencies', 'fire', 0.8)
if img:
    print(f'Loaded: {img.size}')
else:
    print('Asset not found')
"
```
