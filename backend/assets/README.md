# Assets Directory

This directory contains PNG sticker assets for The Daily Collage visualizations.

## Naming Convention

Assets follow the pattern: `{category}_{tag}_{intensity}.png`

### Categories (9 primary)
- `emergencies`
- `crime`
- `festivals`
- `transportation`
- `weather_temp`
- `weather_wet`
- `sports`
- `economics`
- `politics`

### Tags (specific events)
Each category has 3-5 specific tags. Examples:
- **emergencies**: fire, earthquake, evacuation
- **crime**: theft, assault, police
- **festivals**: concert, celebration, crowd
- **transportation**: traffic, congestion, accident
- **weather_temp**: hot, cold
- **weather_wet**: rain, snow, flood
- **sports**: football, hockey, victory
- **economics**: market, business, trade
- **politics**: protest, election, government

### Intensity Levels (3 levels)
- `low`: 0.0 - 0.33 intensity
- `med`: 0.33 - 0.66 intensity
- `high`: 0.66 - 1.0 intensity

## Example Filenames

```
emergencies_fire_low.png
emergencies_fire_med.png
emergencies_fire_high.png
crime_theft_low.png
festivals_concert_high.png
transportation_traffic_med.png
weather_wet_rain_low.png
sports_football_high.png
economics_market_med.png
politics_protest_high.png
```

## Fallback Assets

If a specific `{category}_{tag}_{intensity}.png` is not found, the system falls back to:

1. **Category + Intensity Generic**: `{category}_generic_{intensity}.png`
   - Example: `transportation_generic_high.png`

2. **Ultimate Fallback**: `generic_default.png`

## Generic Fallback Examples

```
transportation_generic_low.png
transportation_generic_med.png
transportation_generic_high.png
weather_temp_generic_low.png
crime_generic_med.png
```

## Atmosphere Assets

Atmosphere assets overlay the entire canvas and follow a different pattern:
`atmosphere_{effect}.png`

Examples:
- `atmosphere_rain.png`
- `atmosphere_snow.png`
- `atmosphere_heat.png`
- `atmosphere_cold.png`
- `atmosphere_celebration.png`
- `atmosphere_fire_glow.png`

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
