# Atmosphere & Storage Enhancement Documentation

This folder contains documentation for the atmosphere effects and Hopsworks storage features.

## Quick Links

- **[ATMOSPHERE_AND_STORAGE_GUIDE.md](./ATMOSPHERE_AND_STORAGE_GUIDE.md)** - Complete user guide
- **[examples_atmosphere_storage.py](./examples_atmosphere_storage.py)** - 7 practical examples
- **[test_atmosphere_features.py](./test_atmosphere_features.py)** - Test suite

## Features Implemented

### Dual Atmosphere Strategies
1. **Asset-Based**: PNG overlays on entire image (deterministic, offline)
2. **Prompt-Based**: AI-generated effects via text prompts (natural blending)

### Hopsworks Integration
- Full artifact store backend for ML ops
- Version control and lineage tracking
- Optional dependency with graceful fallback

## Configuration

```bash
# Asset atmosphere (PNG overlays)
export STABILITY_ATMOSPHERE_STRATEGY=asset

# Prompt atmosphere (AI enhancement)
export STABILITY_ATMOSPHERE_STRATEGY=prompt

# Hopsworks storage
export HOPSWORKS_ENABLED=true
export HOPSWORKS_API_KEY=your_key
export HOPSWORKS_HOST=c.app.hopsworks.ai
```

## Quick Start

```bash
# Run tests
python test_atmosphere_features.py

# See examples
python examples_atmosphere_storage.py
```

## Status

✅ **Production Ready**
- 7/7 tests passing
- Both strategies working
- Complete documentation
- Easy to extend

## Related

- See `../phase2_docs/` for main Phase 2 backend documentation
- See `../../backend/visualization/` for implementation code
