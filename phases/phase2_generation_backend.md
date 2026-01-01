# Phase 2: Hybrid Generation + Backend

Goals: implement the hybrid layout→polish flow with hitboxes, real caching, and API endpoints that serve cached vibes by hash.

## What to change/remove
- Replace the emoji-based `TemplateComposer` in `backend/visualization/composition.py` with a layout engine that places PNG assets and records hitboxes.
- Remove mock signal data from API handlers in `backend/server/main.py`; drive responses from the stored vibe vector.
- Expand caching beyond in-memory dicts; deprecate the current `VisualizationCache` once object storage is wired in.

## What to build
- Asset mapping: category+tag+intensity → asset filename; fallback icons for missing tags; load assets from `backend/assets/`.
- Layout engine: zone-based placement (sky/city/street), scaling by intensity, and hitbox capture `{x,y,w,h}` per element; include atmosphere overlays for weather.
- Polish step: Stability AI Img2Img call with low denoise (~0.35) that preserves layout; keep references to hitboxes unchanged.
- Vibe hash: deterministic key `City_Date_TimeWindow` + discretized scores to fetch/store cached renders; store metadata (hitboxes, source links, timestamps) in DB/object storage (e.g., S3/MinIO + Redis/Postgres lookup).
- API: endpoints read the latest vibe vector from Hopsworks/DB, check cache, generate if miss, return image URL + hitboxes; never expose generation directly to the public API.
- Configuration: add settings module for Stability AI creds, storage buckets, cache TTLs; ensure safe defaults and secrets via env.

## Quality gates
- Tests for cache key determinism and hitbox stability across polish runs.
- Contract tests for API responses (image URL, hitbox list, vibe hash) and error paths (cache miss with generation off, missing vibe vector).
- Small CLI/script to generate a layout+hitbox JSON for a given mocked vibe vector to aid frontend dev.
