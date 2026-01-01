# Phase 1: Data + ML Backbone

Goals: move ingestion to a Hopsworks-backed feature pipeline, adopt the nine canonical signals, and prepare a real model path instead of keyword heuristics.

## What to change/remove
- Deprecate the keyword-based classifier in `ml/utils/classification.py` once a multi-head model is available; keep it only as a fallback or test fixture.
- Replace average-based aggregation with max-pooling per the vibe vector spec; remove any lingering 0-averaging logic.
- Update docs (README, ingestion guide, quick-start) that still reference old paths or the legacy signal list.

## What to build
- Hopsworks integration: feature group for raw articles and an aggregated "vibe vector" keyed by City_Date_TimeWindow (6h). Include schema for score (-1..1) and tag per signal.
- Ingestion job: GDELT + Weather fetchers feeding the feature group; include deduplication/validation from `ml/utils/processing.py` and weather normalization.
- Model: multi-head classifier/regressor that outputs (score, tag) for the 9 signals (`emergencies`, `crime`, `festivals`, `transportation`, `weather_temp`, `weather_wet`, `sports`, `economics`, `politics`). Register in Hopsworks model registry.
- Aggregation: max-pool scores per window, carry forward best tag per signal, and persist the vibe vector for the backend to consume.
- Configuration: centralize API keys/endpoints (GDELT, weather, Hopsworks) via env/pyproject; add sample `.env.example`.

## Quality gates
- Unit tests around normalization, deduping, and aggregation (max-pool correctness, tag selection).
- Smoke test that writes a vibe vector to a test feature group and reads it back.
- Document runbooks for scheduled 6-hour ingestion (e.g., cron, Airflow, or Hopsworks jobs).
