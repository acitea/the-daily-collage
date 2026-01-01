# Phase 0: Structure Alignment

Purpose: document the structural changes just applied and the remaining cleanup so other agents know the new layout.

## Completed
- Created top-level `ml/` package and moved ingestion, models, utils, data, and notebooks under it.
- Added placeholder `backend/assets/` for sticker PNGs/overlays and re-created `backend/utils/` (empty) for backend helpers.
- Updated imports in `backend/server/main.py` and `test_pipeline.py` to use absolute `backend.*` and `ml.*` modules.

## Follow-ups
- Spot-check any remaining docs or scripts for legacy paths after the ml/ migration; update tooling notes as needed.
- Ensure tooling (tasks, CI, Docker) resolves the new module layout; adjust PYTHONPATH/env loading accordingly.
