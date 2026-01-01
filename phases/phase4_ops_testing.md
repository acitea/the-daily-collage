# Phase 4: Ops, DX, and Testing

Goals: harden dev/test workflows, update docs, and align tooling with the new architecture.

## What to change/remove
- Update root README, quick-start scripts, and `ml/ingestion/README.md` to reflect the new layout and Hopsworks-first flow.
- Remove or rewrite obsolete references to emoji templates and the older signal taxonomy.

## What to build
- Docker/uv/Make targets that set PYTHONPATH to project root and run ingestion + API locally.
- CI checks: lint (ruff/flake8), type checking (mypy/pyright), unit tests, and a lightweight pipeline integration test.
- Secrets management: `.env.example`, documented env vars for GDELT/Weather/Hopsworks/Stability, and guidance for local vs CI.
- Monitoring hooks: basic logging/metrics for cache hits/misses, Stability AI latency, and ingestion job success counts.
- Artifacts: sample vibe vector JSON + sample hitbox JSON/image for frontend/dev testing.

## Quality gates
- CI green on lint + tests; build artifacts published (sample JSON/image) on successful main-branch builds.
- Updated docs validated by a contributor walkthrough (fresh clone to running API + sample frontend).
