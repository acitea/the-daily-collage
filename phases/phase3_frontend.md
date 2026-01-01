# Phase 3: Frontend (Interactive Canvas)

Goals: move to the React + Vite app with `<VibeCanvas />` that renders images and clickable hitboxes, using a solid UI kit (Mantine UI preferred) alongside Tailwind CSS for utility styling. React Router may be introduced if multi-view flows emerge.

## What to change/remove
- Replace the static `frontend/index.html` with a Vite React project structure and componentized UI (Mantine UI components + Tailwind utilities for fast layout work).
- Avoid calling generation endpoints; the frontend should only consume the current vibe endpoint that returns image URL + hitboxes + article metadata.

## What to build
- **Scaffold & shell**: Vite + React + Mantine UI + Tailwind utilities; React Router only if multi-page flows appear.
- **Location header**: horizontally scrollable button group for cities (Sweden/Stockholm now, extensible to other countries).
- **Time selector**: horizontal, scrollable timeline with minor (days), medium (weeks), major (months) ticks; click to load the image for that period (Wayback-style).
- **Canvas**: `<VibeCanvas />` renders the vibe image for the selected time/location, overlays invisible hitbox divs from metadata, and opens article modal on click.
- **Signals + headlines**: vertical bar chart of signals (Mantine + recharts/visx or Tailwind-styled bars) showing intensity/tags; below the chart, show the full headline list that contributed to the current image by default. Category header/bar acts as a toggle/filter to narrow that list to the selected category instead of revealing a separate table.
- **State/data layer**: hooks for fetching current vibe and historical snapshots by time window, refreshing on demand, showing cached status/time window; robust error/loading states.
- **UI polish**: responsive layout, clear location selector, badge chips for signal intensities, modal for article lists per hitbox.
- **Config/tooling**: environment-based API base URL, lint/format toolchain, and minimal tests for hitbox overlay math.

## Quality gates
- Snapshot/RTL tests for `<VibeCanvas />` overlays and click handling.
- Smoke E2E (Playwright or Cypress) that loads a mocked vibe response and verifies modal opens on hitbox click.
