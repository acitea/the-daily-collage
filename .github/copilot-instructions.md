# AI Coding Agent Instructions - The Daily Collage

## Project Overview

**The Daily Collage** is a proof-of-concept system that transforms real-time news from geographic locations into a clickable, cartoonish visualization. The system captures the collective "vibe" of a city by ingesting news signals, composing a scene, and polishing it with Generative AI. The system updates every 6 hours per location.

**Core Value**: A "glanceable" understanding of a city's mood. Users see a single image representing the current vibe and can click on specific elements (e.g., a fire, a traffic jam) to read the underlying news articles.

## Architecture Overview

The system architecture has evolved to a **Hybrid Composition** model with a **Feature Store** backbone.

```
project_root/
├── backend/
│   ├── _types/           # All relevant types for the project
│   ├── storage/          # File storage services
│   ├── visualization/    # Layout composition + AI Polish (2 stage image generation)
│   ├── app/              # FastAPI for serving vibes & metadata
│   ├── utils/            # Any relevant or useful scripts for one-off or routinely used
│   └── assets/           # Sticker PNGs and overlays
├── ml/                   # Hopsworks feature definitions & model logic
│   ├── data/             # any data needed for training
│   ├── utils/            # Utility scripts for data processing, for training, evaluation, etc.
│   ├── notebooks/        # notebooks for experimentation, eda, training
│   ├── ingestion/        # GDELT & Weather API fetching -> Feature Store -> Vibe Vector
│   └── models/           # Relevant scripts to define, train, and serve models
└── frontend/             # React + Vite application
```

### Data Flow (The "Vibe" Pipeline)

1.  **Ingestion & Feature Store** (Every 6 hours):
    *   Query GDELT (News) and Weather APIs.
    *   **ML Model**: Classify articles into 9 Signals (Score + Tag).
    *   **Hopsworks**: Store the aggregated "Vibe Vector" (Max-Pooled scores) in the Feature Group.
2.  **Generation** (On Demand / Cached):
    *   Backend checks DB for an existing visualization for the current Vibe Vector.
    *   *Cache Miss*: Trigger **Hybrid Generation** (Layout -> AI Polish). Save to S3.
3.  **Presentation**:
    *   FastAPI serves the Image URL + **Hitbox Metadata**.
    *   React Frontend renders the image and interactive overlays.

## Signal Categories & Data Schema

The system tracks **9 Primary Signals**. The ML model outputs a **Score** (-1.0 to 1.0) and a **Tag** (string) for each.

*   **Categories**: `emergencies`, `crime`, `festivals`, `transportation`, `weather_temp`, `weather_wet`, `sports`, `economics`, `politics`.
*   **Interpretation**:
    *   `Score 0.0`: Irrelevant/Absent.
    *   `Score ~0.9 or ~-0.9`: High Intensity / Active Event.
    *   `Tag`: Descriptive keyword (e.g., "fire", "protest", "snow") used to select assets.

## Technical Stack & Key Dependencies

*   **Language**: Python 3.10+ (Backend/ML), TypeScript (Frontend).
*   **ML & Data**: **Hopsworks** (Feature Store & Model Registry), `gdeltdoc`.
*   **Generation**: `Pillow` (Layout), **Replicate AI API** (Image-to-Image Polish).
*   **Backend**: `fastapi`, some file storage provider.
*   **Frontend**: React, Vite.

## Implementation Guidelines

### 1. Data & ML (Hopsworks Integration)
*   **Refactor Ingestion**: Modify `ml/ingestion/script.py`. Instead of processing locally, push cleaned data to a Hopsworks Feature Group.
*   **Model**: The classifier must be multi-head (Output: Regression Score + Classification Tag).
*   **Aggregation**: When creating the "Vibe Vector" for a 6-hour window, use **Max-Pooling**. Do not average zeros.

### 2. Image Generation Strategy (Hybrid)
*   **Step 1: Layout (Python/Pillow)**:
    *   Map `Category` + `Tag` + `Intensity` -> `Asset PNG` (e.g., "emergency" + "0.8 score" + "fire" -> `backend/assets/fire_major.png`). This is just an example. The exact images will be provided later on, but set up a template for it.
    *   Place assets into Zones (Sky, City, Street) based on logic.
    *   **CRITICAL**: Record the `{x, y, w, h}` coordinates of every placed asset into a `hitboxes` list.
    *   Apply "Atmosphere Overlays" (rain, sun gradients) as the final layer.
*   **Step 2: Polish (Replicate AI)**:
    *   Send the Layout image to Replicate AI `Img2Img`.
    *   **Constraint**: Keep `denoising_strength` (or `image_strength`) low (~0.35).
    *   *Reason*: We need the AI to beautify the style but **preserve the hitbox locations**.

### 3. Caching & Serving (Cost Control)
*   **"Vibe Hash"**: Create a unique key for the request: `City_Date_TimeWindow`.
*   **Logic**:
    *   Check DB for `Vibe Hash`.
    *   If found: Return stored URL + Hitboxes. **(Zero Cost)**
    *   If missing: Generate -> Upload S3 -> Store DB -> Return.
*   **Frontend**: The React app *never* triggers generation directly. It only fetches the "current vibe" endpoint.

### 4. Frontend (Interactive Canvas)
*   **Component**: `<VibeCanvas />`
*   **Logic**:
    *   Render the Image URL.
    *   Render invisible `<div>` overlays using the `hitboxes` metadata.
    *   On Click -> Open Modal with news articles linked to that signal.

## Critical Gotchas & Constraints

1.  **Hitbox Drift**: If the AI Polish strength is too high (>0.5), objects will move, and the clicks will miss. Keep it subtle.
2.  **Cost Management**: Strict caching is mandatory. Do not expose the "Generate" function to the public API.
3.  **Asset Library**: You need a `assets/` folder with transparent PNGs for every potential tag. If a tag is missing, fallback to a generic category icon.
4.  **GDELT Data**: Continue using `gdeltdoc` but ensure the output flows into Hopsworks.
