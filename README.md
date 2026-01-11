# The Daily Collage

A proof-of-concept system that transforms news headlines and events from a specific geographic location into a single, cartoonish visualization that captures the collective "vibe" of what's currently happening in that area.

## Project Overview

**Core Concept**: "A picture is worth a thousand words" - instead of reading through dozens of news articles, users can glance at a generated image to understand the current sentiment and major events in a location (country or city-level). If something in the visualization catches their interest, they can explore the underlying news sources for more details.

**Geographic Scope**: Primarily targeting Sweden and Swedish cities (e.g., Stockholm, Gothenburg, Malmö), with potential expansion to other locations.

**Update Frequency**: Visualizations refresh every 6 hours to capture emerging "hot" news while maintaining computational efficiency.

## Technical Architecture

### Data Pipeline

**Primary Data Source**: GDELT Project (Global Database of Events, Language, and Tone)
- Provides credible, global news headlines with geographic tagging
- Access via GDELT 2.0 DOC API for real-time article monitoring
- Built-in sentiment scoring and event categorization
- Python client available for integration

**Supplementary Data**: Weather API integration to set the atmospheric mood of generated images (e.g., rainy weather could influence color palette or visual elements)

### Signal Categories

News articles will be classified into a fixed set of categories/signals that can be easily represented as cartoon-style visual elements:

**Primary Categories**:
- **Traffic & Transportation**: Road congestion, accidents, public transit disruptions
- **Weather Events**: Storms, heatwaves, snow, flooding
- **Crime & Safety**: Incidents, police activity, emergency services
- **Festivals & Events**: Cultural celebrations, concerts, public gatherings
- **Politics**: Elections, protests, government announcements
- **Sports**: Major games, victories, sporting events
- **Accidents & Emergencies**: Fires, industrial accidents, medical emergencies
- **Economic**: Market news, business developments, employment

Each category will have a corresponding visual "sticker" or template element that can scale in size/intensity based on the signal strength (e.g., small fire icon for minor incident, large flames for major fire).

### Machine Learning Components

**Sentiment Extraction & Classification**:
- Self-hosted language models fine-tuned on news sentiment classification
- Based on state-of-the-art architectures (e.g., BERT-based or similar transformer models)
- Multi-label classification to assign articles to signal categories
- Intensity scoring for each detected signal (0-100 scale)

**Model Training Approach**:
- Fine-tune existing pre-trained models rather than training from scratch
- Focus on Swedish news corpus for localized understanding
- Potential use of LangGraph for orchestration of multi-step classification pipeline

### Image Generation Strategy

**Current Decision Point**: Two competing approaches being evaluated:

**Option A: Template-Based Composition** (Currently Favored)
- Pre-designed cartoon-style templates for each signal category
- Dynamic composition system that arranges and scales elements based on signal intensity
- Lower computational requirements
- Faster iteration and controllable output
- Easier to ensure consistent visual style
- Templates stored as vector graphics or layered image assets

**Option B: Generative AI**
- Full diffusion-model based generation (e.g., Stable Diffusion, Flux)
- Concern: High computational cost and training time
- Concern: Difficulty achieving consistent cartoon style
- May be revisited post-POC if template approach proves limiting

**Image Caching Strategy**:
- Artifact store (database) to cache generated visualizations
- Cache key based on signal combination and intensity levels
- Reduces redundant generation for similar sentiment profiles
- Potential database: MinIO, S3-compatible storage, or PostgreSQL with blob storage

### Backend Stack

**API Framework**: FastAPI
- RESTful endpoints for location-based visualization requests
- Query parameters: location (city/country), date/time range
- Response includes generated image and metadata (underlying news sources, signal breakdown)

**Components**:
- News ingestion service (GDELT API polling every 6 hours)
- Preprocessing pipeline (text cleaning, deduplication)
- ML inference service (sentiment classification)
- Image generation/composition service
- Caching layer

**Database**: Artifact store for image caching, potentially SQLite or PostgreSQL for metadata (news articles, timestamps, signal scores)

### Frontend

**Type**: Web application (framework TBD - potentially React, Vue, or vanilla JavaScript)

**Core Features**:
- Location selector (dropdown or map-based)
- Display current visualization for selected location
- Click-through to view underlying news articles that contributed to each signal
- Simple, clean interface focused on the visualization

**Out of Scope for POC**:
- Historical trend views
- Interactive maps with multiple simultaneous locations
- Mobile-responsive design (desktop-first)

### Deployment

**Containerization**: Docker (not Docker Compose - simple single-container or microservice approach)

**Infrastructure**: 
- Likely single-server deployment for POC
- No Kubernetes orchestration for initial version
- Static hosting for frontend, backend API on VM or cloud instance

## Data Flow

1. **Ingestion** (every 6 hours):
   - Query GDELT API for news articles matching geographic filters
   - Fetch weather data for target locations
   - Store raw articles in processing queue

2. **Processing**:
   - Clean and deduplicate headlines
   - Run through fine-tuned sentiment classification model
   - Extract signal categories and intensity scores
   - Aggregate signals by location

3. **Visualization Generation**:
   - Check cache for similar signal profile
   - If cache miss: compose/generate image based on signal combination
   - Apply weather-based mood adjustments (color palette, background)
   - Store in artifact cache with metadata

4. **API Response**:
   - Serve cached or newly generated visualization
   - Include JSON metadata with signal breakdown and source articles

## Success Criteria (POC)

The POC is considered successful when:
- System can ingest and process news from GDELT for a target location
- Classification model accurately categorizes news into predefined signals
- Visualizations are generated that represent the sentiment combination
- Frontend displays the visualization with the ability to drill down into source articles
- System operates as envisioned by the team

## Visual Style

**Aesthetic**: Cartoonish, illustrative style, or even in a sticker format rather than photorealistic
- Friendly, approachable visual language
- Clear iconography for each signal category
- Consistent color palette across visualizations
- Think: weather app icons, infographic style, or editorial illustrations

## Open Questions & Future Considerations

- **Image generation method**: Final decision between template composition vs. generative AI
- **Multi-language support**: Should non-Swedish news sources be included for Swedish cities?
- **Signal weighting**: How to balance multiple competing signals in a single image (e.g., both festival and traffic jam)?
- **User interaction**: Should users be able to customize which signals they care about?
- **Comparison view**: Post-POC feature to compare vibes between multiple cities
- **Time-of-day variations**: Should morning vs. evening news be weighted differently?

## Technical Risks & Mitigations

**Risk: GDELT API rate limits or downtime**
- Mitigation: Implement exponential backoff, cache raw article data, have fallback to RSS feeds

**Risk: Classification model accuracy on Swedish news**
- Mitigation: Manual labeling of Swedish news dataset for fine-tuning, iterative model improvement

**Risk: Image generation quality/consistency**
- Mitigation: Template-based approach provides more control; establish clear visual guidelines

**Risk: Computational resources for real-time processing**
- Mitigation: 6-hour refresh window allows batch processing; aggressive caching strategy

## Timeline & Milestones

_To be defined by team based on availability and priorities_

Suggested phases:
1. **Data Pipeline Setup**: GDELT integration, data exploration
2. **Model Fine-tuning**: Sentiment classification for Swedish news
3. **Template Design**: Create visual elements for each signal category
4. **Composition Engine**: Build system to arrange templates based on signals
5. **API Development**: FastAPI endpoints and caching layer
6. **Frontend Development**: Basic web interface
7. **Integration & Testing**: End-to-end workflow validation
8. **Demo & Iteration**: Gather feedback, refine approach

## Repository Structure

```

the-daily-collage/
├── backend/
├── backend/
│   ├── server/           \# FastAPI application
│   ├── ingestion/        \# GDELT data fetching
│   ├── models/           \# ML model code and weights
│   ├── utils/            \# Text preprocessing, classification
│   ├── visualization/    \# Image generation/composition
│   └── cache/            \# Caching layer
├── frontend/
│   ├── src/              \# Frontend source code
│   ├── public/           \# Static assets
│   └── dist/             \# Build output
├── templates/            \# Visual template assets (if using template approach)
├── data/
│   ├── raw/              \# Raw GDELT data samples
│   ├── processed/        \# Labeled training data
│   └── cache/            \# Generated image cache
├── notebooks/            \# Jupyter notebooks for exploration
└── docs/

```

## Getting Started

### Model Training Pipeline

The classifier is trained through a multi-stage pipeline from raw articles to a fine-tuned model ready for production inference.

#### Stage 1: Template & Keywords Generation (`ml/utils/generate_templates.py`)

**Purpose**: Create reference data for embedding-based classification

**Process**:
1. Fetches real GDELT articles per category using category-specific keywords
2. Preprocesses Swedish text (normalization, cleaning)
3. Extracts representative phrases from category-specific articles as **SIGNAL_TEMPLATES**
4. Collects observed keywords/tags from titles and descriptions as **TAG_KEYWORDS**
5. Uses LLM (i.e., OpenAI API) to generate additional templates/keywords ensuring comprehensive coverage
6. Outputs JSON files for use in classification pipeline

**Usage**:
```bash
# Generate templates from 100 Swedish articles per category (900 total)
python ml/utils/generate_templates.py --articles-per-category 100 --country sweden

# Generate more templates per category (default 30)
python ml/utils/generate_templates.py --articles-per-category 150 --templates-per-category 50

# Search further back in time
python ml/utils/generate_templates.py --articles-per-category 150 --days-lookback 180
```

**Output**:
- `ml/signal_templates.json` - Templates for each signal category
- `ml/tag_keywords.json` - Keywords associated with each tag

#### Stage 2: Article Labeling (`ml/utils/label_dataset.py`)

**Purpose**: Create labeled training data using embedding-based classification with optional LLM verification

**Process**:
1. Fetches articles from GDELT in batches
2. Classifies articles using **embedding similarity** against SIGNAL_TEMPLATES
3. Assigns (score, tag) pairs for each signal category
4. Optionally verifies labels using LLM for quality assurance
5. Outputs labeled parquet files with structure:
   - `title`, `description`, `url`, `source`
   - `emergencies_score`, `emergencies_tag`
   - `crime_score`, `crime_tag`
   - ... (one pair per signal category)

**Usage**:
```bash
# Generate 500 labeled articles without LLM verification (fast)
python ml/utils/label_dataset.py --articles 500 --country sweden

# Generate 200 articles with LLM verification (slower but higher quality)
python ml/utils/label_dataset.py --articles 200 --country sweden --llm-verify --max-llm-calls 50
```

**Output**:
- `ml/data/labeled_articles_train.parquet` - 70% training set
- `ml/data/labeled_articles_val.parquet` - 30% validation set

#### Stage 3: Model Fine-tuning (`ml/models/quick_finetune.py`)

**Purpose**: Fine-tune a Swedish BERT model on the labeled data

**Architecture**:
- **Base Model**: `KB/bert-base-swedish-cased` (Swedish-specific BERT)
- **Multi-head Design**:
  - 9 parallel "score heads" (regression, outputs -1.0 to 1.0 intensity)
  - 9 parallel "tag heads" (classification, predicts which tag within category)
- **Loss Function**: Weighted combination of regression loss (score) and classification loss (tag)

**Process**:
1. Loads labeled parquet file from Stage 2
2. Creates PyTorch dataset with text tokenization
3. Trains with:
   - Batch size: 32 (GPU) / 8 (CPU)
   - Epochs: 8
   - Learning rate: 2e-5 with cosine annealing
   - Early stopping based on validation loss
4. Saves best checkpoint to `ml/models/checkpoints/best_model.pt`

**Usage**:
```bash
# Fine-tune using labeled articles
python ml/models/quick_finetune.py \
  --train ml/data/labeled_articles_train.parquet \
  --val ml/data/labeled_articles_val.parquet \
  --output ml/models/checkpoints

# Use custom output directory
python ml/models/quick_finetune.py \
  --train ml/data/labeled_articles_train.parquet \
  --val ml/data/labeled_articles_val.parquet \
  --output ./custom_checkpoints
```

**Output**:
- `ml/models/checkpoints/best_model.pt` - Best model checkpoint with state dict

#### Stage 4: Local Testing (`tests/model_inference_smoke.py`)

**Purpose**: Validate model quality with real GDELT articles and LLM adjudication

**Features**:
- Tests all 9 signal categories with real or LLM-generated articles
- Shows top 3 ranked predictions with confidence scores
- Uses OpenAI GPT-4o-mini to verify predictions (ground truth)
- Falls back to LLM-generated articles when GDELT fetch fails for a category
- Provides detailed pass/fail statistics

**Usage**:
```bash
# Run smoke test with LLM verification
export OPENAI_API_KEY="your-key-here"
python tests/model_inference_smoke.py

# Run without OpenAI (uses top-3 ranking as fallback validation)
python tests/model_inference_smoke.py
```

**Example Output**:
```
[1/3] ✅ PASS | Breaking: Major fire in Stockholm...
      Desc: Emergency services responding to downtown fire
      Top 3 predictions:
        → [1] emergencies        score=+0.824 tag=fire
          [2] crime              score=+0.312 tag=
          [3] politics           score=+0.105 tag=
      LLM: ✓ agrees — The article describes an emergency situation (fire)
```

#### Stage 5: Production Deployment (Modal Serverless API)

**Purpose**: Deploy fine-tuned model as scalable HTTP API

**Features**:
- Serverless deployment via Modal
- Persistent volume caching for model & HuggingFace weights
- Returns top-ranked category with score and tag
- Supports single or batch requests

**Deployment**:
```bash
# Deploy to Modal
modal deploy ml/models/serve_modal.py

# Or run locally with Modal
modal run ml/models/serve_modal.py
```

**API Usage**:
```bash
curl -X POST https://your-modal-endpoint.modal.run/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking: Major fire in Stockholm",
    "description": "Emergency services responding to downtown fire"
  }'
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "category": "emergencies",
    "score": 0.824,
    "tag": "fire"
  }
}
```

### Complete Workflow Example

```bash
# 1. Generate templates (one-time setup or refresh)
python ml/utils/generate_templates.py --articles-per-category 100 --country sweden

# 2. Label articles using templates
python ml/utils/label_dataset.py --articles 500 --country sweden --llm-verify --max-llm-calls 100

# 3. Fine-tune model on labeled data
python ml/models/quick_finetune.py \
  --train ml/data/labeled_articles_train.parquet \
  --val ml/data/labeled_articles_val.parquet

# 4. Test model locally
export OPENAI_API_KEY="your-key-here"
python tests/model_inference_smoke.py

# 5. Deploy to Modal
modal deploy ml/models/serve_modal.py
```

### Key Configuration Files

- **`ml/ingestion/hopsworks_pipeline.py`**: Defines SIGNAL_CATEGORIES and TAG_VOCAB (shared across all stages)
- **`ml/signal_templates.json`**: Generated templates (input to labeling stage)
- **`ml/tag_keywords.json`**: Generated keywords (input to labeling stage)
- **`ml/models/checkpoints/best_model.pt`**: Fine-tuned model checkpoint (input to inference)

_To be written once initial implementation begins_

Will include:
- Environment setup instructions
- API key requirements (GDELT, weather API)
- Model training/fine-tuning guide
- Local development workflow
- Deployment instructions

---

**Project Status**: Active Development - Model Training & Inference Complete  
**Last Updated**: January 11, 2026