# The Daily Collage

A proof-of-concept system that transforms Swedish news articles into a single, cartoonish visualization that captures the collective "vibe" of what's currently happening in that area.

## Overview

*"A picture is worth a thousand words"* - instead of reading through dozens of news articles, users can glance at a generated image to understand the current sentiment and major events in Sweden. If something in the visualization catches their interest, they can explore the underlying news sources for more details.

**Geographic Scope**: Primarily targeting Sweden and Swedish cities (e.g., Stockholm, Gothenburg, Malmö).

**Update Frequency**: Visualizations refresh every 6 hours to capture emerging "hot" news.

**Visit the Website**: [The Daily Collage](https://acitea.github.io/the-daily-collage/)

## Technical Architecture

### Data Pipeline

**Data Source**: GDELT Project (Global Database of Events, Language, and Tone)
- Provides credible, global news headlines with geographic tagging
- Access via GDELT 2.0 DOC API for real-time article monitoring
- Python client available

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
- Based on state-of-the-art architectures (i.e., BERT)
- Multi-label classification to assign articles to signal categories
- Intensity scoring for each detected signal (0-1 scale)

**Model Training Approach**:
- Fine-tune existing pre-trained models rather than training from scratch
- Focus on Swedish news corpus for localized understanding

### Image Generation Strategy

**Step 1: Template-Based Composition**
- Pre-designed cartoon-style templates for each signal category
- Dynamic composition system that arranges and scales elements based on signal intensity
- Faster iteration and controllable output
- Easier to ensure consistent visual style
- Templates stored as vector graphics or layered image assets

**Step 2: Refine created template with image-to-image model**
- Enables cohesive blending of elements
- Adjusts color palette based on weather conditions (e.g., gloomy colors for stormy weather
- Enhances overall aesthetic quality

**Image Caching Strategy**:
- Artifact store for generated visualizations
- Cache key based on time window and location
- Potential database: MinIO, S3-compatible storage, or PostgreSQL with blob storage

### Stack

**Components**:
- News ingestion service (GDELT API CRON ingestion every 6 hours)
- ML inference service (sentiment + intensity classification)
- Image generation/composition service
- Caching layer

### Deployment

- Backend API: FastAPI application deployed on Render
- Frontend: React application deployed on Github Pages
- Model hosting: Modal serverless functions for scalable inference
- Job scheduling: Github Actions
- Artifact storage: Hopsworks filesystem
- Feature store: Hopsworks Feature Store for headlines storage and retrieval

## Data Flow

<img width="1222" height="359" alt="Job Pipeline + Image Generation Strategy" src="https://github.com/user-attachments/assets/eb41bd13-c2f0-41da-91ae-83e05836e52c" />

1. **Ingestion** (every 6 hours):
   - Query GDELT API for news articles matching geographic filters & time window
   - Store raw articles in processing quexeue

2. **Processing**:
   - Run through fine-tuned sentiment classification model
   - Extract signal categories and intensity scores
   - Aggregate signals by location

3. **Visualization Generation**:
   - Apply template-based composition based on aggregated signals
   - Apply weather-based mood adjustments (color palette, background)
   - Refine with image-to-image model
   - Store in artifact repository with metadata

4. **API Response**:
   - Serve stored generated visualization
   - Include JSON metadata with signal breakdown and source articles

## Visual Style

**Aesthetic**: Cartoonish, illustrative style, or even in a sticker format rather than photorealistic
- Friendly, approachable visual language
- Clear iconography for each signal category
- Consistent color palette across visualizations
- Think: weather app icons, infographic style, or editorial illustrations

## Repository Structure

```

the-daily-collage/
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
│   ├── ingestion/        # GDELT & Weather API fetching -> Feature Store -> Vibe Vector
│   └── models/           # Relevant scripts to define, train, and serve models
└── frontend/             # React + Vite application
```
## Getting Started

  

### Model Training Pipeline

  

The classifier is trained through a multi-stage pipeline from raw articles to a fine-tuned model ready for production inference.

### Training Architecture
<img width="4651" height="2221" alt="image" src="https://github.com/user-attachments/assets/c204e37c-ae54-4955-b12f-e28a36f38942" />


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
python  ml/utils/generate_templates.py  --articles-per-category  100  --country  sweden

# Generate more templates per category (default 30)
python  ml/utils/generate_templates.py  --articles-per-category  150  --templates-per-category  50

# Search further back in time
python  ml/utils/generate_templates.py  --articles-per-category  150  --days-lookback  180

```

  

**Output**:

-  `ml/signal_templates.json` - Templates for each signal category
-  `ml/tag_keywords.json` - Keywords associated with each tag
  

#### Stage 2: Article Labeling (`ml/utils/label_dataset.py`)


**Purpose**: Create labeled training data using embedding-based classification with optional LLM verification

  

**Process**:

1. Fetches articles from GDELT in batches
2. Classifies articles using **embedding similarity** against SIGNAL_TEMPLATES
3. Assigns (score*, tag**) pairs for each signal category
4. Optionally verifies labels using LLM for quality assurance
5. Outputs labeled parquet files with structure:
-  `title`, `description`, `url`, `source`
-  `emergencies_score`, `emergencies_tag`
-  `crime_score`, `crime_tag`
- ... (one pair per signal category)

**by score we mean the intensity score of individual news article, not the confidence score. Although those two are closely related in our implementation, as we assume that the more negative/positive the news article is the more likely it is that it belongs to a specific news category (e.g., `emergency` with score of -0.732 is negative news with high confidence it being an emergency, while the model would classify the same article also as `sports` category but with score of `+0.00145` which can be interpreted as unlikely of being part of that category.*

***tagging news articles was the initial idea to have even more accurate classification, however due to difficulty assigning such titles to relatively large training set (around 5000 articles) such that is not used by inference.*
  

**Usage**:

```bash
# Generate 500 labeled articles without LLM verification (fast)
python  ml/utils/label_dataset.py  --articles  500  --country  sweden

# Generate 200 articles with LLM verification (slower but higher quality)
python  ml/utils/label_dataset.py  --articles  200  --country  sweden  --llm-verify  --max-llm-calls  50

```



**Output**:

-  `ml/data/labeled_articles_train.parquet` - 70% training set
-  `ml/data/labeled_articles_val.parquet` - 30% validation set
  

#### Stage 3: Model Fine-tuning (`ml/models/quick_finetune.py`)
  

**Purpose**: Fine-tune a Swedish BERT model on the labeled data

  

**Architecture**:

-  **Base Model**: `KB/bert-base-swedish-cased` (Swedish-specific BERT)
-  **Multi-head Design**:
- 9 parallel "score heads" (regression, outputs -1.0 to 1.0 intensity)
- 9 parallel "tag heads" (classification, predicts which tag within category)
-  **Loss Function**: Weighted combination of regression loss (score) and classification loss (tag)
  

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
python  ml/models/quick_finetune.py \
--train ml/data/labeled_articles_train.parquet \
--val  ml/data/labeled_articles_val.parquet \
--output ml/models/checkpoints

# Use custom output directory
python  ml/models/quick_finetune.py \
--train ml/data/labeled_articles_train.parquet \
--val  ml/data/labeled_articles_val.parquet \
--output ./custom_checkpoints

```

  

**Output**:

-  `ml/models/checkpoints/best_model.pt` - Best model checkpoint with state dict

  
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
python  tests/model_inference_smoke.py

# Run without OpenAI (uses top-3 ranking as fallback validation)
python  tests/model_inference_smoke.py

```
  

**Example Output**:

```
[1/3] ✅ PASS | Breaking: Major fire in Stockholm...

Desc: Emergency services responding to downtown fire

Top 3 predictions:
→ [1] emergencies score=+0.824 tag=fire
[2] crime score=+0.312 tag=
[3] politics score=+0.105 tag=
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
modal  deploy  ml/models/serve_modal.py

# Or run locally with Modal
modal  run  ml/models/serve_modal.py
```
  

**API Usage**:

```bash
curl  -X  POST  https://your-modal-endpoint.modal.run/api/predict \
-H "Content-Type: application/json" \
-d  '{
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
python  ml/utils/generate_templates.py  --articles-per-category  100  --country  sweden

# 2. Label articles using templates
python  ml/utils/label_dataset.py  --articles  500  --country  sweden  --llm-verify  --max-llm-calls  100

# 3. Fine-tune model on labeled data
python  ml/models/quick_finetune.py \
--train ml/data/labeled_articles_train.parquet \
--val  ml/data/labeled_articles_val.parquet

# 4. Test model locally
export OPENAI_API_KEY="your-key-here"
python  tests/model_inference_smoke.py

# 5. Deploy to Modal
modal  deploy  ml/models/serve_modal.py
```

  

### Key Configuration Files

  

-  **`ml/ingestion/hopsworks_pipeline.py`**: Defines SIGNAL_CATEGORIES and TAG_VOCAB (shared across all stages)

-  **`ml/signal_templates.json`**: Generated templates (input to labeling stage)

-  **`ml/tag_keywords.json`**: Generated keywords (input to labeling stage)

-  **`ml/models/checkpoints/best_model.pt`**: Fine-tuned model checkpoint (input to inference)

---

 
**Last Updated**: January 14, 2026
