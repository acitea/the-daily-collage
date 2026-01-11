"""
Serve the Daily Collage classifier model using Modal.
Runs the model inference as a serverless HTTP API.

Usage:
  modal run ml/models/serve_modal.py
  
Then Modal will print the endpoint URL to call.

Test with curl:
  curl -X POST https://your-modal-endpoint.modal.run/predict \\
    -H "Content-Type: application/json" \\
    -d '{"title": "Breaking news", "description": "Story details"}'
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import modal
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal image with dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "torch",
    "transformers",
    "hopsworks",
    "pyarrow",
    "polars",
)

# Persistent volume for model caching
model_volume = modal.Volume.from_name("daily-collage-models", create_if_missing=True)
CACHE_ROOT = Path("/mnt/models")
HF_CACHE = CACHE_ROOT / "hf"

app = modal.App("daily-collage-classifier", image=image)


# Signal categories and tags (hardcoded since Modal doesn't have access to ml module)
SIGNAL_CATEGORIES = [
    "emergencies",
    "crime", 
    "festivals",
    "transportation",
    "weather_temp",
    "weather_wet",
    "sports",
    "economics",
    "politics",
]

TAG_VOCAB = {
    "emergencies": ["", "fire", "earthquake", "explosion", "evacuation", "accident"],
    "crime": ["", "theft", "assault", "robbery", "police", "vandalism"],
    "festivals": ["", "concert", "celebration", "parade", "crowd", "event"],
    "transportation": ["", "traffic", "accident", "congestion", "delay", "closure"],
    "weather_temp": ["", "hot", "cold", "heatwave", "freeze"],
    "weather_wet": ["", "rain", "snow", "flood", "storm", "drought"],
    "sports": ["", "football", "hockey", "victory", "championship", "game"],
    "economics": ["", "market", "business", "trade", "employment", "inflation"],
    "politics": ["", "election", "protest", "government", "policy", "vote"],
}


# Global model state (persists across invocations)
_model = None
_tokenizer = None


def load_model(
    api_key: str,
    project: str,
    model_name: str,
    model_version: Optional[int] = None,
):
    """Load model from Hopsworks during container init."""
    global _model, _tokenizer
    
    if _model is not None:
        return  # Already loaded
    
    logger.info(f"Loading model {model_name} v{model_version} (cached if available)...")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache_dir = CACHE_ROOT / f"{model_name}_v{model_version}"
    ckpt_path = cache_dir / "best_model.pt"

    # Import locally to avoid issues outside Modal environment
    import hopsworks
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    
    if not ckpt_path.exists():
        # Login to Hopsworks and download
        project_obj = hopsworks.login(api_key_value=api_key, project=project, engine="python")
        mr = project_obj.get_model_registry()
        model_obj = mr.get_model(name=model_name, version=model_version)
        model_dir = Path(model_obj.download())
        src_ckpt = model_dir / "best_model.pt"
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ckpt, ckpt_path)
        logger.info(f"Cached checkpoint at {ckpt_path}")
    else:
        logger.info(f"Using cached checkpoint at {ckpt_path}")
    
    # Load checkpoint
    # Modal doesn't have access to ml module, so use inline definition
    BASE_MODEL = "KB/bert-base-swedish-cased"
    
    # Define QuickNewsClassifier inline
    import torch.nn as nn
    from transformers import AutoModel
    
    class QuickNewsClassifier(nn.Module):
        """Minimal multi-head classifier."""
        
        def __init__(self, base_model: str = BASE_MODEL, cache_dir: Optional[Path] = None):
            super().__init__()
            self.bert = AutoModel.from_pretrained(base_model, cache_dir=str(cache_dir) if cache_dir else None)
            hidden_size = 768
            
            # Score heads (regression)
            self.score_heads = nn.ModuleDict({
                cat: nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                    nn.Tanh()
                )
                for cat in SIGNAL_CATEGORIES
            })
            
            # Tag heads (classification)
            self.tag_heads = nn.ModuleDict({
                cat: nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, len(TAG_VOCAB[cat]))
                )
                for cat in SIGNAL_CATEGORIES
            })
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output
            
            results = {}
            for cat in SIGNAL_CATEGORIES:
                score = self.score_heads[cat](pooled).squeeze(-1)
                tag_logits = self.tag_heads[cat](pooled)
                results[cat] = (score, tag_logits)
            
            return results
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model = QuickNewsClassifier(BASE_MODEL, cache_dir=HF_CACHE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    _model = model
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=str(HF_CACHE))
    
    logger.info("✓ Model loaded and ready for inference")


def predict_single(title: str, description: str = "") -> Dict[str, Union[float, str]]:
    """Run inference on a single article, return only the top category."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare input
    text = f"{title} [SEP] {description or ''}"
    enc = _tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    # Infer
    with torch.no_grad():
        outputs = _model(input_ids, attention_mask)
    
    # Parse results and find top category
    all_scores = {}
    for cat in SIGNAL_CATEGORIES:
        score_tensor, tag_logits = outputs[cat]
        score = float(score_tensor.squeeze().cpu())
        pred_idx = int(tag_logits.argmax(dim=-1).squeeze().cpu())
        tag = TAG_VOCAB[cat][pred_idx] if 0 <= pred_idx < len(TAG_VOCAB[cat]) else ""
        all_scores[cat] = (score, tag)
    
    # Return only top-ranked category
    top_cat = max(all_scores.items(), key=lambda x: abs(x[1][0]))  # Sort by absolute score
    category, (score, tag) = top_cat
    
    return {"category": category, "score": score, "tag": tag}


@app.function(
    timeout=600,
    memory=4096,
    secrets=[modal.Secret.from_name("hopsworks-credentials")],
    volumes={"/mnt/models": model_volume},
)
def init_model():
    """Initialize model on container startup."""
    import os
    
    api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("API_KEY")
    project = os.environ.get("HOPSWORKS_PROJECT") or os.environ.get("PROJECT", "daily_collage")
    model_name = os.environ.get("MODEL_NAME", "daily_collage_classifier")
    model_version = int(os.environ.get("MODEL_VERSION", "1"))
    
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY or API_KEY environment variable required")
    
    load_model(api_key=api_key, project=project, model_name=model_name, model_version=model_version)


def predict(payload: Dict) -> Dict:
    """
    Inference function (not a Modal remote function - called locally).
    
    Accepts:
      - Single item: {"title": "...", "description": "..."}
      - Batch: {"instances": [{"title": "...", "description": "..."}, ...]}
    
    Returns:
      - Single: {"category": "festivals", "score": 0.85, "tag": "concert"}
      - Batch: {"predictions": [{"category": "...", "score": ..., "tag": "..."}, ...]}
    """
    # Lazy-load model on first invocation
    if _model is None:
        import os
        api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("API_KEY")
        project = os.environ.get("HOPSWORKS_PROJECT") or os.environ.get("PROJECT", "daily_collage")
        model_name = os.environ.get("MODEL_NAME", "daily_collage_classifier")
        model_version = int(os.environ.get("MODEL_VERSION", "1"))
        load_model(api_key=api_key, project=project, model_name=model_name, model_version=model_version)
    
    # Handle batch vs single
    if "instances" in payload:
        instances = payload["instances"]
    elif "title" in payload:
        instances = [payload]
    else:
        return {"error": "Provide 'title' or 'instances' in request"}
    
    results = []
    for item in instances:
        title = item.get("title", "")
        description = item.get("description", "")
        prediction = predict_single(title, description)
        results.append(prediction)
    
    # Return in Hopsworks format if batch, otherwise single
    if len(results) == 1 and "instances" not in payload:
        return results[0]
    else:
        return {"predictions": results}


@app.function(
    timeout=600,
    memory=4096,
    secrets=[modal.Secret.from_name("hopsworks-credentials")],
    concurrency_limit=10,
    volumes={"/mnt/models": model_volume},
)
@modal.web_endpoint(method="POST")
async def api_predict(request: dict) -> dict:
    """
    HTTP POST endpoint for inference.
    
    Example request:
    {
      "title": "Stockholm hosts winter festival",
      "description": "Large crowds expected downtown"
    }
    
    Example response:
    {
      "status": "success",
      "data": {
        "category": "festivals",
        "score": 0.85,
        "tag": "concert"
      }
    }
    """
    try:
        result = predict(request)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# Local testing
if __name__ == "__main__":
    import os
    
    # For local testing, set environment variables:
    # export HOPSWORKS_API_KEY="your-key"
    # export HOPSWORKS_PROJECT="your-project"
    # export MODEL_NAME="test"
    # export MODEL_VERSION="3"
    
    os.environ.setdefault("HOPSWORKS_PROJECT", "terahidro2003")
    os.environ.setdefault("MODEL_NAME", "test")
    os.environ.setdefault("MODEL_VERSION", "3")
    
    # Load model
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("Set HOPSWORKS_API_KEY environment variable")
    
    load_model(
        api_key=api_key,
        project=os.environ.get("HOPSWORKS_PROJECT"),
        model_name=os.environ.get("MODEL_NAME"),
        model_version=int(os.environ.get("MODEL_VERSION")),
    )
    
    # Test prediction
    print("\nTesting inference:")
    result = predict_single(
        title="Stockholm hosts winter festival",
        description="Large crowds expected downtown"
    )
    print(json.dumps(result, indent=2))
