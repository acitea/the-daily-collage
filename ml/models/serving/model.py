import torch
from pathlib import Path
from transformers import AutoTokenizer

try:
    # Import from the project when available in the serving container
    from ml.models.quick_finetune import QuickNewsClassifier, BASE_MODEL
    from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB
except Exception:
    # Fallback imports if module paths differ; adjust if your serving environment uses different paths
    from ..quick_finetune import QuickNewsClassifier, BASE_MODEL  # type: ignore
    from ...ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB  # type: ignore


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_tokenizer = None


def init():
    """Initialize the model and tokenizer. Called once on deployment start."""
    global _model, _tokenizer

    model_dir = Path(__file__).resolve().parent
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = QuickNewsClassifier(BASE_MODEL)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model = model
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def _infer_one(title: str, description: str):
    text = f"{title} [SEP] {description or ''}"
    enc = _tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask)

    result = {}
    for cat in SIGNAL_CATEGORIES:
        score_tensor, tag_logits = outputs[cat]
        score = float(score_tensor.squeeze().cpu())
        pred_idx = int(tag_logits.argmax(dim=-1).squeeze().cpu())
        tag = TAG_VOCAB[cat][pred_idx] if 0 <= pred_idx < len(TAG_VOCAB[cat]) else ""
        result[cat] = {"score": score, "tag": tag}
    return result


def predict(inputs):
    """
    Hopsworks serving entrypoint.
    Accepts either a single object {title, description} or a list of such objects.
    Returns prediction dict(s) mapping each category to {score, tag}.
    """
    if _model is None or _tokenizer is None:
        init()

    # Single item
    if isinstance(inputs, dict):
        title = inputs.get("title", "")
        description = inputs.get("description", "")
        return _infer_one(title, description)

    # Batch
    if isinstance(inputs, list):
        out = []
        for item in inputs:
            title = (item or {}).get("title", "")
            description = (item or {}).get("description", "")
            out.append(_infer_one(title, description))
        return out

    return {"error": "Invalid input format. Provide {title, description} or a list of such objects."}
