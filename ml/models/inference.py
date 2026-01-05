"""
Production inference module for fine-tuned model.
Integrates with existing hopsworks_pipeline.py classification system.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys
import logging
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NewsSignalClassifierInference:
    """
    Production inference wrapper for fine-tuned model.
    Can be used as a drop-in replacement for classify_article().
    """

    def __init__(
        self,
        model_path: str,
        base_model: str = "KB/bert-base-swedish-cased",
        device: Optional[torch.device] = None,
    ):
        """
        Load fine-tuned model for inference.

        Args:
            model_path: Path to best_model.pt checkpoint
            base_model: Base model name
            device: torch device (auto-detected if None)
        """
        self.device = device or DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load model architecture
        self.model = self._build_model(base_model)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Loaded fine-tuned model from {model_path}")

    def _build_model(self, base_model: str) -> nn.Module:
        """Build model architecture (mirrors training model)."""
        bert = AutoModel.from_pretrained(base_model)
        hidden_size = 768

        # Score heads (must match quick_finetune.py architecture exactly)
        score_heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Tanh()
            )
            for cat in SIGNAL_CATEGORIES
        })

        # Tag heads (must match quick_finetune.py architecture exactly)
        tag_heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Dropout(0.1),
                nn.Linear(128, len(TAG_VOCAB[cat]))
            )
            for cat in SIGNAL_CATEGORIES
        })

        # Combine into module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = bert
                self.score_heads = score_heads
                self.tag_heads = tag_heads

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output

                results = {}
                for cat in SIGNAL_CATEGORIES:
                    score = self.score_heads[cat](pooled).squeeze(-1)
                    tag_logits = self.tag_heads[cat](pooled)
                    results[cat] = (score, tag_logits)

                return results

        return Model()

    @torch.no_grad()
    def classify(self, title: str, description: str = "") -> Dict[str, Tuple[float, str]]:
        """
        Classify article using fine-tuned model.

        Returns:
            Dict mapping category -> (score, tag)
            e.g., {"emergencies": (0.8, "fire"), "crime": (0.0, "")}
        """
        text = f"{title} [SEP] {description}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Forward pass
        predictions = self.model(input_ids, attention_mask)

        # Process outputs
        results = {}
        for cat in SIGNAL_CATEGORIES:
            score_tensor, tag_logits = predictions[cat]
            score = float(score_tensor.squeeze().cpu())

            # Get tag
            tag_idx = torch.argmax(tag_logits, dim=-1).item()
            tag = TAG_VOCAB[cat][tag_idx] if tag_idx < len(TAG_VOCAB[cat]) else ""

            # Include if relevant (lowered threshold for sparse training data)
            # For limited training data, use 0.0 threshold to show all non-neutral predictions
            if abs(score) > 0.01 or tag != "":
                results[cat] = (score, tag)

        return results


# Global classifier instance
_classifier: Optional[NewsSignalClassifierInference] = None


def get_fine_tuned_classifier(
    model_path: str = "ml/models/checkpoints/best_model.pt",
) -> NewsSignalClassifierInference:
    """Get or create global classifier instance."""
    global _classifier

    if _classifier is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        _classifier = NewsSignalClassifierInference(model_path)

    return _classifier
