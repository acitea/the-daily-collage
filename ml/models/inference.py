"""
LLM-based classification module for news signal detection.
Uses OpenAI API to classify articles into signal categories with intensity scores.
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not installed. LLM classification will not work.")


class NewsSignalClassifierInference:
    """
    LLM-based inference for news signal classification.
    Uses OpenAI API to classify articles and assign intensity scores.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        """
        Initialize LLM-based classifier.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI model to use
            temperature: Sampling temperature (0-1)
        """
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info(f"✓ Initialized LLM classifier with model: {model}")

    def classify(self, title: str, description: str = "") -> Dict[str, Tuple[float, str]]:
        """
        Classify article using LLM.

        Args:
            title: Article title
            description: Article description/body (optional)

        Returns:
            Dict mapping category -> (score, tag)
            e.g., {"emergencies": (0.8, "fire"), "crime": (0.0, "")}
        """
        # Truncate description to avoid token limits
        desc_truncated = description[:1000] if description else ""
        
        prompt = f"""Analyze this Swedish news article and classify it into signal categories.

ARTICLE:
Title: {title}
Description: {desc_truncated}

SIGNAL CATEGORIES (score range and tags):
- emergencies: Natural disasters, accidents, fires, evacuations (-1 to -0.5 for severe)
  Tags: {', '.join(TAG_VOCAB['emergencies'][1:])}
  
- crime: Criminal incidents, violence, arrests, theft (-1 to -0.5 for serious)
  Tags: {', '.join(TAG_VOCAB['crime'][1:])}
  
- festivals: Cultural events, celebrations, concerts, entertainment (0.3 to 1 for positive)
  Tags: {', '.join(TAG_VOCAB['festivals'][1:])}
  
- transportation: Traffic accidents, congestion, delays, closures (-0.5 to 0 for problems)
  Tags: {', '.join(TAG_VOCAB['transportation'][1:])}
  
- weather_temp: Temperature extremes, heatwaves, cold snaps (-0.8 to 0 for severe)
  Tags: {', '.join(TAG_VOCAB['weather_temp'][1:])}
  
- weather_wet: Rain, flooding, snow, storms (-0.8 to 0 for severe)
  Tags: {', '.join(TAG_VOCAB['weather_wet'][1:])}
  
- sports: Sporting events, competitions, victories (-0.5 to 1)
  Tags: {', '.join(TAG_VOCAB['sports'][1:])}
  
- economics: Markets, business, employment, inflation (-0.8 to 1)
  Tags: {', '.join(TAG_VOCAB['economics'][1:])}
  
- politics: Elections, policy, governance, protests (-0.5 to 1)
  Tags: {', '.join(TAG_VOCAB['politics'][1:])}

SCORING GUIDELINES:
- Score represents intensity/impact (-1 to +1)
- Negative scores for problems/bad events (fires, crime, accidents, storms)
- Positive scores for good events (festivals, victories, celebrations, economic growth)
- 0 = irrelevant/absent
- 0.3 to 0.5 or -0.3 to -0.5 = moderate impact
- 0.6 to 0.9 or -0.6 to -0.9 = major impact
- Only include categories that are clearly relevant to this article

Return ONLY valid JSON:
{{
  "emergencies": {{"score": -0.8, "tag": "fire"}},
  "crime": {{"score": -0.6, "tag": "theft"}},
  ...
}}

If a category is not relevant, omit it entirely from the response."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result_json = json.loads(result_text)
            
            # Convert to expected format
            results = {}
            for category, data in result_json.items():
                if category in SIGNAL_CATEGORIES:
                    score = float(data.get("score", 0.0))
                    tag = str(data.get("tag", ""))
                    
                    # Validate tag is in vocabulary
                    if tag and tag not in TAG_VOCAB[category]:
                        # Find closest match or use empty
                        tag = ""
                    
                    # Only include if score is non-zero or tag is present
                    if abs(score) > 0.01 or tag:
                        results[category] = (score, tag)
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {result_text}")
            return {}
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {}


# Global classifier instance
_classifier: Optional[NewsSignalClassifierInference] = None


def get_fine_tuned_classifier(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> NewsSignalClassifierInference:
    """
    Get or create global LLM classifier instance.
    
    Args:
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        model: OpenAI model to use
    """
    global _classifier
    if _classifier is None:
        _classifier = NewsSignalClassifierInference(api_key=api_key, model=model)
    return _classifier
