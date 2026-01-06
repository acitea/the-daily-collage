"""
Test script to verify embedding_labeling.py can load from JSON files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.utils.embedding_labeling import load_signal_templates, load_tag_keywords

print("Testing JSON loading...")
print()

# Test signal templates loading
print("1. Loading signal templates...")
templates = load_signal_templates()
print(f"   ✓ Loaded {len(templates)} categories")
for cat, temps in list(templates.items())[:3]:
    print(f"     - {cat}: {len(temps)} templates")
print()

# Test tag keywords loading  
print("2. Loading tag keywords...")
keywords = load_tag_keywords()
print(f"   ✓ Loaded {len(keywords)} categories")
for cat, kws in list(keywords.items())[:3]:
    print(f"     - {cat}: {len(kws)} keywords")
print()

# Test classification with loaded templates
print("3. Testing article classification...")
from ml.utils.embedding_labeling import classify_article_embedding

test_article = {
    "title": "Brand i lägenhet – flera personer evakuerade",
    "description": "Räddningstjänsten larmades till en lägenhetsbrand i centrala staden. Flera personer evakuerades från byggnaden."
}

result = classify_article_embedding(
    title=test_article["title"],
    description=test_article["description"],
    similarity_threshold=0.30
)

print(f"   Article: {test_article['title']}")
print(f"   Classification result:")
for cat, (score, tag) in result.items():
    print(f"     - {cat}: score={score:.2f}, tag='{tag}'")

print()
print("✓ All tests passed!")
