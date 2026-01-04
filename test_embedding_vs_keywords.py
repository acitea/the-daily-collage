#!/usr/bin/env python3
"""
Compare keyword-based vs embedding-based article labeling.

Tests both approaches on Swedish articles and shows side-by-side comparison.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import classify_article as keyword_classify
from ml.utils.embedding_labeling import classify_article_embedding

# Test cases: (title, description, expected_category)
TEST_CASES = [
    (
        "Brand utbryter i centrala Stockholm",
        "En kraftig brand har utbrutit på Kungsholmen i Stockholm. Räddningstjänsten är på plats.",
        "emergencies"
    ),
    (
        "Polis letar efter bankrånare",
        "En man är gripen efter ett bankrån på Storgatan. Polisen söker ytterligare en misstänkt.",
        "crime"
    ),
    (
        "Rockfestivalen drar tiotusentals besökare",
        "Årets stora konsert lockar tusentals fans. Festivalen börjar nästa fredag.",
        "festivals"
    ),
    (
        "Trafikstörning på E4",
        "En trafikolycka på E4 norr om Uppsala orsakar långa köer. Vägen är delvis stängd.",
        "transportation"
    ),
    (
        "Värmebölja på väg till Sverige",
        "Meteorologerna varnar för extrem värme nästa vecka. Temperaturerna kan nå 35 grader.",
        "weather_temp"
    ),
    (
        "Översvämningar i Värmland",
        "Kraftiga regn och skyfall orsakar översvämningar i flera områden. Vägar är stängda.",
        "weather_wet"
    ),
    (
        "AIK vinner derbyt mot Djurgården",
        "I en spännande fotbollsmatch vinner AIK mot Djurgården 3-2. Det var en fantastisk seger.",
        "sports"
    ),
    (
        "Börsen stiger på goda nyheter",
        "Aktiemarknad stiger på grund av positiva företagsrapporter. Handelsmännen är optimistiska.",
        "economics"
    ),
    (
        "Riksdag debatterar ny klimatpolitik",
        "Regeringen presenterar en ny klimatpolicy för att minska koldioxid. Oppositionen kritiserar planen.",
        "politics"
    ),
]


def print_comparison(title: str, description: str, expected: str):
    """Print side-by-side comparison of classification methods."""
    print(f"\n{'=' * 90}")
    print(f"Title: {title}")
    print(f"Desc:  {description[:80]}...")
    print(f"Expected: {expected}")
    print(f"{'-' * 90}")
    
    # Keyword-based
    print("Keyword-based:")
    keyword_result = keyword_classify(title, description)
    if keyword_result:
        for cat, (score, tag) in sorted(keyword_result.items()):
            print(f"  {cat:18s}: {score:+.2f}  tag={tag}")
    else:
        print("  (no results)")
    
    # Embedding-based
    print("\nEmbedding-based:")
    try:
        embedding_result = classify_article_embedding(title, description)
        if embedding_result:
            for cat, (score, tag) in sorted(embedding_result.items()):
                print(f"  {cat:18s}: {score:+.2f}  tag={tag}")
        else:
            print("  (no results above threshold)")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("\n" + "=" * 90)
    print("COMPARING KEYWORD-BASED vs EMBEDDING-BASED ARTICLE LABELING")
    print("=" * 90)
    
    print("\nNote: First run will download Swedish BERT model (~500 MB)")
    
    for title, description, expected in TEST_CASES:
        print_comparison(title, description, expected)
    
    print("\n" + "=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)
    print("\n✨ Key differences observed above between keyword-based and embedding-based approaches")


if __name__ == "__main__":
    main()
