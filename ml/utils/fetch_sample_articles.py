"""
Quick script to fetch and inspect real GDELT articles for template creation.

This helps us understand the actual language patterns in GDELT data
so we can create better semantic templates.
"""

import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news

def inspect_articles():
    """Fetch and display sample articles to understand patterns."""
    
    print("\n" + "="*80)
    print("FETCHING GDELT SAMPLE ARTICLES FOR TEMPLATE INSPECTION")
    print("="*80 + "\n")
    
    # Fetch articles
    df = fetch_news(country="sweden", max_articles=200)
    
    print(f"Fetched {len(df)} articles\n")
    print("="*80)
    print("SAMPLE HEADLINES AND DESCRIPTIONS")
    print("="*80 + "\n")
    
    # Display first 100 articles with titles and descriptions
    for idx, row in enumerate(df.head(100).iter_rows(named=True)):
        title = row.get("title", "")
        description = row.get("description") or row.get("content") or row.get("summary") or ""
        
        # Truncate for readability
        desc_short = description[:150] if description else "(no description)"
        
        print(f"{idx+1}. [{row.get('date', '')}]")
        print(f"   Title: {title}")
        print(f"   Desc:  {desc_short}...")
        print()
    
    # Save to file for manual inspection
    output_file = Path("sample_articles.parquet")
    df.head(200).write_parquet(output_file)
    print(f"\n✓ Saved 200 sample articles to {output_file}")
    print("  You can inspect this file to find good template excerpts")

if __name__ == "__main__":
    inspect_articles()
