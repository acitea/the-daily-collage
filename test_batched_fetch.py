#!/usr/bin/env python3
"""
Test script for batched GDELT fetching.
Verifies that we can fetch >250 articles using multiple requests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news_batched

def test_batched_fetch():
    """Test fetching 500 articles from Sweden in batches."""
    print("=" * 60)
    print("Testing Batched GDELT Fetch")
    print("=" * 60)
    
    print("\nFetching 500 articles from Sweden in 250-article batches...")
    print("This will make 2 requests with 0.5s delay between them.\n")
    
    try:
        df = fetch_news_batched(
            country="sweden",
            total_articles=500,
            batch_size=250,
            days_lookback=30,
            batch_delay=0.5
        )
        
        print(f"\n✅ Success! Fetched {len(df)} articles")
        
        if not df.is_empty():
            print("\nFirst 3 articles:")
            for i, row in enumerate(df.head(3).iter_rows(named=True)):
                print(f"\n  {i+1}. {row.get('title', 'N/A')[:70]}")
                print(f"     Source: {row.get('source', 'N/A')}")
                print(f"     URL: {row.get('url', 'N/A')[:60]}")
            
            print(f"\nColumns in DataFrame: {df.columns}")
            print(f"DataFrame shape: {df.shape}")
        else:
            print("⚠️  DataFrame is empty")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error during batched fetch:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batched_fetch()
    sys.exit(0 if success else 1)
