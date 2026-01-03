#!/usr/bin/env python3
"""
Quick test of the updated bootstrap with 500 articles.
This will create training/validation splits with 500 articles total.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.quick_bootstrap import quick_bootstrap

def main():
    print("\n" + "=" * 70)
    print("BOOTSTRAP WITH 500 ARTICLES (BATCHED GDELT FETCH)")
    print("=" * 70 + "\n")
    
    # Fetch and prepare training data
    train_file, val_file = quick_bootstrap(
        countries=["sweden"],
        articles_per_country=500,  # Will make 2 requests of 250 each
        output_dir="ml/data",
        use_batching=True,
        batch_size=250,
        days_lookback=30,
    )
    
    print("\n" + "=" * 70)
    print("BOOTSTRAP COMPLETE")
    print("=" * 70)
    print(f"\n✅ Training set: {train_file}")
    print(f"✅ Validation set: {val_file}")
    
    # Check file sizes
    if Path(train_file).exists():
        train_size = Path(train_file).stat().st_size / 1024 / 1024
        print(f"   Size: {train_size:.2f} MB")
    
    if Path(val_file).exists():
        val_size = Path(val_file).stat().st_size / 1024 / 1024
        print(f"   Size: {val_size:.2f} MB")
    
    print("\n✨ Ready for training! Run: ./train_one_day.sh")

if __name__ == "__main__":
    main()
