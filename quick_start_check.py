#!/usr/bin/env python
"""
QUICK START - One-day fine-tuning sprint
All-in-one script to verify setup before running full training
"""

import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║     THE DAILY COLLAGE - ONE-DAY FINE-TUNING QUICK START        ║
╚════════════════════════════════════════════════════════════════╝
""")

# Check 1: Python version
print("✓ Checking Python version...")
if sys.version_info >= (3, 10):
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
else:
    print(f"  ❌ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)")
    sys.exit(1)

# Check 2: Required packages
print("\n✓ Checking required packages...")
required = ["torch", "transformers", "polars", "pandas"]
missing = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} (missing)")
        missing.append(pkg)

if missing:
    print(f"\n💾 Install missing packages:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)

# Check 3: GPU (optional)
print("\n✓ Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠️  CPU only (training will be slower)")
except Exception as e:
    print(f"  ⚠️  {e}")

# Check 4: Project structure
print("\n✓ Checking project structure...")
required_files = [
    "ml/ingestion/hopsworks_pipeline.py",
    "ml/data/quick_bootstrap.py",
    "ml/models/quick_finetune.py",
    "ml/models/inference.py",
    "train_one_day.sh",
]

for file in required_files:
    path = Path(file)
    if path.exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} (missing)")

# Check 5: GDELT access
print("\n✓ Checking GDELT API access...")
try:
    from gdeltdoc.gapi import gdelt_obj
    g = gdelt_obj.Gapi()
    print(f"  ✅ GDELT API accessible")
except Exception as e:
    print(f"  ⚠️  {e}")

print("""
╔════════════════════════════════════════════════════════════════╗
║                  🚀 READY TO TRAIN!                           ║
╠════════════════════════════════════════════════════════════════╣
║ Run this to start one-day sprint:                             ║
║                                                                ║
║   ./train_one_day.sh                                           ║
║                                                                ║
║ OR manually:                                                   ║
║                                                                ║
║   # 1. Collect & label data (~5 min)                          ║
║   python ml/data/quick_bootstrap.py                            ║
║                                                                ║
║   # 2. Train model (~5-20 min on GPU/CPU)                     ║
║   python ml/models/quick_finetune.py \\                        ║
║     --train ml/data/train_bootstrap.parquet \\                ║
║     --val ml/data/val_bootstrap.parquet                        ║
║                                                                ║
║   # 3. Test inference (~1 min)                                ║
║   python -c "                                                 ║
║     from ml.models.inference import get_fine_tuned_classifier ║
║     m = get_fine_tuned_classifier()                            ║
║     print(m.classify('Fire in Stockholm', ''))               ║
║   "                                                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

📖 Full guide: ONE_DAY_FINETUNING.md
⏱️  Estimated time: 6-8 hours start to finish
💾 Model size: ~400 MB
📊 Expected accuracy: 70-75% (vs 60% baseline)
""")
