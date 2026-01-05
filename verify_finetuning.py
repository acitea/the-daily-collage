"""
Verify fine-tuning setup and run sanity checks
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_environment():
    """Verify all dependencies are installed."""
    print("🔍 Checking environment...")
    
    deps = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "polars": "Polars",
        "gdeltdoc": "GDELT API",
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Install with: pip install {module}")
            all_ok = False
    
    return all_ok

def check_files():
    """Verify training files exist."""
    print("\n🔍 Checking training files...")
    
    files = {
        "ml/data/quick_bootstrap.py": "Data collection script",
        "ml/models/quick_finetune.py": "Training script",
        "ml/models/inference.py": "Inference module",
        "ml/ingestion/hopsworks_pipeline.py": "Pipeline integration",
    }
    
    all_ok = True
    for file, desc in files.items():
        path = PROJECT_ROOT / file
        if path.exists():
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {file} not found")
            all_ok = False
    
    return all_ok

def check_data():
    """Check if training data exists."""
    print("\n🔍 Checking training data...")
    
    data_dir = PROJECT_ROOT / "ml/data"
    
    train_path = data_dir / "train_bootstrap.parquet"
    val_path = data_dir / "val_bootstrap.parquet"
    
    if train_path.exists() and val_path.exists():
        import polars as pl
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path)
        print(f"  ✅ Training data exists")
        print(f"     Train: {len(train_df)} articles")
        print(f"     Val: {len(val_df)} articles")
        return True
    else:
        print(f"  ⚠️  Training data not found (will be created)")
        print(f"     Run: python ml/data/quick_bootstrap.py")
        return False

def check_model():
    """Check if model checkpoint exists."""
    print("\n🔍 Checking model checkpoint...")
    
    model_path = PROJECT_ROOT / "ml/models/checkpoints/best_model.pt"
    
    if model_path.exists():
        print(f"  ✅ Model checkpoint found")
        print(f"     Size: {model_path.stat().st_size / 1e6:.1f} MB")
        
        # Try to load it
        try:
            from ml.models.inference import get_fine_tuned_classifier
            model = get_fine_tuned_classifier(str(model_path))
            print(f"  ✅ Model loads successfully")
            return True
        except Exception as e:
            print(f"  ❌ Model load failed: {e}")
            return False
    else:
        print(f"  ⚠️  Model checkpoint not found (will be created)")
        print(f"     Run: python ml/models/quick_finetune.py --train ml/data/train_bootstrap.parquet --val ml/data/val_bootstrap.parquet")
        return False

def test_inference():
    """Test inference on sample articles."""
    print("\n🔍 Testing inference...")
    
    try:
        from ml.models.inference import get_fine_tuned_classifier
        
        model_path = PROJECT_ROOT / "ml/models/checkpoints/best_model.pt"
        if not model_path.exists():
            print(f"  ⚠️  Model not found, skipping inference test")
            return False
        
        model = get_fine_tuned_classifier(str(model_path))
        
        test_cases = [
            ("Fire in Stockholm", "A major fire broke out in central Stockholm"),
            ("New government policy", "The government announced new climate policy"),
            ("Football championship", "Sweden wins football championship"),
        ]
        
        print(f"  Testing {len(test_cases)} articles...")
        for title, desc in test_cases:
            result = model.classify(title, desc)
            if result:
                print(f"  ✅ {title}")
            else:
                print(f"  ⚠️  {title} - no signals detected")
        
        return True
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")
        return False

def test_pipeline_integration():
    """Test integration with hopsworks_pipeline."""
    print("\n🔍 Testing pipeline integration...")
    
    try:
        from ml.ingestion.hopsworks_pipeline import classify_article
        
        result = classify_article("Fire in Stockholm", "")
        
        if result:
            print(f"  ✅ Pipeline integration working")
            print(f"     Classified article into {len(result)} signals")
            return True
        else:
            print(f"  ⚠️  No signals detected (may be using keyword classifier)")
            return True  # Not a failure
    except Exception as e:
        print(f"  ❌ Pipeline integration failed: {e}")
        return False

def main():
    """Run all checks."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║          ONE-DAY FINE-TUNING - SYSTEM CHECK                    ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    results = {
        "Environment": check_environment(),
        "Files": check_files(),
        "Data": check_data(),
        "Model": check_model(),
        "Pipeline": test_pipeline_integration(),
    }
    
    print(f"\n{'='*64}")
    print("SUMMARY")
    print(f"{'='*64}")
    
    for check, passed in results.items():
        status = "✅" if passed else "⚠️"
        print(f"{status} {check}")
    
    if all(results.values()):
        print("""
╔════════════════════════════════════════════════════════════════╗
║                  ✨ ALL CHECKS PASSED! ✨                     ║
║                                                                ║
║  You're ready to train! Run:                                  ║
║                                                                ║
║    ./train_one_day.sh                                         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
        return 0
    else:
        print("""
╔════════════════════════════════════════════════════════════════╗
║             ⚠️  SETUP INCOMPLETE                              ║
║                                                                ║
║  Fix the issues above, then run:                              ║
║                                                                ║
║    python verify_finetuning.py  (run this again)              ║
║    ./train_one_day.sh           (when ready)                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
        return 1

if __name__ == "__main__":
    sys.exit(main())
