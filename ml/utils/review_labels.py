"""
Interactive label review tool.

Allows manual review and correction of auto-generated labels,
focusing on uncertain cases.

Usage:
    python ml/utils/review_labels.py --input data/train.parquet
"""

import argparse
import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB


def review_labels(
    input_path: str,
    output_path: str = None,
    focus_uncertain: bool = True,
    uncertainty_threshold: float = 0.5
):
    """
    Interactive label review interface.
    
    Args:
        input_path: Path to labeled parquet file
        output_path: Path to save corrected labels (default: input_path with '_reviewed' suffix)
        focus_uncertain: If True, prioritize uncertain labels
        uncertainty_threshold: Score threshold for "uncertain" (default: 0.5)
    """
    df = pl.read_parquet(input_path)
    
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_reviewed{input_path_obj.suffix}")
    
    # Identify uncertain cases
    uncertain_indices = []
    
    if focus_uncertain:
        print("Identifying uncertain cases...")
        for idx, row in enumerate(df.iter_rows(named=True)):
            # Calculate uncertainty (scores near 0.4-0.6 are uncertain)
            max_score = 0
            for category in SIGNAL_CATEGORIES:
                score = abs(row[f"{category}_score"])
                max_score = max(max_score, score)
            
            # Uncertain if max score is near threshold
            if 0.3 < max_score < uncertainty_threshold:
                uncertain_indices.append(idx)
        
        print(f"Found {len(uncertain_indices)} uncertain cases")
        print("Starting with uncertain cases first...\n")
        
        # Prioritize uncertain cases
        review_order = uncertain_indices + [i for i in range(len(df)) if i not in uncertain_indices]
    else:
        review_order = list(range(len(df)))
    
    corrections = {}
    reviewed_count = 0
    
    for idx in review_order:
        row = df.row(idx, named=True)
        
        print(f"\n{'='*100}")
        print(f"Article {reviewed_count + 1}/{len(df)} (Index: {idx})")
        if idx in uncertain_indices:
            print("⚠️  UNCERTAIN CASE")
        print(f"{'='*100}")
        
        print(f"\n📰 Title: {row['title']}")
        desc = row['description']
        if len(desc) > 300:
            desc = desc[:300] + "..."
        print(f"📝 Description: {desc}")
        print(f"🔗 Source: {row['source']}")
        
        # Show detected signals (both positive and negative impact)
        detected_signals = []
        for category in SIGNAL_CATEGORIES:
            score = row[f"{category}_score"]
            tag = row[f"{category}_tag"]
            if abs(score) > 0.01:  # Count both positive and negative scores
                detected_signals.append((category, score, tag))
        
        print(f"\n🎯 Detected signals:")
        if detected_signals:
            for category, score, tag in detected_signals:
                impact = "📈 positive" if score > 0 else "📉 negative"
                print(f"   • {category:20s}: {score:+6.3f} ({impact}, tag: '{tag}')")
        else:
            print("   (no signals detected)")
        
        # Ask for action
        print("\n" + "-"*100)
        print("Actions:")
        print("  [Enter]  = Keep as-is and continue")
        print("  [u]      = Update signal (e.g., 'emergencies 0.85 fire')")
        print("  [d]      = Delete signal (e.g., 'crime')")
        print("  [s]      = Skip this article")
        print("  [q]      = Quit and save all corrections")
        print("  [?]      = Show available tags")
        print("-"*100)
        
        action = input("\n> ").strip().lower()
        
        if action == 'q':
            print("\n💾 Quitting and saving corrections...")
            break
        elif action == 's' or action == '':
            reviewed_count += 1
            continue
        elif action == '?':
            print("\n📋 Available tags per category:")
            for category in SIGNAL_CATEGORIES:
                tags = ", ".join([f"'{t}'" for t in TAG_VOCAB[category] if t])
                print(f"   {category:20s}: {tags}")
            input("\nPress Enter to continue...")
            continue
        elif action == 'd':
            # Delete a signal
            delete_input = input("Category to delete: ").strip().lower()
            if delete_input in SIGNAL_CATEGORIES:
                if idx not in corrections:
                    corrections[idx] = {}
                corrections[idx][delete_input] = (0.0, "")
                print(f"✓ Will delete {delete_input} signal")
            else:
                print(f"❌ Invalid category: {delete_input}")
        elif action == 'u':
            # Update a signal
            update_input = input("Update (format: 'category score tag'): ").strip()
            parts = update_input.split()
            
            if len(parts) >= 2:
                category = parts[0].lower()
                
                if category not in SIGNAL_CATEGORIES:
                    print(f"❌ Invalid category: {category}")
                    continue
                
                try:
                    score = float(parts[1])
                    tag = parts[2] if len(parts) > 2 else ""
                    
                    # Validate score range
                    if score < -1.0 or score > 1.0:
                        print(f"❌ Score must be between -1.0 and 1.0")
                        continue
                    
                    # Validate tag
                    if tag and tag not in TAG_VOCAB[category]:
                        print(f"⚠️  Warning: '{tag}' not in standard tags for {category}")
                        print(f"    Available: {TAG_VOCAB[category]}")
                        confirm = input("    Continue anyway? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    if idx not in corrections:
                        corrections[idx] = {}
                    corrections[idx][category] = (score, tag)
                    print(f"✓ Will update {category} = {score} ('{tag}')")
                    
                except ValueError:
                    print(f"❌ Invalid score: {parts[1]}")
            else:
                print("❌ Invalid format. Use: 'category score tag'")
        
        reviewed_count += 1
    
    # Apply corrections
    if corrections:
        print(f"\n📝 Applying {len(corrections)} corrections...")
        
        # Convert to list of dicts for polars
        rows = df.to_dicts()
        
        for idx, changes in corrections.items():
            for category, (score, tag) in changes.items():
                rows[idx][f"{category}_score"] = score
                rows[idx][f"{category}_tag"] = tag
        
        df_corrected = pl.DataFrame(rows)
        df_corrected.write_parquet(output_path)
        print(f"✓ Saved corrected labels to: {output_path}")
    else:
        print("\n⚠️  No corrections made")
    
    print(f"\n✓ Reviewed {reviewed_count} articles")
    
    return corrections


def main():
    parser = argparse.ArgumentParser(
        description="Review and correct auto-generated labels"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file with labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for corrected labels (default: input_reviewed.parquet)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all articles (not just uncertain ones)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Uncertainty threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    print("="*100)
    print("LABEL REVIEW TOOL")
    print("="*100)
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output or 'auto'}")
    print(f"Mode:       {'All articles' if args.all else 'Focus on uncertain cases'}")
    if not args.all:
        print(f"Threshold:  {args.threshold}")
    print("="*100 + "\n")
    
    review_labels(
        input_path=args.input,
        output_path=args.output,
        focus_uncertain=not args.all,
        uncertainty_threshold=args.threshold
    )
    
    print("\n" + "="*100)
    print("NEXT STEP: Train the model with reviewed labels")
    print("="*100)
    print(f"\npython ml/models/quick_finetune.py \\")
    print(f"  --train {args.output or args.input} \\")
    print(f"  --val data/val.parquet \\")
    print(f"  --epochs 3")
    print()


if __name__ == "__main__":
    main()
