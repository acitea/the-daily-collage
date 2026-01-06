#!/usr/bin/env python3
"""
CLI utility to generate layout + hitbox JSON for a given vibe vector.

Useful for frontend development to preview layouts before integration.

Usage:
    python backend/utils/generate_layout.py --city stockholm --vibe '{"transportation": 0.5, "weather_temp": 0.3}'
    python backend/utils/generate_layout.py --city stockholm --config sample_vibe.json
    python backend/utils/generate_layout.py --sample calm
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import logging

logging.basicConfig(level=logging.DEBUG)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.composition import HybridComposer
from storage import VibeHash


def generate_sample_vibes() -> dict:
    """Generate sample vibe vectors for different scenarios."""
    return {
        "calm": {
            "transportation": 0.1,
            "weather_temp": 0.0,
            "weather_wet": 0.0,
            "crime": 0.0,
            "festivals": 0.0,
            "sports": 0.0,
            "emergencies": 0.0,
            "economics": 0.0,
            "politics": 0.0,
        },
        "active": {
            "transportation": 0.7,
            "weather_temp": 0.2,
            "weather_wet": -0.3,
            "crime": 0.1,
            "festivals": 0.6,
            "sports": 0.5,
            "emergencies": 0.0,
            "economics": 0.3,
            "politics": 0.2,
        },
        "crisis": {
            "transportation": 0.8,
            "weather_temp": 0.9,
            "weather_wet": 0.7,
            "crime": 0.6,
            "festivals": -0.9,
            "sports": -0.5,
            "emergencies": 0.9,
            "economics": -0.4,
            "politics": 0.5,
        },
        "peaceful": {
            "transportation": 0.2,
            "weather_temp": 0.3,
            "weather_wet": -0.2,
            "crime": -0.8,
            "festivals": 0.7,
            "sports": 0.4,
            "emergencies": -0.9,
            "economics": 0.2,
            "politics": 0.1,
        },
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualization layout + hitboxes for vibe vector"
    )

    parser.add_argument(
        "--city",
        default="stockholm",
        help="City name (default: stockholm)",
    )

    parser.add_argument(
        "--vibe",
        type=str,
        help="JSON-encoded vibe vector, e.g. '{\"traffic\": 0.5}'",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON file with vibe vector",
    )

    parser.add_argument(
        "--sample",
        type=str,
        choices=["calm", "active", "crisis", "peaceful"],
        help="Use a sample vibe (calm, active, crisis, peaceful)",
    )

    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Generate layouts for all sample vibes",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file (default: stdout)",
    )

    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available sample vibes",
    )

    parser.add_argument(
        "--image-output",
        type=str,
        default="tmp/daily_collage_layout.png",
        help="Output path for the base compiled image (default: tmp/daily_collage_layout.png)",
    )

    return parser.parse_args()


def load_vibe_vector(args) -> dict:
    """Load vibe vector from arguments."""
    samples = generate_sample_vibes()

    if args.list_samples:
        print("Available sample vibes:")
        for name, vibe in samples.items():
            print(f"  {name}: {json.dumps(vibe, indent=2)}")
        sys.exit(0)

    if args.sample:
        return samples[args.sample]

    if args.config:
        try:
            with open(args.config, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)

    if args.vibe:
        try:
            return json.loads(args.vibe)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --vibe: {e}")
            sys.exit(1)

    # Default
    print("No vibe vector specified. Use --sample, --vibe, --config, or --list-samples")
    sys.exit(1)


def generate_layout(city: str, vibe_vector: dict) -> tuple:
    """Generate layout and return image data + metadata dict."""

    # Create layout composer directly (skip polish by default)
    # Determine assets directory path relative to project root
    composer = HybridComposer()

    image_data, hitboxes = composer.compose(vibe_vector, city)

    # Generate vibe hash
    timestamp = datetime.utcnow()
    vibe_hash = VibeHash.generate(city, timestamp)

    # Build output metadata
    output = {
        "metadata": {
            "city": city,
            "vibe_hash": vibe_hash,
            "timestamp": timestamp.isoformat(),
            "vibe_vector": vibe_vector,
        },
        "hitboxes": [hb.model_dump() for hb in hitboxes],
        "summary": {
            "total_hitboxes": len(hitboxes),
            "signals": list(vibe_vector.keys()),
        },
    }

    return image_data, output


def main():
    """Main entry point."""
    args = parse_args()

    # Handle generating all samples
    if args.all_samples:
        samples = generate_sample_vibes()
        print(f"Generating layouts for all {len(samples)} samples...", file=sys.stderr)
        
        for sample_name, vibe_vector in samples.items():
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Processing: {sample_name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            
            # Generate filenames based on sample name
            image_path = f"tmp/daily_collage_{sample_name}.png"
            json_path = f"tmp/daily_collage_{sample_name}.json"
            
            print(f"Vibe vector: {json.dumps(vibe_vector, indent=2)}", file=sys.stderr)
            
            try:
                image_data, output = generate_layout(args.city, vibe_vector)
            except Exception as e:
                print(f"Error generating layout for {sample_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue
            
            # Save image
            try:
                with open(image_path, "wb") as f:
                    f.write(image_data)
                print(f"✓ Image saved to {image_path}", file=sys.stderr)
            except IOError as e:
                print(f"Error saving image: {e}", file=sys.stderr)
                continue
            
            # Save JSON
            try:
                with open(json_path, "w") as f:
                    f.write(json.dumps(output, indent=2))
                print(f"✓ JSON saved to {json_path}", file=sys.stderr)
            except IOError as e:
                print(f"Error saving JSON: {e}", file=sys.stderr)
                continue
            
            print(f"  Hitboxes: {output['summary']['total_hitboxes']}", file=sys.stderr)
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"✓ Generated all {len(samples)} samples", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        return

    try:
        vibe_vector = load_vibe_vector(args)
    except Exception as e:
        print(f"Error loading vibe vector: {e}")
        sys.exit(1)

    print(f"Generating layout for {args.city}...", file=sys.stderr)
    print(f"Vibe vector: {json.dumps(vibe_vector, indent=2)}", file=sys.stderr)

    try:
        image_data, output = generate_layout(args.city, vibe_vector)
    except Exception as e:
        print(f"Error generating layout: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save the compiled image
    try:
        with open(args.image_output, "wb") as f:
            f.write(image_data)
        print(f"✓ Base image saved to {args.image_output}", file=sys.stderr)
    except IOError as e:
        print(f"Error saving image: {e}", file=sys.stderr)
        sys.exit(1)

    # Output result
    output_json = json.dumps(output, indent=2)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"✓ Layout written to {args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
    else:
        print(output_json)

    # Print summary
    print(f"\n✓ Generated layout for {args.city}", file=sys.stderr)
    print(f"  Image: {args.image_output}", file=sys.stderr)
    print(f"  Vibe hash: {output['metadata']['vibe_hash']}", file=sys.stderr)
    print(f"  Hitboxes: {output['summary']['total_hitboxes']}", file=sys.stderr)
    print(f"  Signals: {', '.join(output['summary']['signals'])}", file=sys.stderr)


if __name__ == "__main__":
    main()

# **CLI Utility for Frontend Development**
# Command-line tool to preview layouts before frontend integration.

# **Usage:**
# ```bash
# # Generate layout for sample vibe
# python backend/utils/generate_layout.py --city stockholm --sample active

# # Use custom vibe vector
# python backend/utils/generate_layout.py --city stockholm --vibe '{"traffic": 0.5, "weather_wet": 0.3}'

# # Load from JSON file
# python backend/utils/generate_layout.py --city stockholm --config my_vibe.json

# # List available samples
# python backend/utils/generate_layout.py --list-samples

# # Save to file
# python backend/utils/generate_layout.py --sample crisis --output layout.json
# ```
