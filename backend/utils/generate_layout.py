#!/usr/bin/env python3
"""
CLI utility to generate layout + hitbox JSON for a given vibe vector.

Useful for frontend development to preview layouts before integration.

Usage:
    python backend/utils/generate_layout.py --city stockholm --vibe '{"traffic": 0.5, "weather": 0.3}'
    python backend/utils/generate_layout.py --city stockholm --config sample_vibe.json
    python backend/utils/generate_layout.py --sample
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.visualization.composition import HybridComposer
from backend.storage import VibeHash


def generate_sample_vibes() -> dict:
    """Generate sample vibe vectors for different scenarios."""
    return {
        "calm": {
            "traffic": 0.1,
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
            "traffic": 0.7,
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
            "traffic": 0.8,
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
            "traffic": 0.2,
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
        "--output",
        type=str,
        help="Output JSON file (default: stdout)",
    )

    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available sample vibes",
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


def generate_layout(city: str, vibe_vector: dict) -> dict:
    """Generate layout and return as dict."""
    composer = HybridComposer()

    # Generate image and hitboxes
    image_data, hitboxes = composer.compose(vibe_vector, city)

    # Generate vibe hash
    timestamp = datetime.utcnow()
    vibe_hash = VibeHash.generate(city, timestamp)

    # Build output
    output = {
        "metadata": {
            "city": city,
            "vibe_hash": vibe_hash,
            "timestamp": timestamp.isoformat(),
            "vibe_vector": vibe_vector,
        },
        "layout": {
            "image_width": 1024,
            "image_height": 768,
        },
        "hitboxes": hitboxes,
        "summary": {
            "total_hitboxes": len(hitboxes),
            "signals": list(vibe_vector.keys()),
        },
    }

    return output


def main():
    """Main entry point."""
    args = parse_args()

    try:
        vibe_vector = load_vibe_vector(args)
    except Exception as e:
        print(f"Error loading vibe vector: {e}")
        sys.exit(1)

    print(f"Generating layout for {args.city}...", file=sys.stderr)
    print(f"Vibe vector: {json.dumps(vibe_vector, indent=2)}", file=sys.stderr)

    try:
        output = generate_layout(args.city, vibe_vector)
    except Exception as e:
        print(f"Error generating layout: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"  Vibe hash: {output['metadata']['vibe_hash']}", file=sys.stderr)
    print(f"  Hitboxes: {output['summary']['total_hitboxes']}", file=sys.stderr)
    print(f"  Signals: {', '.join(output['summary']['signals'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
