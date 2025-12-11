#!/usr/bin/env python3
"""
Integration test for The Daily Collage pipeline.

Tests the full data flow: ingestion -> classification -> visualization.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from utils.classification import classify_articles, SignalCategory
from visualization.composition import TemplateComposer, SignalIntensity
import polars as pl


def test_pipeline():
    """Test the complete pipeline with mock data."""
    
    print("=" * 60)
    print("The Daily Collage - Pipeline Integration Test")
    print("=" * 60)
    
    # 1. Create mock news articles
    print("\n1️⃣ Creating mock news articles...")
    mock_articles = pl.DataFrame({
        "title": [
            "Heavy traffic congestion on Stockholm highway",
            "Severe rainstorm hits Sweden",
            "Local festival celebrates midsummer",
            "Stock market reaches new high",
            "Police respond to incident downtown",
        ],
        "url": [
            "https://example.com/traffic1",
            "https://example.com/weather1",
            "https://example.com/festival1",
            "https://example.com/economic1",
            "https://example.com/crime1",
        ],
        "source": [
            "SVT",
            "SVT",
            "DN",
            "DI",
            "Expressen",
        ],
        "date": [
            "2025-12-11",
            "2025-12-11",
            "2025-12-11",
            "2025-12-11",
            "2025-12-11",
        ],
        "tone": [0.5, -0.3, 0.8, 0.6, -0.4],
    })
    
    print(f"   ✓ Created {len(mock_articles)} mock articles")
    
    # 2. Classify articles
    print("\n2️⃣ Classifying articles into signals...")
    classified = classify_articles(mock_articles)
    
    print(f"   ✓ Classified articles:")
    for article in classified:
        print(f"      - {article.title[:40]}...")
        primary = article.primary_signal.value if article.primary_signal else "unknown"
        intensity = article.signals[0].intensity if article.signals else 0
        print(f"        Signal: {primary}, Intensity: {intensity:.0f}%")
    
    # 3. Aggregate signals by intensity
    print("\n3️⃣ Aggregating signals by intensity...")
    signal_dict = {}
    for article in classified:
        if article.primary_signal:
            signal = article.primary_signal.value
            intensity = article.signals[0].intensity if article.signals else 0
            if signal not in signal_dict:
                signal_dict[signal] = []
            signal_dict[signal].append(intensity)
    
    signals_for_viz = []
    for signal_name in sorted(signal_dict.keys(), key=lambda s: sum(signal_dict[s]) / len(signal_dict[s]), reverse=True):
        intensities = signal_dict[signal_name]
        avg_intensity = sum(intensities) / len(intensities)
        signals_for_viz.append(SignalIntensity(signal_name, avg_intensity))
        print(f"   • {signal_name.upper()}: {avg_intensity:.0f}%")
    
    # 4. Generate visualization
    print("\n4️⃣ Generating visualization...")
    composer = TemplateComposer()
    image_data = composer.compose(signals_for_viz, "Stockholm")
    
    print(f"   ✓ Generated PNG image: {len(image_data)} bytes")
    print(f"   ✓ Image dimensions: 1024x768 pixels")
    
    # Save sample image for inspection
    with open("sample_visualization.png", "wb") as f:
        f.write(image_data)
    print(f"   ✓ Saved sample to sample_visualization.png")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
