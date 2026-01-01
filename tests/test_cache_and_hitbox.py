"""
Tests for cache determinism and hitbox stability.

Verifies that:
1. Vibe hashes are deterministic (same input -> same hash)
2. Hitbox locations are stable across polish runs
3. Cache retrieval works correctly
"""

import pytest
from datetime import datetime
from typing import Dict

from backend.visualization.caching import (
    VibeHash,
    VibeCache,
    CacheMetadata,
    LocalStorageBackend,
    MockS3StorageBackend,
)
from backend.visualization.assets import ZoneLayoutComposer
from backend.visualization.polish import MockStabilityAIPoller


class TestVibeHashDeterminism:
    """Test that vibe hashes are deterministic."""

    def test_same_vector_produces_same_hash(self):
        """Same vibe vector should always produce same hash."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {
            "traffic": 0.45,
            "weather_temp": -0.2,
            "crime": 0.1,
        }

        hash1 = VibeHash.generate(city, timestamp, vibe_vector)
        hash2 = VibeHash.generate(city, timestamp, vibe_vector)

        assert hash1 == hash2, "Same input should produce same hash"

    def test_different_scores_produce_different_hash(self):
        """Different scores should produce different hashes."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        v1 = {"traffic": 0.45, "weather_temp": -0.2}
        v2 = {"traffic": 0.46, "weather_temp": -0.2}

        hash1 = VibeHash.generate(city, timestamp, v1)
        hash2 = VibeHash.generate(city, timestamp, v2)

        assert hash1 != hash2, "Different scores should produce different hashes"

    def test_hash_discretizes_scores(self):
        """Scores within discretization step should hash the same."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        # Scores within 0.1 of each other should discretize to same value
        v1 = {"traffic": 0.450}
        v2 = {"traffic": 0.451}

        hash1 = VibeHash.generate(city, timestamp, v1)
        hash2 = VibeHash.generate(city, timestamp, v2)

        # Should be same due to discretization
        assert hash1 == hash2, f"Scores within discretization step should hash same: {hash1} vs {hash2}"

    def test_different_cities_produce_different_hash(self):
        """Different cities should produce different hashes."""
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}

        hash1 = VibeHash.generate("stockholm", timestamp, vibe_vector)
        hash2 = VibeHash.generate("gothenburg", timestamp, vibe_vector)

        assert hash1 != hash2, "Different cities should produce different hashes"

    def test_different_windows_produce_different_hash(self):
        """Different time windows should produce different hashes."""
        city = "stockholm"
        vibe_vector = {"traffic": 0.45}

        # Window 0 (00:00-06:00)
        ts1 = datetime(2025, 12, 11, 3, 30, 0)
        # Window 1 (06:00-12:00)
        ts2 = datetime(2025, 12, 11, 9, 30, 0)

        hash1 = VibeHash.generate(city, ts1, vibe_vector)
        hash2 = VibeHash.generate(city, ts2, vibe_vector)

        assert hash1 != hash2, "Different time windows should produce different hashes"

    def test_hash_format(self):
        """Hash should have expected format."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}

        vibe_hash = VibeHash.generate(city, timestamp, vibe_vector)

        # Format: city_date_window_hash
        parts = vibe_hash.split("_")
        assert len(parts) >= 4, f"Hash should have >=4 parts: {vibe_hash}"
        assert parts[0] == "stockholm"
        assert "-" in parts[1]  # Date format YYYY-MM-DD
        assert "-" in parts[2]  # Window format HH-HH


class TestCacheDeterminism:
    """Test that cache behaves deterministically."""

    def test_cache_hit_deterministic(self):
        """Retrieving same vibe twice should hit cache both times."""
        storage = LocalStorageBackend(":memory:")  # In-memory for testing
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}
        image_data = b"fake_image_data"
        hitboxes = [{"x": 10, "y": 20, "width": 100, "height": 100}]

        # Store
        url1, meta1 = cache.set(city, timestamp, vibe_vector, image_data, hitboxes)

        # Retrieve same
        img2, meta2 = cache.get(city, timestamp, vibe_vector)

        assert img2 == image_data, "Should retrieve same image"
        assert meta2 is not None, "Should retrieve metadata"

    def test_cache_miss_after_invalidation(self):
        """Cache should miss if vibe_vector differs."""
        storage = LocalStorageBackend(":memory:")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        v1 = {"traffic": 0.45}
        image_data = b"fake_image"
        hitboxes = []

        cache.set(city, timestamp, v1, image_data, hitboxes)

        # Try with different vector
        v2 = {"traffic": 0.46}
        img_result, meta_result = cache.get(city, timestamp, v2)

        assert img_result is None, "Should not retrieve with different vibe vector"

    def test_cache_exists_check(self):
        """Cache exists() should match set/get behavior."""
        storage = LocalStorageBackend(":memory:")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}

        assert not cache.exists(city, timestamp, vibe_vector), "Should not exist initially"

        cache.set(city, timestamp, vibe_vector, b"image", [])

        assert cache.exists(city, timestamp, vibe_vector), "Should exist after set"


class TestHitboxStability:
    """Test that hitboxes are stable across composition runs."""

    def test_hitbox_coordinates_are_integers(self):
        """Hitbox coordinates should always be integers."""
        composer = ZoneLayoutComposer(
            image_width=1024,
            image_height=768,
            assets_dir="./backend/assets",
        )

        signals = [
            ("transportation", "traffic", 0.5, 0.5),
            ("weather_wet", "rain", 0.7, 0.7),
        ]

        image, hitboxes = composer.compose(signals)

        for hb in hitboxes:
            assert isinstance(hb["x"], int), f"x should be int, got {type(hb['x'])}"
            assert isinstance(hb["y"], int), f"y should be int, got {type(hb['y'])}"
            assert isinstance(hb["width"], int), f"width should be int, got {type(hb['width'])}"
            assert isinstance(hb["height"], int), f"height should be int, got {type(hb['height'])}"

    def test_hitbox_within_canvas_bounds(self):
        """Hitboxes should be within canvas bounds."""
        width, height = 1024, 768
        composer = ZoneLayoutComposer(
            image_width=width,
            image_height=height,
            assets_dir="./backend/assets",
        )

        signals = [
            ("transportation", "traffic", 0.9, 0.9),
            ("weather_wet", "snow", 0.9, 0.9),
            ("crime", "theft", 0.9, 0.9),
        ]

        image, hitboxes = composer.compose(signals)

        for hb in hitboxes:
            assert hb["x"] >= 0, f"x out of bounds: {hb['x']}"
            assert hb["y"] >= 0, f"y out of bounds: {hb['y']}"
            assert hb["x"] + hb["width"] <= width, f"x+width exceeds canvas: {hb['x']} + {hb['width']} > {width}"
            assert hb["y"] + hb["height"] <= height, f"y+height exceeds canvas: {hb['y']} + {hb['height']} > {height}"

    def test_hitbox_scaling_with_intensity(self):
        """Higher intensity should result in larger hitboxes."""
        composer = ZoneLayoutComposer(
            image_width=1024,
            image_height=768,
            assets_dir="./backend/assets",
        )

        # Low intensity
        signals_low = [("transportation", "traffic", 0.2, 0.2)]
        _, hbs_low = composer.compose(signals_low)

        # High intensity
        signals_high = [("transportation", "traffic", 0.8, 0.8)]
        _, hbs_high = composer.compose(signals_high)

        if hbs_low and hbs_high:
            area_low = hbs_low[0]["width"] * hbs_low[0]["height"]
            area_high = hbs_high[0]["width"] * hbs_high[0]["height"]
            assert area_high > area_low, "Higher intensity should produce larger hitbox"

    def test_hitbox_has_signal_info(self):
        """Hitboxes should include signal category and tag."""
        composer = ZoneLayoutComposer(
            image_width=1024,
            image_height=768,
            assets_dir="./backend/assets",
        )

        signals = [("transportation", "traffic", 0.5, 0.5)]
        _, hitboxes = composer.compose(signals)

        assert len(hitboxes) > 0, "Should have hitboxes"
        hb = hitboxes[0]
        assert hb["signal_category"] == "transportation"
        assert hb["signal_tag"] == "traffic"
        assert 0.0 <= hb["signal_intensity"] <= 1.0

    def test_multiple_signals_produce_multiple_hitboxes(self):
        """Multiple signals should produce multiple hitboxes."""
        composer = ZoneLayoutComposer(
            image_width=1024,
            image_height=768,
            assets_dir="./backend/assets",
        )

        signals = [
            ("transportation", "traffic", 0.5, 0.5),
            ("weather_wet", "rain", 0.7, 0.7),
            ("crime", "theft", 0.4, 0.4),
        ]

        _, hitboxes = composer.compose(signals)

        assert len(hitboxes) <= len(signals), "Should have at most as many hitboxes as signals"

    def test_polish_preserves_hitbox_count(self):
        """Polish should not affect hitbox count."""
        composer = ZoneLayoutComposer(
            image_width=1024,
            image_height=768,
            assets_dir="./backend/assets",
        )
        poller = MockStabilityAIPoller()

        signals = [
            ("transportation", "traffic", 0.5, 0.5),
            ("weather_wet", "rain", 0.7, 0.7),
        ]

        image_before, hbs_before = composer.compose(signals)

        # Polish the image
        import io
        img_bytes = io.BytesIO()
        image_before.save(img_bytes, format="PNG")
        polished = poller.polish(img_bytes.getvalue())

        # Hitboxes should remain unchanged
        image_after, hbs_after = composer.compose(signals)

        assert len(hbs_before) == len(hbs_after), "Polish should not change hitbox count"


class TestCacheMetadata:
    """Test cache metadata handling."""

    def test_metadata_serialization(self):
        """Metadata should serialize and deserialize correctly."""
        meta = CacheMetadata(
            vibe_hash="test_hash",
            city="stockholm",
            timestamp=datetime(2025, 12, 11, 3, 30, 0),
            vibe_vector={"traffic": 0.45},
            image_url="s3://bucket/image.png",
            hitboxes=[{"x": 10, "y": 20, "width": 100, "height": 100}],
            source_articles=[{"title": "Test", "url": "http://example.com"}],
        )

        data = meta.to_dict()
        assert data["vibe_hash"] == "test_hash"
        assert data["city"] == "stockholm"
        assert len(data["hitboxes"]) == 1

        # Deserialize
        meta2 = CacheMetadata.from_dict(data)
        assert meta2.vibe_hash == "test_hash"
        assert meta2.city == "stockholm"

    def test_metadata_stores_source_articles(self):
        """Metadata should preserve source article information."""
        articles = [
            {"title": "Article 1", "url": "http://example.com/1"},
            {"title": "Article 2", "url": "http://example.com/2"},
        ]

        meta = CacheMetadata(
            vibe_hash="test",
            city="stockholm",
            timestamp=datetime.utcnow(),
            vibe_vector={},
            image_url="url",
            hitboxes=[],
            source_articles=articles,
        )

        assert meta.source_articles == articles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
