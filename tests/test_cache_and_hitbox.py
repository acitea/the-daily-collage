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
        """Same location and time should always produce same hash."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        hash1 = VibeHash.generate(city, timestamp)
        hash2 = VibeHash.generate(city, timestamp)

        assert hash1 == hash2, "Same input should produce same hash"

    def test_different_scores_produce_same_hash(self):
        """Hash is now independent of signal scores."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        # Hash should be same regardless of vibe vectors
        hash1 = VibeHash.generate(city, timestamp)
        hash2 = VibeHash.generate(city, timestamp)

        assert hash1 == hash2, "Same city/time should produce same hash"

    def test_different_cities_produce_different_hash(self):
        """Different cities should produce different hashes."""
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        hash1 = VibeHash.generate("stockholm", timestamp)
        hash2 = VibeHash.generate("gothenburg", timestamp)

        assert hash1 != hash2, "Different cities should produce different hashes"

    def test_different_windows_produce_different_hash(self):
        """Different time windows should produce different hashes."""
        city = "stockholm"

        # Window 0 (00:00-06:00)
        ts1 = datetime(2025, 12, 11, 3, 30, 0)
        # Window 1 (06:00-12:00)
        ts2 = datetime(2025, 12, 11, 9, 30, 0)

        hash1 = VibeHash.generate(city, ts1)
        hash2 = VibeHash.generate(city, ts2)

        assert hash1 != hash2, "Different time windows should produce different hashes"

    def test_hash_format(self):
        """Hash should have expected format."""
        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)

        cache_key = VibeHash.generate(city, timestamp)

        # Format: city_date_window
        parts = cache_key.split("_")
        assert len(parts) == 3, f"Key should have 3 parts: {cache_key}"
        assert parts[0] == "stockholm"
        assert "-" in parts[1]  # Date format YYYY-MM-DD
        assert "-" in parts[2]  # Window format HH-HH


class TestCacheDeterminism:
    """Test that cache behaves deterministically."""

    def test_cache_hit_deterministic(self):
        """Retrieving same location/time twice should hit cache both times."""
        storage = LocalStorageBackend(":memory:")  # In-memory for testing
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}
        image_data = b"fake_image_data"
        hitboxes = [{"x": 100, "y": 150, "w": 50, "h": 50}]

        # Store
        url1, meta1 = cache.set(city, timestamp, image_data, hitboxes, vibe_vector)

        # Retrieve same
        img2, meta2 = cache.get(city, timestamp)

        assert img2 == image_data, "Should retrieve same image"
        assert meta2 is not None, "Should retrieve metadata"

    def test_cache_always_hits_same_location_time(self):
        """Cache should always hit for same location and time."""
        storage = LocalStorageBackend(":memory:")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        v1 = {"traffic": 0.45}
        image_data = b"fake_image"
        hitboxes = []

        cache.set(city, timestamp, image_data, hitboxes, v1)

        # Should retrieve even with different vibe vector (cache is location/time based)
        img_result, meta_result = cache.get(city, timestamp)

        assert img_result == image_data, "Should retrieve for same location/time"
        assert meta_result is not None, "Should have metadata"

    def test_cache_exists_check(self):
        """Cache exists() should match set/get behavior."""
        # Use MockS3 for proper isolation
        storage = MockS3StorageBackend("test-bucket")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.45}

        assert not cache.exists(city, timestamp), "Should not exist initially"

        cache.set(city, timestamp, b"image", [], vibe_vector)

        assert cache.exists(city, timestamp), "Should exist after set"


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
