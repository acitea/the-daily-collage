"""
Integration tests for the complete visualization pipeline.

Tests end-to-end functionality including layout composition,
polishing, caching, and API contracts.
"""

import pytest
import json
from datetime import datetime
from typing import Dict

from backend.visualization.composition import HybridComposer, VisualizationService
from backend.storage import VibeCache, LocalStorageBackend
from backend.settings import settings


class TestHybridComposerPipeline:
    """Test the complete hybrid composition pipeline."""

    def test_compose_generates_image_bytes(self):
        """Compose should return valid PNG bytes."""
        composer = HybridComposer()

        vibe_vector = {
            "traffic": 0.45,
            "weather_temp": -0.2,
            "weather_wet": 0.3,
        }

        image_data, hitboxes = composer.compose(vibe_vector, "stockholm")

        assert isinstance(image_data, bytes), "Should return bytes"
        assert len(image_data) > 0, "Image should not be empty"
        assert image_data[:4] == b"\x89PNG", "Should be valid PNG"

    def test_compose_returns_hitboxes(self):
        """Compose should return hitbox metadata."""
        composer = HybridComposer()

        vibe_vector = {
            "transportation": 0.5,
            "weather_wet": 0.7,
        }

        image_data, hitboxes = composer.compose(vibe_vector, "gothenburg")

        assert isinstance(hitboxes, list), "Should return list of hitboxes"
        for hb in hitboxes:
            assert "x" in hb
            assert "y" in hb
            assert "width" in hb
            assert "height" in hb
            assert "signal_category" in hb
            assert "signal_tag" in hb

    def test_compose_with_extreme_scores(self):
        """Should handle extreme score values."""
        composer = HybridComposer()

        vibe_vector = {
            "traffic": 1.0,  # Maximum
            "crime": -1.0,   # Minimum
            "festivals": 0.0,  # Neutral
        }

        image_data, hitboxes = composer.compose(vibe_vector, "malmo")

        assert len(image_data) > 0
        assert isinstance(hitboxes, list)

    def test_compose_with_empty_vibe_vector(self):
        """Should handle empty vibe vector."""
        composer = HybridComposer()

        image_data, hitboxes = composer.compose({}, "stockholm")

        # Should still produce a valid image (even if blank)
        assert len(image_data) > 0
        assert image_data[:4] == b"\x89PNG"


class TestVisualizationServiceIntegration:
    """Test the full VisualizationService."""

    def test_generate_or_get_new(self):
        """Should generate new visualization if not cached."""
        service = VisualizationService()

        vibe_vector = {
            "traffic": 0.45,
            "weather_wet": 0.3,
        }

        image_data, metadata = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        assert image_data is not None
        assert len(image_data) > 0
        assert not metadata["cached"], "First call should not be cached"
        assert metadata["vibe_hash"] is not None
        assert metadata["image_url"] is not None
        assert "hitboxes" in metadata
        assert metadata["vibe_vector"] == vibe_vector

    def test_generate_or_get_cache_hit(self):
        """Should hit cache on second call with same vibe."""
        service = VisualizationService()

        vibe_vector = {
            "traffic": 0.45,
            "weather_wet": 0.3,
        }

        # First call
        image_data_1, metadata_1 = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        # Second call
        image_data_2, metadata_2 = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        assert image_data_1 == image_data_2, "Should return same image"
        assert metadata_2["cached"], "Second call should be cached"
        assert metadata_1["vibe_hash"] == metadata_2["vibe_hash"]

    def test_generate_or_get_force_regenerate(self):
        """Force regenerate should skip cache."""
        service = VisualizationService()

        vibe_vector = {
            "traffic": 0.45,
        }

        # First call
        image_data_1, metadata_1 = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        # Second call with force regenerate
        image_data_2, metadata_2 = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
            force_regenerate=True,
        )

        assert not metadata_2["cached"], "Force regenerate should not use cache"

    def test_generate_or_get_with_source_articles(self):
        """Should preserve source articles in metadata."""
        service = VisualizationService()

        articles = [
            {"title": "Article 1", "url": "http://example.com/1"},
        ]

        vibe_vector = {"traffic": 0.5}

        _, metadata = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
            source_articles=articles,
        )

        # Retrieve from storage to verify persistence
        vibe_hash = metadata["vibe_hash"]
        storage_meta = service.cache.storage.get_metadata(vibe_hash)

        assert storage_meta is not None
        assert len(storage_meta.source_articles) > 0

    def test_different_cities_different_hashes(self):
        """Different cities should produce different hashes."""
        service = VisualizationService()

        vibe_vector = {"traffic": 0.5}

        _, meta_stockholm = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        _, meta_gothenburg = service.generate_or_get(
            city="gothenburg",
            vibe_vector=vibe_vector,
        )

        assert meta_stockholm["vibe_hash"] != meta_gothenburg["vibe_hash"]


class TestAPIContractCompliance:
    """Test that responses comply with API contracts."""

    def test_vibe_vector_scoring_range(self):
        """Vibe vector scores should be -1.0 to 1.0."""
        service = VisualizationService()

        vibe_vector = {
            "traffic": 0.45,
            "crime": -0.8,
            "festivals": 0.0,
        }

        _, metadata = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        for cat, score in metadata["vibe_vector"].items():
            assert -1.0 <= score <= 1.0, f"{cat} score {score} out of range"

    def test_hitbox_coordinates_non_negative(self):
        """Hitbox coordinates should be non-negative."""
        service = VisualizationService()

        vibe_vector = {"traffic": 0.5, "weather_wet": 0.7}

        _, metadata = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        for hb in metadata["hitboxes"]:
            assert hb["x"] >= 0
            assert hb["y"] >= 0
            assert hb["width"] > 0
            assert hb["height"] > 0

    def test_vibe_hash_format_consistency(self):
        """Vibe hash should follow consistent format."""
        service = VisualizationService()

        vibe_vector = {"traffic": 0.5}

        _, metadata = service.generate_or_get(
            city="stockholm",
            vibe_vector=vibe_vector,
        )

        vibe_hash = metadata["vibe_hash"]

        # Format: city_date_window_hash
        parts = vibe_hash.split("_")
        assert len(parts) >= 4, f"Invalid vibe hash format: {vibe_hash}"
        assert parts[0].lower() == "stockholm"
        assert len(parts[1]) == 10  # YYYY-MM-DD
        assert "-" in parts[2]  # HH-HH window


class TestCacheStorageConsistency:
    """Test that cache storage is consistent."""

    def test_metadata_persists_after_retrieve(self):
        """Metadata should be retrievable after storage."""
        storage = LocalStorageBackend(":memory:")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.5}
        image_data = b"test_image_data"
        hitboxes = [{"x": 10, "y": 20, "width": 50, "height": 50}]
        articles = [{"title": "Test", "url": "http://example.com"}]

        url, meta = cache.set(
            city=city,
            timestamp=timestamp,
            image_data=image_data,
            hitboxes=hitboxes,
            vibe_vector=vibe_vector,
            source_articles=articles,
        )

        # Retrieve
        retrieved_meta = storage.get_metadata(meta.vibe_hash)

        assert retrieved_meta is not None
        assert retrieved_meta.city == city
        assert len(retrieved_meta.hitboxes) == 1
        assert len(retrieved_meta.source_articles) == 1

    def test_image_persists_after_retrieve(self):
        """Image data should be retrievable after storage."""
        storage = LocalStorageBackend(":memory:")
        cache = VibeCache(storage)

        city = "stockholm"
        timestamp = datetime(2025, 12, 11, 3, 30, 0)
        vibe_vector = {"traffic": 0.5}
        image_data = b"test_image_data_xyz"
        hitboxes = []

        url, meta = cache.set(
            city=city,
            timestamp=timestamp,
            image_data=image_data,
            hitboxes=hitboxes,
            vibe_vector=vibe_vector,
        )

        # Retrieve
        retrieved_image = storage.get_image(meta.vibe_hash)

        assert retrieved_image == image_data


class TestErrorHandling:
    """Test error handling in visualization pipeline."""

    def test_missing_asset_fallback(self):
        """Should handle missing assets gracefully."""
        composer = HybridComposer()

        # Use signals that might not have exact asset matches
        vibe_vector = {
            "transportation": 0.5,
            "weather_wet": 0.7,
        }

        # Should not raise exception
        image_data, hitboxes = composer.compose(vibe_vector, "stockholm")

        assert len(image_data) > 0

    def test_zero_vibe_vector_handling(self):
        """Should handle all-zero vibe vector."""
        composer = HybridComposer()

        vibe_vector = {
            "traffic": 0.0,
            "weather_wet": 0.0,
            "crime": 0.0,
        }

        image_data, hitboxes = composer.compose(vibe_vector, "stockholm")

        assert len(image_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
