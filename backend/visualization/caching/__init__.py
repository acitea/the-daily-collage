"""
Caching system for The Daily Collage visualizations.

Provides:
- VibeCache: High-level cache interface
- StorageBackend: Abstract interface for storage implementations
- Storage backends: LocalStorageBackend, MockS3StorageBackend, HopsworksStorageBackend
- VibeHash: Deterministic cache key generation
- CacheMetadata: Metadata about cached visualizations
"""

from backend.visualization.caching.cache import VibeCache
from backend.visualization.caching.core import CacheMetadata, StorageBackend, VibeHash
from backend.visualization.caching.factory import create_storage_backend
from backend.visualization.caching.storage import (
    HopsworksStorageBackend,
    LocalStorageBackend,
    MockS3StorageBackend,
)

__all__ = [
    "VibeCache",
    "StorageBackend",
    "VibeHash",
    "CacheMetadata",
    "create_storage_backend",
    "LocalStorageBackend",
    "MockS3StorageBackend",
    "HopsworksStorageBackend",
]
