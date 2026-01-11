"""
Caching and storage system for The Daily Collage.

Provides:
- VibeCache: High-level cache interface
- StorageBackend: Abstract interface for storage implementations
- Storage backends: LocalStorageBackend, MockS3StorageBackend, HopsworksStorageBackend
- VibeHash: Deterministic cache key generation
- CacheMetadata: Metadata about cached visualizations
"""

from .cache import VibeCache
from .factory import create_storage_backend
from .providers import (
    HopsworksStorageBackend,
    LocalStorageBackend,
    MockS3StorageBackend,
)

__all__ = [
    "VibeCache",
    "create_storage_backend",
    "LocalStorageBackend",
    "MockS3StorageBackend",
    "HopsworksStorageBackend",
]
