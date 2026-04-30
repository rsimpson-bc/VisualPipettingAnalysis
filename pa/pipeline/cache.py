"""
Layer 0 — ProcessedImageCache.

Per-step cache for expensive image operations (blur, gradient, A:B diff).
Multiple analysis modes running on the same step will reuse cached results
rather than recomputing them.

Lifetime: created at the start of _dispatch.run(), garbage-collected when
the step returns. Do NOT hold a reference to this across steps.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


# Cache key = (pipette_index, operation_name, frozenset of params)
_CacheKey = Tuple[int, str, frozenset]


class ProcessedImageCache:
    """
    Keyed by (pipette_index, operation, params).
    Thread-safe for read access; write access must be done before parallel
    per-pipette dispatch begins (i.e. during the single-threaded setup phase).
    """

    def __init__(self):
        self._store: Dict[_CacheKey, Any] = {}

    def _key(self, pipette_index: int, operation: str, params: dict) -> _CacheKey:
        return (pipette_index, operation, frozenset(params.items()))

    def get(
        self,
        pipette_index: int,
        operation: str,
        params: dict,
    ) -> Any | None:
        """Return cached result or None if not present."""
        return self._store.get(self._key(pipette_index, operation, params))

    def put(
        self,
        pipette_index: int,
        operation: str,
        params: dict,
        value: Any,
    ) -> None:
        """Store a computed result."""
        self._store[self._key(pipette_index, operation, params)] = value

    def get_or_compute(
        self,
        pipette_index: int,
        operation: str,
        params: dict,
        fn,
    ) -> Any:
        """
        Return cached value if present, otherwise call fn(), cache, and return.

        Usage:
            result = cache.get_or_compute(
                pipette_index=1,
                operation="blur_gaussian",
                params={"kernel_size": 5},
                fn=lambda: blur_gaussian(img, kernel_size=5),
            )
        """
        key = self._key(pipette_index, operation, params)
        if key not in self._store:
            self._store[key] = fn()
        return self._store[key]

    def clear(self) -> None:
        self._store.clear()
