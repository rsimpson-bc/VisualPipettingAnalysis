"""
Layer 0 — Multiprocessing Pool Manager.

A single warm Pool is created at PA startup (before the server accepts requests)
and reused for all per-pipette parallel dispatch.

Why warm it early:
  The first Pool() call spawns N worker processes and imports all modules in each.
  This costs ~1 second. Paying it at startup (not at the first method step) keeps
  step latency predictable.

Usage:
    # In pa/__main__.py at startup:
    pool_manager.initialize(n_workers=8)

    # In analysis dispatch:
    pool = pool_manager.get()
    results = pool.map(analyze_one_pipette, image_sets)
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Optional


_pool: Optional[Pool] = None


def initialize(n_workers: Optional[int] = None) -> None:
    """
    Spawn worker processes and warm them up.
    Call once at PA process startup, before accepting HTTP requests.
    """
    global _pool
    if _pool is not None:
        return  # already initialized
    workers = n_workers or cpu_count()
    _pool = Pool(processes=workers)
    # Warm the workers: submit a trivial job to force module imports in each process.
    _pool.map(_noop, range(workers))
    print(f"[PA] Worker pool ready: {workers} processes.")


def get() -> Pool:
    """Return the warm pool. Raises if initialize() was not called."""
    if _pool is None:
        raise RuntimeError("pool_manager.initialize() has not been called.")
    return _pool


def shutdown() -> None:
    """Gracefully terminate the pool. Called at process exit."""
    global _pool
    if _pool is not None:
        _pool.terminate()
        _pool.join()
        _pool = None


def _noop(_):
    """Trivial worker function used to warm up worker processes."""
    return True
