"""
Schema loader for pipeline parameter schemas.

Resolves JSON Schema files relative to the ``schemas/`` directory in the
workspace root (two levels above this file: pa_gui/params/ → pa_gui/ → root/).
Uses the ``param_schema_index.json`` index to map mode names and integrator
types to their schema files.

Handles ``allOf`` composition (used by LineContinuity modes that extend
RidgeExtractor.schema.json) by merging the referenced schema's properties
into the resolved schema before returning it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# Workspace root = two directories above this file (…/pa_gui/params/schema_loader.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMAS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "schemas"))
_INDEX_PATH = os.path.join(_SCHEMAS_DIR, "param_schema_index.json")

# Cache so schemas are only read from disk once per session.
_schema_cache: Dict[str, Dict[str, Any]] = {}
_index_cache: Optional[Dict[str, Any]] = None


def _load_index() -> Dict[str, Any]:
    global _index_cache
    if _index_cache is None:
        with open(_INDEX_PATH, encoding="utf-8") as f:
            _index_cache = json.load(f)
    return _index_cache


def _load_schema_file(rel_path: str) -> Dict[str, Any]:
    """Load a schema file by path relative to schemas/."""
    abs_path = os.path.join(_SCHEMAS_DIR, rel_path)
    if abs_path not in _schema_cache:
        with open(abs_path, encoding="utf-8") as f:
            _schema_cache[abs_path] = json.load(f)
    return dict(_schema_cache[abs_path])  # shallow copy so callers can modify


def _resolve_allof(schema: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    Flatten one level of allOf composition.

    If the schema contains ``allOf: [{ "$ref": "SomeFile.schema.json" }]``,
    load the referenced schema's properties and merge them *under* the current
    schema's properties (current schema wins on key collision).
    """
    if "allOf" not in schema:
        return schema

    merged_props: Dict[str, Any] = {}
    for item in schema["allOf"]:
        ref = item.get("$ref", "")
        if ref:
            ref_abs = os.path.join(base_dir, ref)
            # Normalise to a path relative to _SCHEMAS_DIR for the cache key
            rel = os.path.relpath(ref_abs, _SCHEMAS_DIR)
            ref_schema = _load_schema_file(rel)
            merged_props.update(ref_schema.get("properties", {}))

    # Current schema's own properties override referenced ones
    merged_props.update(schema.get("properties", {}))
    result = dict(schema)
    result["properties"] = merged_props
    result.pop("allOf", None)
    return result


def load_mode_schema(mode_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the resolved JSON Schema dict for the named pipeline mode, or None
    if no schema is registered for that mode.
    """
    index = _load_index()
    rel = index.get("modes", {}).get(mode_name)
    if not rel:
        return None
    schema = _load_schema_file(rel)
    base_dir = os.path.dirname(os.path.join(_SCHEMAS_DIR, rel))
    return _resolve_allof(schema, base_dir)


def load_integrator_schema(analysis_type: str) -> Optional[Dict[str, Any]]:
    """
    Return the resolved JSON Schema dict for the integrator of the given
    analysis_type ("LiquidMeasurement" or "TipIdentification"), or None.
    """
    index = _load_index()
    rel = index.get("integrators", {}).get(analysis_type)
    if not rel:
        return None
    schema = _load_schema_file(rel)
    base_dir = os.path.dirname(os.path.join(_SCHEMAS_DIR, rel))
    return _resolve_allof(schema, base_dir)


def list_mode_names() -> list[str]:
    """Return the list of all registered mode names in the index."""
    return list(_load_index().get("modes", {}).keys())


def invalidate_cache() -> None:
    """Clear the in-memory schema cache (useful after editing schema files on disk)."""
    global _schema_cache, _index_cache
    _schema_cache.clear()
    _index_cache = None
