from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.abc_geometry import (
    extract_bbox_dimensions,
    extract_circle_radii,
    holes_with_depths,
    load_yaml,
)


_ABC_META_CACHE: dict[tuple[str, str, str, float], tuple[float, dict[str, Any]]] = {}


def get_part_meta(
    part_number: str,
    *,
    stats_root: str,
    features_root: str,
    scale_to_mm: float = 1.0,
) -> dict[str, Any]:
    """
    Load part dimensions (bbox) + hole info from ABC dataset yml files.
    Returns dict with: length_mm, width_mm, height_mm, holes (list[dict]).

    If files are missing/unreadable, returns {}.
    """
    part_number = str(part_number)
    stats_dir = Path(stats_root) / part_number
    feat_dir = Path(features_root) / part_number
    if not stats_dir.is_dir() or not feat_dir.is_dir():
        return {}

    stats_files = sorted(stats_dir.glob(f"{part_number}_*_stats_*.yml"))
    if not stats_files:
        return {}
    sf = stats_files[0]
    suffix = sf.stem.split("_stats_")[-1]

    ff_candidates = sorted(feat_dir.glob(f"{part_number}_*_features_{suffix}.yml"))
    if not ff_candidates:
        ff_candidates = sorted(feat_dir.glob(f"{part_number}_*_features_*.yml"))
    if not ff_candidates:
        return {}
    ff = ff_candidates[0]

    cache_key = (str(sf), str(ff), part_number, float(scale_to_mm))
    mtime = max(sf.stat().st_mtime, ff.stat().st_mtime)
    cached = _ABC_META_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        stats = load_yaml(sf)
        feats = load_yaml(ff)
        dims = extract_bbox_dimensions(stats, scale_to_mm=scale_to_mm)
        circles = extract_circle_radii(feats, scale_to_mm=scale_to_mm)
        holes = holes_with_depths(circles, stats.get("bbox"))

        out: dict[str, Any] = {
            "length_mm": dims.get("length_x_mm"),
            "width_mm": dims.get("length_y_mm"),
            "height_mm": dims.get("length_z_mm"),
            "holes": holes,
            "stats_path": str(sf),
            "features_path": str(ff),
        }
    except Exception:
        out = {}

    _ABC_META_CACHE[cache_key] = (mtime, out)
    return out

