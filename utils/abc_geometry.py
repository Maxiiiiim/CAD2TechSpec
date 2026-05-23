from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

# Tolerances (mm)
RADIUS_TOLERANCE_MM = 0.01
CENTER_TOLERANCE_MM = 0.5


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_bbox_dimensions(stats: dict[str, Any], *, scale_to_mm: float = 1.0) -> dict[str, float]:
    """
    Extract overall dimensions from `bbox` in stats.
    bbox: [x_min, y_min, z_min, x_max, y_max, z_max, size_x, size_y, size_z]
    """
    bbox = stats.get("bbox")
    if not bbox or len(bbox) < 9:
        return {}
    size_x, size_y, size_z = bbox[6], bbox[7], bbox[8]
    lx = round(size_x * scale_to_mm, 4)
    ly = round(size_y * scale_to_mm, 4)
    lz = round(size_z * scale_to_mm, 4)
    thickness_mm = min(lx, ly, lz)
    thickness_axis = ("x", "y", "z")[(lx, ly, lz).index(thickness_mm)]
    return {
        "length_x_mm": lx,
        "length_y_mm": ly,
        "length_z_mm": lz,
        "thickness_mm": thickness_mm,
        "thickness_axis": thickness_axis,
        "diagonal_mm": round(float(stats.get("diag", 0)) * scale_to_mm, 4),
    }


def _normalize(v: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in v))
    if s < 1e-12:
        return [0.0, 0.0, 0.0]
    return [x / s for x in v]


def _axis_key(axis: list[float], tol: float = 1e-6) -> tuple[float, float, float]:
    a = _normalize(axis)
    if a[2] < -tol or (
        abs(a[2]) <= tol
        and (a[1] < -tol or (abs(a[1]) <= tol and a[0] < -tol))
    ):
        a = [-x for x in a]
    return (
        round(a[0] / tol) * tol,
        round(a[1] / tol) * tol,
        round(a[2] / tol) * tol,
    )


def _bbox_extent_along_axis(bbox: list[float], center: list[float], axis: list[float]) -> float | None:
    if len(bbox) < 6:
        return None
    x0, y0, z0 = bbox[0], bbox[1], bbox[2]
    x1, y1, z1 = bbox[3], bbox[4], bbox[5]
    cx, cy, cz = center[0], center[1], center[2]
    ax, ay, az = _normalize(axis)

    t_min, t_max = float("-inf"), float("inf")
    if abs(ax) > 1e-12:
        t_min = max(t_min, (x0 - cx) / ax)
        t_max = min(t_max, (x1 - cx) / ax)
        if ax < 0:
            t_min, t_max = t_max, t_min
    if abs(ay) > 1e-12:
        t_min = max(t_min, (y0 - cy) / ay)
        t_max = min(t_max, (y1 - cy) / ay)
        if ay < 0:
            t_min, t_max = t_max, t_min
    if abs(az) > 1e-12:
        t_min = max(t_min, (z0 - cz) / az)
        t_max = min(t_max, (z1 - cz) / az)
        if az < 0:
            t_min, t_max = t_max, t_min
    if t_min > t_max:
        return None
    return max(0.0, t_max - t_min)


def extract_circle_radii(features: dict[str, Any], *, scale_to_mm: float = 1.0) -> list[dict[str, Any]]:
    curves = features.get("curves") or []
    circles: list[dict[str, Any]] = []
    for c in curves:
        if c.get("type") != "Circle":
            continue
        r = c.get("radius")
        if r is None:
            continue
        loc = c.get("location") or [0, 0, 0]
        z_axis = c.get("z_axis")
        if not z_axis or len(z_axis) != 3:
            z_axis = [0.0, 0.0, 1.0]
        z_axis = _normalize([float(x) * scale_to_mm for x in z_axis])
        center = [round(float(x) * scale_to_mm, 4) for x in loc]
        r_mm = float(r) * scale_to_mm
        circles.append(
            {
                "radius_mm": round(r_mm, 4),
                "diameter_mm": round(2 * r_mm, 4),
                "center": center,
                "z_axis": z_axis,
                "sharp": bool(c.get("sharp", False)),
            }
        )
    return circles


def _perpendicular_axes(axis: list[float]) -> tuple[list[float], list[float]]:
    ax, ay, az = axis[0], axis[1], axis[2]
    if abs(az) <= 0.9:
        perp1 = _normalize([-ay, ax, 0])
    else:
        perp1 = _normalize([0, -az, ay])
    perp2 = _normalize(
        [
            ay * perp1[2] - az * perp1[1],
            az * perp1[0] - ax * perp1[2],
            ax * perp1[1] - ay * perp1[0],
        ]
    )
    return (perp1, perp2)


def _hole_id(center: list[float], axis: list[float], tol: float) -> tuple[float, float]:
    perp1, perp2 = _perpendicular_axes(axis)
    u = center[0] * perp1[0] + center[1] * perp1[1] + center[2] * perp1[2]
    v = center[0] * perp2[0] + center[1] * perp2[1] + center[2] * perp2[2]
    return (round(u / tol) * tol, round(v / tol) * tol)


def _filter_same_depth_keep_min_radius(holes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not holes:
        return holes
    by_depth: dict[float | None, list[dict[str, Any]]] = {}
    for h in holes:
        d = h.get("depth_mm")
        by_depth.setdefault(d, []).append(h)

    out: list[dict[str, Any]] = []
    for depth, group in by_depth.items():
        if len(group) <= 1:
            out.extend(group)
            continue
        singletons = [g for g in group if g.get("count", 0) == 1]
        multi = [g for g in group if g.get("count", 0) != 1]
        out.extend(singletons)
        if not multi:
            continue
        if len(multi) == 1:
            out.extend(multi)
            continue
        r_min = min(float(g["radius_mm"]) for g in multi)
        for g in multi:
            if abs(float(g["radius_mm"]) - r_min) <= RADIUS_TOLERANCE_MM * 2:
                out.append(g)
    return sorted(out, key=lambda x: (x["radius_mm"], x["depth_mm"] or 0))


def holes_with_depths(
    circles: list[dict[str, Any]],
    bbox: list[float] | None,
    *,
    radius_tol: float = RADIUS_TOLERANCE_MM,
    center_tol: float = CENTER_TOLERANCE_MM,
    depth_tol: float = 0.01,
    concentric_keep_inner: bool = True,
    same_depth_min_radius_only: bool = True,
) -> list[dict[str, Any]]:
    if not circles:
        return []

    by_axis_hole: dict[
        tuple[tuple[float, float, float], tuple[float, float]], list[dict[str, Any]]
    ] = {}
    for c in circles:
        axis = c["z_axis"]
        a_key = _axis_key(axis)
        hid = _hole_id(c["center"], axis, center_tol)
        by_axis_hole.setdefault((a_key, hid), []).append(c)

    filtered: list[dict[str, Any]] = []
    for (_a_key, _hid), group in by_axis_hole.items():
        if not group:
            continue
        if concentric_keep_inner:
            r_min = min(float(c["radius_mm"]) for c in group)
            r_key = round(r_min / radius_tol) * radius_tol
            r_key = round(r_key, 4)
            group = [c for c in group if abs(float(c["radius_mm"]) - r_key) <= radius_tol * 2]
        filtered.extend(group)

    by_radius_axis: dict[tuple[float, tuple[float, float, float]], list[dict[str, Any]]] = {}
    for c in filtered:
        r = float(c["radius_mm"])
        axis = c["z_axis"]
        r_key = round(r / radius_tol) * radius_tol
        r_key = round(r_key, 4)
        a_key = _axis_key(axis)
        by_radius_axis.setdefault((r_key, a_key), []).append(c)

    holes: list[dict[str, Any]] = []
    for (r, _), group in by_radius_axis.items():
        axis = group[0]["z_axis"]
        by_hole: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for c in group:
            hid = _hole_id(c["center"], axis, center_tol)
            by_hole.setdefault(hid, []).append(c)

        for _hid, rim_circles in by_hole.items():
            ts = [
                float(c["center"][0]) * axis[0]
                + float(c["center"][1]) * axis[1]
                + float(c["center"][2]) * axis[2]
                for c in rim_circles
            ]
            depth_mm: float | None = round(max(ts) - min(ts), 4) if ts else None
            if depth_mm is not None and depth_mm < 1e-6:
                depth_mm = None
            if depth_mm is None and bbox and len(rim_circles) == 1:
                ext = _bbox_extent_along_axis(bbox, rim_circles[0]["center"], axis)
                depth_mm = round(ext, 4) if ext is not None else None
            holes.append(
                {
                    "radius_mm": round(float(r), 4),
                    "diameter_mm": round(2 * float(r), 4),
                    "depth_mm": depth_mm,
                    "center_sample": rim_circles[0]["center"],
                }
            )

    by_r_depth: dict[tuple[float, float | None], dict[str, Any]] = {}
    for h in holes:
        r = float(h["radius_mm"])
        depth = h["depth_mm"]
        r_round = round(r / radius_tol) * radius_tol
        r_round = round(r_round, 4)
        depth_key = round(depth / depth_tol) * depth_tol if depth is not None else None
        depth_key = round(depth_key, 4) if depth_key is not None else None
        key = (r_round, depth_key)
        if key not in by_r_depth:
            by_r_depth[key] = {
                "radius_mm": r_round,
                "diameter_mm": round(2 * r_round, 4),
                "depth_mm": depth_key,
                "count": 0,
                "centers_sample": [],
            }
        by_r_depth[key]["count"] += 1
        if len(by_r_depth[key]["centers_sample"]) < 5:
            by_r_depth[key]["centers_sample"].append(h["center_sample"])

    result = [
        {
            "radius_mm": v["radius_mm"],
            "diameter_mm": v["diameter_mm"],
            "depth_mm": v["depth_mm"],
            "count": v["count"],
            "centers_sample": v["centers_sample"],
        }
        for v in sorted(by_r_depth.values(), key=lambda x: (x["radius_mm"], x["depth_mm"] or 0))
    ]
    if same_depth_min_radius_only:
        result = _filter_same_depth_keep_min_radius(result)
    return result

