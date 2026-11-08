"""Shared Y-up camera grids and exported pinhole camera specifications."""
from __future__ import annotations

import math


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number")
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must have three finite numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} must have three finite numbers")
    return list(value)


def make_view_specs(camera_sampling, export_camera, resolution, export_resolution=None):
    """Resolve shared Y-up cameras; never fit a camera to method output bounds."""
    import numpy as np

    allowed = {"mode", "elevations_deg", "azimuths_deg", "distance_multipliers", "target_y_up_m", "radius_m", "anchor_view_id"}
    if not isinstance(camera_sampling, dict) or set(camera_sampling) - allowed:
        raise ValueError("Unsupported camera_sampling fields")
    if "anchor_view_id" in camera_sampling and (
            not isinstance(camera_sampling["anchor_view_id"], str)
            or not camera_sampling["anchor_view_id"].strip()):
        raise ValueError("camera_sampling.anchor_view_id must be a non-empty string")
    mode = camera_sampling.get("mode", "export_camera")
    if mode == "export_camera":
        if not export_camera:
            raise ValueError("export_camera requires canonical_camera.json in scene export")
        # Preserve the input-view pixel grid. The generic rendering resolution is
        # only used for orbit search; export cameras already carry calibration.
        return [{"id": "export_camera", "camera": _export_view(export_camera, export_resolution)}]
    if mode != "orbit_grid":
        raise ValueError("camera_sampling.mode must be export_camera or orbit_grid")
    target = np.asarray(_vector(camera_sampling.get("target_y_up_m"), "target_y_up_m"), dtype=float)
    radius = _positive(camera_sampling.get("radius_m"), "radius_m")
    arrays = {}
    for key, default in (("elevations_deg", [30]), ("azimuths_deg", [0]), ("distance_multipliers", [2])):
        values = camera_sampling.get(key, default)
        if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list of unique finite numbers")
        arrays[key] = values
    if any(not -90 <= v <= 90 for v in arrays["elevations_deg"]) or any(v <= 0 for v in arrays["distance_multipliers"]):
        raise ValueError("Elevations must be in [-90,90] and distance multipliers positive")
    result = []
    for ei, elevation in enumerate(arrays["elevations_deg"]):
        for ai, azimuth in enumerate(arrays["azimuths_deg"]):
            for di, multiplier in enumerate(arrays["distance_multipliers"]):
                e, a = math.radians(elevation), math.radians(azimuth)
                back = np.array([math.cos(e)*math.sin(a), math.sin(e), math.cos(e)*math.cos(a)])
                # The analytic right vector remains defined at the poles.
                right = np.array([math.cos(a), 0, -math.sin(a)])
                up = np.cross(back, right)
                result.append({"id": f"e{ei:02d}_a{ai:02d}_d{di:02d}", "camera": {
                    "position_y_up_m": (target + radius*multiplier*back).tolist(),
                    "target_y_up_m": target.tolist(), "up_y_up": up.tolist(),
                    "elevation_deg": elevation, "azimuth_deg": azimuth,
                    "distance_radius": multiplier, "radius_m": radius,
                }})
    return result


def _export_view(camera, resolution=None):
    import numpy as np
    pose = np.asarray(camera["camera_to_world"], dtype=float)
    k = np.asarray(camera["intrinsics_pixels"], dtype=float).copy()
    width, height = camera["width"], camera["height"]
    effective = list(resolution) if resolution is not None else [width, height]
    scale_x, scale_y = effective[0]/width, effective[1]/height
    k[0, :] *= scale_x
    k[1, :] *= scale_y
    return {"position_y_up_m": pose[:3, 3].tolist(),
            "target_y_up_m": (pose[:3, 3]-pose[:3, 2]).tolist(),
            "up_y_up": pose[:3, 1].tolist(), "intrinsics_pixels": k.tolist(),
            "resolution": effective, "source_resolution": [width, height],
            "source_intrinsics_pixels": camera["intrinsics_pixels"],
            "intrinsics_scale_xy": [scale_x, scale_y],
            "resolution_policy": "native" if resolution is None else "full_image_resize_no_crop_assumption",
            "near_m": camera["near"],
            "far_m": camera["far"], "source": "canonical_camera.json"}
