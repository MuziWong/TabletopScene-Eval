"""Resolve a shared orbit from support-table geometry without changing a scene.

Only the explicitly marked support table participates in framing. All supplied
methods contribute to one shared rig; a method's object layout never controls
its own camera. The resulting ordinary ``orbit_grid`` specification can be
used by any renderer with the same pinhole camera convention.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path


def _finite(value, name, *, minimum=None):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value)
                 and (minimum is None or value >= minimum))
    except OverflowError:
        valid = False
    if not valid:
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be a finite number{suffix}")
    return float(value)


def validate_framing_options(framing):
    """Validate and resolve table-framing defaults without native dependencies.

    ``target_height='tabletop'`` targets the highest table vertex among all
    methods. ``'bounds_center'`` instead targets the vertical midpoint of the
    combined table bounds. Neither choice changes geometry or scene poses.
    """
    allowed = {"mode", "target_height", "target_height_offset_m", "margin"}
    if not isinstance(framing, dict) or set(framing) - allowed:
        raise ValueError("camera_sampling.framing has unsupported fields")
    mode = framing.get("mode", "shared_table_projection")
    if mode != "shared_table_projection":
        raise ValueError("camera_sampling.framing.mode must be shared_table_projection")
    height = framing.get("target_height", "bounds_center")
    if not isinstance(height, str) or height not in {"tabletop", "bounds_center"}:
        raise ValueError("camera_sampling.framing.target_height must be tabletop or bounds_center")
    return {
        "mode": mode,
        "target_height": height,
        "target_height_offset_m": _finite(
            framing.get("target_height_offset_m", 0.0), "target_height_offset_m"),
        "margin": _finite(framing.get("margin", 1.08), "framing.margin", minimum=1.0),
    }


def _frustum(resolution, vertical_fov_deg):
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in resolution)):
        raise ValueError("Table camera framing requires [width, height] positive integer resolution")
    vertical = _finite(vertical_fov_deg, "vertical_fov_deg")
    if not 0 < vertical < 179:
        raise ValueError("vertical_fov_deg must be strictly between 0 and 179")
    half_vertical = math.radians(vertical) / 2
    half_horizontal = math.atan(math.tan(half_vertical) * resolution[0] / resolution[1])
    return half_vertical, half_horizontal


def _projection_distance(vertices, target, sampling, margin,
                         half_vertical, half_horizontal, np):
    """Fit all table vertices in each requested view, with one shared distance.

    A point relative to the target has camera-right/up coordinates x/y and
    back coordinate z. At orbit distance D its positive forward depth is D-z,
    so |x|/(D-z) <= tan(horizontal_half_fov)/margin (and likewise for y).
    One distance satisfies these linear constraints for every requested view.
    """
    arrays = {}
    for name, default in (("elevations_deg", [30]), ("azimuths_deg", [0])):
        values = sampling.get(name, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Projection framing requires a non-empty {name} list")
        checked = [_finite(value, name) for value in values]
        if len(set(checked)) != len(checked):
            raise ValueError(f"Projection framing requires unique {name}")
        arrays[name] = checked
    if any(not -90 <= value <= 90 for value in arrays["elevations_deg"]):
        raise ValueError("Projection framing elevations_deg must be in [-90,90]")
    relative = vertices - target
    with np.errstate(over="ignore", invalid="ignore"):
        geometry_scale = float(np.linalg.norm(relative, axis=1).max())
    if not math.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("Support-table extent relative to the target must be positive and finite")
    clearance = 1e-9 * max(1.0, geometry_scale)
    records = []
    for elevation in arrays["elevations_deg"]:
        for azimuth in arrays["azimuths_deg"]:
            e, a = math.radians(elevation), math.radians(azimuth)
            back = np.array([math.cos(e)*math.sin(a), math.sin(e), math.cos(e)*math.cos(a)])
            right = np.array([math.cos(a), 0, -math.sin(a)])
            up = np.cross(back, right)
            z = relative @ back
            horizontal_bound = float(np.max(z + margin*np.abs(relative @ right)/math.tan(half_horizontal)))
            vertical_bound = float(np.max(z + margin*np.abs(relative @ up)/math.tan(half_vertical)))
            front_bound = float(z.max()) + clearance
            distance = max(horizontal_bound, vertical_bound, front_bound)
            records.append({"elevation_deg": elevation, "azimuth_deg": azimuth,
                            "horizontal_min_distance_m": horizontal_bound,
                            "vertical_min_distance_m": vertical_bound,
                            "positive_depth_min_distance_m": front_bound,
                            "minimum_distance_m": distance})
    return max(record["minimum_distance_m"] for record in records), records, clearance


def _table_vertices(objects_by_method):
    import numpy as np
    import trimesh

    if (not isinstance(objects_by_method, dict) or not objects_by_method
            or any(not isinstance(name, str) or not name for name in objects_by_method)):
        raise ValueError("Table camera framing requires a non-empty method-to-object mapping")
    clouds, records = [], []
    for method, objects in sorted(objects_by_method.items()):
        if not isinstance(objects, list):
            raise ValueError(f"{method}: table camera objects must be a list")
        tables = []
        for obj in objects:
            if not isinstance(obj, dict) or not isinstance(obj.get("is_table", False), bool):
                raise ValueError(f"{method}: objects require boolean is_table flags")
            if obj.get("is_table", False):
                tables.append(obj)
        if len(tables) != 1:
            raise ValueError(f"{method}: table camera framing requires exactly one is_table object")
        table = tables[0]
        if not isinstance(table.get("id"), str) or not table["id"]:
            raise ValueError(f"{method}: support table requires a non-empty id")
        mesh_value = table.get("mesh")
        if not isinstance(mesh_value, (str, Path)) or not str(mesh_value).strip():
            raise ValueError(f"{method}: support table requires a mesh path")
        path = Path(mesh_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Support-table mesh does not exist: {path}")
        try:
            transform = np.asarray(table.get("transform", np.eye(4)), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{method}: support-table transform must be a finite 4x4 affine matrix") from exc
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.array_equal(transform[3], [0, 0, 0, 1])
                or np.linalg.matrix_rank(transform[:3, :3]) < 3):
            raise ValueError(f"{method}: support-table transform must be a finite 4x4 affine matrix")
        loaded = trimesh.load(path, force="scene", process=False)
        mesh = loaded.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"{method}: support table contains no triangle geometry")
        # Loading and transforming a new mesh leaves caller data and source
        # assets untouched. to_geometry() already bakes GLTF scene-node poses.
        mesh.apply_transform(transform)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if not np.isfinite(vertices).all():
            raise ValueError(f"{method}: support-table world vertices must be finite")
        bounds = np.asarray(mesh.bounds, dtype=float)
        if not np.isfinite(bounds).all() or bounds[1, 1] <= bounds[0, 1]:
            raise ValueError(f"{method}: support table must have positive vertical extent")
        clouds.append(vertices)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({
            "method": method, "table_id": table["id"], "mesh": str(path),
            "mesh_sha256": digest.hexdigest(), "transform": transform.tolist(),
            "vertex_count": len(vertices), "bounds_y_up_m": bounds.tolist(),
        })
    return np.concatenate(clouds, axis=0), records


def resolve_shared_table_camera_sampling(camera_sampling, objects_by_method,
                                        resolution, vertical_fov_deg=45.0):
    """Return ``(ordinary_orbit_sampling, framing_details_or_None)``.

    ``objects_by_method`` maps names to canonical Y-up object-manifest lists.
    Every method must contain exactly one table flagged ``is_table=True``.
    One union of table vertices determines the common rig. The table's legs
    remain in this union even when the target is explicitly its tabletop.

    Solve exact per-vertex pinhole bounds at every requested azimuth/elevation,
    then take one common distance. The projection framing
    margin multiplies projected extents, limiting normalized image coordinates
    to 1/margin. Resolved ``radius_m`` is the *base orbit distance* and requested
    distance multipliers (>=1) scale it. Only the table determines framing.

    Without a ``framing`` block, return a deep copy unchanged and ``None``;
    no geometry or optional geometry package is loaded in that case.
    """
    if not isinstance(camera_sampling, dict):
        raise ValueError("camera_sampling must be an object")
    sampling = deepcopy(camera_sampling)
    if "framing" not in sampling:
        return sampling, None
    if sampling.get("mode") != "orbit_grid":
        raise ValueError("Table camera framing requires camera_sampling.mode=orbit_grid")
    if "target_y_up_m" in sampling or "radius_m" in sampling:
        raise ValueError("Automatic table framing cannot also specify target_y_up_m or radius_m")
    framing = validate_framing_options(sampling.pop("framing"))
    multipliers = sampling.get("distance_multipliers", [1.0])
    if not isinstance(multipliers, list) or not multipliers:
        raise ValueError("Automatic table framing requires a non-empty distance_multipliers list")
    for value in multipliers:
        _finite(value, "Automatic table framing distance multiplier", minimum=1.0)
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("Automatic table framing requires unique distance multipliers")
    half_vertical, half_horizontal = _frustum(resolution, vertical_fov_deg)
    import numpy as np
    vertices, records = _table_vertices(objects_by_method)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    target = bounds.mean(axis=0)
    if framing["target_height"] == "tabletop":
        target[1] = bounds[1, 1]
    target[1] += framing["target_height_offset_m"]
    distance, projection_records, depth_clearance = _projection_distance(
        vertices, target, sampling, framing["margin"], half_vertical, half_horizontal, np)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Fitted table-camera distance must be positive and finite")
    sampling.update(target_y_up_m=target.tolist(), radius_m=distance,
                    distance_multipliers=deepcopy(multipliers))
    details = {
        "schema": "tabletopscene-eval.shared-table-camera/v1",
        "mode": framing["mode"], "table_only": True,
        "coordinate_system": "glb_y_up_world", "units": "meters",
        "scope": "one_shared_rig_across_all_supplied_methods",
        "source_tables": records, "combined_table_bounds_y_up_m": bounds.tolist(),
        "target_height": framing["target_height"],
        "target_height_rule": ("combined_max_y" if framing["target_height"] == "tabletop"
                               else "combined_bounds_vertical_midpoint"),
        "target_height_offset_m": framing["target_height_offset_m"],
        "target_y_up_m": target.tolist(), "margin": framing["margin"],
        "base_camera_distance_m": distance,
        "resolved_radius_m_semantics": "base_camera_distance",
        "distance_multipliers": deepcopy(multipliers),
        "vertical_fov_deg": float(vertical_fov_deg),
        "horizontal_fov_deg": math.degrees(2 * half_horizontal),
        "resolution": list(resolution),
        "geometry_modified": False,
        "normalization_performed_by_camera_helper": False,
        "fit_guarantee": "all_table_vertices_fit_angular_pinhole_frusta_before_near_far_clipping",
        "non_table_object_fit_guaranteed": False,
        "distance_rule": "maximum_per_vertex_pinhole_constraint_over_all_requested_angles",
        "projection_margin_ndc_limit": 1.0 / framing["margin"],
        "projection_positive_depth_clearance_m": depth_clearance,
        "projection_view_constraints": projection_records,
    }
    return sampling, details
