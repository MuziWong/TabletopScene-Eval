"""Visual-only uniform normalization from a scene's support-table geometry.

Source assets, canonical export validation and physical coordinates are never
changed. A copied object manifest receives one shared similarity transform.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path


def validate_visual_normalization(options=None):
    """Validate an explicit policy; omitted low-level options preserve geometry."""
    if options is None:
        options = {}
    allowed = {"enabled", "mode", "target_diameter_m"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("visual_normalization supports enabled, mode and target_diameter_m")
    if not isinstance(options.get("enabled", False), bool):
        raise ValueError("visual_normalization.enabled must be boolean")
    if options.get("mode", "tabletop_diameter") != "tabletop_diameter":
        raise ValueError("visual_normalization.mode must be tabletop_diameter")
    diameter = options.get("target_diameter_m", 1.)
    if (isinstance(diameter, bool) or not isinstance(diameter, (float, int))
            or not math.isfinite(diameter) or diameter <= 0):
        raise ValueError("visual_normalization.target_diameter_m must be finite and positive")
    return {"enabled": options.get("enabled", False), "mode": "tabletop_diameter",
            "target_diameter_m": float(diameter)}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_scene_objects(objects, options=None):
    """Copy objects and apply one ``scale @ translation @ original_pose``.

    Translation moves the support table's world XZ bounds midpoint and maximum
    Y to the origin. One positive scale makes max(X extent,Z extent) equal the
    requested diameter. Table legs, object layouts, heights and all relative
    offsets receive exactly the same uniform scale. Mesh files stay unchanged.
    """
    options = validate_visual_normalization(options)
    copied = deepcopy(objects)
    if not options["enabled"]:
        return copied, {**options, "geometry_modified": False, "physics_geometry_unchanged": True,
                        "scope": "visual_rendering_only"}
    import numpy as np
    import trimesh
    if not isinstance(objects, list) or not objects:
        raise ValueError("Visual normalization requires a nonempty object list")
    ids, tables, hashes, poses = set(), [], {}, []
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"] or obj["id"] in ids:
            raise ValueError("Visual normalization requires unique nonempty object IDs")
        ids.add(obj["id"])
        if not isinstance(obj.get("is_table", False), bool):
            raise ValueError("Visual normalization requires boolean is_table flags")
        path = Path(obj["mesh"]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Missing visual-normalization source mesh: {path}")
        hashes[str(path)] = _sha(path)
        try:
            pose = np.asarray(obj.get("transform", np.eye(4)), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Visual-normalization object transforms must be finite affine 4x4 matrices") from exc
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                or np.linalg.matrix_rank(pose[:3, :3]) < 3):
            raise ValueError("Visual-normalization object transforms must be finite affine 4x4 matrices")
        poses.append(pose)
        if obj.get("is_table", False):
            tables.append((obj, path, pose))
    if len(tables) != 1:
        raise ValueError("Visual normalization requires exactly one is_table object")
    table, path, table_pose = tables[0]
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = loaded.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Support-table source must contain triangle geometry")
    mesh.apply_transform(table_pose)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Support-table world bounds must be finite")
    extents = bounds[1] - bounds[0]
    diameter = float(max(extents[0], extents[2]))
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Support table must have positive XZ diameter")
    center = bounds.mean(axis=0)
    center[1] = bounds[1, 1]
    scale = options["target_diameter_m"] / diameter
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Visual normalization requires a positive finite uniform scale")
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    if not np.isfinite(transform).all():
        raise ValueError("Visual normalization would create a nonfinite world transform")
    for obj, pose in zip(copied, poses):
        result_pose = transform @ pose
        if not np.isfinite(result_pose).all():
            raise ValueError("Visual normalization would create a nonfinite object transform")
        obj["transform"] = result_pose.tolist()
    for source, digest in hashes.items():
        if _sha(source) != digest:
            raise RuntimeError("A source mesh changed during visual normalization")
    details = {
        **options, "schema_version": "tabletopscene-eval.visual-normalization/v1",
        "scope": "visual_rendering_only", "source_coordinate_system": "glb_y_up_world",
        "target_coordinate_system": "normalized_glb_y_up_world",
        "source_table_id": table["id"], "source_table_mesh": str(path),
        "source_table_mesh_sha256": hashes[str(path)],
        "source_table_world_transform": table_pose.tolist(),
        "source_table_bounds_y_up_m": bounds.tolist(), "source_table_dimensions_m": extents.tolist(),
        "source_table_diameter_m": diameter, "diameter_definition": "max_world_x_extent_world_z_extent",
        "source_tabletop_center_y_up_m": center.tolist(), "uniform_scale": float(scale),
        "normalized_world_from_source_world": transform.tolist(),
        "normalized_table_bounds_y_up_m": (bounds * scale - center * scale).tolist(),
        "normalization_rule": "shared_uniform_scale_after_tabletop_center_translation_for_every_object",
        "source_asset_sha256": hashes, "source_hashes_before_after_equal": True,
        "object_ids": [obj["id"] for obj in objects], "geometry_modified": True,
        "source_mesh_files_modified": False, "relative_pose_policy": "single_common_similarity_transform",
        "physics_geometry_unchanged": True,
    }
    return copied, details


def normalize_camera(camera, details):
    """Transform an export camera with its scene; preserve rotation and K.

    Orbit specs already use the normalized scene's shared coordinate frame and
    must not be passed through this function. Near/far and recorded distances
    scale with the scene so the original pinhole image remains unchanged.
    """
    result = deepcopy(camera)
    if camera is None or not details or not details.get("enabled", False):
        return result
    import numpy as np
    transform = np.asarray(details["normalized_world_from_source_world"], dtype=np.float64)
    scale = details["uniform_scale"]
    pose = np.asarray(camera["camera_to_world"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Export camera requires a finite camera_to_world 4x4 matrix")
    normalized_pose = pose.copy()
    normalized_pose[:3, 3] = transform[:3, :3] @ pose[:3, 3] + transform[:3, 3]
    result["camera_to_world"] = normalized_pose.tolist()
    if "target_world" in camera:
        result["target_world"] = (transform[:3, :3] @ np.asarray(camera["target_world"]) + transform[:3, 3]).tolist()
    for key in ("near", "far", "distance"):
        if key in camera:
            result[key] = float(camera[key] * scale)
    return result
