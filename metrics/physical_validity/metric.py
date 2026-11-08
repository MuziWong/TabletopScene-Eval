"""Object-pair collision percentages using mesh-derived convex solids.

This is an explicitly approximate VHACD/FCL protocol. It does not claim to
reproduce an unpublished TabletopGen collision implementation. Convex mesh
assets bypass decomposition; other assets retain the original surface for
provenance and use VHACD only to construct collision geometry.
"""
import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path

RUNTIME_DETAILS = {}

DEFAULT_DECOMPOSITION = {
    "max_convex_hulls": 128,
    "max_vertices_per_hull": 64,
    "resolution": 400000,
    "minimum_volume_percent_error": 1.0,
    "max_recursion_depth": 10,
    "shrink_wrap": True,
    "fill_mode": "flood",
    "min_edge_length": 2,
    "find_best_plane": False,
}
_NATIVE_NAMES = {
    "max_convex_hulls": "maxConvexHulls",
    "max_vertices_per_hull": "maxNumVerticesPerCH",
    "resolution": "resolution",
    "minimum_volume_percent_error": "minimumVolumePercentErrorAllowed",
    "max_recursion_depth": "maxRecursionDepth",
    "shrink_wrap": "shrinkWrap",
    "fill_mode": "fillMode",
    "min_edge_length": "minEdgeLength",
    "find_best_plane": "findBestPlane",
}
_CACHE_SCHEMA = "tabletopscene-eval.vhacd-cache/v1"


def _finite_number(value, minimum=0):
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= minimum)
    except OverflowError:
        return False


def validate_options(options):
    """Validate configuration without importing optional native packages."""
    if not isinstance(options, dict):
        raise ValueError("Physical metric options must be an object")
    unknown = set(options) - {
        "min_penetration_m", "numerical_tolerance_m", "decomposition", "cache_dir",
        "penetration_thresholds_m"
    }
    if unknown:
        raise ValueError(f"Unsupported vhacd_fcl options: {sorted(unknown)}")
    for name, default in (("min_penetration_m", 0.001), ("numerical_tolerance_m", 1e-9)):
        if not _finite_number(options.get(name, default)):
            raise ValueError(f"{name} must be a non-negative finite number")
    if not math.isfinite(options.get("min_penetration_m", 0.001) +
                         options.get("numerical_tolerance_m", 1e-9)):
        raise ValueError("The combined penetration threshold must be finite")
    if "penetration_thresholds_m" in options:
        from .convex_geometry import validate_penetration_thresholds
        if options["penetration_thresholds_m"] is None:
            raise ValueError("penetration_thresholds_m must be a non-empty list")
        validate_penetration_thresholds(options["penetration_thresholds_m"],
                                       options.get("numerical_tolerance_m", 1e-9))
    if "cache_dir" in options and (
            not isinstance(options["cache_dir"], str) or not options["cache_dir"].strip()):
        raise ValueError("cache_dir must be a non-empty path string")
    decomposition = options.get("decomposition", {})
    if not isinstance(decomposition, dict):
        raise ValueError("decomposition must be an object")
    unknown = set(decomposition) - set(DEFAULT_DECOMPOSITION)
    if unknown:
        raise ValueError(f"Unsupported VHACD decomposition options: {sorted(unknown)}")
    params = {**DEFAULT_DECOMPOSITION, **decomposition}
    for name, minimum in (("max_convex_hulls", 1), ("max_vertices_per_hull", 4),
                          ("resolution", 1), ("max_recursion_depth", 1),
                          ("min_edge_length", 1)):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"decomposition.{name} must be an integer >= {minimum}")
    if not _finite_number(params["minimum_volume_percent_error"]):
        raise ValueError("decomposition.minimum_volume_percent_error must be finite and >= 0")
    for name in ("shrink_wrap", "find_best_plane"):
        if not isinstance(params[name], bool):
            raise ValueError(f"decomposition.{name} must be a boolean")
    if params["fill_mode"] not in ("flood", "raycast", "surface"):
        raise ValueError("decomposition.fill_mode must be flood, raycast or surface")


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_exact(mesh, np, trimesh):
    """Merge identical coordinates only; retain every nonzero-area triangle."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or
            not np.isfinite(vertices).all()):
        raise ValueError("Collision asset must contain finite triangle geometry")
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    triangles = vertices[faces]
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
    if not np.isfinite(cross).all():
        raise ValueError("Collision mesh coordinates exceed numeric range")
    keep = np.any(cross != 0.0, axis=1)
    removed_faces = int((~keep).sum())
    faces = faces[keep]
    if not len(faces):
        raise ValueError("Collision asset has no non-degenerate triangles")
    used, face_inverse = np.unique(faces, return_inverse=True)
    cleaned = trimesh.Trimesh(vertices=vertices[used], faces=face_inverse.reshape((-1, 3)),
                              process=False)
    return cleaned, {
        "original_vertex_count": int(len(mesh.vertices)),
        "original_face_count": int(len(mesh.faces)),
        "cleaned_vertex_count": int(len(cleaned.vertices)),
        "cleaned_face_count": int(len(cleaned.faces)),
        "removed_zero_area_faces": removed_faces,
        "vertex_merge": "exact_coordinate_equality_only",
        "proximity_merge_distance_m": 0.0,
        "hole_filling": False,
    }


def _geometry_digest(mesh, np):
    digest = hashlib.sha256()
    for array in (np.asarray(mesh.vertices, dtype="<f8"),
                  np.asarray(mesh.faces, dtype="<u8")):
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _convex_part(vertices, faces, np, trimesh):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or len(faces) < 4
            or faces.dtype.kind not in "iu" or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError("Invalid convex decomposition geometry")
    # Convexify each native part separately, never the whole nonconvex object.
    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
    if (not part.is_watertight or not part.is_convex or not part.is_volume or
            not math.isfinite(float(part.volume)) or part.volume <= 0):
        raise ValueError("Convex collision part must be a closed positive-volume solid")
    return part


def _load_cache(path, expected_key, np, trimesh):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes().decode("utf-8"))
        if metadata.get("cache_key") != expected_key or metadata.get("schema") != _CACHE_SCHEMA:
            raise ValueError("VHACD cache metadata mismatch")
        count = metadata.get("part_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("VHACD cache contains an invalid part count")
        parts = [_convex_part(data[f"vertices_{i}"], data[f"faces_{i}"], np, trimesh)
                 for i in range(count)]
    return parts


def _write_cache(path, parts, key, np):
    payload = {}
    for index, part in enumerate(parts):
        payload[f"vertices_{index}"] = np.asarray(part.vertices, dtype=np.float64)
        payload[f"faces_{index}"] = np.asarray(part.faces, dtype=np.uint32)
    metadata = {"schema": _CACHE_SCHEMA, "cache_key": key, "part_count": len(parts)}
    payload["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode("utf-8"),
                                        dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace prevents partially written caches from concurrent readers.
    descriptor, temporary = tempfile.mkstemp(prefix=".vhacd-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _local_parts(mesh, decomposition, native_params, versions, cache_dir, np, trimesh, vhacdx):
    geometry_sha = _geometry_digest(mesh, np)
    # Collision solver versions are reported in results but do not affect the
    # generated geometry. Reuse decomposition when only FCL changes.
    geometry_versions = {key: value for key, value in versions.items() if key != "python-fcl"}
    key_input = {"schema": _CACHE_SCHEMA, "geometry_sha256": geometry_sha,
                 "versions": geometry_versions, "parameters": native_params,
                 "cleanup": "exact_vertices_and_zero_area_faces_v1"}
    key = hashlib.sha256(json.dumps(key_input, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()
    detail = {"geometry_sha256": geometry_sha, "cache_key": key,
              "cache_hit": False, "cache_path": None, "cache_sha256": None,
              "cache_key_inputs": key_input}
    # trimesh's general is_convex predicate uses a scale-relative planar
    # tolerance. Tighten this bypass to floating-point roundoff so a shallow
    # concavity cannot silently be replaced by its complete-object hull.
    convex_roundoff = 64 * np.finfo(np.float64).eps * max(
        1.0, float(np.abs(mesh.vertices).max()), float(mesh.scale))
    exact_convex = (mesh.is_watertight and mesh.is_volume and mesh.is_convex and
                    bool(np.all(mesh.face_adjacency_projections <= convex_roundoff)))
    detail["convex_bypass_roundoff_local_units"] = convex_roundoff
    if exact_convex:
        parts = [_convex_part(mesh.vertices, mesh.faces, np, trimesh)]
        detail["construction"] = "exact_convex_bypass"
    else:
        detail["construction"] = "vhacd_approximation"
        path = cache_dir / f"{key}.npz"
        detail["cache_path"] = str(path)
        if path.is_file():
            parts = _load_cache(path, key, np, trimesh)
            detail["cache_hit"] = True
        else:
            # vhacdx expects VTK cells [3, i, j, k, 3, ...], NOT an Nx3 array.
            # Supplying raw triangle indices silently corrupts decomposition.
            cells = np.column_stack((np.full(len(mesh.faces), 3, dtype=np.uint32),
                                     np.asarray(mesh.faces, dtype=np.uint32))).ravel()
            native_parts = vhacdx.compute_vhacd(
                np.asarray(mesh.vertices, dtype=np.float64), cells, **native_params)
            if not native_parts:
                raise ValueError("VHACD returned no convex collision parts")
            parts = [_convex_part(vertices, faces, np, trimesh)
                     for vertices, faces in native_parts]
            if len(parts) > decomposition["max_convex_hulls"]:
                raise ValueError("VHACD returned more parts than max_convex_hulls")
            _write_cache(path, parts, key, np)
        detail["cache_sha256"] = _file_digest(path)
    detail["part_count"] = len(parts)
    detail["part_vertex_counts"] = [int(len(part.vertices)) for part in parts]
    detail["source_volume_local_units3"] = float(mesh.volume) if mesh.is_volume else None
    detail["sum_convex_part_volumes_local_units3"] = float(sum(part.volume for part in parts))
    return parts, detail


def evaluate_physical(scene, options=None):
    """Evaluate initial non-table object pairs with a strict penetration limit.

    Input units must explicitly be meters. Source scale metadata is preserved;
    declaring meters does not establish that an export was physically calibrated.
    Nodes are baked before decomposition, and the complete per-object affine
    transform is then applied to part vertices, including nonuniform scale.
    """
    options = {} if options is None else options
    RUNTIME_DETAILS.clear()
    validate_options(options)
    try:
        import numpy as np
        import trimesh
        import vhacdx
        import fcl
        from .convex_geometry import evaluate_convex_objects
    except ImportError as exc:
        raise ImportError("vhacd_fcl requires numpy, scipy, trimesh, vhacdx and python-fcl") from exc
    decomposition = {**DEFAULT_DECOMPOSITION, **options.get("decomposition", {})}
    native_params = {_NATIVE_NAMES[key]: value for key, value in decomposition.items()}
    native_params["asyncACD"] = False
    versions = {"vhacdx": vhacdx.__version__, "trimesh": trimesh.__version__,
                "numpy": np.__version__, "python-fcl": fcl.__version__}
    cache_dir = Path(options.get("cache_dir", "~/.cache/tabletopscene-eval/vhacd")).expanduser().resolve()
    scene = Path(scene).expanduser().resolve()
    with scene.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), list):
        raise ValueError("Physical scene manifest must contain an 'objects' list")
    if manifest.get("units") != "meters":
        raise ValueError("vhacd_fcl requires manifest units='meters'; convert geometry explicitly first")
    selected, excluded, ids, records = [], [], set(), []
    objects = manifest["objects"]
    # Validate identities and the evaluated population before expensive VHACD.
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"]:
            raise ValueError("Each object requires a non-empty string id")
        if obj["id"] in ids:
            raise ValueError(f"Duplicate object id: {obj['id']}")
        ids.add(obj["id"])
        if not isinstance(obj.get("is_table", False), bool):
            raise ValueError(f"is_table must be a boolean for {obj['id']}")
        if obj.get("is_table", False):
            excluded.append(obj["id"])
    if len(objects) - len(excluded) < 2:
        raise ValueError("Physical validity requires at least two non-table objects")
    if "cache_dir" in options:
        # A caller-chosen cache must not modify the input manifest/asset tree.
        # Check all paths before generating the first cached decomposition.
        source_directories = [scene.parent]
        for obj in objects:
            if isinstance(obj.get("mesh"), str) and obj["mesh"]:
                source_path = Path(obj["mesh"]).expanduser()
                source_path = source_path if source_path.is_absolute() else scene.parent / source_path
                source_directories.append(source_path.resolve().parent)
        for directory in source_directories:
            if cache_dir == directory or directory in cache_dir.parents:
                raise ValueError("cache_dir must be outside the source manifest and mesh directories")
    for obj in sorted(objects, key=lambda item: item["id"]):
        if obj.get("is_table", False):
            continue
        if not isinstance(obj.get("mesh"), str) or not obj["mesh"]:
            raise ValueError(f"Missing collision mesh for {obj['id']}")
        path = Path(obj["mesh"]).expanduser()
        path = (path if path.is_absolute() else scene.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Collision mesh not found: {path}")
        try:
            transform = np.asarray(obj.get("transform", np.eye(4)), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid 4x4 affine transform for {obj['id']}") from exc
        if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
                not np.array_equal(transform[3], [0, 0, 0, 1]) or
                np.linalg.matrix_rank(transform[:3, :3]) < 3):
            raise ValueError(f"Invalid 4x4 affine transform for {obj['id']}")
        # Disabling trimesh processing prevents tolerance-based vertex merging.
        loaded = trimesh.load(path, force="scene", process=False)
        mesh = loaded.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"No triangle mesh in {path}")
        mesh, cleanup = _clean_exact(mesh, np, trimesh)
        try:
            parts, construction = _local_parts(mesh, decomposition, native_params, versions,
                                               cache_dir, np, trimesh, vhacdx)
        except ValueError as exc:
            raise ValueError(
                f"Invalid collision geometry for {obj['id']} "
                f"(source is_watertight={bool(mesh.is_watertight)}): {exc}; "
                "supply valid closed collision geometry or explicitly revise decomposition settings"
            ) from exc
        world_parts = []
        for part in parts:
            world = part.copy()
            world.apply_transform(transform)
            world_parts.append(_convex_part(world.vertices, world.faces, np, trimesh))
        selected.append({"id": obj["id"], "parts": world_parts})
        records.append({"id": obj["id"], "mesh": str(path), "sha256": _file_digest(path),
                        "transform": transform.tolist(), "is_watertight": bool(mesh.is_watertight),
                        "is_convex": bool(mesh.is_convex), "is_volume": bool(mesh.is_volume),
                        "cleanup": cleanup, **construction,
                        "approximation_warning": ("open_mesh_vhacd_has_no_exact_solid_interior"
                                                   if not mesh.is_watertight else None)})
    values, collision_details = evaluate_convex_objects(
        selected, min_penetration_m=options.get("min_penetration_m", 0.001),
        numerical_tolerance_m=options.get("numerical_tolerance_m", 1e-9),
        penetration_thresholds_m=options.get("penetration_thresholds_m"))
    RUNTIME_DETAILS.update(collision_details)
    RUNTIME_DETAILS.update({
        "backend": "vhacd_fcl", "collision_geometry": "mesh_derived_convex_solids",
        "geometry_approximation": "vhacd_except_closed_convex_meshes",
        "units": "meters", "state": "initial_exported_configuration",
        "min_penetration_m": options.get("min_penetration_m", 0.001),
        "numerical_tolerance_m": options.get("numerical_tolerance_m", 1e-9),
        "non_table_object_count": len(selected), "total_pairs": math.comb(len(selected), 2),
        "excluded_table_ids": excluded, "meshes": records, "versions": versions,
        "decomposition": decomposition, "native_decomposition_parameters": native_params,
        "cache_dir": str(cache_dir), "settling": False,
        "source_scale_metadata": deepcopy(manifest.get("source_export", {})),
        "scale_caveat": "units_declaration_does_not_establish_real_world_calibration",
    })
    return values
