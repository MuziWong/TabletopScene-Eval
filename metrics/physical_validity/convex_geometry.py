"""Object-pair collision percentages from a mesh's convex decomposition.

Each input part is a closed convex triangle mesh in scene coordinates measured
in meters. This module never substitutes a single hull for a nonconvex object.
The measured depth is the maximum penetration of its convex-part pairs; it is
not a minimum translation distance for the complete nonconvex mesh union.
"""
import hashlib
import math
from itertools import combinations


def _finite_nonnegative(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite nonnegative number in meters")
    return float(value)


def validate_penetration_thresholds(thresholds, numerical_tolerance_m=1e-9):
    """Return a sorted, unique sweep or None, without native dependencies."""
    tolerance = _finite_nonnegative(numerical_tolerance_m, "numerical_tolerance_m")
    if thresholds is None:
        return None
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("penetration_thresholds_m must be a non-empty list")
    values = [_finite_nonnegative(value, "penetration_thresholds_m") for value in thresholds]
    if len(set(values)) != len(values):
        raise ValueError("penetration_thresholds_m must contain unique values")
    if any(not math.isfinite(value + tolerance) or not math.isfinite(value * 1000)
           for value in values):
        raise ValueError("Each combined penetration threshold and its millimeter value must be finite")
    return sorted(values)


def _summarize_threshold(pair_records, object_count, threshold, tolerance):
    """Reuse completed maximum-depth queries; no geometry or FCL calls here."""
    decisions = [{"pair": list(record["pair"]),
                  "raw_intersects": record["raw_intersects"],
                  "intersects": record["max_penetration_depth_m"] > threshold + tolerance,
                  "max_penetration_depth_m": record["max_penetration_depth_m"]}
                 for record in pair_records]
    intersecting = [record["pair"] for record in decisions if record["intersects"]]
    filtered = [record["pair"] for record in decisions
                if record["raw_intersects"] and not record["intersects"]]
    total_pairs = math.comb(object_count, 2)
    count = len(intersecting)
    return {
        "threshold_m": threshold, "threshold_mm": threshold * 1000,
        "numerical_tolerance_m": tolerance,
        "threshold_comparison": "depth > threshold_m + numerical_tolerance_m",
        "non_table_object_count": object_count, "total_pairs": total_pairs,
        "intersecting_pair_count": count, "intersecting_pairs": intersecting,
        "filtered_pair_count": len(filtered), "filtered_pairs": filtered,
        "pair_decisions": decisions,
        "ColO": 100.0 * count / total_pairs, "ColS": 100.0 if count else 0.0,
    }


def evaluate_convex_objects(objects, min_penetration_m=0.001,
                            numerical_tolerance_m=1e-9, penetration_thresholds_m=None):
    """Return ``(metric_values, audit_details)`` for already selected objects.

    ``objects`` is a list of ``{"id": str, "parts": [trimesh.Trimesh, ...]}``.
    The caller excludes the support table and applies scene/unit transforms.
    All distinct object pairs are counted once. A pair collides when any part
    pair has depth greater than ``min_penetration_m + numerical_tolerance_m``.
    The latter is a comparison guard, not a statement about solver accuracy.
    An optional penetration_thresholds_m sweep reuses these same part queries;
    it never reruns FCL and does not change the primary min_penetration_m score.

    Each AABB-overlapping part pair receives its own FCL Convex/Convex query,
    using LIBCCD's MPR penetration estimate and one requested contact. FCL
    0.7's generic LIBCCD shape intersection returns one penetration contact
    for Convex/Convex (see
    ``gjk_solver_libccd-inl.h``, ShapeIntersectLibccdImpl). No global collision
    manager, triangle-BVH depth, contact truncation across parts, simulation,
    mesh displacement, or support-pair exemption is used.

    MPR is not an exact minimum-translation oracle, even for a convex pair;
    degenerate symmetric configurations can overestimate the depth. In
    particular, numerical_tolerance_m does not bound this approximation error.
    """
    threshold = _finite_nonnegative(min_penetration_m, "min_penetration_m")
    tolerance = _finite_nonnegative(numerical_tolerance_m, "numerical_tolerance_m")
    sweep_thresholds = validate_penetration_thresholds(penetration_thresholds_m, tolerance)
    if not math.isfinite(threshold + tolerance):
        raise ValueError("The combined penetration threshold must be finite")
    if not isinstance(objects, list) or len(objects) < 2:
        raise ValueError("Convex collision evaluation requires at least two objects")
    ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"]:
            raise ValueError("Each object requires a non-empty string id")
        if obj["id"] in ids:
            raise ValueError(f"Duplicate object id: {obj['id']}")
        ids.add(obj["id"])
        if not isinstance(obj.get("parts"), list) or not obj["parts"]:
            raise ValueError(f"Each object requires a non-empty parts list: {obj['id']}")
    try:
        import fcl
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise ImportError("Convex collision evaluation requires numpy, trimesh and python-fcl") from exc

    prepared, mesh_records = [], []
    for obj in sorted(objects, key=lambda item: item["id"]):
        parts = []
        for index, supplied in enumerate(obj["parts"]):
            label = f"{obj['id']} part {index}"
            if not isinstance(supplied, trimesh.Trimesh):
                raise ValueError(f"Expected a trimesh.Trimesh for {label}")
            # Work on a copy so trimesh's topology/normal caches and any future
            # validation changes cannot alter caller-owned geometry.
            mesh = supplied.copy()
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                    or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4
                    or not np.isfinite(vertices).all()
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError(f"Invalid finite triangle mesh for {label}")
            volume = float(mesh.volume)
            if not math.isfinite(volume) or volume <= 0 or not mesh.is_volume or not mesh.is_convex:
                raise ValueError(f"Each part must be a closed, outward-oriented convex solid: {label}")
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
            geometry_hash = digest.hexdigest()
            # Explicit Convex construction: never let trimesh choose BVH.
            polygons = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces))
            geometry = fcl.Convex(vertices, len(faces), polygons.ravel())
            collision_object = fcl.CollisionObject(geometry)
            parts.append({"sha256": geometry_hash, "bounds": mesh.bounds.copy(),
                          "geometry": geometry, "collision_object": collision_object,
                          "vertex_count": len(vertices), "face_count": len(faces)})
        # Geometry hashes make part IDs and query order independent of input
        # list order; equal meshes remain distinct parts with sequential IDs.
        parts.sort(key=lambda part: part["sha256"])
        records = []
        for index, part in enumerate(parts):
            part["id"] = f"part_{index:04d}"
            records.append({"id": part["id"], "geometry_sha256": part["sha256"],
                            "bounds_m": part["bounds"].tolist(),
                            "vertex_count": part["vertex_count"], "face_count": part["face_count"]})
        prepared.append({"id": obj["id"], "parts": parts})
        mesh_records.append({"id": obj["id"], "part_count": len(parts), "parts": records})

    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True,
                                   gjk_solver_type=fcl.GJKSolverType.GST_LIBCCD)
    pair_records, raw_pairs, filtered_pairs, intersecting_pairs = [], [], [], []
    for first, second in combinations(prepared, 2):
        pair = [first["id"], second["id"]]
        part_records = []
        max_depth, deepest_part_pair = 0.0, None
        raw_intersects, contact_count = False, 0
        for first_part in first["parts"]:
            for second_part in second["parts"]:
                bounds_a, bounds_b = first_part["bounds"], second_part["bounds"]
                if not (np.all(bounds_a[0] <= bounds_b[1]) and np.all(bounds_b[0] <= bounds_a[1])):
                    continue
                part_pair = [first_part["id"], second_part["id"]]
                result = fcl.CollisionResult()
                try:
                    fcl.collide(first_part["collision_object"], second_part["collision_object"],
                                request, result)
                except Exception as exc:
                    raise RuntimeError(f"FCL convex query failed for {pair}, parts {part_pair}") from exc
                intersects = bool(result.is_collision)
                contacts = result.contacts
                if intersects != bool(contacts):
                    raise RuntimeError(f"FCL collision/contact disagreement for {pair}, parts {part_pair}")
                depths = []
                for contact in contacts:
                    depth = float(contact.penetration_depth)
                    if not math.isfinite(depth) or depth < -tolerance:
                        raise RuntimeError(f"Invalid FCL penetration depth for {pair}, parts {part_pair}: {depth}")
                    depths.append(max(0.0, depth))
                depth = max(depths, default=0.0)
                raw_intersects = raw_intersects or intersects
                contact_count += len(contacts)
                if intersects and (deepest_part_pair is None or depth > max_depth):
                    max_depth, deepest_part_pair = depth, part_pair
                part_records.append({"parts": part_pair, "raw_intersects": intersects,
                                     "max_penetration_depth_m": depth, "contact_count": len(contacts)})
        retained = max_depth > threshold + tolerance
        if raw_intersects:
            raw_pairs.append(pair)
        if retained:
            intersecting_pairs.append(pair)
        elif raw_intersects:
            filtered_pairs.append(pair)
        total_part_pairs = len(first["parts"]) * len(second["parts"])
        pair_records.append({"pair": pair, "raw_intersects": raw_intersects,
                             "intersects": retained, "max_penetration_depth_m": max_depth,
                             "deepest_part_pair": deepest_part_pair,
                             "total_part_pairs": total_part_pairs,
                             "aabb_candidate_part_pairs": len(part_records),
                             "aabb_rejected_part_pairs": total_part_pairs - len(part_records),
                             "contact_count": contact_count, "part_pair_queries": part_records})

    total_pairs = math.comb(len(prepared), 2)
    count = len(intersecting_pairs)
    details = {
        "collision_mode": "convex_part_penetration", "collision_engine": "python-fcl",
        "fcl_geometry_type": "Convex", "gjk_solver": "GST_LIBCCD",
        "solver": "GST_LIBCCD",
        "penetration_solver": "libccd_mpr", "penetration_depth_is_exact_mtd": False,
        "num_max_contacts_per_part_pair": 1,
        "depth_definition": "maximum_convex_part_pair_penetration",
        "depth_semantics": "maximum_libccd_mpr_estimate_across_convex_part_pairs_not_exact_mesh_union_mtd",
        "min_penetration_m": threshold, "numerical_tolerance_m": tolerance,
        "threshold_comparison": "depth > min_penetration_m + numerical_tolerance_m",
        "units": "meters", "non_table_object_count": len(prepared),
        "total_pairs": total_pairs, "intersecting_pair_count": count,
        "intersecting_pairs": intersecting_pairs,
        "raw_intersecting_pair_count": len(raw_pairs), "raw_intersecting_pairs": raw_pairs,
        "filtered_pair_count": len(filtered_pairs), "filtered_pairs": filtered_pairs,
        "objects": mesh_records, "pair_records": pair_records,
    }
    if sweep_thresholds is not None:
        details["threshold_sweep"] = [
            _summarize_threshold(pair_records, len(prepared), value, tolerance)
            for value in sweep_thresholds
        ]
        details["threshold_sweep_geometry_queries_reused"] = True
    return {"ColO": 100.0 * count / total_pairs, "ColS": 100.0 if count else 0.0}, details
