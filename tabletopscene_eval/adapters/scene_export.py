"""Read AgenTable's portable rigid scene-export package without changing it.

Geometry and poses come from canonical_layout.json (Y-up, meters). The separately
exported scene_config.json is Z-up and is checked for agreement; its transforms
must never be applied on top of the canonical layout transforms.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


_Y_TO_Z = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
_IDENTITY = [[float(i == j) for j in range(4)] for i in range(4)]


class SceneExportError(ValueError):
    """An export is incomplete, changed, or incompatible with the rigid schema."""


@dataclass(frozen=True)
class SceneExport:
    """Validated, read-only package description in canonical GLB coordinates."""

    root: Path
    result_path: Path
    scene_config_path: Path
    layout_path: Path
    camera_path: Path | None
    objects: list[dict]
    camera: dict | None
    layout: dict
    scene_config: dict
    provenance: dict

    def collision_manifest(self) -> dict:
        """Describe original triangle meshes, before any collision approximation.

        This contains individual objects, including the table flag so a metric
        can explicitly exclude the table. No convex hulls or bodies are generated.
        """
        return {
            "schema_version": "tabletopscene-eval.object-meshes/v1",
            "units": "meters",
            "coordinate_system": "glb_y_up_world",
            "objects": deepcopy(self.objects),
            "source_export": deepcopy(self.provenance),
        }


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneExportError(f"Cannot read scene export file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SceneExportError(f"{path.name} must contain a JSON object")
    return value


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SceneExportError(f"{label} must be a finite number")
    return float(value)


def _vector(value, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SceneExportError(f"{label} must contain {length} numbers")
    return [_number(item, label) for item in value]


def _matrix(value, size: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != size:
        raise SceneExportError(f"{label} must be a {size}x{size} matrix")
    return [_vector(row, size, label) for row in value]


def _close(a, b, tolerance=2e-6) -> bool:
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tolerance) for x, y in zip(a, b))
    return abs(a - b) <= tolerance


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def _rigid(value, label: str):
    matrix = _matrix(value, 4, label)
    rotation = [row[:3] for row in matrix[:3]]
    identity = [[float(i == j) for j in range(3)] for i in range(3)]
    det = sum(rotation[0][i] * (
        rotation[1][(i + 1) % 3] * rotation[2][(i + 2) % 3]
        - rotation[1][(i + 2) % 3] * rotation[2][(i + 1) % 3]
    ) for i in range(3))
    if (not _close(matrix[3], [0, 0, 0, 1]) or not _close(det, 1)
            or not _close(_multiply(_transpose(rotation), rotation), identity)):
        raise SceneExportError(f"{label} must be a proper rigid transform; scale must already be baked")
    return matrix


def _asset_path(value, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SceneExportError("Each exported asset requires a relative GLB path")
    path = Path(value)
    if path.is_absolute():
        raise SceneExportError("Exported GLB paths must be relative to the portable package")
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise SceneExportError("Exported asset path leaves the scene package")
    if path.suffix.lower() != ".glb" or not path.is_file():
        raise SceneExportError(f"Exported GLB asset is missing: {path}")
    return path


def _euler_xyz(angles):
    x, y, z = [math.radians(value) for value in angles]
    sx, cx, sy, cy, sz, cz = math.sin(x), math.cos(x), math.sin(y), math.cos(y), math.sin(z), math.cos(z)
    # scipy Rotation.from_euler('XYZ', ..., degrees=True): intrinsic XYZ.
    return [[cy * cz, -cy * sz, sy],
            [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy]]


def _check_engine_config(config: dict, objects: list[dict], root: Path):
    if config.get("format") != "embodichain.scene-export/v1" or config.get("articulation") != []:
        raise SceneExportError("scene_config.json must use the rigid embodichain.scene-export/v1 format")
    groups = [config.get("background"), config.get("rigid_object")]
    if any(not isinstance(group, list) for group in groups):
        raise SceneExportError("scene_config.json requires background and rigid_object lists")
    entries = {}
    for index, group in enumerate(groups):
        for entry in group:
            if not isinstance(entry, dict) or not isinstance(entry.get("uid"), str) or entry["uid"] in entries:
                raise SceneExportError("scene_config.json has an invalid or duplicate uid")
            entries[entry["uid"]] = entry
            if (entry["uid"] == "table") != (index == 0):
                raise SceneExportError("Only the table may be an exported background object")
    if set(entries) != {item["id"] for item in objects}:
        raise SceneExportError("Object IDs differ between canonical_layout.json and scene_config.json")
    for item in objects:
        entry = entries[item["id"]]
        shape = entry.get("shape", {})
        if (not isinstance(shape, dict) or shape.get("shape_type") != "Mesh"
                or str(_asset_path(shape.get("fpath"), root)) != item["mesh"]):
            raise SceneExportError(f"{item['id']}: engine and canonical mesh paths differ")
        if entry.get("is_articulated") is not False:
            raise SceneExportError(f"{item['id']}: only rigid exported objects are supported")
        if entry.get("body_type") != ("kinematic" if item["is_table"] else "dynamic"):
            raise SceneExportError(f"{item['id']}: unexpected exported body type")
        if not _close(_vector(entry.get("body_scale"), 3, "body_scale"), [1, 1, 1]):
            raise SceneExportError(f"{item['id']}: body scale must already be baked")
        pose = item["transform"]
        position = [pose[0][3], -pose[2][3], pose[1][3]]
        rotation = _multiply(_multiply(_Y_TO_Z, [row[:3] for row in pose[:3]]), _transpose(_Y_TO_Z))
        if (not _close(_vector(entry.get("init_pos"), 3, "init_pos"), position)
                or not _close(_euler_xyz(_vector(entry.get("init_rot"), 3, "init_rot")), rotation)):
            raise SceneExportError(f"{item['id']}: engine Z-up pose disagrees with canonical Y-up pose")


def _camera(value: dict):
    if value.get("projection") != "pinhole":
        raise SceneExportError("Exported camera must use pinhole projection")
    for name in ("width", "height"):
        if isinstance(value.get(name), bool) or not isinstance(value.get(name), int) or value[name] <= 0:
            raise SceneExportError(f"Exported camera {name} must be a positive integer")
    _rigid(value.get("camera_to_world"), "camera_to_world")
    _vector(value.get("target_world"), 3, "target_world")
    intrinsics = _matrix(value.get("intrinsics_pixels"), 3, "intrinsics_pixels")
    if (intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0
            or not _close(intrinsics[2], [0, 0, 1])):
        raise SceneExportError("Exported camera has invalid pinhole intrinsics")
    near, far = (_number(value.get(name), f"camera {name}") for name in ("near", "far"))
    if not 0 < near < far:
        raise SceneExportError("Exported camera requires 0 < near < far")
    return value


def load_scene_export(path) -> SceneExport:
    """Load an export directory or its result.json, resolving relocated packages.

    Absolute output paths in result.json are historical provenance. Files are
    always resolved inside the supplied package. Original source files need not
    remain available. Their recorded hashes are kept separately from hashes of
    the actual exported assets used for evaluation.
    """
    supplied = Path(path).expanduser().resolve()
    if supplied.is_dir():
        root = supplied
    elif supplied.is_file() and supplied.name == "result.json":
        root = supplied.parent
    else:
        raise SceneExportError("scene_export must point to an export directory or its result.json")
    result_path, layout_path, config_path = (root / name for name in ("result.json", "canonical_layout.json", "scene_config.json"))
    result, layout, config = _json(result_path), _json(layout_path), _json(config_path)
    if result.get("schema_version") != "agentable.scene-export/v1" or result.get("success") is not True or result.get("status") != "scene_exported":
        raise SceneExportError("result.json must describe a successful agentable.scene-export/v1 package")
    if (layout.get("schema_version") != "agentable.canonical-scene-layout/v1"
            or layout.get("coordinate_system") != "glb_y_up_world"
            or layout.get("layout_units") != "meters"
            or layout.get("scale_applied") is not True or layout.get("table_id") != "table"):
        raise SceneExportError("Canonical layout must use agentable.canonical-scene-layout/v1, Y-up meters, baked scales and table_id='table'")
    instances = layout.get("instances")
    if not isinstance(instances, list) or not instances:
        raise SceneExportError("Canonical layout requires a nonempty instances list")
    objects, seen, hashes = [], set(), {}
    source_assets = result.get("assets")
    if not isinstance(source_assets, dict):
        raise SceneExportError("result.json requires asset provenance and exported hashes")
    for instance in instances:
        identity = instance.get("object_id") if isinstance(instance, dict) else None
        if (not isinstance(identity, str) or not identity.strip() or identity in seen
                or identity in {".", ".."} or any(c in identity for c in ("/", "\\", "\x00", ":"))):
            raise SceneExportError("Canonical object IDs must be unique, nonempty safe filenames")
        seen.add(identity)
        if not _close(_vector(instance.get("scale"), 3, f"{identity} scale"), [1, 1, 1]):
            raise SceneExportError(f"{identity}: instance scale must already be baked")
        mesh = _asset_path(instance.get("geometry_path"), root)
        pose = _rigid(instance.get("matrix"), f"{identity} matrix")
        if identity == "table" and not _close(pose, _IDENTITY):
            raise SceneExportError("Exported table pose must be baked into its GLB, with an identity layout pose")
        asset = source_assets.get(identity)
        actual_hash = _digest(mesh)
        if not isinstance(asset, dict) or asset.get("export_sha256") != actual_hash:
            raise SceneExportError(f"{identity}: exported mesh SHA256 differs from result.json")
        hashes[str(mesh)] = actual_hash
        objects.append({"id": identity, "mesh": str(mesh), "transform": pose, "is_table": identity == "table"})
    if "table" not in seen or set(source_assets) != seen:
        raise SceneExportError("Canonical table or exported asset identities are inconsistent")
    _check_engine_config(config, objects, root)
    camera_path = root / "canonical_camera.json"
    declared_camera = result.get("canonical_camera_path") is not None
    camera = _camera(_json(camera_path)) if declared_camera else None
    if not declared_camera:
        camera_path = None
    metadata_paths = [result_path, layout_path, config_path]
    graph_path = root / "scene_graph.json"
    if graph_path.is_file():
        metadata_paths.append(graph_path)
    if camera_path:
        metadata_paths.append(camera_path)
    hashes.update({str(item): _digest(item) for item in metadata_paths})
    provenance = {
        "schema_version": result["schema_version"], "package_root": str(root),
        "scene_id": config.get("scene_id"), "sha256": hashes,
        "source_assets": deepcopy(source_assets),
        **{key: deepcopy(result.get(key)) for key in (
            "meters_per_unit", "source_layout_units", "scale_policy", "physics_parameter_source",
            "support_graph_policy", "physics_rerun", "source_world_to_export_world",
            "source_world_to_canonical_world", "object_id_mapping",
        )},
        "canonical_coordinates": "GLB Y-up, meters; camera looks along local -Z with local +Y up",
        "engine_coordinates": "Z-up; scene_config init_rot is intrinsic XYZ in degrees",
    }
    return SceneExport(root, result_path, config_path, layout_path, camera_path, objects, camera, layout, config, provenance)
