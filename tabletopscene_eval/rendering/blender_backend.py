"""External Blender controller for immutable, material-preserving scene renders.

The normal evaluation environment needs no bpy installation. Blender runs its own
Python worker and returns RGB(A), first-hit object masks and a camera audit.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess


def _number(value, label, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"{label} must be a finite {'nonnegative' if zero else 'positive'} number")
    return value


def _vector(value, label, *, nonnegative=False):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value)
            or (nonnegative and any(x < 0 for x in value))):
        raise ValueError(f"{label} requires three finite {'nonnegative ' if nonnegative else ''}numbers")
    return list(value)


def validate_rendering_options(options):
    """Validate Blender's rendering schema and explicit device selection."""
    allowed = {"backend", "executable", "resolution", "renderer", "device", "samples", "denoise",
               "world", "lights", "color_management", "transparent_background", "timeout_seconds",
               "export_camera_resolution", "camera", "threads"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unsupported Blender rendering options")
    if options.get("backend", "blender") != "blender":
        raise ValueError("Blender rendering requires backend=blender")
    if options.get("renderer", "CYCLES") != "CYCLES":
        raise ValueError("The tested Blender renderer is CYCLES")
    if options.get("device", "OPTIX") not in {"OPTIX", "CUDA", "CPU"}:
        raise ValueError("Blender device must be OPTIX, CUDA or CPU")
    if not isinstance(options.get("executable", "/snap/bin/blender"), str) or not options.get("executable", "/snap/bin/blender").strip():
        raise ValueError("Blender executable must be a nonempty command or path")
    resolution = options.get("resolution", [512, 512])
    if (not isinstance(resolution, list) or len(resolution) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 16 for x in resolution)):
        raise ValueError("Blender resolution must be [width,height], integers >=16")
    for name, default in (("samples", 32), ("threads", 4)):
        if isinstance(options.get(name, default), bool) or not isinstance(options.get(name, default), int) or options.get(name, default) < 1:
            raise ValueError(f"Blender {name} must be a positive integer")
    for name, default in (("denoise", True), ("transparent_background", True)):
        if not isinstance(options.get(name, default), bool):
            raise ValueError(f"Blender {name} must be boolean")
    _number(options.get("timeout_seconds", 900), "Blender timeout_seconds")
    if options.get("export_camera_resolution", "native") not in {"native", "reference"}:
        raise ValueError("export_camera_resolution must be native or reference")
    world = options.get("world", {})
    if not isinstance(world, dict) or set(world) - {"color", "strength"}:
        raise ValueError("Blender world supports color and strength")
    _vector(world.get("color", [1, 1, 1]), "world.color", nonnegative=True)
    _number(world.get("strength", .6), "world.strength", zero=True)
    lights = options.get("lights", [])
    if not isinstance(lights, list):
        raise ValueError("Blender lights must be a list")
    for light in lights:
        if (not isinstance(light, dict) or set(light) - {"type", "position_y_up_m", "target_y_up_m", "color", "energy", "size"}
                or light.get("type", "AREA") != "AREA"):
            raise ValueError("Blender studio lights support AREA, position_y_up_m, target_y_up_m, color, energy and size")
        position = _vector(light.get("position_y_up_m"), "light.position_y_up_m")
        target = _vector(light.get("target_y_up_m", [0, 0, 0]), "light.target_y_up_m")
        if sum((a-b)**2 for a, b in zip(position, target)) <= 1e-18:
            raise ValueError("AREA light position and target must differ")
        _vector(light.get("color", [1, 1, 1]), "light.color", nonnegative=True)
        _number(light.get("energy", 400), "light.energy", zero=True)
        _number(light.get("size", 3), "light.size")
    color = options.get("color_management", {})
    if not isinstance(color, dict) or set(color) - {"view_transform", "look", "exposure", "gamma"}:
        raise ValueError("color_management supports view_transform, look, exposure and gamma")
    for key, default in (("view_transform", "AgX"), ("look", "None")):
        if not isinstance(color.get(key, default), str) or not color.get(key, default):
            raise ValueError(f"color_management.{key} must be a nonempty Blender enum")
    exposure = color.get("exposure", 0)
    if isinstance(exposure, bool) or not isinstance(exposure, (float, int)) or not math.isfinite(exposure):
        raise ValueError("color_management.exposure must be finite")
    _number(color.get("gamma", 1), "color_management.gamma")
    camera = options.get("camera", {})
    if not isinstance(camera, dict) or set(camera) - {"near_m", "far_m", "vertical_fov_deg"}:
        raise ValueError("Blender camera supports near_m, far_m and vertical_fov_deg")
    near = _number(camera.get("near_m", .01), "camera.near_m")
    far = _number(camera.get("far_m", 100), "camera.far_m")
    fov = _number(camera.get("vertical_fov_deg", 45), "camera.vertical_fov_deg")
    if far <= near or fov >= 179:
        raise ValueError("Require far_m > near_m and vertical_fov_deg <179")


def y_up_pose_to_blender(pose):
    """Conjugate a Y-up scene pose for Blender's imported Z-up local geometry."""
    import numpy as np
    conversion = np.eye(4)
    conversion[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    value = np.asarray(pose, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Object transform must be finite 4x4")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Object transform must be affine")
    return conversion @ value @ conversion.T


def render_objects(objects, output_dir, view_specs, rendering=None, foreground_options=None):
    """Render explicit shared cameras through a bounded external bpy worker."""
    from ..foreground import validate_foreground_options
    rendering = dict(rendering or {})
    validate_rendering_options(rendering)
    foreground = validate_foreground_options({} if foreground_options is None else foreground_options)
    if not isinstance(objects, list) or not objects:
        raise ValueError("Blender requires a nonempty object list")
    if not isinstance(view_specs, list) or not view_specs:
        raise ValueError("Blender requires explicit view specifications")
    ids = [view.get("id") for view in view_specs]
    if any(not isinstance(x, str) or not x or any(c in x for c in '/\\\x00') or x in {'.', '..'} for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Blender view IDs must be unique safe filenames")
    object_ids = [obj.get("id") for obj in objects]
    if any(not isinstance(x, str) or not x for x in object_ids) or len(set(object_ids)) != len(object_ids):
        raise ValueError("Blender object IDs must be unique nonempty strings")
    output = Path(output_dir).expanduser().resolve()
    for obj in objects:
        source = Path(obj["mesh"]).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Missing Blender visual asset: {source}")
        if output.is_relative_to(source.parent):
            raise ValueError("Blender output must be outside source asset directories")
        y_up_pose_to_blender(obj["transform"])
    output.mkdir(parents=True, exist_ok=True)
    request = {"objects": objects, "rendering": rendering, "views": view_specs,
               "foreground": foreground, "output_dir": str(output)}
    request_path = output / "blender_request.json"
    response_path = output / "blender_response.json"
    response_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
    worker = Path(__file__).with_name("blender_worker.py")
    log_path = output / "blender.log"
    env = dict(os.environ)
    # Blender carries its own bundled Python and must not inherit a foreign
    # Python runtime's startup path or home.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    timeout = rendering.get("timeout_seconds", 900)
    try:
        with log_path.open("w") as log:
            process = subprocess.run([rendering.get("executable", "/snap/bin/blender"), "--background", "--factory-startup",
                                      "--python-exit-code", "1", "--python", str(worker), "--", str(request_path), str(response_path)],
                                     env=env, cwd=str(output), stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Blender exceeded {timeout}s; see {log_path}") from exc
    if not response_path.is_file():
        raise RuntimeError(f"Blender exited {process.returncode} without a response; see {log_path}")
    result = json.loads(response_path.read_text())
    if process.returncode or "error" in result:
        raise RuntimeError(f"Blender failed: {result.get('error', process.returncode)}; see {log_path}")
    result["details"]["worker_log"] = str(log_path)
    return result


def prepare_scene_export(export_path, output_dir, camera_sampling=None, rendering=None,
                         foreground_options=None, visual_normalization=None):
    """Render an AgenTable export, retaining a separate export-camera GPT anchor."""
    from ..adapters.scene_export import load_scene_export
    from ..normalization import normalize_scene_objects, normalize_camera
    from ..foreground import validate_foreground_options
    from .cameras import make_view_specs, _export_view
    exported = load_scene_export(export_path)
    visual_objects, normalization = normalize_scene_objects(exported.objects, visual_normalization)
    visual_camera = normalize_camera(exported.camera, normalization)
    options = dict(rendering or {})
    validate_rendering_options(options)
    foreground = validate_foreground_options({} if foreground_options is None else foreground_options)
    resolution = options.get("resolution", [512, 512])
    export_resolution = resolution if options.get("export_camera_resolution", "native") == "reference" else None
    sampling = dict(camera_sampling or {"mode": "export_camera"})
    requested_anchor = sampling.pop("anchor_view_id", None)
    views = make_view_specs(sampling, visual_camera, resolution, export_resolution)
    if requested_anchor is not None:
        anchor = next((view for view in views if view["id"] == requested_anchor), None)
        if anchor is None:
            raise ValueError(f"Configured anchor_view_id is absent from camera candidates: {requested_anchor}")
    else:
        anchor = {"id": "export_camera", "camera": _export_view(visual_camera, export_resolution)} if visual_camera else None
    specs = list(views)
    if anchor and anchor["id"] not in {v["id"] for v in specs}:
        specs.append(anchor)
    result = render_objects(visual_objects, output_dir, specs, options, foreground)
    result["details"]["scene_export"] = exported.provenance
    result["details"]["visual_normalization"] = normalization
    rendered = {v["id"]: v for v in result["all_views"]}
    first = rendered[anchor["id"] if anchor else views[0]["id"]]
    result["render"] = first["render"]
    if foreground["enabled"]:
        result["render_mask"] = first["mask"]
    if not anchor:
        result["details"]["gpt_anchor_warning"] = "No export camera; anchor is first configured view"
    result["views"] = [rendered[v["id"]] for v in views]
    result.pop("all_views", None)
    (Path(output_dir) / "preparation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
