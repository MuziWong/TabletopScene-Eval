"""Blender-owned Python worker; execute only through the external controller."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tabletopscene_eval.rendering.blender_backend import validate_rendering_options, y_up_pose_to_blender
from tabletopscene_eval.foreground import validate_foreground_options

C = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _look_at(position, target, up):
    position, target, up = [Vector(x) for x in (position, target, up)]
    back = position - target
    if back.length < 1e-10:
        raise ValueError("Camera position and target must differ")
    back.normalize()
    right = up.cross(back)
    if right.length < 1e-10:
        raise ValueError("Camera up must not be parallel to its view direction")
    right.normalize()
    up = back.cross(right).normalized()
    matrix = Matrix((right, up, back)).transposed().to_4x4()
    matrix.translation = position
    return matrix


def _configure_camera(scene, camera, spec, options):
    settings = options.get("camera", {})
    resolution = spec.get("resolution", options.get("resolution", [512, 512]))
    width, height = resolution
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    data = camera.data
    data.type = "PERSP"
    data.clip_start = spec.get("near_m", settings.get("near_m", .01))
    data.clip_end = spec.get("far_m", settings.get("far_m", 100))
    data.shift_x = data.shift_y = 0
    if "intrinsics_pixels" in spec:
        k = np.array(spec["intrinsics_pixels"], dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[0, 1]) > 1e-10:
            raise ValueError("Camera intrinsics require finite positive focal lengths and zero skew")
        if not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid homogeneous camera intrinsics")
        aspect = k[0, 0] / k[1, 1]
        if aspect >= 1:
            scene.render.pixel_aspect_y = aspect
        else:
            scene.render.pixel_aspect_x = 1 / aspect
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36
        data.lens = k[0, 0] * data.sensor_width / width
        data.shift_x = (width / 2 - k[0, 2]) / width
        data.shift_y = (k[1, 2] - height / 2) * aspect / width
    else:
        data.sensor_fit = "VERTICAL"
        data.sensor_height = 24
        data.lens = data.sensor_height / (2 * math.tan(math.radians(settings.get("vertical_fov_deg", 45)) / 2))
    position = C[:3, :3] @ np.array(spec["position_y_up_m"], dtype=float)
    target = C[:3, :3] @ np.array(spec["target_y_up_m"], dtype=float)
    up = C[:3, :3] @ np.array(spec.get("up_y_up", [0, 1, 0]), dtype=float)
    camera.matrix_world = _look_at(position, target, up)
    scene.camera = camera
    bpy.context.view_layer.update()
    projection = np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height,
                                                scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y))
    actual_k = [[float(projection[0, 0] * width / 2), 0., float(width / 2 * (1-projection[0, 2]))],
                [0., float(projection[1, 1] * height / 2), float(height / 2 * (1+projection[1, 2]))], [0., 0., 1.]]
    if "intrinsics_pixels" in spec and not np.allclose(actual_k, k, atol=2e-3, rtol=1e-6):
        raise RuntimeError(f"Blender camera K does not match requested intrinsics: {actual_k}")
    return {"actual_camera_to_world_z_up": [list(row) for row in camera.matrix_world],
            "actual_projection_matrix": projection.tolist(), "actual_intrinsics_pixels": actual_k,
            "actual_resolution": [width, height], "near_m": float(data.clip_start), "far_m": float(data.clip_end)}


def _configure_scene(options):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = options.get("samples", 32)
    scene.cycles.use_denoising = options.get("denoise", True)
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = options.get("threads", 4)
    device = options.get("device", "OPTIX")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "CPU":
        scene.cycles.device = "CPU"
        enabled = [{"type": "CPU", "name": "Blender CPU device"}]
    else:
        prefs.compute_device_type = device
        prefs.get_devices()
        for item in prefs.devices:
            item.use = item.type == device
        enabled = [{"type": item.type, "name": item.name, "id": item.id} for item in prefs.devices if item.use]
        if not enabled:
            raise RuntimeError(f"Requested Cycles {device} device is unavailable; choose an available explicit device")
        scene.cycles.device = "GPU"
    scene.render.film_transparent = options.get("transparent_background", True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.dither_intensity = 0
    color = options.get("color_management", {})
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = color.get("view_transform", "AgX")
    scene.view_settings.look = color.get("look", "None")
    scene.view_settings.exposure = color.get("exposure", 0)
    scene.view_settings.gamma = color.get("gamma", 1)
    world = bpy.data.worlds.new("EvaluationWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    spec = options.get("world", {})
    bg.inputs["Color"].default_value = (*spec.get("color", [1, 1, 1]), 1)
    bg.inputs["Strength"].default_value = spec.get("strength", .6)
    lights = []
    for index, spec in enumerate(options.get("lights", [])):
        data = bpy.data.lights.new(f"eval_light_{index}", "AREA")
        data.energy, data.size = spec.get("energy", 400), spec.get("size", 3)
        data.color = spec.get("color", [1, 1, 1])
        actor = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(actor)
        position = C[:3, :3] @ np.array(spec["position_y_up_m"])
        target = C[:3, :3] @ np.array(spec.get("target_y_up_m", [0, 0, 0]))
        actor.matrix_world = _look_at(position, target, [0, 0, 1] if abs(position[2]-target[2]) < np.linalg.norm(position-target)*.99 else [0, 1, 0])
        lights.append({"name": actor.name, "type": "AREA", "energy": float(data.energy),
                       "size": float(data.size), "color": list(data.color), "world_pose_z_up": [list(row) for row in actor.matrix_world]})
    return scene, {"device": device, "enabled_devices": enabled, "actual_lights": lights}


def _import_objects(objects):
    records, monitored, index_to_id, excluded = [], [], {}, []
    for index, obj in enumerate(objects, 1):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=obj["mesh"])
        imported = set(bpy.data.objects) - before
        roots = [node for node in imported if node.parent not in imported]
        transform = Matrix(y_up_pose_to_blender(obj["transform"]).tolist())
        for node in roots:
            node.matrix_world = transform @ node.matrix_world
        meshes = [node for node in imported if node.type == "MESH"]
        if not meshes:
            raise ValueError(f"Visual asset contains no mesh: {obj['id']}")
        for node in meshes:
            node.pass_index = index
            if node.rigid_body is not None:
                raise ValueError("Imported rigid bodies are unsupported for static evaluation")
        bpy.context.view_layer.update()
        point_groups = []
        for node in meshes:
            coordinates = np.empty(len(node.data.vertices) * 3, dtype=np.float32)
            node.data.vertices.foreach_get("co", coordinates)
            matrix = np.array(node.matrix_world)
            point_groups.append(coordinates.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
        points = np.concatenate(point_groups)
        index_to_id[index] = obj["id"]
        if obj.get("is_table", False): excluded.append(index)
        root_records = [{"name": node.name, "matrix_world_z_up": [list(row) for row in node.matrix_world]} for node in roots]
        for node in imported:
            monitored.append((node, np.array(node.matrix_world)))
        records.append({"id": obj["id"], "mesh": obj["mesh"], "mesh_sha256": _sha(obj["mesh"]),
                        "object_index": index, "is_table": bool(obj.get("is_table", False)),
                        "mesh_node_names": [node.name for node in meshes], "root_nodes": root_records,
                        "mesh_vertex_count": sum(len(node.data.vertices) for node in meshes),
                        "world_bounds_z_up": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                        "material_names": sorted({material.name for node in meshes for material in node.data.materials if material})})
    return records, monitored, index_to_id, excluded


def _compositor(scene, output, selected_indices):
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    links = scene.node_tree.links
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(layers.outputs["Image"], composite.inputs["Image"])
    # IndexOB is the visible first-hit object pass: nonselected table geometry
    # still occludes selected objects. No RGB/color threshold is involved.
    masks = []
    for index in selected_indices:
        node = nodes.new("CompositorNodeIDMask")
        node.index = index
        node.use_antialiasing = False
        links.new(layers.outputs["IndexOB"], node.inputs[0])
        masks.append(node.outputs["Alpha"])
    if masks:
        union = masks[0]
        for current in masks[1:]:
            node = nodes.new("CompositorNodeMath")
            node.operation = "MAXIMUM"
            links.new(union, node.inputs[0]); links.new(current, node.inputs[1])
            union = node.outputs[0]
    else:
        node = nodes.new("CompositorNodeValue")
        node.outputs[0].default_value = 0
        union = node.outputs[0]
    mask_file = nodes.new("CompositorNodeOutputFile")
    mask_file.base_path = str(output)
    mask_file.format.file_format = "PNG"
    mask_file.format.color_mode = "BW"
    mask_file.format.color_depth = "8"
    links.new(union, mask_file.inputs[0])
    id_file = nodes.new("CompositorNodeOutputFile")
    id_file.base_path = str(output)
    id_file.format.file_format = "OPEN_EXR"
    id_file.format.color_mode = "BW"
    id_file.format.color_depth = "32"
    links.new(layers.outputs["IndexOB"], id_file.inputs[0])
    return mask_file, id_file


def _read_ids(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    channels = image.channels
    ids = np.rint(values.reshape(-1, channels)[:, 0]).astype(np.int32)
    bpy.data.images.remove(image)
    present, counts = np.unique(ids, return_counts=True)
    return dict(zip(present.tolist(), counts.tolist()))


def run(request):
    options = dict(request["rendering"])
    validate_rendering_options(options)
    foreground = validate_foreground_options(request.get("foreground") or {})
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_hashes = {obj["mesh"]: _sha(obj["mesh"]) for obj in request["objects"]}
    scene, details = _configure_scene(options)
    actors, monitored, actor_ids, table_indices = _import_objects(request["objects"])
    selected = [index for index in actor_ids if foreground["include_table"] or index not in table_indices]
    mask_output, id_output = _compositor(scene, output, selected)
    data = bpy.data.cameras.new("EvaluationCamera")
    camera = bpy.data.objects.new("EvaluationCamera", data)
    scene.collection.objects.link(camera)
    views, view_details = [], []
    for view in request["views"]:
        identity = view["id"]
        info = _configure_camera(scene, camera, view["camera"], options)
        render_path = output / f"{identity}.png"
        mask_path = output / f"{identity}_mask.png"
        ids_path = output / f"{identity}_object_ids.exr"
        mask_output.file_slots[0].path = f"{identity}_mask_"
        id_output.file_slots[0].path = f"{identity}_ids_"
        scene.render.filepath = str(render_path)
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        mask_generated = output / f"{identity}_mask_{scene.frame_current:04d}.png"
        ids_generated = output / f"{identity}_ids_{scene.frame_current:04d}.exr"
        mask_generated.replace(mask_path)
        ids_generated.replace(ids_path)
        visible = _read_ids(ids_path)
        record = {**view, "render": str(render_path), "mask": str(mask_path)}
        views.append(record)
        view_details.append({**record, **info, "object_ids_exr": str(ids_path),
                             "visible_object_indices": list(visible),
                             "visible_object_pixel_counts": {str(k): v for k, v in visible.items()},
                             "visible_object_ids": [actor_ids[k] for k in visible if k in actor_ids],
                             "foreground_pixel_count": sum(v for k, v in visible.items() if k in selected),
                             "seconds": time.perf_counter()-start})
    max_drift = 0.
    for node, initial in monitored:
        max_drift = max(max_drift, float(np.abs(np.array(node.matrix_world)-initial).max()))
    if max_drift > 1e-7:
        raise RuntimeError(f"Static imported object pose changed during rendering: {max_drift}")
    for path, original in initial_hashes.items():
        if _sha(path) != original:
            raise RuntimeError("A source visual asset changed during rendering")
    details.update({"backend": "blender", "renderer": "CYCLES", "blender_version": bpy.app.version_string,
                    "blender_binary": bpy.app.binary_path, "python_version": sys.version,
                    "blender_build_hash": bpy.app.build_hash.decode(),
                    "samples": scene.cycles.samples, "denoise": scene.cycles.use_denoising,
                    "seed": scene.cycles.seed, "transparent_background": scene.render.film_transparent,
                    "world": options.get("world", {"color": [1, 1, 1], "strength": .6}),
                    "color_management": {"view_transform": scene.view_settings.view_transform,
                                         "look": scene.view_settings.look, "exposure": scene.view_settings.exposure,
                                         "gamma": scene.view_settings.gamma},
                    "actors": actors, "views": view_details, "physics_enabled": False,
                    "simulation_steps": 0, "ground_created": False,
                    "source_hashes_before_after_equal": True,
                    "max_pose_element_drift": max_drift, "after_all_views_pose_verified": True,
                    "materials": "source GLB node hierarchies and PBR materials",
                    "foreground": {**foreground, "source": "Cycles IndexOB first-hit object-index pass",
                                   "actor_ids": {str(k): v for k, v in actor_ids.items()},
                                   "selected_object_ids": [actor_ids[k] for k in selected],
                                   "visibility": "excluded objects remain occluders; no color threshold",
                                   "mask_antialiasing": False}})
    return {"views": views, "all_views": views, "details": {"rendering": details}}


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--")+1:]
    request_path, response_path = map(Path, args)
    try:
        result = run(json.loads(request_path.read_text()))
        response_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as exc:
        response_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
