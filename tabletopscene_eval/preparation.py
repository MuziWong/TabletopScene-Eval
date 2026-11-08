"""Prepare read-only scene exports for Blender images and original-scale physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4


def _slug(value):
    return (re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")[:48] or "item") + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _file(value):
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"Renderer did not produce {path}")
    return path


def prepare_evaluation(config):
    from .adapters.scene_export import load_scene_export
    from .camera_geometry import resolve_shared_table_camera_sampling
    from .normalization import normalize_scene_objects, validate_visual_normalization
    from .foreground import validate_foreground_options, composite_foreground
    from .rendering.blender_backend import prepare_scene_export
    working = deepcopy(config)
    categories = working["metrics"]
    foreground = validate_foreground_options(working.get("foreground", {}))
    normalization = validate_visual_normalization({"enabled": True, **working.get("visual_normalization", {})})
    working.update(foreground=foreground, visual_normalization=normalization)
    visual = categories["input_view"]["enabled"] or categories["gpt_evaluation"]["enabled"]
    exports = {(s["id"], name): load_scene_export(result["scene_export"])
               for s in working["samples"] for name, result in s["methods"].items()}
    source_roots = [export.root for export in exports.values()]
    base = Path(working.get("rendering", {}).get("output_dir") or Path(working.get("config_path", Path.cwd() / "experiment.json")).parent / "runs").expanduser().resolve()
    cache = categories["physical_validity"]["options"].get("cache_dir")
    for destination in [base] + ([Path(cache).expanduser().resolve()] if cache else []):
        if any(destination.is_relative_to(root) for root in source_roots):
            raise ValueError("Output/cache directories must be outside every source scene_export")
    artifacts = base / ("run_" + uuid4().hex[:12])
    artifacts.mkdir(parents=True, exist_ok=False)
    working["artifact_directory"] = str(artifacts)
    for sample in working["samples"]:
        sample_dir = artifacts / _slug(sample["id"])
        sample_dir.mkdir()
        options = {k: v for k, v in working.get("rendering", {}).items() if k != "output_dir"}
        if visual and options.get("resolution") == "reference":
            from PIL import Image
            with Image.open(sample["reference"]) as image:
                options["resolution"] = list(image.size)
        sampling = sample.get("camera_sampling", working.get("camera_sampling", {"mode": "export_camera"}))
        framing = None
        if visual and "framing" in sampling:
            objects = {name: normalize_scene_objects(exports[(sample["id"], name)].objects, normalization)[0]
                       for name in sample["methods"]}
            sampling, framing = resolve_shared_table_camera_sampling(sampling, objects,
                options.get("resolution", [512, 512]), options.get("camera", {}).get("vertical_fov_deg", 45))
        reference_details = None
        if visual and foreground["enabled"]:
            destination = sample_dir / "reference_foreground.png"
            reference_details = composite_foreground(sample["reference"], sample["reference_mask"], destination, foreground)
            sample["foreground_reference"] = destination
        for name, result in sample["methods"].items():
            exported = exports[(sample["id"], name)]
            output = sample_dir / _slug(name)
            output.mkdir()
            result["preparation"] = {"scene_export": exported.provenance}
            if framing:
                result["preparation"]["camera_framing"] = framing
            if categories["physical_validity"]["enabled"]:
                manifest = output / "object_meshes.json"
                manifest.write_text(json.dumps(exported.collision_manifest(), indent=2, allow_nan=False))
                result["scene"] = manifest
            if visual:
                prepared = prepare_scene_export(exported.root, output, sampling, options,
                    foreground_options=foreground, visual_normalization=normalization)
                result["preparation"]["backend"] = prepared.get("details", {})
                result["render"] = _file(prepared["render"])
                if result.pop("generate_views", False):
                    result["views"] = [{**v, "render": _file(v["render"])} for v in prepared["views"]]
                if foreground["enabled"]:
                    result["render_mask"] = _file(prepared["render_mask"])
                    result["foreground_render"] = output / "foreground_render.png"
                    composite_foreground(result["render"], result["render_mask"], result["foreground_render"], foreground,
                        allow_empty=True, expected_size=reference_details["size"])
                    if result.get("views"):
                        result["foreground_views"] = []
                        for view in result["views"]:
                            path = output / ("foreground_" + _slug(view["id"]) + ".png")
                            composite_foreground(view["render"], view["mask"], path, foreground,
                                allow_empty=True, expected_size=reference_details["size"])
                            result["foreground_views"].append({**view, "source_render": str(view["render"]), "render": path})
            result.pop("generate_views", None)
        if categories["input_view"]["enabled"] and categories["input_view"]["view_selection"] != "fixed":
            signature = None
            for result in sample["methods"].values():
                views = result.get("views", [])
                current = {v["id"]: v["camera"] for v in views}
                expected = categories["input_view"].get("expected_view_count")
                if not views or len(current) != len(views) or (expected is not None and len(views) != expected):
                    raise ValueError("Renderer returned an invalid candidate count or duplicate IDs")
                if signature is not None and current != signature:
                    raise ValueError("Methods must share identical candidate cameras")
                signature = current
    return working
