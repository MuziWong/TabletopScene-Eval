"""Validate the single supported input: portable AgenTable scene_export packages."""
from pathlib import Path

from .profiles import load_composed_config


class ConfigError(ValueError):
    """An experiment cannot be evaluated with the selected protocol."""


CATEGORIES = ("input_view", "gpt_evaluation", "physical_validity")
INPUT_METRICS = ("LPIPS", "DINOv2", "CLIP")


def _mapping(value, location):
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be an object")
    return value


def _path(value, base, location):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty path")
    path = (base / Path(value).expanduser()).resolve()
    if not path.is_file():
        raise ConfigError(f"{location}: file does not exist: {path}")
    return path


def load_config(filename):
    """Load a composed experiment; resolve and validate every source package."""
    path = Path(filename).expanduser().resolve()
    try:
        raw = load_composed_config(path)
        from .foreground import validate_foreground_options
        from .normalization import validate_visual_normalization
        from .rendering.blender_backend import validate_rendering_options
        from .camera_geometry import validate_framing_options
        from .rendering.cameras import make_view_specs
        foreground = validate_foreground_options(raw.get("foreground", {}))
        normalization = validate_visual_normalization({"enabled": True, **_mapping(raw.get("visual_normalization", {}), "visual_normalization")})
        rendering = dict(_mapping(raw.get("rendering", {}), "rendering"))
        checked = {k: v for k, v in rendering.items() if k != "output_dir"}
        if checked.get("resolution") == "reference":
            checked["resolution"] = [512, 512]
        validate_rendering_options(checked)
        sampling = _mapping(raw.get("camera_sampling", {"mode": "export_camera"}), "camera_sampling")
        metrics = _mapping(raw.get("metrics"), "metrics")
        if set(metrics) - set(CATEGORIES):
            raise ConfigError("Unknown metric categories")
        categories = {}
        for name in CATEGORIES:
            entry = _mapping(metrics.get(name, {}), f"metrics.{name}")
            enabled = entry.get("enabled", False)
            if not isinstance(enabled, bool):
                raise ConfigError(f"metrics.{name}.enabled must be boolean")
            options = dict(_mapping(entry.get("options", {}), f"metrics.{name}.options"))
            category = {"enabled": enabled, "options": options}
            if name == "input_view":
                from tabletopscene_eval.image_metrics import validate_options
                validate_options(options)
                names = entry.get("metrics", list(INPUT_METRICS))
                if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                        or any(n not in INPUT_METRICS for n in names)):
                    raise ConfigError("input_view.metrics must contain unique LPIPS, DINOv2 or CLIP names")
                selection = entry.get("view_selection", "fixed")
                if selection not in ("fixed", "best_per_metric", "mean"):
                    raise ConfigError("input_view.view_selection must be fixed, best_per_metric or mean")
                category.update(metrics=names, view_selection=selection)
                count = entry.get("expected_view_count")
                if count is not None:
                    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                        raise ConfigError("expected_view_count must be a positive integer")
                    category["expected_view_count"] = count
            elif name == "physical_validity":
                from metrics.physical_validity.metric import validate_options
                if entry.get("backend", "vhacd_fcl") != "vhacd_fcl":
                    raise ConfigError("physical_validity.backend must be vhacd_fcl")
                validate_options(options)
                category["backend"] = "vhacd_fcl"
            else:
                from metrics.gpt_evaluation import validate_options
                validate_options(options)
                for field in ("env_file", "prompt_file"):
                    if options.get(field) and (enabled or field == "prompt_file"):
                        options[field] = str(_path(options[field], path.parent, f"gpt_evaluation.{field}"))
                if options.get("audit_dir"):
                    options["audit_dir"] = str((path.parent / Path(options["audit_dir"]).expanduser()).resolve())
            categories[name] = category
        if not any(c["enabled"] for c in categories.values()):
            raise ConfigError("Enable at least one metric category")
        visual = categories["input_view"]["enabled"] or categories["gpt_evaluation"]["enabled"]
        if not visual:
            foreground["enabled"] = False
        samples = raw.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ConfigError("samples must be a non-empty list")
        resolved, ids, method_set = [], set(), None
        from .adapters.scene_export import load_scene_export
        for index, sample in enumerate(samples):
            location = f"samples[{index}]"
            sample = _mapping(sample, location)
            identity = sample.get("id")
            if not isinstance(identity, str) or not identity.strip() or identity in ids:
                raise ConfigError(f"{location}.id must be unique and non-empty")
            ids.add(identity)
            methods = _mapping(sample.get("methods"), f"{location}.methods")
            if not methods or any(not isinstance(n, str) or not n.strip() for n in methods):
                raise ConfigError(f"{location}.methods requires named scene_export methods")
            if method_set is not None and set(methods) != method_set:
                raise ConfigError("All samples must use the same method set")
            method_set = set(methods)
            camera = {**sampling, **_mapping(sample.get("camera_sampling", {}), f"{location}.camera_sampling")}
            preview = dict(camera)
            if "framing" in preview:
                validate_framing_options(preview.pop("framing"))
                if camera.get("mode") != "orbit_grid" or "target_y_up_m" in camera or "radius_m" in camera:
                    raise ConfigError("Automatic table framing requires orbit_grid without explicit target/radius")
                preview.update(target_y_up_m=[0, 0, 0], radius_m=1)
            if visual and categories["gpt_evaluation"]["enabled"] and camera.get("mode") == "orbit_grid" and not camera.get("anchor_view_id"):
                raise ConfigError("GPT orbit evaluation requires an explicit shared anchor_view_id")
            entry = {"id": identity, "reference": None, "methods": {}, "camera_sampling": camera}
            if visual:
                entry["reference"] = _path(sample.get("reference"), path.parent, f"{location}.reference")
                if foreground["enabled"]:
                    entry["reference_mask"] = _path(sample.get("reference_mask"), path.parent, f"{location}.reference_mask")
            for name, source in methods.items():
                source = _mapping(source, f"{location}.methods.{name}")
                if set(source) != {"scene_export"}:
                    raise ConfigError(f"{location}.methods.{name}: only scene_export is supported")
                value = source["scene_export"]
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError("scene_export must be a non-empty path")
                exported = load_scene_export((path.parent / Path(value).expanduser()).resolve())
                item = {"scene_export": exported.root}
                if visual:
                    views = make_view_specs(preview, exported.camera, [512, 512])
                    if camera.get("anchor_view_id") and camera["anchor_view_id"] not in {v["id"] for v in views}:
                        raise ConfigError("anchor_view_id is not a configured view")
                    selection = categories["input_view"].get("view_selection", "fixed")
                    if categories["input_view"]["enabled"]:
                        if selection == "fixed" and len(views) != 1 and not camera.get("anchor_view_id"):
                            raise ConfigError("Fixed evaluation needs an export camera or an explicit anchor")
                        count = categories["input_view"].get("expected_view_count")
                        actual = 1 if selection == "fixed" else len(views)
                        if count is not None and actual != count:
                            raise ConfigError(f"Expected {count} views, got {actual}")
                        if selection != "fixed":
                            item["generate_views"] = True
                entry["methods"][name] = item
            resolved.append(entry)
        return {"metrics": categories, "samples": resolved, "config_path": path,
                "rendering": rendering, "camera_sampling": sampling, "foreground": foreground,
                "visual_normalization": normalization, "config_sources": raw.get("_config_sources", [str(path)])}
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
