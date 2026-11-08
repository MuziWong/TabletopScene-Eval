"""Discover standard scene exports and evaluate complete scenes sequentially.

Only scene_export packages are accepted. Reference/mask data is shared across
methods; partial method joins and failed scenes never enter aggregate scores.
"""
from copy import deepcopy
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


class BatchPlanError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(f"Batch input validation failed ({len(issues)} issues): " +
                         "; ".join(item["message"] for item in issues[:4]))


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(exc, **context):
    # SDK exception messages may contain a gateway URL; logs need the cause,
    # not transport credentials or an endpoint address.
    message = re.sub(r"https?://\S+", "<URL omitted>", str(exc))
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "<credential omitted>", message)
    return {**context, "error_type": type(exc).__name__, "message": message}


def _scene_path(scene_id):
    path = Path(scene_id)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Unsafe relative scene ID: {scene_id}")
    return path


def _discover(root, output, method):
    from .adapters.scene_export import load_scene_export
    root = Path(root).expanduser().resolve()
    if root.is_file() and root.name == "result.json":
        root = root.parent
    if not root.is_dir():
        return {}, [_error(FileNotFoundError(f"Input root not found: {root}"), method=method, stage="discovery")]
    if root == output or root.is_relative_to(output):
        raise ValueError("The output directory must not contain an input root")
    if root.name == "scene_export":
        files, base = [root/"result.json"], root.parent
    else:
        files, base = sorted(root.rglob("scene_export/result.json")), root
    found, issues = {}, []
    for path in files:
        # This is a destination error, never a skippable input error. Check
        # before loading or collecting issues so even invalid/continued exports
        # cannot receive discovery logs or automatically generated masks.
        if output.is_relative_to(path.parent.resolve()):
            raise ValueError("Output must be outside every source scene_export package")
        if path.is_relative_to(output):
            continue
        case = path.parent.parent
        relative = case.relative_to(base)
        identity = case.name if str(relative) == "." else relative.as_posix()
        try:
            exported = load_scene_export(path)
            if identity in found:
                raise ValueError(f"Duplicate relative scene ID: {identity}")
            found[identity] = exported
        except Exception as exc:
            issues.append(_error(exc, stage="discovery", method=method,
                                 scene_id=identity, path=str(path)))
    if not files:
        issues.append(_error(ValueError(f"No scene_export/result.json found under {root}"),
                             stage="discovery", method=method))
    return found, issues


def _match(root, pattern, scene_id, label):
    root = Path(root).expanduser().resolve()
    formatted = pattern.format(scene_id=scene_id, scene_name=Path(scene_id).name)
    if Path(formatted).is_absolute() or ".." in Path(formatted).parts:
        raise ValueError(f"{label} pattern must stay relative to its root")
    matches = sorted(path.resolve() for path in root.glob(formatted) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one match for {root / formatted}, found {len(matches)}")
    return matches[0]


def _auto_mask(case, reference, destination, foreground):
    import numpy as np
    from PIL import Image
    analysis = case/"input_analysis"/"success"
    original, graph_path = analysis/"original.png", analysis/"scene_graph.json"
    if not original.is_file() or _sha(original) != _sha(reference):
        raise ValueError("Automatic masks require input_analysis/success/original.png identical to the shared reference")
    graph = json.loads(graph_path.read_text())
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("Automatic masks require a nonempty scene_graph.json nodes mapping")
    with Image.open(reference) as image:
        size = image.size
    union, records = np.zeros(size[::-1], dtype=bool), []
    for identity, node in sorted(nodes.items()):
        if node.get("type") == "subscene":
            continue
        table = (identity == "table" or node.get("is_table") is True
                 or str(node.get("category", "")).lower() == "table")
        if table and not foreground["include_table"]:
            continue
        value = node.get("mask_path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Automatic mask is missing mask_path for node {identity}")
        path = Path(value)
        path = path if path.is_absolute() else analysis/path
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Automatic mask dimensions differ from the reference: {path}")
            union |= np.asarray(image.convert("L")) > foreground["mask_threshold"]
        records.append({"id": identity, "path": str(path.resolve()), "sha256": _sha(path), "is_table": table})
    if not records or not union.any():
        raise ValueError("Automatic reference foreground mask is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype("uint8")*255).save(destination)
    provenance = {"automatic_pipeline_mask_not_human_gt": True,
                  "include_table": foreground["include_table"],
                  "reference_sha256": _sha(reference), "graph_sha256": _sha(graph_path),
                  "mask_sources": records, "mask_sha256": _sha(destination),
                  "foreground_fraction": float(union.mean())}
    _save(destination.with_suffix(".provenance.json"), provenance)
    return provenance


def build_batch_plan(base_config, methods, output_dir, *, reference_root=None,
                     reference_pattern="{scene_id}.png", mask_root=None,
                     mask_pattern="{scene_id}.png", auto_mask=False,
                     reference=None, reference_mask=None, invalid_policy="error"):
    """Return a plan with scene_export-only samples and separate input metadata.

    A method root may be one case, its scene_export directory, or an output
    hierarchy. IDs are deterministic relative case paths (e.g. ``2/02-01``).
    Explicit reference/mask files are permitted only when one case is found.
    No implicit mask filename is guessed and no old score cache is resumed.
    """
    from .foreground import validate_foreground_options
    if not isinstance(methods, dict) or not methods or any(not isinstance(m, str) or not m.strip() for m in methods):
        raise ValueError("methods must map nonempty method names to input roots")
    if invalid_policy not in {"error", "continue"}:
        raise ValueError("invalid_policy must be error or continue")
    if not isinstance(auto_mask, bool):
        raise ValueError("auto_mask must be boolean")
    if reference is not None and reference_root is not None:
        raise ValueError("Use reference or reference_root, not both")
    if sum([reference_mask is not None, mask_root is not None, auto_mask]) > 1:
        raise ValueError("Use only one of reference_mask, mask_root or auto_mask")
    if not isinstance(base_config, dict) or base_config.get("samples"):
        raise ValueError("Batch base_config must be a protocol without samples")
    output = Path(output_dir).expanduser().resolve()
    if any((output/name).exists() for name in ("results.json", "summary.json")):
        raise FileExistsError("Output already contains evaluation results; choose a fresh --output (resume is unsupported)")
    base = deepcopy(base_config)
    config_sources = base.pop("_config_sources", base.pop("config_sources", []))
    base.pop("config_path", None)
    foreground = validate_foreground_options(base.get("foreground", {}))
    needs_images = any(base.get("metrics", {}).get(category, {}).get("enabled", False)
                       for category in ("input_view", "gpt_evaluation"))
    if not needs_images:
        # A physics-only run must not depend on image or mask availability.
        foreground["enabled"] = False
    base["foreground"] = foreground
    exports, issues = {}, []
    for method, root in methods.items():
        exports[method], errors = _discover(root, output, method)
        issues.extend(errors)
    all_ids = sorted(set().union(*(set(found) for found in exports.values())))
    if needs_images and (reference is not None or reference_mask is not None) and len(all_ids) != 1:
        raise ValueError("Explicit --reference/--mask files require one scene; use --reference-root/--mask-root for a batch")
    samples, metadata = [], {}
    primary = next(iter(methods))
    for scene_id in all_ids:
        try:
            missing = [method for method in methods if scene_id not in exports[method]]
            if missing:
                raise ValueError(f"Scene {scene_id} is missing valid exports for methods: {', '.join(missing)}")
            case = exports[primary][scene_id].root.parent
            sample = {"id": scene_id, "methods": {
                method: {"scene_export": str(exports[method][scene_id].root)} for method in methods}}
            record = {}
            if needs_images:
                ref = (Path(reference).expanduser().resolve() if reference is not None else
                       _match(reference_root, reference_pattern, scene_id, "Reference") if reference_root is not None else
                       case/"input_analysis"/"success"/"original.png")
                if not ref.is_file():
                    raise FileNotFoundError(f"Shared reference not found: {ref}; provide --reference or --reference-root")
                sample["reference"] = str(ref)
                record.update(reference_sha256=_sha(ref), reference_source_method=primary)
            if foreground["enabled"]:
                if reference_mask is not None:
                    mask = Path(reference_mask).expanduser().resolve()
                elif mask_root is not None:
                    mask = _match(mask_root, mask_pattern, scene_id, "Reference mask")
                elif auto_mask:
                    relative = _scene_path(scene_id)
                    mask = output/"masks"/relative.parent/(relative.name + ".png")
                    record["auto_mask"] = _auto_mask(case, ref, mask, foreground)
                else:
                    raise ValueError("Foreground evaluation needs --mask, --mask-root or explicit --auto-mask; no mask filename is guessed")
                if not mask.is_file():
                    raise FileNotFoundError(f"Shared reference mask not found: {mask}")
                from PIL import Image
                with Image.open(ref) as image, Image.open(mask) as masked:
                    if image.size != masked.size:
                        raise ValueError("Shared reference and mask must have identical dimensions")
                sample["reference_mask"] = str(mask)
                record["reference_mask_sha256"] = _sha(mask)
            samples.append(sample)
            metadata[scene_id] = record
        except Exception as exc:
            issues.append(_error(exc, stage="sample", scene_id=scene_id))
    if not samples and not issues:
        issues.append(_error(ValueError("No complete valid scenes were discovered"), stage="discovery"))
    discovery = {"input_roots": {m: str(Path(root).expanduser().resolve()) for m, root in methods.items()},
                 "scene_ids": [sample["id"] for sample in samples], "issues": issues,
                 "input_metadata": metadata, "invalid_policy": invalid_policy,
                 "config_sources": config_sources}
    _save(output/"discovery.json", discovery)
    if issues and invalid_policy == "error":
        raise BatchPlanError(issues)
    return {"base_config": base, "samples": samples, "output_dir": str(output), "discovery": discovery}


def _complete_values(result, sample, config):
    rows = result.get("samples", [])
    if len(rows) != 1 or rows[0].get("id") != sample["id"] or set(rows[0].get("methods", {})) != set(sample["methods"]):
        raise ValueError("Evaluator must return exactly the requested complete scene and every method")
    expected = {"input_view": config.get("metrics", {}).get("input_view", {}).get("metrics", ["LPIPS", "DINOv2", "CLIP"]),
                "gpt_evaluation": ["VF", "IA", "PP", "Avg.", "OR"],
                "physical_validity": ["ColO", "ColS"]}
    values = []
    for method, groups in rows[0]["methods"].items():
        for category, keys in expected.items():
            if not config.get("metrics", {}).get(category, {}).get("enabled", False):
                continue
            scores = groups.get(category)
            if not isinstance(scores, dict) or set(scores) != set(keys):
                raise ValueError(f"Incomplete {category} scores for {method}")
            for metric, value in scores.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {metric} score for {method}")
                values.append(((method, category, metric), float(value)))
    sweeps = []
    options = config.get("metrics", {}).get("physical_validity", {}).get("options", {})
    requested = options.get("penetration_thresholds_m")
    result_sweeps = result.get("physical_validity_threshold_sweep", {})
    if config.get("metrics", {}).get("physical_validity", {}).get("enabled") and requested is not None:
        if set(result_sweeps) != set(sample["methods"]):
            raise ValueError("Physical threshold results must include every method")
        for method, entries in result_sweeps.items():
            if sorted(entry["threshold_m"] for entry in entries) != sorted(requested):
                raise ValueError(f"Incomplete physical thresholds for {method}")
            for entry in entries:
                for metric in ("ColO", "ColS"):
                    stats = entry[metric]
                    value = stats.get("mean")
                    if stats.get("count") != 1 or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Expected one finite scene score per physical threshold")
                    sweeps.append(((method, entry["threshold_m"], metric), float(value)))
    return values, sweeps


def run_batch(plan, run_callable, *, continue_on_error=False):
    """Run one fresh scene config at a time; macro-average complete successes.

    ``run_callable(config_dict)`` returns the ordinary evaluator result with one
    samples row. The caller owns config loading and model/API construction.
    """
    output = Path(plan["output_dir"])
    if any((output/name).exists() for name in ("results.json", "summary.json")):
        raise FileExistsError("Output already contains evaluation results; resume is unsupported")
    collected, sweeps = defaultdict(list), defaultdict(list)
    completed, errors = [], deepcopy(plan["discovery"]["issues"])
    for sample in plan["samples"]:
        directory = output/"scenes"/_scene_path(sample["id"])
        config = deepcopy(plan["base_config"])
        config["samples"] = [deepcopy(sample)]
        config.setdefault("rendering", {})["output_dir"] = str(directory/"artifacts")
        _save(directory/"config.json", config)
        try:
            result = run_callable(config)
            values, thresholds = _complete_values(result, sample, config)
            _save(directory/"result.json", {"status": "complete", "scene_id": sample["id"], "result": result})
            for key, value in values:
                collected[key].append(value)
            for key, value in thresholds:
                sweeps[key].append(value)
            completed.append(sample["id"])
        except Exception as exc:
            record = _error(exc, stage="evaluation", scene_id=sample["id"])
            errors.append(record)
            _save(directory/"result.json", {"status": "failed", "scene_id": sample["id"], "error": record})
            if not continue_on_error:
                break
    summary = {}
    for (method, category, metric), values in sorted(collected.items()):
        summary.setdefault(method, {}).setdefault(category, {})[metric] = {"mean": sum(values)/len(values), "count": len(values)}
    threshold_summary = {}
    for method in plan["samples"][0]["methods"] if plan["samples"] else []:
        rows = []
        for threshold in sorted({key[1] for key in sweeps if key[0] == method}):
            row = {"threshold_m": threshold, "threshold_mm": threshold*1000}
            for metric in ("ColO", "ColS"):
                values = sweeps[(method, threshold, metric)]
                row[metric] = {"mean": sum(values)/len(values), "count": len(values)}
            rows.append(row)
        if rows:
            threshold_summary[method] = rows
    report = {"status": ("completed_with_errors" if completed and errors else "complete" if completed else "failed"),
              "output_dir": str(output), "discovery": plan["discovery"],
              "completed_scenes": completed, "errors": errors,
              "not_run_scenes": [sample["id"] for sample in plan["samples"]
                                 if sample["id"] not in completed and not any(error.get("scene_id") == sample["id"] for error in errors)],
              "summary": summary, "physical_validity_threshold_sweep": threshold_summary,
              "aggregation": "unweighted macro mean across complete successful scenes only"}
    _save(output/"summary.json", report)
    return report
