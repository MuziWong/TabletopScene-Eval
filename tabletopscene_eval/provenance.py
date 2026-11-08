"""Run provenance without importing heavy models or recording credentials."""
import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect_provenance(config):
    packages = {}
    for name in ("numpy", "scipy", "Pillow", "torch", "torchvision", "lpips", "transformers", "openai", "python-dotenv", "trimesh", "python-fcl", "vhacdx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = set()
    for sample in config["samples"]:
        for key in ("reference", "reference_mask", "foreground_reference"):
            if sample.get(key):
                files.add(Path(sample[key]))
        for result in sample["methods"].values():
            for key in ("render", "render_mask", "foreground_render", "scene"):
                if result.get(key):
                    files.add(Path(result[key]))
            files.update(Path(view["render"]) for view in result.get("views", []))
            files.update(Path(view["mask"]) for view in result.get("views", []) if view.get("mask"))
            files.update(Path(view["render"]) for view in result.get("foreground_views", []))
    source = config.get("config_path")
    if source:
        files.add(Path(source))
    files.update(Path(source) for source in config.get("config_sources", []))
    protocols = {}
    for category, entry in config["metrics"].items():
        if entry["enabled"]:
            protocols[category] = {k: v for k, v in entry.items() if k != "options"}
            protocols[category]["options"] = {k: v for k, v in entry["options"].items() if k not in {"base_url", "audit_dir"}}
    from tabletopscene_eval.image_metrics import DEFAULT_OPTIONS, RUNTIME_INFO
    if "input_view" in protocols:
        protocols["input_view"]["options"] = {**DEFAULT_OPTIONS, **protocols["input_view"]["options"]}
        protocols["input_view"]["runtime"] = dict(RUNTIME_INFO)
    if "gpt_evaluation" in protocols:
        from metrics.gpt_evaluation import RUNTIME_JUDGMENTS
        opts = protocols["gpt_evaluation"]["options"]
        effective_model = RUNTIME_JUDGMENTS[-1]["requested_model"] if RUNTIME_JUDGMENTS else opts.get("model")
        protocols["gpt_evaluation"]["options"] = {"image_detail": "high", **opts, "order_seed": opts.get("order_seed", opts.get("seed", 0))}
        if effective_model:
            protocols["gpt_evaluation"]["options"]["model"] = effective_model
        protocols["gpt_evaluation"]["runtime"] = list(RUNTIME_JUDGMENTS)
        prompt_file = protocols["gpt_evaluation"]["options"].get("prompt_file")
        if prompt_file:
            files.add(Path(prompt_file))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "code_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): _digest(path)
            for package in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / "metrics")
            for path in sorted(package.rglob("*.py"))
        },
        "protocols": protocols,
        "rendering": config.get("rendering", {}),
        "camera_sampling": config.get("camera_sampling", {}),
        "foreground": config.get("foreground", {"enabled": False}),
        "visual_normalization": config.get("visual_normalization", {"enabled": True}),
        "config_sources": config.get("config_sources", []),
        "artifact_directory": config.get("artifact_directory"),
        "aggregation": "arithmetic mean over the same scenes for every method (macro average)",
        "sha256": {str(path): _digest(path) for path in sorted(files)},
        "hash_scope": "config, reference/render images, masks, foreground composites, explicit prompt files and manifests; physical details also record collision mesh SHA256; remote model weights are not hashed",
    }
