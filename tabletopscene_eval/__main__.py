"""One command for a scene_export folder, a batch, or an explicit experiment."""
import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .profiles import load_composed_config


def _default_config():
    source = Path(__file__).resolve().parents[1] / "configs/default.json"
    return source if source.is_file() else Path(sys.prefix) / "share/tabletopscene-eval/configs/default.json"


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _check_output(output, config):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output must be a fresh or empty directory; choose a fresh --output")
    for sample in config.get("samples", []):
        for result in sample.get("methods", {}).values():
            value = result.get("scene_export")
            if not isinstance(value, str):
                continue
            source = Path(value).expanduser().resolve()
            if source.name == "result.json":
                source = source.parent
            if output.is_relative_to(source):
                raise ValueError("Output must be outside every source scene_export package")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tabletopscene-eval", description="Automatically evaluate AgenTable scene_export folders")
    parser.add_argument("input", nargs="?", help="scene_export folder, result.json, or an output root with --batch")
    parser.add_argument("--batch", action="store_true", help="Recursively discover and evaluate scene_export packages")
    parser.add_argument("--method", action="append", default=[], metavar="NAME=ROOT", help="Named method root; repeat to compare methods")
    parser.add_argument("--config", default=str(_default_config()), help="Protocol JSON or experiment with samples")
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    parser.add_argument("--reference", type=Path, help="Original image for one scene")
    parser.add_argument("--reference-root", type=Path, help="Original image directory for a batch")
    parser.add_argument("--reference-pattern", default="{scene_id}.png", help="Image path pattern relative to reference-root")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--mask", type=Path, help="Foreground mask for one scene")
    masks.add_argument("--mask-root", type=Path, help="Foreground mask directory for a batch")
    masks.add_argument("--auto-mask", action="store_true", help="Explicitly union existing AgenTable input-analysis masks")
    parser.add_argument("--mask-pattern", default="{scene_id}.png", help="Mask path pattern relative to mask-root")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures and continue; exit remains nonzero if any scene fails")
    parser.add_argument("--no-gpt", action="store_true", help="Skip remote GPT evaluation")
    parser.add_argument("--physics-only", action="store_true", help="Evaluate original-scale collisions without images, Blender or API")
    parser.add_argument("--env-file", type=Path, help="Private API environment file override")
    parser.add_argument("--check-api", action="store_true", help="Run a small sanitized text/JSON API connectivity probe")
    args = parser.parse_args(argv)
    try:
        base = load_composed_config(args.config)
        if args.env_file:
            base.setdefault("metrics", {}).setdefault("gpt_evaluation", {}).setdefault("options", {})["env_file"] = str(args.env_file.expanduser().resolve())
        if args.check_api:
            from .api import check_api_connection
            result = check_api_connection(base.get("metrics", {}).get("gpt_evaluation", {}).get("options", {}))
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["ok"] else 1
        if not args.output:
            parser.error("--output is required for evaluation")
        output = args.output.expanduser().resolve()
        _check_output(output, base)
        if args.no_gpt or args.physics_only:
            base.setdefault("metrics", {}).setdefault("gpt_evaluation", {})["enabled"] = False
        if args.physics_only:
            base["metrics"].setdefault("input_view", {})["enabled"] = False
            base["metrics"].setdefault("physical_validity", {})["enabled"] = True
            base.setdefault("foreground", {})["enabled"] = False
        from .runner import evaluate
        from .batch import build_batch_plan, run_batch, _error
        if not args.input and not args.method and base.get("samples"):
            sources = base.pop("_config_sources", [])
            base.setdefault("rendering", {})["output_dir"] = str(output/"artifacts")
            _save(output/"config.json", base)
            result = evaluate(load_config(output/"config.json"))
            result["provenance"]["source_protocol_files"] = sources
            _save(output/"results.json", result)
            print(f"Results: {output/'results.json'}")
            return 0
        if args.input and args.method:
            raise ValueError("Use positional input or --method NAME=ROOT, not both")
        methods = {}
        if args.input:
            methods["AgenTable"] = args.input
        for item in args.method:
            name, separator, root = item.partition("=")
            if not separator or not name.strip() or not root.strip() or name in methods:
                raise ValueError("--method requires a unique NAME=ROOT")
            methods[name] = root
        if not methods:
            parser.error("Provide an input folder, --method NAME=ROOT, or a --config with samples")
        if not args.batch:
            for root in methods.values():
                p = Path(root).expanduser()
                if not (p.name in {"scene_export", "result.json"} or (p/"scene_export/result.json").is_file()):
                    raise ValueError("An output hierarchy requires --batch; one scene accepts its case or scene_export folder")
        base.pop("samples", None)
        plan = build_batch_plan(base, methods, output,
            reference_root=args.reference_root, reference_pattern=args.reference_pattern,
            mask_root=args.mask_root, mask_pattern=args.mask_pattern, auto_mask=args.auto_mask,
            reference=args.reference, reference_mask=args.mask,
            invalid_policy="continue" if args.continue_on_error else "error")
        if not args.batch and len(plan["samples"]) > 1:
            raise ValueError("Multiple scenes found; add --batch")
        def run(config):
            path = Path(config["rendering"]["output_dir"]).parent/"config.json"
            checked = load_config(path)
            result = evaluate(checked)
            result["provenance"]["source_protocol_files"] = plan["discovery"].get("config_sources", [])
            print(f"Completed: {config['samples'][0]['id']}", flush=True)
            return result
        report = run_batch(plan, run, continue_on_error=args.continue_on_error)
        print(f"Completed {len(report['completed_scenes'])} scene(s), {len(report['errors'])} error(s). Results: {output/'summary.json'}")
        return 0 if report["status"] == "complete" else 1
    except Exception as exc:
        from .batch import _error
        error = _error(exc)
        print(f"TabletopScene-Eval: {error['error_type']}: {error['message']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
