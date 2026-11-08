"""Evaluate configured samples and aggregate scene-level metrics by method."""
from __future__ import annotations

import math
from collections import defaultdict

from .view_selection import evaluate_views


class EvaluationError(RuntimeError):
    """An evaluator failed or returned incomplete results."""


def _checked_values(values, expected, location):
    if not isinstance(values, dict) or set(values) != set(expected):
        raise EvaluationError(f"{location}: expected metric keys {list(expected)}, got {list(values) if isinstance(values, dict) else type(values).__name__}")
    checked = {}
    for key in expected:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise EvaluationError(f"{location}: {key} must be a finite number, got {value!r}")
        checked[key] = float(value)
    return checked


def _checked_threshold_sweep(details, options, location):
    expected = options.get("penetration_thresholds_m")
    if expected is None:
        return []
    sweep = details.get("threshold_sweep")
    if not isinstance(sweep, list):
        raise EvaluationError(f"{location}: configured penetration threshold sweep is missing")
    checked, seen = [], set()
    for row in sweep:
        if not isinstance(row, dict):
            raise EvaluationError(f"{location}: threshold sweep rows must be objects")
        threshold = row.get("threshold_m")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold not in expected or threshold in seen):
            raise EvaluationError(f"{location}: invalid or duplicate threshold in sweep")
        seen.add(threshold)
        values = _checked_values({key: row.get(key) for key in ("ColO", "ColS")},
                                 ("ColO", "ColS"), f"{location}, threshold {threshold} m")
        if not 0 <= values["ColO"] <= 100 or values["ColS"] not in (0, 100):
            raise EvaluationError(f"{location}: threshold sweep needs scene-level ColO in [0,100] and binary ColS")
        checked.append((float(threshold), values))
    if seen != set(expected):
        raise EvaluationError(f"{location}: threshold sweep does not cover every configured threshold")
    return checked


def evaluate(config: dict) -> dict:
    from .preparation import prepare_evaluation
    try:
        config = prepare_evaluation(config)
    except Exception as exc:
        raise EvaluationError(f"Scene preparation failed: {exc}") from exc
    categories = config["metrics"]
    # Import only enabled evaluators; their optional dependencies are loaded on use.
    if categories["input_view"]["enabled"]:
        from tabletopscene_eval.image_metrics import evaluate_input_view, RUNTIME_INFO
        RUNTIME_INFO.clear()
    if categories["gpt_evaluation"]["enabled"]:
        from metrics.gpt_evaluation import evaluate_gpt, RUNTIME_JUDGMENTS
        RUNTIME_JUDGMENTS.clear()
    if categories["physical_validity"]["enabled"]:
        from metrics.physical_validity.metric import evaluate_physical, RUNTIME_DETAILS
        RUNTIME_DETAILS.clear()
    rows = []
    collected = defaultdict(lambda: defaultdict(list))
    physical_sweeps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample in config["samples"]:
        method_results = {name: {} for name in sample["methods"]}
        details = {}
        sample_id = sample["id"]
        for name, result in sample["methods"].items():
            if result.get("preparation"):
                details.setdefault(name, {})["preparation"] = result["preparation"]
            if categories["input_view"]["enabled"]:
                loc = f"sample {sample_id!r}, method {name!r}, input_view"
                try:
                    image_result = result
                    reference = sample["reference"]
                    if config.get("foreground", {}).get("enabled"):
                        reference = sample["foreground_reference"]
                        image_result = {"render": result.get("foreground_render"), "views": result.get("foreground_views", [])}
                    values, view_details = evaluate_views(reference, image_result, categories["input_view"]["metrics"], categories["input_view"]["options"], categories["input_view"].get("view_selection", "fixed"), evaluate_input_view)
                    view_details["image_scope"] = "foreground_composite" if config.get("foreground", {}).get("enabled") else "full_frame"
                    details.setdefault(name, {})["input_view"] = view_details
                    method_results[name]["input_view"] = _checked_values(values, categories["input_view"]["metrics"], loc)
                except Exception as exc:
                    raise EvaluationError(f"{loc}: {exc}") from exc
            if categories["physical_validity"]["enabled"]:
                loc = f"sample {sample_id!r}, method {name!r}, physical_validity"
                try:
                    RUNTIME_DETAILS.clear()
                    values = evaluate_physical(result["scene"], categories["physical_validity"]["options"])
                    method_results[name]["physical_validity"] = _checked_values(values, ("ColO", "ColS"), loc)
                    physical_details = dict(RUNTIME_DETAILS)
                    details.setdefault(name, {})["physical_validity"] = physical_details
                    for threshold, sweep_values in _checked_threshold_sweep(
                            physical_details, categories["physical_validity"]["options"], loc):
                        for metric, value in sweep_values.items():
                            physical_sweeps[name][threshold][metric].append(value)
                except Exception as exc:
                    raise EvaluationError(f"{loc}: {exc}") from exc
        if categories["gpt_evaluation"]["enabled"]:
            loc = f"sample {sample_id!r}, gpt_evaluation"
            try:
                if config.get("foreground", {}).get("enabled"):
                    values = evaluate_gpt(
                        sample["foreground_reference"],
                        {name: result["foreground_render"] for name, result in sample["methods"].items()},
                        categories["gpt_evaluation"]["options"],
                        context_images={"reference": sample["reference"], "renders": {name: result["render"] for name, result in sample["methods"].items()}},
                    )
                else:
                    values = evaluate_gpt(sample["reference"], {name: result["render"] for name, result in sample["methods"].items()}, categories["gpt_evaluation"]["options"])
                if not isinstance(values, dict) or set(values) != set(method_results):
                    raise EvaluationError(f"expected results for methods {list(method_results)}")
                for name in method_results:
                    method_results[name]["gpt_evaluation"] = _checked_values(values[name], ("VF", "IA", "PP", "Avg.", "OR"), f"{loc}, method {name!r}")
            except Exception as exc:
                raise EvaluationError(f"{loc}: {exc}") from exc
        for name, groups in method_results.items():
            for group, values in groups.items():
                for key, value in values.items():
                    collected[name][(group, key)].append(value)
        rows.append({"id": sample_id, "methods": method_results, "details": details})
    summary = {}
    for name, values in collected.items():
        groups = {}
        for (category, metric), samples in values.items():
            groups.setdefault(category, {})[metric] = {"mean": sum(samples) / len(samples), "count": len(samples)}
        summary[name] = groups
    from .provenance import collect_provenance
    result = {"samples": rows, "summary": summary, "provenance": collect_provenance(config)}
    if physical_sweeps:
        # Average per-scene rates rather than pooling object pairs. Binary
        # scene ColS values therefore yield the percentage of affected scenes.
        result["physical_validity_threshold_sweep"] = {
            name: [{"threshold_m": threshold, "threshold_mm": threshold * 1000,
                    "numerical_tolerance_m": categories["physical_validity"]["options"].get("numerical_tolerance_m", 1e-9),
                    **{metric: {"mean": sum(values) / len(values), "count": len(values)}
                       for metric, values in metrics.items()}}
                   for threshold, metrics in sorted(thresholds.items())]
            for name, thresholds in physical_sweeps.items()
        }
    return result
