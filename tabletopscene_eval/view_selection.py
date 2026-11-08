"""Explicit fixed, per-metric best-of-view, and mean-over-view protocols."""
import math


def evaluate_views(reference, result, metric_names, options, selection, evaluator):
    """Select no hidden views; preserve every candidate score and selection.

    LPIPS is minimized, DINOv2/CLIP are maximized. Each metric may select a
    different view. Ties use the lexicographically first view ID.
    """
    if selection == "fixed":
        views = [{"id": "fixed", "render": result["render"], "camera": {}}]
    else:
        views = sorted(result["views"], key=lambda view: view["id"])
    candidates = []
    for view in views:
        values = evaluator(reference, view["render"], metric_names, options)
        if not isinstance(values, dict) or set(values) != set(metric_names):
            raise ValueError(f"view {view['id']!r}: expected metrics {metric_names}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError(f"view {view['id']!r}: all metrics must be finite numbers")
        candidates.append({**view, "render": str(view["render"]), "scores": values})
    selected, scores = {}, {}
    for metric in metric_names:
        if selection == "mean":
            scores[metric] = sum(v["scores"][metric] for v in candidates) / len(candidates)
        else:
            best = min(candidates, key=lambda v: v["scores"][metric]) if metric == "LPIPS" else max(candidates, key=lambda v: v["scores"][metric])
            selected[metric] = best["id"]
            scores[metric] = best["scores"][metric]
    return scores, {"selection": selection, "candidate_count": len(candidates), "selected_view_ids": selected, "candidates": candidates}
