"""Independent best-per-metric selection and deterministic ties."""
import unittest
from tabletopscene_eval.view_selection import evaluate_views

class ViewSelectionTests(unittest.TestCase):
    def test_each_metric_selects_its_own_view_and_keeps_all_scores(self):
        result = {"views": [{"id": name, "render": name, "camera": {}} for name in ("b", "a")]}
        scores = {"a": {"LPIPS": .2, "DINOv2": .7, "CLIP": .8}, "b": {"LPIPS": .4, "DINOv2": .9, "CLIP": .8}}
        values, details = evaluate_views("ref", result, list(scores["a"]), {}, "best_per_metric", lambda ref, render, names, opts: scores[render])
        self.assertEqual(values, {"LPIPS": .2, "DINOv2": .9, "CLIP": .8})
        self.assertEqual(details["selected_view_ids"], {"LPIPS": "a", "DINOv2": "b", "CLIP": "a"})
        self.assertEqual(len(details["candidates"]), 2)
        mean, details = evaluate_views("ref", result, list(scores["a"]), {}, "mean", lambda ref, render, names, opts: scores[render])
        self.assertAlmostEqual(mean["LPIPS"], .3)
        self.assertEqual(details["selected_view_ids"], {})
