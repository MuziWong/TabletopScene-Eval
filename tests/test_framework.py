"""Strict public input and orchestration contracts, without models or services."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from metrics.gpt_evaluation import RUNTIME_JUDGMENTS, evaluate_gpt
from tabletopscene_eval.config import ConfigError, load_config
from tabletopscene_eval.runner import EvaluationError, evaluate
from tests import test_scene_export as export_fixtures


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        fixture = export_fixtures.SceneExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root.parent
        Image.new("RGB", (16, 16), (30, 40, 50)).save(self.root / "reference.png")
        self.path = self.root / "experiment.json"
        self.raw = {
            "visual_normalization": {"enabled": False},
            "metrics": {
                "input_view": {"enabled": True, "metrics": ["LPIPS"]},
                "gpt_evaluation": {"enabled": True},
                "physical_validity": {"enabled": True}},
            "samples": [{"id": identity, "reference": "reference.png", "methods": {
                "A": {"scene_export": "scene_export"},
                "B": {"scene_export": "scene_export/result.json"}}}
                for identity in ("one", "two")],
        }

    def config(self):
        self.path.write_text(json.dumps(self.raw))
        return load_config(self.path)

    def render(self, export, output, sampling, options, **kwargs):
        image = output / "anchor.png"
        Image.new("RGB", (16, 16)).save(image)
        return {"render": str(image), "views": [], "details": {"renderer": "offline-test"}}

    def test_config_resolves_standard_export_and_shared_method_set(self):
        cfg = self.config()
        self.assertEqual(cfg["samples"][0]["reference"], self.root / "reference.png")
        self.assertEqual(cfg["samples"][0]["methods"]["B"]["scene_export"], self.fixture.root)
        self.assertEqual(cfg["metrics"]["physical_validity"]["backend"], "vhacd_fcl")
        del self.raw["samples"][1]["methods"]["B"]
        with self.assertRaisesRegex(ConfigError, "same method set"):
            self.config()

    def test_only_scene_export_is_a_public_method_input(self):
        baseline = deepcopy(self.raw)
        for source in ({"render": "reference.png"}, {"scene": "scene.json"},
                       {"views": []}, {"physical_values": {"ColO": 0, "ColS": 0}},
                       {"scene_export": "scene_export", "render": "reference.png"},
                       {"scene_export": "scene_export", "camera_sampling": {}},
                       {"scene_export": "scene_export", "rendering": {}}):
            with self.subTest(source=source):
                self.raw = deepcopy(baseline)
                self.raw["samples"][0]["methods"]["A"] = source
                with self.assertRaisesRegex(ConfigError, "only scene_export"):
                    self.config()

    def test_missing_files_duplicate_ids_and_invalid_exports_fail_early(self):
        baseline = deepcopy(self.raw)
        for mutation, message in (
            (lambda: self.raw["samples"][0].update(reference="missing.png"), "file does not exist"),
            (lambda: self.raw["samples"][1].update(id="one"), "unique"),
            (lambda: self.raw["samples"][0]["methods"]["A"].update(scene_export="missing"), "export directory"),
        ):
            with self.subTest(message=message):
                self.raw = deepcopy(baseline)
                mutation()
                with self.assertRaisesRegex(ConfigError, message):
                    self.config()
        self.raw = baseline
        self.fixture.result["success"] = False
        self.fixture.save()
        with self.assertRaisesRegex(ConfigError, "successful"):
            self.config()

    def test_enabled_categories_and_metric_names_are_strict(self):
        baseline = deepcopy(self.raw)
        for metrics in ({}, {"unknown": {"enabled": True}},
                        {"input_view": {"enabled": 1}},
                        {"input_view": {"enabled": True, "metrics": ["CLIP", "CLIP"]}},
                        {"input_view": {"enabled": True, "metrics": ["unknown"]}},
                        {"physical_validity": {"enabled": True, "backend": "fcl"}}):
            with self.subTest(metrics=metrics):
                self.raw = {**baseline, "metrics": metrics}
                with self.assertRaises(ConfigError):
                    self.config()

    def test_foreground_reference_mask_and_shared_orbit_anchor_are_explicit(self):
        self.raw["foreground"] = {"enabled": True}
        with self.assertRaisesRegex(ConfigError, "reference_mask"):
            self.config()
        Image.new("L", (16, 16), 255).save(self.root / "mask.png")
        for sample in self.raw["samples"]:
            sample["reference_mask"] = "mask.png"
        self.raw["metrics"]["input_view"].update(view_selection="best_per_metric", expected_view_count=2)
        self.raw["camera_sampling"] = {
            "mode": "orbit_grid", "target_y_up_m": [0, 0, 0], "radius_m": 2,
            "elevations_deg": [25], "azimuths_deg": [0, 180], "distance_multipliers": [1]}
        with self.assertRaisesRegex(ConfigError, "explicit shared anchor_view_id"):
            self.config()
        self.raw["camera_sampling"]["anchor_view_id"] = "missing"
        with self.assertRaisesRegex(ConfigError, "not a configured view"):
            self.config()
        self.raw["camera_sampling"]["anchor_view_id"] = "e00_a00_d00"
        self.assertTrue(self.config()["samples"][0]["methods"]["A"]["generate_views"])
        self.raw["metrics"]["input_view"]["expected_view_count"] = 3
        with self.assertRaisesRegex(ConfigError, "Expected 3 views, got 2"):
            self.config()

    def test_runner_uses_scene_macro_means_and_keeps_direct_judged_ranks(self):
        judgments = [
            {"A": {"VF": 7, "IA": 7, "PP": 7, "Avg.": 7, "OR": 2},
             "B": {"VF": 1, "IA": 1, "PP": 1, "Avg.": 1, "OR": 1}},
            {"A": {"VF": 4, "IA": 4, "PP": 4, "Avg.": 4, "OR": 1},
             "B": {"VF": 3, "IA": 3, "PP": 3, "Avg.": 3, "OR": 2}},
        ]
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render), \
             patch("tabletopscene_eval.image_metrics.evaluate_input_view", side_effect=[{"LPIPS": v} for v in (.1, .2, .3, .4)]), \
             patch("metrics.physical_validity.metric.evaluate_physical", side_effect=[
                 {"ColO": c, "ColS": s} for c, s in ((10, 100), (0, 0), (0, 0), (20, 100))]) as physical, \
             patch("metrics.gpt_evaluation.evaluate_gpt", side_effect=judgments):
            result = evaluate(self.config())
        self.assertEqual(physical.call_count, 4)
        summary = result["summary"]["A"]
        self.assertAlmostEqual(summary["input_view"]["LPIPS"]["mean"], .2)
        self.assertEqual(summary["physical_validity"]["ColS"], {"mean": 50, "count": 2})
        self.assertEqual(summary["gpt_evaluation"]["OR"], {"mean": 1.5, "count": 2})
        self.assertEqual(result["samples"][0]["methods"]["A"]["gpt_evaluation"]["OR"], 2)
        protocol = result["provenance"]["protocols"]["physical_validity"]
        self.assertEqual(protocol["backend"], "vhacd_fcl")
        self.assertIn("macro average", result["provenance"]["aggregation"])

    def test_runner_rejects_missing_nonfinite_and_boolean_metric_values(self):
        self.raw["metrics"] = {"input_view": {"enabled": True, "metrics": ["LPIPS"]}}
        cfg = self.config()
        for values in ({}, {"LPIPS": float("nan")}, {"LPIPS": True}, {"LPIPS": .1, "CLIP": .9}):
            with self.subTest(values=values), \
                 patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render), \
                 patch("tabletopscene_eval.image_metrics.evaluate_input_view", return_value=values):
                with self.assertRaisesRegex(EvaluationError, "sample 'one'.*method 'A'"):
                    evaluate(cfg)

    def test_gpt_requires_exactly_the_configured_methods(self):
        self.raw["metrics"] = {"gpt_evaluation": {"enabled": True}}
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render), \
             patch("metrics.gpt_evaluation.evaluate_gpt", return_value={"A": {}}):
            with self.assertRaisesRegex(EvaluationError, "expected results for methods"):
                evaluate(self.config())

    def test_single_method_gpt_is_supported_without_inventing_a_comparison(self):
        self.raw["metrics"] = {"gpt_evaluation": {"enabled": True}}
        for sample in self.raw["samples"]:
            sample["methods"] = {"only": {"scene_export": "scene_export"}}
        self.assertEqual(set(self.config()["samples"][0]["methods"]), {"only"})
        RUNTIME_JUDGMENTS.clear()
        self.addCleanup(RUNTIME_JUDGMENTS.clear)
        answer = {"candidate_1": {"VF": 6, "IA": 5, "PP": 4, "OR": 1}}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])))))
        with patch("metrics.gpt_evaluation.metric.create_api_client", return_value=(client, SimpleNamespace(model="offline-model"))):
            scores = evaluate_gpt(self.root / "reference.png", {"only": self.root / "reference.png"})
        self.assertEqual(scores["only"], {"VF": 6, "IA": 5, "PP": 4, "Avg.": 5, "OR": 1})
        self.assertFalse(RUNTIME_JUDGMENTS[-1]["ranking_comparative"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][0]["text"]
        self.assertIn("SINGLE-CANDIDATE REQUEST", prompt)
        self.assertIn("not a comparative ranking", prompt)


if __name__ == "__main__":
    unittest.main()
