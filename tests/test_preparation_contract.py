"""Read-only exports, shared Blender cameras, masks and original-scale physics."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from tabletopscene_eval.config import ConfigError, load_config
from tabletopscene_eval.preparation import prepare_evaluation
from tabletopscene_eval.rendering.cameras import make_view_specs
from tabletopscene_eval.runner import evaluate
from tests import test_scene_export as export_fixtures


HAS_GEOMETRY = all(importlib.util.find_spec(name) is not None for name in ("numpy", "trimesh"))


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = export_fixtures.SceneExportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.parent
        self.path = self.root / "experiment.json"
        self.raw = {
            "metrics": {"physical_validity": {"enabled": True}},
            "samples": [{"id": "sample", "methods": {"method": {"scene_export": "scene_export"}}}],
        }

    def config(self):
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        return load_config(self.path)

    def snapshot(self):
        return {str(path.relative_to(self.fixture.root)): path.read_bytes()
                for path in self.fixture.root.rglob("*") if path.is_file()}

    def visual(self, *, orbit=False, foreground=False, methods=("method",)):
        Image.new("RGB", (20, 16), (30, 50, 70)).save(self.root / "reference.png")
        self.raw["samples"][0].update(reference="reference.png", methods={
            name: {"scene_export": "scene_export"} for name in methods})
        self.raw["visual_normalization"] = {"enabled": False}
        self.raw["rendering"] = {"backend": "blender", "resolution": "reference", "device": "CPU"}
        self.raw["metrics"]["input_view"] = {"enabled": True, "metrics": ["LPIPS"]}
        if orbit:
            self.raw["camera_sampling"] = {
                "mode": "orbit_grid", "target_y_up_m": [0, 0, 0], "radius_m": 2,
                "azimuths_deg": [0, 180], "elevations_deg": [25], "distance_multipliers": [1],
                "anchor_view_id": "e00_a00_d00"}
            self.raw["metrics"]["input_view"].update(view_selection="best_per_metric", expected_view_count=2)
        if foreground:
            self.raw["foreground"] = {"enabled": True, "include_table": False}
            mask = Image.new("L", (20, 16), 0)
            mask.putpixel((1, 1), 255)
            mask.save(self.root / "reference_mask.png")
            self.raw["samples"][0]["reference_mask"] = "reference_mask.png"

    def render(self, export, output, sampling, options, **kwargs):
        self.assertEqual(options["backend"], "blender")
        self.assertEqual(options["resolution"], [20, 16])
        self.assertEqual(set(kwargs), {"foreground_options", "visual_normalization"})
        views = make_view_specs(sampling, self.fixture.camera, [20, 16])
        for index, view in enumerate(views):
            image, mask = output / f"view_{index}.png", output / f"mask_{index}.png"
            Image.new("RGB", (20, 16), (200, 10, 20)).save(image)
            pixels = Image.new("L", (20, 16), 0)
            if index == 0:
                pixels.putpixel((3, 2), 255)
            pixels.save(mask)
            view.update(render=str(image), mask=str(mask))
        return {"render": views[0]["render"], "render_mask": views[0]["mask"],
                "views": views, "details": {"rendering": {"backend": "blender"}}}

    def test_visual_normalization_defaults_on_and_partial_options_preserve_default(self):
        self.assertTrue(self.config()["visual_normalization"]["enabled"])
        self.raw["visual_normalization"] = {"target_diameter_m": 2}
        options = self.config()["visual_normalization"]
        self.assertTrue(options["enabled"])
        self.assertEqual(options["target_diameter_m"], 2)
        self.raw["visual_normalization"] = {"enabled": False}
        self.assertFalse(self.config()["visual_normalization"]["enabled"])

    def test_physics_only_generates_original_manifest_without_camera_or_renderer(self):
        self.fixture.result["canonical_camera_path"] = None
        self.fixture.save()
        self.raw["foreground"] = {"enabled": True}
        before = self.snapshot()
        config = self.config()
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=AssertionError("unneeded renderer")), \
             patch("tabletopscene_eval.normalization.normalize_scene_objects", side_effect=AssertionError("physics must not normalize")):
            prepared = prepare_evaluation(config)
        method = prepared["samples"][0]["methods"]["method"]
        manifest = json.loads(method["scene"].read_text())
        self.assertEqual(manifest["units"], "meters")
        self.assertTrue(manifest["objects"][0]["is_table"])
        self.assertEqual(manifest["objects"][1]["transform"], self.fixture.pose)
        self.assertTrue(all(Path(obj["mesh"]).is_file() for obj in manifest["objects"]))
        self.assertTrue(method["scene"].is_relative_to(self.root / "runs"))
        self.assertNotIn("render", method)
        self.assertIsNone(prepared["samples"][0]["reference"])
        self.assertFalse(config["foreground"]["enabled"])
        self.assertNotIn("scene", config["samples"][0]["methods"]["method"])
        self.assertEqual(self.snapshot(), before)

    def test_runner_passes_original_manifest_to_vhacd_fcl(self):
        before = self.snapshot()
        def physical(scene, options):
            manifest = json.loads(Path(scene).read_text())
            self.assertEqual(manifest["objects"][0]["transform"], export_fixtures.IDENTITY)
            self.assertEqual(manifest["objects"][1]["transform"], self.fixture.pose)
            self.assertEqual(options, {})
            return {"ColO": 0, "ColS": 0}
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=AssertionError("unneeded renderer")), \
             patch("metrics.physical_validity.metric.evaluate_physical", side_effect=physical) as metric:
            result = evaluate(self.config())
        metric.assert_called_once()
        self.assertEqual(result["summary"]["method"]["physical_validity"]["ColO"], {"mean": 0, "count": 1})
        self.assertEqual(self.snapshot(), before)

    def test_standard_export_visuals_are_deferred_to_blender_and_do_not_mutate_sources(self):
        self.visual()
        before = self.snapshot()
        config = self.config()
        self.assertNotIn("render", config["samples"][0]["methods"]["method"])
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render) as renderer:
            prepared = prepare_evaluation(config)
        renderer.assert_called_once()
        method = prepared["samples"][0]["methods"]["method"]
        self.assertTrue(method["render"].is_file())
        self.assertTrue(method["scene"].is_file())
        self.assertEqual(method["preparation"]["backend"]["rendering"]["backend"], "blender")
        self.assertEqual(self.snapshot(), before)

    def test_normalization_precedes_one_joint_framing_for_every_method(self):
        self.visual(orbit=True, methods=("a", "b"))
        self.raw["visual_normalization"] = {"enabled": True}
        sampling = self.raw["camera_sampling"]
        del sampling["target_y_up_m"], sampling["radius_m"]
        sampling["framing"] = {"mode": "shared_table_projection", "target_height": "bounds_center"}
        events = []
        def normalize(objects, options):
            self.assertTrue(options["enabled"])
            events.append("normalize")
            normalized = deepcopy(objects)
            for item in normalized:
                item["normalized_fixture"] = True
            return normalized, {"enabled": True}
        def framing(sampling, objects, resolution, fov):
            events.append("frame")
            self.assertEqual(events, ["normalize", "normalize", "frame"])
            self.assertEqual(set(objects), {"a", "b"})
            self.assertTrue(all(obj["normalized_fixture"] for rows in objects.values() for obj in rows))
            self.assertEqual(resolution, [20, 16])
            resolved = {k: v for k, v in sampling.items() if k != "framing"}
            return {**resolved, "target_y_up_m": [0, -.3, 0], "radius_m": 1.6}, {"joint_methods": ["a", "b"]}
        with patch("tabletopscene_eval.normalization.normalize_scene_objects", side_effect=normalize), \
             patch("tabletopscene_eval.camera_geometry.resolve_shared_table_camera_sampling", side_effect=framing) as fit, \
             patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render) as renderer:
            prepared = prepare_evaluation(self.config())
        fit.assert_called_once()
        self.assertEqual(renderer.call_count, 2)
        self.assertEqual(renderer.call_args_list[0].args[2], renderer.call_args_list[1].args[2])
        for method in prepared["samples"][0]["methods"].values():
            self.assertEqual(method["preparation"]["camera_framing"]["joint_methods"], ["a", "b"])
            self.assertEqual(json.loads(method["scene"].read_text())["objects"][1]["transform"], self.fixture.pose)

    def test_candidate_cameras_must_match_and_ids_must_be_unique(self):
        self.visual(orbit=True, methods=("a", "b"))
        for failure in ("different_camera", "duplicate_id", "missing_view"):
            calls = 0
            def render(*args, **kwargs):
                nonlocal calls
                result = self.render(*args, **kwargs)
                calls += 1
                if calls == 2:
                    if failure == "different_camera":
                        result["views"][0]["camera"]["radius_m"] = 99
                    elif failure == "duplicate_id":
                        result["views"][1]["id"] = result["views"][0]["id"]
                    else:
                        result["views"].pop()
                return result
            with self.subTest(failure=failure), patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=render):
                with self.assertRaisesRegex(ValueError, "identical candidate cameras|invalid candidate count"):
                    prepare_evaluation(self.config())

    def test_foreground_uses_each_own_mask_without_moving_pixels_and_keeps_empty_failures(self):
        self.visual(orbit=True, foreground=True, methods=("a", "b"))
        self.raw["metrics"]["gpt_evaluation"] = {"enabled": True}
        before = self.snapshot()
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render) as renderer:
            prepared = prepare_evaluation(self.config())
        self.assertFalse(renderer.call_args.kwargs["foreground_options"]["include_table"])
        sample = prepared["samples"][0]
        with Image.open(sample["foreground_reference"]) as image:
            self.assertEqual(image.size, (20, 16))
            self.assertEqual(image.getpixel((1, 1)), (30, 50, 70))
            self.assertEqual(image.getpixel((3, 2)), (255, 255, 255))
        for method in sample["methods"].values():
            with Image.open(method["foreground_render"]) as image:
                self.assertEqual(image.getpixel((3, 2)), (200, 10, 20))
                self.assertEqual(image.getpixel((1, 1)), (255, 255, 255))
            self.assertEqual(len(method["foreground_views"]), 2)
            with Image.open(method["foreground_views"][1]["render"]) as image:
                self.assertEqual(image.getextrema(), ((255, 255),) * 3)
        self.assertEqual(self.snapshot(), before)
        judgment = {name: {"VF": 4, "IA": 4, "PP": 4, "Avg.": 4, "OR": index + 1}
                    for index, name in enumerate(("a", "b"))}
        with patch("tabletopscene_eval.preparation.prepare_evaluation", return_value=prepared), \
             patch("tabletopscene_eval.image_metrics.evaluate_input_view", return_value={"LPIPS": .5}), \
             patch("metrics.physical_validity.metric.evaluate_physical", return_value={"ColO": 0, "ColS": 0}), \
             patch("metrics.gpt_evaluation.evaluate_gpt", return_value=judgment) as gpt:
            evaluate(prepared)
        self.assertEqual(gpt.call_args.args[0], sample["foreground_reference"])
        self.assertEqual(gpt.call_args.kwargs["context_images"]["reference"], sample["reference"])
        self.assertEqual(set(gpt.call_args.kwargs["context_images"]["renders"]), {"a", "b"})

    def test_foreground_rejects_render_canvas_or_mask_dimension_mismatches(self):
        self.visual(foreground=True)
        for failure in ("canvas", "mask"):
            def render(*args, **kwargs):
                result = self.render(*args, **kwargs)
                if failure == "canvas":
                    Image.new("RGB", (19, 16)).save(result["render"])
                Image.new("L", (19, 16), 255).save(result["render_mask"])
                return result
            with self.subTest(failure=failure), patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=render):
                with self.assertRaisesRegex(ValueError, "canvas differs|mask dimensions differ"):
                    prepare_evaluation(self.config())

    @unittest.skipUnless(HAS_GEOMETRY, "requires numpy and trimesh")
    def test_actual_normalization_fits_unit_table_and_keeps_original_physics(self):
        import trimesh
        self.visual(orbit=True)
        self.raw["visual_normalization"] = {"enabled": True}
        sampling = self.raw["camera_sampling"]
        del sampling["target_y_up_m"], sampling["radius_m"]
        sampling["framing"] = {"mode": "shared_table_projection", "target_height": "bounds_center", "margin": 1.08}
        for instance in self.fixture.layout["instances"]:
            path = self.fixture.root / instance["geometry_path"]
            trimesh.creation.box(extents=(2, 1.4, 2) if instance["object_id"] == "table" else (.2, .2, .2)).export(path)
            self.fixture.result["assets"][instance["object_id"]]["export_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fixture.save()
        before = self.snapshot()
        with patch("tabletopscene_eval.rendering.blender_backend.prepare_scene_export", side_effect=self.render):
            method = prepare_evaluation(self.config())["samples"][0]["methods"]["method"]
        bounds = method["preparation"]["camera_framing"]["combined_table_bounds_y_up_m"]
        self.assertAlmostEqual(bounds[1][0] - bounds[0][0], 1)
        self.assertAlmostEqual(bounds[1][1], 0)
        manifest = json.loads(method["scene"].read_text())
        self.assertEqual(manifest["objects"][0]["transform"], export_fixtures.IDENTITY)
        self.assertEqual(manifest["objects"][1]["transform"], self.fixture.pose)
        self.assertEqual(self.snapshot(), before)

    def test_inherited_export_paths_keep_their_source_origin(self):
        profiles = self.root / "profiles"
        profiles.mkdir()
        inherited = deepcopy(self.raw)
        inherited["samples"][0]["methods"]["method"]["scene_export"] = "../scene_export/result.json"
        (profiles / "samples.json").write_text(json.dumps(inherited))
        self.path.write_text(json.dumps({"extends": "profiles/samples.json"}))
        method = prepare_evaluation(load_config(self.path))["samples"][0]["methods"]["method"]
        self.assertEqual(method["scene_export"], self.fixture.root)
        self.assertTrue(method["scene"].is_file())

    def test_output_and_cache_cannot_write_inside_source_even_through_symlinks(self):
        before = self.snapshot()
        alias = self.root / "source_alias"
        alias.symlink_to(self.fixture.root, target_is_directory=True)
        for kind in ("output", "cache"):
            for target in (self.fixture.root, self.fixture.root / "not_created", alias / "not_created"):
                with self.subTest(kind=kind, target=str(target)):
                    self.raw.pop("rendering", None)
                    self.raw["metrics"]["physical_validity"]["options"] = {}
                    if kind == "output":
                        self.raw["rendering"] = {"output_dir": str(target)}
                    else:
                        self.raw["metrics"]["physical_validity"]["options"]["cache_dir"] = str(target)
                    with self.assertRaisesRegex(ValueError, "outside every source"):
                        prepare_evaluation(self.config())
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
