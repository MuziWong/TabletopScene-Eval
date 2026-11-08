"""Batch joins, explicit masks, immutable inputs and complete-scene averaging."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from tabletopscene_eval.batch import BatchPlanError, build_batch_plan, run_batch
from tests import test_scene_export as scene_export_tests


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = scene_export_tests.SceneExportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = {"foreground": {"enabled": False}, "metrics": {
            "input_view": {"enabled": True, "metrics": ["CLIP"]},
            "physical_validity": {"enabled": False}, "gpt_evaluation": {"enabled": False}}}

    def case(self, root, identity):
        case = root/identity
        shutil.copytree(self.fixture.root, case/"scene_export")
        analysis = case/"input_analysis"/"success"
        analysis.mkdir(parents=True)
        Image.new("RGB", (4, 4), "white").save(analysis/"original.png")
        return case

    def test_discovery_is_sorted_and_excludes_its_output_subtree(self):
        inputs = self.root/"outputs"
        self.case(inputs, "2/02-02")
        self.case(inputs, "2/02-01")
        output = inputs/"evaluation"
        self.case(output, "should_not_be_evaluated")
        plan = build_batch_plan(self.base, {"A": inputs}, output)
        self.assertEqual([x["id"] for x in plan["samples"]], ["2/02-01", "2/02-02"])
        self.assertTrue(all(set(x["methods"]["A"]) == {"scene_export"} for x in plan["samples"]))
        self.assertEqual(self.base.get("samples"), None)

    def test_composed_config_metadata_does_not_pollute_saved_scene_configs(self):
        case = self.case(self.root/"inputs", "case")
        base = {**self.base, "_config_sources": ["protocol.json"], "config_path": "old.json"}
        plan = build_batch_plan(base, {"A": case}, self.root/"out")
        self.assertNotIn("_config_sources", plan["base_config"])
        self.assertNotIn("config_path", plan["base_config"])
        self.assertEqual(plan["discovery"]["config_sources"], ["protocol.json"])
        self.assertEqual(base["_config_sources"], ["protocol.json"])

    def test_existing_config_or_batch_results_prevent_plan_and_execution(self):
        case = self.case(self.root/"inputs", "case")
        for filename in ("results.json", "summary.json"):
            with self.subTest(filename=filename):
                output = self.root/filename.removesuffix(".json")
                plan = build_batch_plan(self.base, {"A": case}, output)
                sentinel = output/filename
                sentinel.write_text('{"previous_run": true}')
                before = {str(path.relative_to(output)): path.read_bytes()
                          for path in output.rglob("*") if path.is_file()}
                with self.assertRaisesRegex(FileExistsError, "already contains evaluation results"):
                    build_batch_plan(self.base, {"A": case}, output)
                calls = []
                with self.assertRaisesRegex(FileExistsError, "already contains evaluation results"):
                    run_batch(plan, lambda config: calls.append(config))
                self.assertFalse(calls)
                self.assertEqual(before, {str(path.relative_to(output)): path.read_bytes()
                                          for path in output.rglob("*") if path.is_file()})
                self.assertFalse((output/"scenes").exists())

    def test_output_inside_any_source_export_is_fatal_before_writing(self):
        inputs = self.root/"inputs"
        case = self.case(inputs, "2/02-01")
        source = case/"scene_export"
        before = {str(path.relative_to(source)): path.read_bytes()
                  for path in source.rglob("*") if path.is_file()}
        source_directories = sorted(str(path) for path in source.rglob("*") if path.is_dir())
        for input_root in (inputs, case, source, source/"result.json"):
            for policy in ("error", "continue"):
                output = source/"new"/"evaluation"
                with self.subTest(input_root=str(input_root), policy=policy):
                    with self.assertRaisesRegex(ValueError, "outside every source"):
                        build_batch_plan(self.base, {"A": input_root}, output,
                                         invalid_policy=policy)
                    self.assertFalse((source/"new").exists())
                    self.assertEqual(before, {str(path.relative_to(source)): path.read_bytes()
                                              for path in source.rglob("*") if path.is_file()})
                    self.assertEqual(source_directories, sorted(str(path) for path in source.rglob("*") if path.is_dir()))

    def test_destination_guard_cannot_be_skipped_for_invalid_or_later_method(self):
        a = self.case(self.root/"a", "case")
        b = self.case(self.root/"b", "case")
        source = b/"scene_export"
        (source/"result.json").write_text('{"success": false}')
        alias = self.root/"source_alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias/"new"
        with self.assertRaisesRegex(ValueError, "outside every source"):
            build_batch_plan(self.base, {"A": a, "B": b}, output,
                             invalid_policy="continue")
        self.assertFalse((source/"new").exists())

    def test_invalid_export_is_listed_and_only_explicit_continue_skips_it(self):
        inputs = self.root/"inputs"
        self.case(inputs, "valid")
        bad = self.case(inputs, "invalid")/"scene_export"/"result.json"
        result = json.loads(bad.read_text())
        result["success"] = False
        bad.write_text(json.dumps(result))
        with self.assertRaises(BatchPlanError) as captured:
            build_batch_plan(self.base, {"A": inputs}, self.root/"strict")
        self.assertEqual(captured.exception.issues[0]["scene_id"], "invalid")
        self.assertTrue((self.root/"strict"/"discovery.json").is_file())
        plan = build_batch_plan(self.base, {"A": inputs}, self.root/"continued", invalid_policy="continue")
        self.assertEqual([x["id"] for x in plan["samples"]], ["valid"])
        self.assertTrue(plan["discovery"]["issues"])

    def test_multiple_methods_join_strictly_with_shared_explicit_reference_root(self):
        a, b, images = self.root/"a", self.root/"b", self.root/"images"
        self.case(a, "2/01")
        self.case(b, "2/01")
        self.case(a, "2/02")
        (images/"2").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(images/"2/01.png")
        with self.assertRaises(BatchPlanError):
            build_batch_plan(self.base, {"A": a, "B": b}, self.root/"strict", reference_root=images)
        plan = build_batch_plan(self.base, {"A": a, "B": b}, self.root/"continued",
                                reference_root=images, invalid_policy="continue")
        self.assertEqual(len(plan["samples"]), 1)
        self.assertEqual(plan["samples"][0]["reference"], str(images/"2/01.png"))
        self.assertEqual(set(plan["samples"][0]["methods"]), {"A", "B"})
        self.assertIn("B", plan["discovery"]["issues"][0]["message"])

    def test_explicit_single_scene_files_and_ambiguous_glob(self):
        inputs = self.root/"inputs"
        case = self.case(inputs, "02-01")
        reference = case/"input_analysis/success/original.png"
        mask = self.root/"mask.png"
        Image.new("L", (4, 4), 255).save(mask)
        config = deepcopy(self.base)
        config["foreground"]["enabled"] = True
        plan = build_batch_plan(config, {"A": case/"scene_export/result.json"}, self.root/"one",
                                reference=reference, reference_mask=mask)
        self.assertEqual(plan["samples"][0]["id"], "02-01")
        self.assertEqual(plan["samples"][0]["reference_mask"], str(mask))
        self.case(inputs, "02-02")
        with self.assertRaisesRegex(ValueError, "require one scene"):
            build_batch_plan(config, {"A": inputs}, self.root/"many", reference=reference)
        images = self.root/"images"
        images.mkdir()
        Image.new("RGB", (4, 4)).save(images/"02-01.png")
        Image.new("RGB", (4, 4)).save(images/"02-01.jpg")
        with self.assertRaises(BatchPlanError) as captured:
            build_batch_plan(self.base, {"A": case}, self.root/"ambiguous",
                             reference_root=images, reference_pattern="{scene_id}.*")
        self.assertIn("found 2", str(captured.exception))

    def test_foreground_requires_an_explicit_mask_policy(self):
        case = self.case(self.root/"inputs", "case")
        config = deepcopy(self.base)
        config["foreground"]["enabled"] = True
        with self.assertRaisesRegex(BatchPlanError, "--auto-mask"):
            build_batch_plan(config, {"A": case}, self.root/"out")

    def test_physics_only_needs_no_reference_or_mask(self):
        case = self.case(self.root/"inputs", "case")
        shutil.rmtree(case/"input_analysis")
        config = deepcopy(self.base)
        config["metrics"]["input_view"]["enabled"] = False
        config["metrics"]["physical_validity"]["enabled"] = True
        config["foreground"]["enabled"] = True
        plan = build_batch_plan(config, {"A": case}, self.root/"out", auto_mask=True)
        self.assertNotIn("reference", plan["samples"][0])
        self.assertNotIn("reference_mask", plan["samples"][0])
        self.assertFalse(plan["base_config"]["foreground"]["enabled"])

    def test_auto_mask_is_shared_excludes_table_when_requested_and_preserves_sources(self):
        case = self.case(self.root/"inputs", "case")
        analysis = case/"input_analysis/success"
        for filename, pixel in (("table.png", (0, 0)), ("cup.png", (2, 2))):
            data = np.zeros((4, 4), dtype=np.uint8)
            data[pixel] = 255
            Image.fromarray(data).save(analysis/filename)
        (analysis/"scene_graph.json").write_text(json.dumps({"nodes": {
            "table": {"category": "table", "mask_path": "table.png"},
            "cup_1": {"category": "cup", "mask_path": "cup.png"},
            "group": {"type": "subscene", "mask_path": "not_read.png"}}}))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in case.rglob("*") if p.is_file()}
        config = deepcopy(self.base)
        config["foreground"] = {"enabled": True, "include_table": False}
        plan = build_batch_plan(config, {"A": case}, self.root/"out", auto_mask=True)
        mask = np.asarray(Image.open(plan["samples"][0]["reference_mask"]))
        self.assertEqual(int((mask > 0).sum()), 1)
        self.assertEqual(mask[2, 2], 255)
        provenance = plan["discovery"]["input_metadata"]["case"]["auto_mask"]
        self.assertTrue(provenance["automatic_pipeline_mask_not_human_gt"])
        self.assertEqual([x["id"] for x in provenance["mask_sources"]], ["cup_1"])
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in case.rglob("*") if p.is_file()})

    def test_automatic_masks_keep_complete_dotted_scene_ids(self):
        inputs, output = self.root/"inputs", self.root/"out"
        for identity, pixel in (("group/foo.1", (1, 1)), ("group/foo.2", (2, 2))):
            case = self.case(inputs, identity)
            analysis = case/"input_analysis/success"
            mask = Image.new("L", (4, 4), 0)
            mask.putpixel(pixel, 255)
            mask.save(analysis/"cup.png")
            (analysis/"scene_graph.json").write_text(json.dumps({"nodes": {
                "cup": {"mask_path": "cup.png"}}}))
        config = deepcopy(self.base)
        config["foreground"]["enabled"] = True
        plan = build_batch_plan(config, {"A": inputs}, output, auto_mask=True)
        paths = [Path(sample["reference_mask"]) for sample in plan["samples"]]
        self.assertEqual(paths, [output/"masks/group/foo.1.png", output/"masks/group/foo.2.png"])
        for path, foreground, background in ((paths[0], (1, 1), (2, 2)),
                                             (paths[1], (2, 2), (1, 1))):
            with Image.open(path) as image:
                self.assertEqual(image.getpixel(foreground), 255)
                self.assertEqual(image.getpixel(background), 0)
            self.assertTrue(path.with_suffix(".provenance.json").is_file())

    def test_sequential_complete_scene_macro_average_and_five_thresholds(self):
        inputs = self.root/"inputs"
        for identity in ("a", "b", "c"):
            self.case(inputs, identity)
        config = deepcopy(self.base)
        thresholds = [0, .0001, .0003, .0005, .001]
        config["metrics"]["physical_validity"] = {"enabled": True, "options": {"penetration_thresholds_m": thresholds}}
        plan = build_batch_plan(config, {"A": inputs}, self.root/"out")
        calls = []
        def evaluate(one):
            self.assertEqual(len(one["samples"]), 1)
            identity = one["samples"][0]["id"]
            calls.append(identity)
            if identity == "b":
                raise RuntimeError("deliberate scene failure")
            rate, clip = (10, .2) if identity == "a" else (30, .8)
            row = {"id": identity, "methods": {"A": {"input_view": {"CLIP": clip},
                   "physical_validity": {"ColO": rate, "ColS": 100 if identity == "a" else 0}}}}
            sweep = [{"threshold_m": threshold, "ColO": {"mean": rate, "count": 1},
                      "ColS": {"mean": 100 if identity == "a" else 0, "count": 1}} for threshold in thresholds]
            return {"samples": [row], "physical_validity_threshold_sweep": {"A": sweep}}
        result = run_batch(plan, evaluate, continue_on_error=True)
        self.assertEqual(calls, ["a", "b", "c"])
        self.assertEqual(result["completed_scenes"], ["a", "c"])
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["summary"]["A"]["input_view"]["CLIP"], {"mean": .5, "count": 2})
        self.assertEqual(len(result["physical_validity_threshold_sweep"]["A"]), 5)
        for row in result["physical_validity_threshold_sweep"]["A"]:
            self.assertEqual(row["ColO"], {"mean": 20, "count": 2})
            self.assertEqual(row["ColS"], {"mean": 50, "count": 2})
        self.assertTrue((self.root/"out/scenes/b/result.json").is_file())
        self.assertTrue((self.root/"out/summary.json").is_file())
        with self.assertRaises(FileExistsError):
            run_batch(plan, evaluate)

    def test_incomplete_method_result_fails_whole_scene_and_failfast_stops(self):
        a, b = self.root/"a", self.root/"b"
        for identity in ("one", "two"):
            self.case(a, identity)
            self.case(b, identity)
        plan = build_batch_plan(self.base, {"A": a, "B": b}, self.root/"out")
        calls = []
        def partial(config):
            sample = config["samples"][0]
            calls.append(sample["id"])
            return {"samples": [{"id": sample["id"], "methods": {"A": {"input_view": {"CLIP": .9}}}}]}
        result = run_batch(plan, partial)
        self.assertEqual(calls, ["one"])
        self.assertFalse(result["completed_scenes"])
        self.assertEqual(result["summary"], {})
        self.assertEqual(result["not_run_scenes"], ["two"])
        self.assertIn("every method", result["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
