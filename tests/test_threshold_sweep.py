"""One geometry evaluation, five depth thresholds, and scene-macro summaries."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics.physical_validity.convex_geometry import evaluate_convex_objects, validate_penetration_thresholds
from metrics.physical_validity.metric import RUNTIME_DETAILS, evaluate_physical, validate_options
from tabletopscene_eval.config import ConfigError, load_config
from tabletopscene_eval.profiles import load_composed_config
from tabletopscene_eval.runner import EvaluationError, evaluate
from tests import test_scene_export as export_fixtures


THRESHOLDS = [0, .0001, .0003, .0005, .001]
GEOMETRY_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "trimesh", "fcl"))
VHACD_AVAILABLE = GEOMETRY_AVAILABLE and importlib.util.find_spec("vhacdx")


class ThresholdSweepContractTests(unittest.TestCase):
    def test_sweep_validation_and_default_profiles(self):
        invalid = [[], (), "0.001", [0, 0], [True], [-.001], [float("nan")],
                   [float("inf")], [10**1000], [1e308], [None]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options({"penetration_thresholds_m": value})
        with self.assertRaises(ValueError):
            validate_options({"penetration_thresholds_m": None})
        self.assertEqual(validate_penetration_thresholds(list(reversed(THRESHOLDS))), THRESHOLDS)
        root = Path(__file__).resolve().parents[1]
        for path in (root/"metrics"/"physical_validity"/"config.json", root/"configs"/"default.json"):
            options = load_composed_config(path)["metrics"]["physical_validity"]["options"]
            self.assertEqual(options["penetration_thresholds_m"], THRESHOLDS)
            self.assertEqual(options["min_penetration_m"], .001)
            validate_options(options)

    def config(self, root, thresholds=THRESHOLDS):
        fixture = export_fixtures.SceneExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = {
            "metrics": {"physical_validity": {"enabled": True, "backend": "vhacd_fcl",
                        "options": {"min_penetration_m": .001,
                                    "penetration_thresholds_m": thresholds}}},
            "samples": [{"id": name, "methods": {"method": {"scene_export": str(fixture.root)}}}
                        for name in ("small", "large")],
        }
        path = root/"config.json"
        path.write_text(json.dumps(raw))
        return load_config(path)

    def test_runner_keeps_primary_and_averages_each_scene_equally(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            # Scene one: 1/1 pairs at 0 and .1 mm. Scene two: 1/10 pairs
            # through .3 mm. Pooling pairs would produce 2/11, not 55%.
            counts = iter(([100, 100, 0, 0, 0], [10, 10, 10, 0, 0]))
            def fake_evaluate(scene, options):
                rates = next(counts)
                RUNTIME_DETAILS.update({"threshold_sweep": [
                    {"threshold_m": threshold, "ColO": value,
                     "ColS": 100 if value else 0}
                    for threshold, value in zip(THRESHOLDS, rates)]})
                return {"ColO": 0, "ColS": 0}
            with patch("metrics.physical_validity.metric.evaluate_physical", side_effect=fake_evaluate) as metric:
                result = evaluate(config)
        self.assertEqual(metric.call_count, 2)
        self.assertEqual(result["summary"]["method"]["physical_validity"],
                         {"ColO": {"mean": 0, "count": 2}, "ColS": {"mean": 0, "count": 2}})
        rows = result["physical_validity_threshold_sweep"]["method"]
        self.assertEqual([row["threshold_mm"] for row in rows], [0, .1, .3, .5, 1])
        self.assertEqual([row["ColO"]["mean"] for row in rows], [55, 55, 5, 0, 0])
        self.assertEqual([row["ColS"]["mean"] for row in rows], [100, 100, 50, 0, 0])
        self.assertTrue(all(row[metric]["count"] == 2 for row in rows for metric in ("ColO", "ColS")))
        self.assertEqual(result["samples"][0]["details"]["method"]["physical_validity"]["threshold_sweep"][0]["ColO"], 100)

    def test_runner_rejects_incomplete_or_invalid_sweep(self):
        cases = [None, [], [{"threshold_m": 0, "ColO": 0, "ColS": 0}]*5,
                 [{"threshold_m": t, "ColO": float("nan"), "ColS": 0} for t in THRESHOLDS],
                 [{"threshold_m": t, "ColO": 0, "ColS": 50} for t in THRESHOLDS]]
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            for rows in cases:
                def fake_evaluate(scene, options):
                    RUNTIME_DETAILS.update({"threshold_sweep": rows})
                    return {"ColO": 0, "ColS": 0}
                with self.subTest(rows=rows), patch("metrics.physical_validity.metric.evaluate_physical", side_effect=fake_evaluate):
                    with self.assertRaisesRegex(EvaluationError, "threshold"):
                        evaluate(config)

    def test_config_rejects_sweep_on_original_mesh_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config(root)
            path = root/"config.json"
            data = json.loads(path.read_text())
            data["metrics"]["physical_validity"] = {
                "enabled": True, "backend": "fcl", "options": {"penetration_thresholds_m": THRESHOLDS}}
            path.write_text(json.dumps(data))
            with self.assertRaises(ConfigError):
                load_config(path)


@unittest.skipUnless(GEOMETRY_AVAILABLE, "requires numpy, trimesh and python-fcl")
class ThresholdSweepGeometryTests(unittest.TestCase):
    def test_five_analytic_depths_share_only_five_fcl_queries(self):
        import fcl
        import trimesh
        objects = []
        # Five isolated pairs: .05, .2, .4, .8 and 1.2 mm penetration.
        for index, depth in enumerate((.00005, .0002, .0004, .0008, .0012)):
            for side, x in (("a", 0), ("b", 1-depth)):
                mesh = trimesh.creation.box()
                mesh.apply_translation([x, index*10, 0])
                objects.append({"id": f"{index}_{side}", "parts": [mesh]})
        with patch("fcl.collide", wraps=fcl.collide) as collide:
            primary, details = evaluate_convex_objects(objects, penetration_thresholds_m=THRESHOLDS)
        self.assertEqual(collide.call_count, 5)
        self.assertEqual(primary["ColS"], 100)
        self.assertAlmostEqual(primary["ColO"], 100/45)
        rows = details["threshold_sweep"]
        self.assertEqual([row["intersecting_pair_count"] for row in rows], [5, 4, 3, 2, 1])
        self.assertTrue(details["threshold_sweep_geometry_queries_reused"])
        self.assertTrue(all(len(row["pair_decisions"]) == 45 for row in rows))
        self.assertTrue(all(row["total_pairs"] == 45 for row in rows))
        self.assertEqual(rows[-1]["ColO"], primary["ColO"])
        self.assertEqual(rows[-1]["intersecting_pairs"], details["intersecting_pairs"])
        json.dumps(details, allow_nan=False)

    def test_threshold_boundary_uses_same_numerical_guard(self):
        import trimesh
        objects = [{"id": name, "parts": [trimesh.creation.box()]} for name in ("a", "b")]
        objects[1]["parts"][0].apply_translation([1-.0003, 0, 0])
        _, details = evaluate_convex_objects(objects, penetration_thresholds_m=THRESHOLDS)
        self.assertEqual([row["intersecting_pair_count"] for row in details["threshold_sweep"]],
                         [1, 1, 0, 0, 0])


@unittest.skipUnless(VHACD_AVAILABLE, "requires VHACDx and FCL")
class ThresholdSweepPipelineTests(unittest.TestCase):
    def test_nonconvex_mesh_decomposed_once_for_all_thresholds(self):
        import trimesh
        import vhacdx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"inputs"
            source.mkdir()
            first = trimesh.creation.box(extents=[.2, .2, .2])
            first.apply_translation([-.5, 0, 0])
            second = first.copy()
            second.apply_translation([1, 0, 0])
            trimesh.util.concatenate([first, second]).export(source/"nonconvex.obj")
            trimesh.creation.box(extents=[.1, .1, .1]).export(source/"probe.obj")
            manifest = source/"scene.json"
            manifest.write_text(json.dumps({"units": "meters", "objects": [
                {"id": "frame", "mesh": "nonconvex.obj"},
                {"id": "probe", "mesh": "probe.obj"},
                {"id": "table", "mesh": "not-loaded.obj", "is_table": True}]}))
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            with patch("vhacdx.compute_vhacd", wraps=vhacdx.compute_vhacd) as decompose:
                values = evaluate_physical(manifest, {
                    "penetration_thresholds_m": THRESHOLDS, "cache_dir": str(root/"cache"),
                    "decomposition": {"resolution": 10000, "max_convex_hulls": 16}})
            self.assertEqual(decompose.call_count, 1)
            self.assertEqual(values, {"ColO": 0, "ColS": 0})
            self.assertEqual(len(RUNTIME_DETAILS["threshold_sweep"]), 5)
            self.assertEqual(RUNTIME_DETAILS["excluded_table_ids"], ["table"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})


if __name__ == "__main__":
    unittest.main()
