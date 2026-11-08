"""Analytic solid geometry checks for the convex-part collision metric."""
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from metrics.physical_validity.convex_geometry import evaluate_convex_objects


AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("numpy", "trimesh", "fcl"))


class ConvexInputValidationTest(unittest.TestCase):
    def test_rejects_invalid_thresholds_and_object_lists(self):
        for value in (True, False, -0.001, float("nan"), float("inf"), 10**1000, "0.001", None):
            for option in ("min_penetration_m", "numerical_tolerance_m"):
                with self.subTest(value=value, option=option), self.assertRaisesRegex(ValueError, option):
                    evaluate_convex_objects([], **{option: value})
        for objects in (None, [], [{"id": "a", "parts": []}], [{"id": "a", "parts": [None]}] * 2,
                        [{"id": "", "parts": []}, {"id": "b", "parts": []}],
                        [{"id": "a", "parts": []}, {"id": "b", "parts": []}]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                evaluate_convex_objects(objects)


@unittest.skipUnless(AVAILABLE, "requires numpy, trimesh and python-fcl")
class ConvexCollisionTest(unittest.TestCase):
    @staticmethod
    def box(extents=(1, 1, 1), translation=(0, 0, 0)):
        import trimesh
        mesh = trimesh.creation.box(extents)
        mesh.apply_translation(translation)
        return mesh

    def pair(self, depth):
        return [{"id": "a", "parts": [self.box()]},
                {"id": "b", "parts": [self.box(translation=(1 - depth, 0, 0))]}]

    def test_analytic_box_depths_and_one_millimeter_boundary(self):
        for depth, expected, raw in ((-0.01, 0, False), (0, 0, False),
                                     (0.0005, 0, True), (0.001, 0, True),
                                     (0.0010001, 100, True), (0.002, 100, True)):
            with self.subTest(depth=depth):
                values, details = evaluate_convex_objects(self.pair(depth))
                self.assertEqual(values, {"ColO": expected, "ColS": expected})
                self.assertEqual(details["raw_intersecting_pair_count"], int(raw))
                self.assertEqual(details["filtered_pair_count"], int(raw and expected == 0))
                self.assertAlmostEqual(details["pair_records"][0]["max_penetration_depth_m"], max(0, depth), places=10)
                self.assertEqual(details["pair_records"][0]["contact_count"], int(raw))
                self.assertEqual(details["fcl_geometry_type"], "Convex")
                json.dumps(details, allow_nan=False)

    def test_full_containment_is_detected_without_surface_crossing(self):
        objects = [{"id": "outer", "parts": [self.box((2, 2, 2))]},
                   {"id": "inner", "parts": [self.box((0.2, 0.2, 0.2))]}]
        values, details = evaluate_convex_objects(objects)
        self.assertEqual(values, {"ColO": 100, "ColS": 100})
        self.assertGreater(details["pair_records"][0]["max_penetration_depth_m"], 0.001)
        # MPR estimates depth: a centered containment can select an oblique
        # direction and overestimate the exact 1.1 m minimum translation.
        self.assertFalse(details["penetration_depth_is_exact_mtd"])
        objects[1]["parts"][0].apply_translation([0.001, 0, 0])
        self.assertAlmostEqual(evaluate_convex_objects(objects)[1]["pair_records"][0]["max_penetration_depth_m"], 1.099)

    def test_concave_object_gap_is_preserved_by_separate_parts(self):
        import trimesh
        parts = [self.box((0.2, 1, 1), (-0.5, 0, 0)), self.box((0.2, 1, 1), (0.5, 0, 0))]
        objects = [{"id": "frame", "parts": parts},
                   {"id": "inside_gap", "parts": [self.box((0.2, 0.2, 0.2))]}]
        values, details = evaluate_convex_objects(objects)
        self.assertEqual(values, {"ColO": 0, "ColS": 0})
        self.assertEqual(details["pair_records"][0]["aabb_rejected_part_pairs"], 2)
        # A single whole-object convex hull would fill this intentionally empty gap.
        objects[0]["parts"] = [trimesh.util.concatenate(parts).convex_hull]
        self.assertEqual(evaluate_convex_objects(objects)[0]["ColO"], 100)

    def test_all_part_pairs_checked_and_each_object_pair_counted_once(self):
        first_parts = [self.box(translation=(0, 0, 0)), self.box(translation=(0, 2, 0))]
        second_parts = [self.box(translation=(0.9995, 0, 0)), self.box(translation=(0.998, 2, 0))]
        objects = [{"id": "a", "parts": first_parts}, {"id": "b", "parts": second_parts},
                   {"id": "c", "parts": [self.box(translation=(10, 0, 0))]}]
        values, details = evaluate_convex_objects(objects)
        self.assertAlmostEqual(values["ColO"], 100 / 3)
        self.assertEqual(values["ColS"], 100)
        self.assertEqual(details["total_pairs"], 3)
        self.assertEqual(details["intersecting_pairs"], [["a", "b"]])
        self.assertEqual(len(details["pair_records"]), 3)
        record = details["pair_records"][0]
        self.assertAlmostEqual(record["max_penetration_depth_m"], 0.002)
        self.assertEqual(record["total_part_pairs"], 4)
        self.assertEqual(record["aabb_candidate_part_pairs"], 2)
        self.assertEqual(record["contact_count"], 2)

    def test_input_reversal_preserves_results_and_input_geometry(self):
        import numpy as np
        objects = [{"id": "b", "parts": [self.box(translation=(0.998, 0, 0)), self.box(translation=(5, 0, 0))]},
                   {"id": "a", "parts": [self.box(), self.box(translation=(10, 0, 0))]}]
        before = [[part.vertices.copy() for part in obj["parts"]] for obj in objects]
        normal = evaluate_convex_objects(objects)
        reverse = evaluate_convex_objects([{"id": obj["id"], "parts": list(reversed(obj["parts"]))}
                                          for obj in reversed(objects)])
        self.assertEqual(normal, reverse)
        self.assertEqual([obj["id"] for obj in objects], ["b", "a"])
        for obj, vertices in zip(objects, before):
            for part, original in zip(obj["parts"], vertices):
                np.testing.assert_array_equal(part.vertices, original)

    def test_rejects_open_nonconvex_and_nonfinite_parts(self):
        import trimesh
        opened = self.box()
        opened.update_faces([True] * 11 + [False])
        nonconvex = trimesh.util.concatenate([self.box(), self.box(translation=(2, 0, 0))])
        invalid = self.box()
        invalid.vertices[0, 0] = float("nan")
        for mesh in (opened, nonconvex, invalid, None):
            with self.subTest(mesh=type(mesh)), self.assertRaises(ValueError):
                evaluate_convex_objects([{"id": "a", "parts": [mesh]}, {"id": "b", "parts": [self.box()]}])

    def test_invalid_collision_contacts_fail_explicitly(self):
        for is_collision, depths in ((True, []), (False, [0.001]), (True, [float("nan")]),
                                    (True, [float("inf")]), (True, [-0.01])):
            def fake_collide(a, b, request, result):
                result.is_collision = is_collision
                result.contacts = [SimpleNamespace(penetration_depth=depth) for depth in depths]
            with self.subTest(is_collision=is_collision, depths=depths):
                with patch("fcl.collide", side_effect=fake_collide), self.assertRaises(RuntimeError):
                    evaluate_convex_objects(self.pair(0.002))


if __name__ == "__main__":
    unittest.main()
