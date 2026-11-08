"""Integration checks of geometry, units and persisted VHACD preprocessing."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from metrics.physical_validity.metric import RUNTIME_DETAILS, evaluate_physical, validate_options

_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "trimesh", "vhacdx", "fcl"))


class MeshPenetrationOptionsTests(unittest.TestCase):
    def test_invalid_options_rejected_without_native_query(self):
        invalid = [None, [], {"unexpected": 1}, {"min_penetration_m": True},
                   {"min_penetration_m": -1}, {"min_penetration_m": float("nan")},
                   {"numerical_tolerance_m": float("inf")}, {"cache_dir": ""},
                   {"decomposition": []}, {"decomposition": {"asyncACD": True}},
                   {"decomposition": {"resolution": True}},
                   {"decomposition": {"max_vertices_per_hull": 3}},
                   {"decomposition": {"max_convex_hulls": 0}},
                   {"decomposition": {"shrink_wrap": 1}},
                   {"decomposition": {"fill_mode": "invalid"}},
                   {"decomposition": {"minimum_volume_percent_error": float("nan")}}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(options)


@unittest.skipUnless(_AVAILABLE, "requires real VHACDx and FCL")
class MeshPenetrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import trimesh
        self.np, self.trimesh = np, trimesh
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.cache = self.root / "cache"
        self.manifest = self.inputs / "scene.json"
        self.box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        self.box.export(self.inputs / "box.obj")
        self.options = {"cache_dir": str(self.cache), "decomposition": {
            "resolution": 10000, "max_convex_hulls": 16}}

    def _write_manifest(self, objects, **extras):
        self.manifest.write_text(json.dumps({"units": "meters", "objects": objects, **extras}))
        return self.manifest

    def _objects(self, x=0.2):
        transform = self.np.eye(4)
        transform[0, 3] = x
        return [{"id": "a", "mesh": "box.obj"},
                {"id": "b", "mesh": "box.obj", "transform": transform.tolist()}]

    def _l_shape(self):
        np = self.np
        polygon = np.asarray([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
        vertices = np.vstack((np.column_stack((polygon, np.zeros(6))),
                              np.column_stack((polygon, np.ones(6)))))
        cap = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
        faces = [[c, b, a] for a, b, c in cap] + [[a + 6, b + 6, c + 6] for a, b, c in cap]
        for i in range(6):
            j = (i + 1) % 6
            faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
        mesh = self.trimesh.Trimesh(vertices, faces, process=False)
        self.assertTrue(mesh.is_volume)
        self.assertFalse(mesh.is_convex)
        mesh.export(self.inputs / "l.obj")
        return mesh

    def test_exact_convex_threshold_and_table_exclusion(self):
        for depth, expected in ((0.0005, 0.0), (0.001, 0.0), (0.0015, 100.0)):
            with self.subTest(depth=depth):
                objects = self._objects(x=0.1 - depth)
                objects.append({"id": "table", "mesh": "not-loaded.obj", "is_table": True})
                self._write_manifest(objects)
                with patch("vhacdx.compute_vhacd", side_effect=AssertionError("convex bypass expected")):
                    values = evaluate_physical(self.manifest, self.options)
                self.assertEqual(values, {"ColO": expected, "ColS": expected})
                self.assertEqual(RUNTIME_DETAILS["excluded_table_ids"], ["table"])
                self.assertEqual(RUNTIME_DETAILS["total_pairs"], 1)
                self.assertTrue(all(record["construction"] == "exact_convex_bypass"
                                    for record in RUNTIME_DETAILS["meshes"]))

    def test_full_affine_scaling_applied_to_vertices(self):
        objects = self._objects(x=0.1485)
        transform = self.np.diag([2.0, 1.5, 0.75, 1.0])
        objects[0]["transform"] = transform.tolist()
        self._write_manifest(objects)
        values = evaluate_physical(self.manifest, self.options)
        self.assertEqual(values["ColO"], 100.0)
        self.assertAlmostEqual(RUNTIME_DETAILS["pair_records"][0]["max_penetration_depth_m"],
                               0.0015, places=7)

    def test_glb_scene_node_transform_is_baked(self):
        # Source mesh is shifted by the node, then manifest shifts it back.
        scene = self.trimesh.Scene()
        node = self.np.eye(4)
        node[0, 3] = 1.0
        scene.add_geometry(self.box, transform=node)
        scene.export(self.inputs / "node.glb")
        objects = self._objects(x=0.098)
        objects[0]["mesh"] = "node.glb"
        outer = self.np.eye(4)
        outer[0, 3] = -1.0
        objects[0]["transform"] = outer.tolist()
        self._write_manifest(objects)
        values = evaluate_physical(self.manifest, self.options)
        self.assertEqual(values["ColO"], 100.0)
        self.assertAlmostEqual(RUNTIME_DETAILS["pair_records"][0]["max_penetration_depth_m"],
                               0.002, places=6)

    def test_real_nonconvex_vhacd_preserves_cutout_and_reuses_cache(self):
        original = self._l_shape()
        small = self.trimesh.creation.box(extents=[0.4, 0.4, 0.4])
        small.export(self.inputs / "small.obj")
        transform = self.np.eye(4)
        transform[:3, 3] = [1.5, 1.5, 0.5]
        objects = [{"id": "l", "mesh": "l.obj"},
                   {"id": "probe", "mesh": "small.obj", "transform": transform.tolist()}]
        self._write_manifest(objects)
        original_sha = hashlib.sha256((self.inputs / "l.obj").read_bytes()).hexdigest()
        values = evaluate_physical(self.manifest, self.options)
        self.assertEqual(values["ColO"], 0.0)
        record = RUNTIME_DETAILS["meshes"][0]
        self.assertEqual(record["construction"], "vhacd_approximation")
        self.assertGreater(record["part_count"], 1)
        self.assertFalse(record["cache_hit"])
        self.assertAlmostEqual(record["sum_convex_part_volumes_local_units3"],
                               original.volume, delta=original.volume * 0.2)
        cache_key = record["cache_key"]
        cache_sha = record["cache_sha256"]
        with patch("vhacdx.compute_vhacd", side_effect=AssertionError("cache must be reused")):
            second = evaluate_physical(self.manifest, self.options)
        self.assertEqual(second, values)
        self.assertTrue(RUNTIME_DETAILS["meshes"][0]["cache_hit"])
        self.assertEqual(RUNTIME_DETAILS["meshes"][0]["cache_key"], cache_key)
        self.assertEqual(RUNTIME_DETAILS["meshes"][0]["cache_sha256"], cache_sha)
        changed = deepcopy(self.options)
        changed["decomposition"]["resolution"] = 12000
        evaluate_physical(self.manifest, changed)
        self.assertNotEqual(RUNTIME_DETAILS["meshes"][0]["cache_key"], cache_key)
        self.assertFalse(RUNTIME_DETAILS["meshes"][0]["cache_hit"])
        self.assertEqual(hashlib.sha256((self.inputs / "l.obj").read_bytes()).hexdigest(), original_sha)
        self.assertFalse(RUNTIME_DETAILS["native_decomposition_parameters"]["asyncACD"])

    def test_units_and_affine_contract(self):
        for units in (None, "centimeters", "m"):
            self._write_manifest(self._objects(), units=units)
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, "units='meters'"):
                evaluate_physical(self.manifest, self.options)
        for transform in ([[1, 2], [3, 4]], self.np.zeros((4, 4)).tolist(),
                          (self.np.eye(4) * float("nan")).tolist()):
            objects = self._objects()
            objects[0]["transform"] = transform
            self._write_manifest(objects)
            with self.subTest(transform=transform), self.assertRaisesRegex(ValueError, "affine"):
                evaluate_physical(self.manifest, self.options)

    def test_open_surface_does_not_bypass_and_degenerate_native_parts_fail(self):
        import vhacdx
        open_mesh = self.box.copy()
        open_mesh.update_faces(self.np.arange(len(open_mesh.faces) - 1))
        self.assertFalse(open_mesh.is_watertight)
        open_mesh.export(self.inputs / "open.obj")
        objects = self._objects()
        objects[0]["mesh"] = "open.obj"
        self._write_manifest(objects)
        # Native flood+shrinkWrap on this open shell returns planar parts.
        # Do not discard those parts or add unrequested thickness to score it.
        with patch("vhacdx.compute_vhacd", wraps=vhacdx.compute_vhacd) as native:
            with self.assertRaisesRegex(ValueError, "source is_watertight=False.*positive-volume"):
                evaluate_physical(self.manifest, self.options)
            self.assertEqual(native.call_count, 1)
        self.assertFalse(any(self.cache.glob("*.npz")))

    def test_scale_provenance_and_source_cache_protection(self):
        source = {"source_layout_units": "service_relative_not_calibrated_meters",
                  "meters_per_unit": 1, "coordinate_and_units": {"calibrated": False}}
        self._write_manifest(self._objects(), source_export=source)
        evaluate_physical(self.manifest, self.options)
        self.assertEqual(RUNTIME_DETAILS["source_scale_metadata"], source)
        for directory in (self.inputs, self.inputs / "cache"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "outside"):
                evaluate_physical(self.manifest, {"cache_dir": str(directory)})

    def test_exact_cleanup_does_not_merge_nearby_vertices(self):
        from metrics.physical_validity.metric import _clean_exact
        # Two boxes separated by 0.1 mm must retain distinct close vertices.
        first = self.trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        second = first.copy()
        second.apply_translation([0.1001, 0, 0])
        mesh = self.trimesh.util.concatenate([first, second])
        cleaned, record = _clean_exact(mesh, self.np, self.trimesh)
        self.assertEqual(len(cleaned.vertices), 16)
        self.assertEqual(record["proximity_merge_distance_m"], 0.0)
        self.assertEqual(record["removed_zero_area_faces"], 0)


if __name__ == "__main__":
    unittest.main()
