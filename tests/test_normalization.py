"""Visual similarities preserve projection, geometry ratios and physical inputs."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from tabletopscene_eval.normalization import (normalize_camera,
    normalize_scene_objects, validate_visual_normalization)


class VisualNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root/"assets").mkdir()
        self.table = self.root/"assets/table.glb"
        self.prop = self.root/"assets/prop.glb"
        # Bake a nontrivial local node transform into the GLB's scene graph.
        scene = trimesh.Scene()
        local = trimesh.transformations.translation_matrix([.1, .4, -.2])
        scene.add_geometry(trimesh.creation.box([2., .8, 1.5]), transform=local)
        scene.export(self.table)
        trimesh.creation.box([.2, .4, .1]).export(self.prop)
        pose = trimesh.transformations.rotation_matrix(.3, [0, 1, 0])
        pose[:3, 3] = [4., .1, -2.]
        prop_pose = trimesh.transformations.rotation_matrix(.2, [1, 0, 0])
        prop_pose[:3, 3] = [4.3, 1.1, -2.2]
        self.objects = [{"id": "table", "mesh": str(self.table), "transform": pose.tolist(), "is_table": True},
                        {"id": "prop", "mesh": str(self.prop), "transform": prop_pose.tolist(), "is_table": False}]
        self.options = {"enabled": True, "mode": "tabletop_diameter", "target_diameter_m": 1.}

    def tearDown(self):
        self.temporary.cleanup()

    def bounds(self, obj):
        mesh = trimesh.load(obj["mesh"], force="scene", process=False).to_geometry()
        mesh.apply_transform(obj["transform"])
        return mesh.bounds

    def test_policy_rejects_ambiguous_scale_and_axis_controls(self):
        for bad in ({"enabled": 1}, {"target_diameter_m": True}, {"target_diameter_m": 0},
                    {"target_diameter_m": float("nan")}, {"scale_y": 1}, {"mode": "per_object"}):
            with self.subTest(options=bad), self.assertRaises(ValueError):
                validate_visual_normalization(bad)

    def test_disabled_policy_copies_without_reading_geometry(self):
        missing = [{"id": "no_file", "mesh": "/missing/input.glb", "nested": {"value": 1}}]
        copied, details = normalize_scene_objects(missing)
        self.assertEqual(copied, missing)
        copied[0]["nested"]["value"] = 2
        self.assertEqual(missing[0]["nested"]["value"], 1)
        self.assertFalse(details["enabled"])

    def test_one_common_uniform_similarity_centers_table_and_preserves_scene_ratios(self):
        original = deepcopy(self.objects)
        hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.table, self.prop)}
        normalized, details = normalize_scene_objects(self.objects, self.options)
        bounds = self.bounds(normalized[0]); extent = bounds[1]-bounds[0]
        self.assertAlmostEqual(float(max(extent[0], extent[2])), 1., places=7)
        np.testing.assert_allclose((bounds[0]+bounds[1])[[0, 2]]/2, [0, 0], atol=1e-7)
        self.assertAlmostEqual(float(bounds[1, 1]), 0, places=7)
        scale = details["uniform_scale"]
        world = np.array(details["normalized_world_from_source_world"])
        np.testing.assert_allclose(world[:3, :3], scale*np.eye(3), atol=1e-12)
        before_points, after_points = [], []
        for before, after in zip(self.objects, normalized):
            np.testing.assert_allclose(after["transform"], world @ np.array(before["transform"]), atol=1e-12)
            point = np.array([.04, .02, -.06, 1])
            before_points.append((np.array(before["transform"])@point)[:3])
            after_points.append((np.array(after["transform"])@point)[:3])
        self.assertAlmostEqual(np.linalg.norm(after_points[0]-after_points[1]),
                               scale*np.linalg.norm(before_points[0]-before_points[1]), places=10)
        self.assertAlmostEqual(float(extent[1]), float((self.bounds(original[0])[1, 1]-self.bounds(original[0])[0, 1])*scale), places=7)
        self.assertEqual(self.objects, original)
        self.assertEqual(hashes, details["source_asset_sha256"])
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.table, self.prop)})
        self.assertTrue(details["physics_geometry_unchanged"])

    def test_exactly_one_table_is_required(self):
        for objects in ([], [self.objects[1]], [self.objects[0], {**self.objects[0], "id": "second_table"}]):
            with self.subTest(objects=len(objects)), self.assertRaisesRegex(ValueError, "nonempty|exactly one"):
                normalize_scene_objects(objects, self.options)

    def camera(self):
        pose = trimesh.transformations.rotation_matrix(.2, [1, 0, 0]); pose[:3, 3] = [4., 2., 1.]
        return {"width": 160, "height": 120, "camera_to_world": pose.tolist(),
                "intrinsics_pixels": [[180, 0, 73], [0, 170, 64], [0, 0, 1]],
                "target_world": [4, .9, -2], "distance": 3.2, "near": .01, "far": 100}

    def test_export_camera_preserves_rigidity_intrinsics_and_pixel_projection(self):
        _, details = normalize_scene_objects(self.objects, self.options)
        camera = self.camera(); original = deepcopy(camera)
        result = normalize_camera(camera, details)
        self.assertEqual(result["intrinsics_pixels"], camera["intrinsics_pixels"])
        before_pose, after_pose = np.array(camera["camera_to_world"]), np.array(result["camera_to_world"])
        np.testing.assert_array_equal(after_pose[:3, :3], before_pose[:3, :3])
        np.testing.assert_allclose(after_pose[:3, :3].T@after_pose[:3, :3], np.eye(3), atol=1e-12)
        point = np.array([4.1, 1., -2.1, 1.])
        transformed = np.array(details["normalized_world_from_source_world"])@point
        before_camera = np.linalg.inv(before_pose)@point
        after_camera = np.linalg.inv(after_pose)@transformed
        np.testing.assert_allclose(after_camera[:3], details["uniform_scale"]*before_camera[:3], atol=1e-12)
        np.testing.assert_allclose(after_camera[:2]/after_camera[2], before_camera[:2]/before_camera[2], atol=1e-12)
        for field in ("near", "far", "distance"):
            self.assertEqual(result[field], camera[field]*details["uniform_scale"])
        self.assertEqual(camera, original)

    def test_blender_controller_normalizes_only_render_copy_and_export_camera(self):
        from tabletopscene_eval.rendering.blender_backend import prepare_scene_export
        exported = SimpleNamespace(objects=self.objects, camera=self.camera(), provenance={})
        captured = {}
        def render(objects, output, views, options, foreground):
            captured.update(objects=objects, views=views)
            result = [{**v, "render": str(Path(output)/(v["id"]+".png"))} for v in views]
            return {"all_views": result, "views": result, "details": {}}
        original = deepcopy(self.objects)
        with patch("tabletopscene_eval.adapters.scene_export.load_scene_export", return_value=exported), \
             patch("tabletopscene_eval.rendering.blender_backend.render_objects", side_effect=render):
            prepared = prepare_scene_export(self.root, self.root, {"mode": "export_camera"},
                                            visual_normalization=self.options)
        expected, details = normalize_scene_objects(original, self.options)
        self.assertEqual(captured["objects"], expected)
        self.assertEqual(self.objects, original)
        self.assertEqual(prepared["details"]["visual_normalization"]["uniform_scale"], details["uniform_scale"])
        camera = captured["views"][0]["camera"]
        expected_camera = normalize_camera(exported.camera, details)
        self.assertEqual(camera["position_y_up_m"], [row[3] for row in expected_camera["camera_to_world"][:3]])

    @unittest.skipUnless(os.environ.get("TABLETOPSCENE_TEST_BLENDER_RENDER") == "1", "enable real Blender rendering")
    def test_real_blender_retains_uniform_scale_and_common_object_offsets(self):
        from tabletopscene_eval.rendering.blender_backend import render_objects
        normalized, details = normalize_scene_objects(self.objects, self.options)
        views = [{"id": "normalized", "camera": {"position_y_up_m": [0, .8, 2],
                    "target_y_up_m": [0, -.1, 0], "up_y_up": [0, 1, 0]}}]
        result = render_objects(normalized, self.root/"render", views,
                                {"resolution": [64, 64], "device": "CPU", "samples": 1, "denoise": False},
                                {"enabled": True})
        conversion = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        for obj, actor in zip(normalized, result["details"]["rendering"]["actors"]):
            expected = trimesh.load(obj["mesh"], force="scene", process=False).to_geometry()
            expected.apply_transform(conversion @ np.array(obj["transform"]))
            np.testing.assert_allclose(actor["world_bounds_z_up"], expected.bounds, atol=2e-6)
        self.assertGreater(result["details"]["rendering"]["views"][0]["foreground_pixel_count"], 0)
        self.assertEqual(result["details"]["rendering"]["max_pose_element_drift"], 0)


if __name__ == "__main__":
    unittest.main()
