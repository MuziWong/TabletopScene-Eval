"""Blender config and optional real cameras, transforms and occlusion checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tabletopscene_eval.rendering.blender_backend import (prepare_scene_export,
    render_objects, validate_rendering_options, y_up_pose_to_blender)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_MESH = HAS_NUMPY and importlib.util.find_spec("trimesh") is not None
RUN_NATIVE = HAS_MESH and os.environ.get("TABLETOPSCENE_TEST_BLENDER_RENDER") == "1"


class BlenderOptionsTests(unittest.TestCase):
    def test_independent_schema_and_explicit_device(self):
        validate_rendering_options({"backend": "blender", "device": "OPTIX", "resolution": [80, 64]})
        for bad in ({"lighting": {}}, {"python_executable": "python"}, {"renderer": "HYBRID"},
                    {"device": "AUTOMATIC"}, {"resolution": "reference"}, {"samples": True},
                    {"denoise": 1}, {"world": {"strength": float("nan")}},
                    {"color_management": {"exposure": float("inf")}},
                    {"camera": {"near_m": 1, "far_m": .1}}):
            with self.subTest(options=bad), self.assertRaises(ValueError):
                validate_rendering_options(bad)

    def test_studio_light_geometry_requires_a_defined_direction(self):
        good = {"lights": [{"type": "AREA", "position_y_up_m": [1, 2, 3],
                             "target_y_up_m": [0, 0, 0], "energy": 400, "size": 3}]}
        validate_rendering_options(good)
        for invalid in ({"type": "SUN", "position_y_up_m": [1, 2, 3]},
                        {"position_y_up_m": [0, 0, 0]},
                        {"position_y_up_m": [0, float("nan"), 0]}):
            with self.subTest(light=invalid), self.assertRaises(ValueError):
                validate_rendering_options({"lights": [invalid]})

@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class BlenderTransformTests(unittest.TestCase):
    def test_conjugation_preserves_points_and_a_rotated_plane_normal(self):
        import numpy as np
        # A tilted source plane's normal and an off-origin point must rotate
        # with its rigid pose, after the importer has changed the local basis.
        angle = .7
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = [.2, .4, -.3]
        c = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        point, normal = np.array([.1, .2, .3, 1]), np.array([0, 1, 0, 0])
        result = y_up_pose_to_blender(pose)
        np.testing.assert_allclose(result @ (c @ point), c @ (pose @ point), atol=1e-12)
        np.testing.assert_allclose(result @ (c @ normal), c @ (pose @ normal), atol=1e-12)
        np.testing.assert_array_equal(pose[:3, 3], [.2, .4, -.3])

    def test_anchor_id_uses_the_requested_candidate_without_export_camera(self):
        import numpy as np
        camera = {"width": 80, "height": 64, "camera_to_world": np.eye(4).tolist(),
                  "intrinsics_pixels": [[80, 0, 40], [0, 80, 32], [0, 0, 1]], "near": .01, "far": 10}
        exported = SimpleNamespace(objects=[], camera=camera, provenance={"fixture": True})
        sampling = {"mode": "orbit_grid", "elevations_deg": [25], "azimuths_deg": [0, 180],
                    "distance_multipliers": [3], "target_y_up_m": [0, 0, 0], "radius_m": .5,
                    "anchor_view_id": "e00_a01_d00"}
        captured = {}
        def worker(objects, output, specs, options, foreground):
            captured["specs"] = specs
            rendered = [{**view, "render": str(Path(output)/(view["id"]+".png")),
                         "mask": str(Path(output)/(view["id"]+"_mask.png"))} for view in specs]
            return {"views": rendered, "all_views": rendered, "details": {}}
        with tempfile.TemporaryDirectory() as output, \
             patch("tabletopscene_eval.adapters.scene_export.load_scene_export", return_value=exported), \
             patch("tabletopscene_eval.rendering.blender_backend.render_objects", side_effect=worker):
            result = prepare_scene_export(output, output, sampling, {"resolution": [80, 64]},
                                          foreground_options={"enabled": True})
        self.assertEqual(len(captured["specs"]), 2)
        self.assertTrue(result["render"].endswith("e00_a01_d00.png"))
        self.assertTrue(result["render_mask"].endswith("e00_a01_d00_mask.png"))
        self.assertNotIn("export_camera", {view["id"] for view in captured["specs"]})


@unittest.skipUnless(RUN_NATIVE, "set TABLETOPSCENE_TEST_BLENDER_RENDER=1 for real Blender")
class BlenderNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable = os.environ.get("TABLETOPSCENE_BLENDER_EXECUTABLE", "/snap/bin/blender")

    def test_real_anisotropic_intrinsics_and_table_occludes_excluded_foreground(self):
        import numpy as np
        from PIL import Image
        import trimesh
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"assets").mkdir()
            asset = root/"assets/box.glb"
            trimesh.creation.box(extents=[.3, .1, .2]).export(asset)
            objects = []
            for identity, z, table in (("table", .15, True), ("box", 0, False)):
                pose = np.eye(4); pose[2, 3] = z
                objects.append({"id": identity, "mesh": str(asset), "transform": pose.tolist(), "is_table": table})
            original = copy.deepcopy(objects)
            k = [[80., 0., 43.], [0., 100., 29.], [0., 0., 1.]]
            views = [{"id": label, "camera": {"position_y_up_m": [0, 0, z], "target_y_up_m": [0, 0, 0],
                                              "up_y_up": [0, 1, 0], "intrinsics_pixels": k,
                                              "resolution": [96, 64]}} for label, z in (("front", 2), ("back", -2))]
            result = render_objects(objects, root/"render", views,
                                    {"executable": self.executable, "resolution": [96, 64], "samples": 2,
                                     "device": "CPU", "denoise": False},
                                    {"enabled": True, "include_table": False})
            front, back = result["details"]["rendering"]["views"]
            np.testing.assert_allclose(front["actual_intrinsics_pixels"], k, atol=2e-3)
            with Image.open(front["mask"]) as image: self.assertEqual(np.count_nonzero(np.asarray(image)), 0)
            with Image.open(back["mask"]) as image: self.assertGreater(np.count_nonzero(np.asarray(image)), 0)
            self.assertEqual(front["visible_object_ids"], ["table"])
            self.assertIn("box", back["visible_object_ids"])
            self.assertEqual(objects, original)
            self.assertEqual(result["details"]["rendering"]["max_pose_element_drift"], 0)

    def test_real_imported_node_transform_and_object_pose_apply_once(self):
        import numpy as np
        import trimesh
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"assets").mkdir()
            scene = trimesh.Scene()
            node = trimesh.transformations.rotation_matrix(.6, [0, 0, 1]); node[:3, 3] = [.2, .3, -.1]
            scene.add_geometry(trimesh.creation.box([.2, .1, .3]), node_name="offset_child", transform=node)
            asset = root/"assets/hierarchy.glb"; scene.export(asset)
            pose = trimesh.transformations.rotation_matrix(.4, [0, 1, 0]); pose[:3, 3] = [-.1, .1, .2]
            expected = scene.to_geometry(); expected.apply_transform(pose)
            c = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            expected.apply_transform(c)
            obj = {"id": "offset", "mesh": str(asset), "transform": pose.tolist(), "is_table": False}
            view = {"id": "fixed", "camera": {"position_y_up_m": [0, 1, 3], "target_y_up_m": [0, .2, 0], "up_y_up": [0, 1, 0]}}
            result = render_objects([obj], root/"render", [view],
                                    {"executable": self.executable, "resolution": [64, 64], "samples": 1,
                                     "device": "CPU", "denoise": False}, {"enabled": True})
            actor = result["details"]["rendering"]["actors"][0]
            np.testing.assert_allclose(actor["world_bounds_z_up"], expected.bounds, atol=2e-6)
            self.assertGreater(result["details"]["rendering"]["views"][0]["foreground_pixel_count"], 0)

    def test_real_32_view_budget_binary_masks_and_white_foreground_composite(self):
        import numpy as np
        from PIL import Image
        import trimesh
        from tabletopscene_eval.foreground import composite_foreground
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"assets").mkdir()
            asset = root/"assets/box.glb"
            trimesh.creation.box([.3, .3, .3]).export(asset)
            obj = {"id": "box", "mesh": str(asset), "transform": np.eye(4).tolist(), "is_table": False}
            views = []
            for index in range(32):
                angle = 2*np.pi*index/32
                views.append({"id": f"orbit_{index:02}", "camera": {
                    "position_y_up_m": [2*np.sin(angle), .5, 2*np.cos(angle)],
                    "target_y_up_m": [0, 0, 0], "up_y_up": [0, 1, 0]}})
            result = render_objects([obj], root/"render", views,
                                    {"executable": self.executable, "resolution": [32, 32], "samples": 1,
                                     "device": "CPU", "denoise": False}, {"enabled": True})
            self.assertEqual(len(result["views"]), 32)
            self.assertEqual(len({v["render"] for v in result["views"]}), 32)
            for view in result["views"]:
                with Image.open(view["mask"]) as image:
                    mask = np.asarray(image)
                    self.assertEqual(set(np.unique(mask)), {0, 255})
                with Image.open(view["render"]) as image:
                    self.assertEqual(image.mode, "RGBA")
                    rgba = np.asarray(image)
                self.assertFalse(np.any((mask > 127) & (rgba[:, :, 3] == 0)))
                destination = root/(view["id"]+"_foreground.png")
                composite_foreground(view["render"], view["mask"], destination, {"enabled": True})
                with Image.open(destination) as image: rgb = np.asarray(image)
                self.assertTrue((rgb[mask == 0] == 255).all())
                np.testing.assert_array_equal(rgb[mask == 255], rgba[mask == 255, :3])
            self.assertEqual(result["details"]["rendering"]["max_pose_element_drift"], 0)


if __name__ == "__main__":
    unittest.main()
