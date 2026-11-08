"""Shared support-table framing, full-circle coverage and immutable geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from tabletopscene_eval.camera_geometry import (
    resolve_shared_table_camera_sampling, validate_framing_options,
)
from tabletopscene_eval.rendering.cameras import make_view_specs, _export_view


class SharedTableCameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile = json.loads((Path(__file__).resolve().parents[1]
                                   / "configs/camera.json").read_text())["camera_sampling"]

    def tearDown(self):
        self.temporary.cleanup()

    def table(self, name, extents=(1.0, .75, 1.0), offset=(0, 0, 0)):
        path = self.root / f"{name}.glb"
        trimesh.creation.box(extents=extents).export(path)
        pose = np.eye(4)
        pose[:3, 3] = [offset[0], extents[1]/2 + offset[1], offset[2]]
        return {"id": name, "mesh": str(path), "transform": pose.tolist(), "is_table": True}

    def assert_table_fits(self, table, sampling, resolution, vertical_fov_deg):
        mesh = trimesh.load(table["mesh"], force="scene", process=False).to_geometry()
        mesh.apply_transform(table["transform"])
        tangent_y = np.tan(np.radians(vertical_fov_deg)/2)
        tangent_x = tangent_y * resolution[0]/resolution[1]
        views = make_view_specs(sampling, None, resolution)
        for view in views:
            camera = view["camera"]
            position = np.asarray(camera["position_y_up_m"])
            back = position - np.asarray(camera["target_y_up_m"])
            back /= np.linalg.norm(back)
            up = np.asarray(camera["up_y_up"])
            right = np.cross(up, back)
            relative = np.asarray(mesh.vertices) - position
            depth = -(relative @ back)
            self.assertTrue((depth > 0).all())
            self.assertTrue((np.abs(relative @ right) <= tangent_x*depth + 1e-10).all())
            self.assertTrue((np.abs(relative @ up) <= tangent_y*depth + 1e-10).all())

    def test_single_table_bounds_center_and_offset_are_explicit(self):
        table = self.table("table", (1, .8, 1), (.3, .2, -.7))
        profile = deepcopy(self.profile)
        profile["framing"].update(target_height="bounds_center", target_height_offset_m=.15)
        sampling, details = resolve_shared_table_camera_sampling(profile, {"only": [table]}, [256, 256])
        np.testing.assert_allclose(sampling["target_y_up_m"], [.3, .75, -.7], atol=1e-6)
        self.assertEqual(details["target_height_rule"], "combined_bounds_vertical_midpoint")
        self.assert_table_fits(table, sampling, [256, 256], 45)

    def test_resolving_same_union_is_independent_of_method_order(self):
        a, b = self.table("a"), self.table("b", (1.3, .9, 1.1))
        first = resolve_shared_table_camera_sampling(self.profile, {"A": [a], "B": [b]}, [300, 200])
        second = resolve_shared_table_camera_sampling(self.profile, {"B": [b], "A": [a]}, [300, 200])
        self.assertEqual(first, second)

    def test_no_framing_preserves_explicit_protocol_without_loading_geometry(self):
        original = {"mode": "orbit_grid", "target_y_up_m": [1, 2, 3], "radius_m": 2,
                    "elevations_deg": [40], "azimuths_deg": [0]}
        copied, details = resolve_shared_table_camera_sampling(original, None, None, None)
        self.assertEqual(copied, original)
        self.assertIsNone(details)
        copied["target_y_up_m"][0] = 999
        self.assertEqual(original["target_y_up_m"][0], 1)

    def test_anchor_metadata_never_changes_camera_geometry(self):
        table = self.table("table")
        sampling, _ = resolve_shared_table_camera_sampling(self.profile, {"a": [table]}, [128, 96])
        with_anchor = make_view_specs(sampling, None, [128, 96])
        sampling.pop("anchor_view_id")
        self.assertEqual(with_anchor, make_view_specs(sampling, None, [128, 96]))
        for invalid in (None, "", " ", 12, True):
            with self.subTest(anchor=invalid), self.assertRaisesRegex(ValueError, "anchor_view_id"):
                make_view_specs({**sampling, "anchor_view_id": invalid}, None, [128, 96])

    def test_projection_profile_full_circle_preserves_shared_rig_and_margin(self):
        a = self.table("a", (1.0, .72, .98))
        b = self.table("b", (.99, .74, 1.0))
        # Place both tabletops at zero, as a visual-normalization caller does.
        a["transform"][1][3] = -.72/2
        b["transform"][1][3] = -.74/2
        methods = {"A": [a], "B": [b]}
        original = deepcopy(methods)
        digests = {table["mesh"]: hashlib.sha256(Path(table["mesh"]).read_bytes()).hexdigest()
                   for table in (a, b)}
        for resolution in ([1312, 1199], [200, 400], [400, 200]):
            sampling, details = resolve_shared_table_camera_sampling(
                self.profile, methods, resolution, 45)
            self.assertEqual(details["mode"], "shared_table_projection")
            self.assertEqual(details["target_height"], "bounds_center")
            np.testing.assert_allclose(sampling["target_y_up_m"], [0, -.37, 0], atol=1e-6)
            self.assertFalse(details["normalization_performed_by_camera_helper"])
            self.assertEqual(len(details["projection_view_constraints"]), 16)
            self.assertAlmostEqual(sampling["radius_m"], max(
                item["minimum_distance_m"] for item in details["projection_view_constraints"]))
            views = make_view_specs(sampling, None, resolution)
            self.assertEqual(len(views), 16)
            self.assertEqual({view["camera"]["elevation_deg"] for view in views}, {25, 35})
            anchor = next(view for view in views if view["id"] == sampling["anchor_view_id"])
            self.assertEqual(anchor["camera"]["elevation_deg"], 35)
            self.assertEqual(anchor["camera"]["azimuth_deg"], 0)
            for table in (a, b):
                self.assert_table_fits(table, sampling, resolution, 45)
        self.assertEqual(methods, original)
        self.assertEqual(digests, {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in digests})

    def test_projection_distance_is_minimal_for_its_prescribed_margin(self):
        table = self.table("table", (1.0, .72, 1.0))
        table["transform"][1][3] = -.36
        sampling, details = resolve_shared_table_camera_sampling(
            self.profile, {"a": [table]}, [320, 240], 45)
        mesh = trimesh.load(table["mesh"], force="scene", process=False).to_geometry()
        mesh.apply_transform(table["transform"])
        limit = details["projection_margin_ndc_limit"]
        def maximum_ndc(distance):
            sample = {**sampling, "radius_m": distance}
            tangent_y = np.tan(np.radians(45)/2)
            tangent_x = tangent_y * 320/240
            maximum = 0.0
            for view in make_view_specs(sample, None, [320, 240]):
                camera = view["camera"]
                position = np.asarray(camera["position_y_up_m"])
                back = position - np.asarray(camera["target_y_up_m"])
                back /= np.linalg.norm(back)
                up = np.asarray(camera["up_y_up"])
                right = np.cross(up, back)
                relative = mesh.vertices - position
                depth = -(relative @ back)
                self.assertTrue((depth > 0).all())
                maximum = max(maximum, float(np.abs(relative @ right / (depth*tangent_x)).max()),
                              float(np.abs(relative @ up / (depth*tangent_y)).max()))
            return maximum
        self.assertAlmostEqual(maximum_ndc(sampling["radius_m"]), limit, places=10)
        # A smaller distance violates the prescribed padding, even if it may
        # still fit the image edges. This is an independent projection check.
        self.assertGreater(maximum_ndc(sampling["radius_m"]*.9999), limit)

    def test_projection_never_fits_an_oversized_non_table_object(self):
        table = self.table("table")
        huge = {"id": "oversized", "mesh": str(self.root/"missing.glb"), "is_table": False}
        first = resolve_shared_table_camera_sampling(self.profile, {"a": [table]}, [128, 96])
        second = resolve_shared_table_camera_sampling(self.profile, {"a": [table, huge]}, [128, 96])
        self.assertEqual(first, second)

    def test_projection_rejects_invalid_angle_grids(self):
        table = self.table("table")
        for update in ({"elevations_deg": []}, {"elevations_deg": [91]},
                       {"elevations_deg": [25, 25]}, {"elevations_deg": [True]},
                       {"azimuths_deg": [float("nan")]}, {"azimuths_deg": [0, 0]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                resolve_shared_table_camera_sampling({**self.profile, **update},
                                                     {"a": [table]}, [128, 96])

    def test_invalid_framing_and_ambiguous_explicit_settings_fail(self):
        for invalid in ({"margin": .99}, {"margin": True}, {"margin": float("nan")},
                        {"target_height": "center"}, {"target_height_offset_m": float("inf")},
                        {"mode": "per_method"}, {"fit_all_objects": True}, []):
            with self.subTest(framing=invalid), self.assertRaises(ValueError):
                validate_framing_options(invalid)
        for updates in ({"target_y_up_m": [0, 0, 0]}, {"radius_m": 1},
                        {"mode": "export_camera"}, {"distance_multipliers": [.9]},
                        {"distance_multipliers": [True]}, {"distance_multipliers": [1, 1]}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                resolve_shared_table_camera_sampling({**self.profile, **updates}, {}, [128, 96])

    def test_missing_or_duplicate_table_and_invalid_camera_fail(self):
        table = self.table("table")
        for methods in ({}, {"a": []}, {"a": [table, deepcopy(table)]}):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                resolve_shared_table_camera_sampling(self.profile, methods, [128, 96])
        for resolution, fov in (([0, 96], 45), ([128, True], 45), ([128.0, 96], 45),
                                ([128, 96], 0), ([128, 96], 179), ([128, 96], True)):
            with self.subTest(resolution=resolution, fov=fov), self.assertRaises(ValueError):
                resolve_shared_table_camera_sampling(self.profile, {"a": [table]}, resolution, fov)


class CameraSpecificationTests(unittest.TestCase):
    def test_export_calibration_survives_nonsquare_resize_and_roll(self):
        angle = .2
        pose = np.eye(4)
        pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        pose[:3, 3] = [1, 2, 3]
        camera = {"width": 160, "height": 120, "camera_to_world": pose.tolist(),
                  "intrinsics_pixels": [[180, 0, 75], [0, 170, 63], [0, 0, 1]], "near": .01, "far": 100}
        original = deepcopy(camera)
        native = make_view_specs({"mode": "export_camera"}, camera, [512, 512])[0]["camera"]
        resized = _export_view(camera, [320, 360])
        self.assertEqual(native["resolution"], [160, 120])
        self.assertEqual(native["intrinsics_pixels"], camera["intrinsics_pixels"])
        self.assertEqual(resized["intrinsics_pixels"], [[360, 0, 150], [0, 510, 189], [0, 0, 1]])
        self.assertEqual(resized["position_y_up_m"], [1, 2, 3])
        np.testing.assert_array_equal(resized["up_y_up"], pose[:3, 1])
        self.assertEqual(camera, original)

    def test_orbit_poles_have_finite_orthonormal_camera_bases(self):
        sampling = {"mode": "orbit_grid", "elevations_deg": [90, -90, 45],
                    "azimuths_deg": [0, 90], "distance_multipliers": [2],
                    "target_y_up_m": [1, 2, 3], "radius_m": .7}
        views = make_view_specs(sampling, None, [128, 96])
        self.assertEqual(len(views), 6)
        for view in views:
            camera = view["camera"]
            offset = np.array(camera["position_y_up_m"])-np.array(camera["target_y_up_m"])
            self.assertAlmostEqual(float(np.linalg.norm(offset)), 1.4)
            back = offset/np.linalg.norm(offset)
            up = np.array(camera["up_y_up"])
            self.assertAlmostEqual(float(np.linalg.norm(up)), 1)
            self.assertAlmostEqual(float(np.dot(up, back)), 0)
            self.assertTrue(np.isfinite(np.cross(up, back)).all())


if __name__ == "__main__":
    unittest.main()
