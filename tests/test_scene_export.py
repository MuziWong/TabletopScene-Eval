"""Portable export ingestion, identity mapping, units and frame checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tabletopscene_eval.adapters.scene_export import SceneExportError, load_scene_export


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class SceneExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "scene_export"
        self.root.mkdir()
        self.pose = [[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]]
        self.layout = {
            "schema_version": "agentable.canonical-scene-layout/v1",
            "coordinate_system": "glb_y_up_world", "layout_units": "meters",
            "scale_applied": True, "table_id": "table", "instances": [],
        }
        self.config = {"format": "embodichain.scene-export/v1", "scene_id": "fixture",
                       "background": [], "rigid_object": [], "articulation": []}
        self.result = {
            "schema_version": "agentable.scene-export/v1", "success": True,
            "status": "scene_exported", "output_directory": "/old/location",
            "canonical_camera_path": "/old/location/canonical_camera.json",
            "meters_per_unit": .01, "source_layout_units": "service_relative_not_calibrated_meters",
            "scale_policy": "explicit_conversion_factor_not_a_real_size_calibration", "assets": {},
        }
        for identity in ("table", "cup_1"):
            relative = f"mesh_assets/{identity}/{identity}.glb"
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"mesh fixture bytes " + identity.encode())
            self.layout["instances"].append({"object_id": identity, "geometry_path": relative,
                                              "matrix": IDENTITY if identity == "table" else self.pose,
                                              "scale": [1, 1, 1]})
            self.result["assets"][identity] = {
                "source_object_id": "source_" + identity, "source_geometry_path": "/unavailable/source.glb",
                "source_sha256": "source_hash_from_export",
                "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            entry = {"uid": identity, "shape": {"shape_type": "Mesh", "fpath": relative},
                     "body_scale": [1, 1, 1], "is_articulated": False,
                     "body_type": "kinematic" if identity == "table" else "dynamic",
                     "init_pos": [0, 0, 0] if identity == "table" else [1, -3, 2],
                     "init_rot": [0, 0, 0] if identity == "table" else [0, 0, 90]}
            self.config["background" if identity == "table" else "rigid_object"].append(entry)
        self.camera = {"projection": "pinhole", "width": 640, "height": 480,
                       "camera_to_world": IDENTITY, "target_world": [0, 0, -1],
                       "intrinsics_pixels": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                       "near": .01, "far": 100, "horizontal_fov_deg": 56.1449738717}
        self.save()

    def save(self):
        for filename, value in (("result.json", self.result), ("canonical_layout.json", self.layout),
                                ("scene_config.json", self.config), ("canonical_camera.json", self.camera)):
            (self.root / filename).write_text(json.dumps(value), encoding="utf-8")

    def test_canonical_geometry_not_double_transformed_and_sources_preserved(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        exported = load_scene_export(self.root)
        self.assertEqual(exported.objects[1]["transform"], self.pose)
        self.assertEqual(exported.objects[0]["transform"], IDENTITY)
        self.assertEqual(exported.objects[0]["is_table"], True)
        self.assertEqual(exported.objects[1]["mesh"], str(self.root / "mesh_assets/cup_1/cup_1.glb"))
        self.assertEqual(exported.camera, self.camera)
        self.assertEqual(exported.provenance["source_assets"], self.result["assets"])
        manifest = exported.collision_manifest()
        self.assertEqual(manifest["units"], "meters")
        manifest["objects"][1]["transform"][0][3] = 999
        self.assertEqual(exported.objects[1]["transform"][0][3], 1)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_result_json_and_relocated_package_ignore_stale_output_paths(self):
        relocated = self.root.parent / "relocated"
        shutil.copytree(self.root, relocated)
        self.root.rename(self.root.parent / "original_unavailable")
        exported = load_scene_export(relocated / "result.json")
        self.assertEqual(exported.root, relocated)
        self.assertEqual(exported.camera_path, relocated / "canonical_camera.json")
        self.assertTrue(all(Path(item["mesh"]).is_relative_to(relocated) for item in exported.objects))

    def test_reject_changed_asset_and_incomplete_export(self):
        mesh = self.root / "mesh_assets/cup_1/cup_1.glb"
        mesh.write_bytes(b"changed mesh")
        with self.assertRaisesRegex(SceneExportError, "SHA256"):
            load_scene_export(self.root)
        self.result["success"] = False
        self.save()
        with self.assertRaisesRegex(SceneExportError, "successful"):
            load_scene_export(self.root)

    def test_reject_unit_scale_and_pose_mismatches(self):
        original = deepcopy(self.layout)
        for change, message in ((lambda: self.layout.update(layout_units="unspecified"), "Y-up meters"),
                                (lambda: self.layout["instances"][1].update(scale=[2, 1, 1]), "baked"),
                                (lambda: self.layout["instances"][1]["matrix"][0].__setitem__(0, 2), "rigid")):
            self.layout = deepcopy(original)
            change()
            self.save()
            with self.assertRaisesRegex(SceneExportError, message):
                load_scene_export(self.root)
        self.layout = original
        self.config["rigid_object"][0]["init_pos"] = [1, 2, 3]
        self.save()
        with self.assertRaisesRegex(SceneExportError, "Z-up pose"):
            load_scene_export(self.root)

    def test_reject_duplicate_objects_and_external_mesh(self):
        self.layout["instances"].append(deepcopy(self.layout["instances"][1]))
        self.save()
        with self.assertRaisesRegex(SceneExportError, "unique"):
            load_scene_export(self.root)
        self.layout["instances"].pop()
        self.layout["instances"][1]["geometry_path"] = "../../external.glb"
        self.save()
        with self.assertRaisesRegex(SceneExportError, "leaves"):
            load_scene_export(self.root)

    def test_camera_is_optional_but_declared_camera_must_exist(self):
        (self.root / "canonical_camera.json").unlink()
        with self.assertRaisesRegex(SceneExportError, "Cannot read"):
            load_scene_export(self.root)
        self.result["canonical_camera_path"] = None
        self.save()
        exported = load_scene_export(self.root)
        self.assertIsNone(exported.camera)
        self.assertIsNone(exported.camera_path)


if __name__ == "__main__":
    unittest.main()
