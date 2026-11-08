"""Configuration fragments keep their own path origins across inheritance."""
import json
from pathlib import Path
import tempfile
import unittest

from tabletopscene_eval.profiles import ConfigCompositionError, load_composed_config


class ProfileCompositionTests(unittest.TestCase):
    def test_merge_and_origin_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profiles").mkdir()
            (root / "experiments").mkdir()
            parent = root / "profiles" / "base.json"
            parent.write_text(json.dumps({
                "rendering": {"resolution": [512, 512], "executable": "./engine/bin/blender", "lighting": {"environment_intensity": 1}},
                "metrics": {"gpt_evaluation": {"options": {"env_file": "private.env", "model": "gpt-5"}}},
                "samples": [{"reference": "input.png", "methods": {"a": {"scene_export": "scene", "views": [{"render": "v.png"}]}}}],
            }), encoding="utf-8")
            child = root / "experiments" / "run.json"
            child.write_text(json.dumps({"extends": "../profiles/base.json", "rendering": {"resolution": [640, 480], "lighting": {"lights": []}}, "metrics": {"gpt_evaluation": {"options": {"audit_dir": "audit"}}}}), encoding="utf-8")
            raw = load_composed_config(child)
            self.assertEqual(raw["rendering"]["resolution"], [640, 480])
            self.assertEqual(raw["rendering"]["lighting"], {"environment_intensity": 1, "lights": []})
            self.assertEqual(raw["rendering"]["executable"], str(root / "profiles/engine/bin/blender"))
            options = raw["metrics"]["gpt_evaluation"]["options"]
            self.assertEqual(options["env_file"], str(root / "profiles/private.env"))
            self.assertEqual(options["audit_dir"], str(root / "experiments/audit"))
            self.assertEqual(raw["samples"][0]["methods"]["a"]["scene_export"], str(root / "profiles/scene"))
            self.assertEqual(raw["_config_sources"], [str(parent), str(child)])

    def test_ordered_parents_replace_lists_and_preserve_bare_blender(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.json").write_text('{"rendering":{"executable":"blender","resolution":[512,512]},"samples":[1]}', encoding="utf-8")
            (root / "b.json").write_text('{"rendering":{"resolution":[256,256]},"samples":[2]}', encoding="utf-8")
            (root / "run.json").write_text('{"extends":["a.json","b.json"]}', encoding="utf-8")
            raw = load_composed_config(root / "run.json")
            self.assertEqual(raw["rendering"], {"executable": "blender", "resolution": [256, 256]})
            self.assertEqual(raw["samples"], [2])

    def test_executable_dispatch_symlink_is_not_resolved_to_its_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "dispatcher"
            target.write_text("fixture")
            alias = root / "blender"
            alias.symlink_to(target)
            config = root / "run.json"
            config.write_text(json.dumps({"rendering": {"executable": "./blender"}}))
            raw = load_composed_config(config)
            self.assertEqual(raw["rendering"]["executable"], str(alias))
            self.assertNotEqual(raw["rendering"]["executable"], str(target))

    def test_cycle_and_invalid_fragments_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.json").write_text('{"extends":"b.json"}', encoding="utf-8")
            (root / "b.json").write_text('{"extends":"a.json"}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigCompositionError, "cycle"):
                load_composed_config(root / "a.json")
            for content in ('[]', '{"extends":[3]}', '{"_config_sources":[]}'):
                (root / "bad.json").write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigCompositionError):
                    load_composed_config(root / "bad.json")


if __name__ == "__main__":
    unittest.main()
