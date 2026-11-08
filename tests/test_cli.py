"""Reject unsafe CLI destinations before creating any run files."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tabletopscene_eval.__main__ import main


class OutputGuardTests(unittest.TestCase):
    def invoke(self, output, config):
        with patch("tabletopscene_eval.__main__.load_composed_config", return_value=config), redirect_stderr(StringIO()):
            return main(["--config", "unused.json", "--output", str(output)])

    def test_manual_config_cannot_write_inside_source_or_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scene_export"
            source.mkdir()
            (source / "result.json").write_text("source data")
            alias = root / "source_alias"
            alias.symlink_to(source, target_is_directory=True)
            for supplied in (source, source / "result.json", alias):
                with self.subTest(source=supplied):
                    output = source / "new_run"
                    config = {"samples": [{"methods": {"AgenTable": {"scene_export": str(supplied)}}}]}
                    self.assertEqual(self.invoke(output, config), 1)
                    self.assertFalse(output.exists())
                    self.assertEqual((source / "result.json").read_text(), "source data")

    def test_existing_outputs_are_never_mixed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for filename in ("results.json", "summary.json", "discovery.json"):
                with self.subTest(filename=filename):
                    sentinel = output / filename
                    sentinel.write_text("original result")
                    self.assertEqual(self.invoke(output, {}), 1)
                    self.assertEqual(sentinel.read_text(), "original result")
                    self.assertFalse((output / "config.json").exists())
                    sentinel.unlink()


if __name__ == "__main__":
    unittest.main()
