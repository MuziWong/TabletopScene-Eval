"""Opt-in numerical smoke tests using real pretrained image metric models.

Run from the repository root:
  TABLETOP_EVAL_REAL_MODELS=1 python -m unittest tests.test_real_input_view -v

This downloads weights on first use. It never calls the GPT API. The synthetic
fixtures check numerical behavior, not reproduction of a paper's results.
"""
import math
import os
from pathlib import Path
import tempfile
import unittest

from tabletopscene_eval.image_metrics import evaluate_input_view


@unittest.skipUnless(os.environ.get("TABLETOP_EVAL_REAL_MODELS") == "1",
                     "set TABLETOP_EVAL_REAL_MODELS=1 to load pretrained models")
class RealInputViewTest(unittest.TestCase):
    def test_identity_and_changed_images(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = Image.new("RGB", (256, 256), "#e8e2d5")
            draw = ImageDraw.Draw(reference)
            draw.rectangle((20, 130, 235, 230), fill="#a67747")
            draw.ellipse((40, 80, 125, 165), fill="#df3927")
            draw.rectangle((155, 60, 210, 150), fill="#2376b8")
            reference.save(root / "reference.png")
            altered = Image.new("RGB", (256, 256), "#111c3d")
            draw = ImageDraw.Draw(altered)
            for x in range(0, 256, 32):
                draw.line((x, 0, 255 - x, 255), fill="#7afa63", width=12)
            altered.save(root / "altered.png")

            options = {
                "device": "cpu",
                "dinov2_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
                "clip_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            }
            for metric in ("LPIPS", "DINOv2", "CLIP"):
                with self.subTest(metric=metric):
                    identical = evaluate_input_view(root / "reference.png", root / "reference.png", [metric], options)[metric]
                    changed = evaluate_input_view(root / "reference.png", root / "altered.png", [metric], options)[metric]
                    self.assertTrue(math.isfinite(identical) and math.isfinite(changed))
                    if metric == "LPIPS":
                        self.assertAlmostEqual(identical, 0.0, delta=1e-5)
                        self.assertGreater(changed, identical + 1e-4)
                    else:
                        self.assertAlmostEqual(identical, 1.0, delta=1e-5)
                        self.assertLess(changed, identical - 1e-4)


if __name__ == "__main__":
    unittest.main()
