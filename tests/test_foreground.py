"""Foreground protocol tests with real pixels and offline evaluator boundaries."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tabletopscene_eval.foreground import composite_foreground


class ForegroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, background, foreground=None):
        image = Image.new("RGB", (8, 8), background)
        for xy, color in (foreground or {}).items():
            image.putpixel(xy, color)
        path = self.root / name
        image.save(path)
        return path

    def mask(self, name, points):
        mask = Image.new("L", (8, 8), 0)
        for xy in points:
            mask.putpixel(xy, 255)
        path = self.root / name
        mask.save(path)
        return path



    def test_background_removed_without_hiding_prediction_extras_or_moving_pixels(self):
        ref = self.image("ref.png", (20, 30, 40), {(2, 3): (255, 0, 0)})
        pred = self.image("pred.png", (100, 110, 120), {(2, 3): (255, 0, 0), (6, 6): (0, 0, 255)})
        ref_mask = self.mask("ref_mask.png", [(2, 3)])
        pred_mask = self.mask("pred_mask.png", [(2, 3), (6, 6)])
        before = hashlib.sha256(pred.read_bytes()).hexdigest()
        a, b = self.root / "a.png", self.root / "b.png"
        composite_foreground(ref, ref_mask, a, {})
        detail = composite_foreground(pred, pred_mask, b, {}, allow_empty=True)
        with Image.open(a) as reference, Image.open(b) as rendered:
            self.assertEqual(reference.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(rendered.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(rendered.getpixel((2, 3)), reference.getpixel((2, 3)))
            self.assertEqual(rendered.getpixel((6, 6)), (0, 0, 255))
            self.assertEqual(reference.getpixel((6, 6)), (255, 255, 255))
        self.assertEqual(detail["foreground_pixels"], 2)
        self.assertEqual(hashlib.sha256(pred.read_bytes()).hexdigest(), before)

    def test_mask_validation_rejects_empty_reference_and_size_mismatch_but_keeps_empty_prediction(self):
        image = self.image("ref.png", (0, 0, 0))
        empty = self.mask("empty.png", [])
        output = self.root / "out.png"
        with self.assertRaisesRegex(ValueError, "Reference foreground mask is empty"):
            composite_foreground(image, empty, output, {})
        detail = composite_foreground(image, empty, output, {}, allow_empty=True)
        self.assertEqual(detail["foreground_pixels"], 0)
        with Image.open(output) as result:
            self.assertEqual(result.getextrema(), ((255, 255),) * 3)
        Image.new("L", (4, 4), 255).save(self.root / "wrong.png")
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            composite_foreground(image, self.root / "wrong.png", output, {})
        with self.assertRaisesRegex(ValueError, "colored segmentation map"):
            composite_foreground(image, self.image("colors.png", (255, 0, 0)), output, {})








if __name__ == "__main__":
    unittest.main()
