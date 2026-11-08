"""Weight-free image metric boundaries and separate algorithm semantics."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from tabletopscene_eval.image_metrics import evaluate_input_view, RUNTIME_INFO
from tabletopscene_eval.profiles import load_composed_config


class ImageMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference, self.prediction = self.root/"reference.png", self.root/"prediction.png"
        Image.new("RGB", (32, 32), (255, 0, 0)).save(self.reference)
        Image.new("RGB", (32, 40), (0, 0, 255)).save(self.prediction)
        RUNTIME_INFO.clear()

    def tearDown(self):
        self.temporary.cleanup()

    def test_lpips_dimension_guard_and_explicit_shared_resize_normalization(self):
        with patch("metrics.lpips.metric.load_model") as loader:
            with self.assertRaisesRegex(ValueError, "LPIPS image dimensions differ"):
                evaluate_input_view(self.reference, self.prediction, ["LPIPS"], {"device": "cpu"})
            loader.assert_not_called()
        def model(ref, pred):
            self.assertEqual(list(ref.shape), [1, 3, 32, 64])
            self.assertEqual(list(pred.shape), [1, 3, 32, 64])
            self.assertTrue(torch.is_inference_mode_enabled())
            torch.testing.assert_close(ref[0, :, 0, 0], torch.tensor([1., -1., -1.]))
            torch.testing.assert_close(pred[0, :, 0, 0], torch.tensor([-1., -1., 1.]))
            return torch.tensor(.25)
        with patch("metrics.lpips.metric.load_model", return_value=model):
            result = evaluate_input_view(self.reference, self.prediction, ["LPIPS"],
                                         {"device": "cpu", "image_size": [64, 32]})
        self.assertEqual(result, {"LPIPS": .25})
        self.assertEqual(RUNTIME_INFO["LPIPS"]["image_size"], [64, 32])
        with Image.open(self.prediction) as image: self.assertEqual(image.size, (32, 40))

    def test_dinov2_uses_final_cls_and_clip_uses_projected_image_features(self):
        class Processor:
            def __call__(self, images, return_tensors):
                self.images = images
                return {"pixel_values": torch.zeros((2, 3, 2, 2))}
            def to_dict(self): return {"fixture": "processor"}
        class Dino:
            config = SimpleNamespace(_commit_hash="dino-fixture")
            def __call__(self, **batch):
                return SimpleNamespace(last_hidden_state=torch.tensor([[[1., 0.], [1., 1.]], [[0., 1.], [1., 1.]]]))
        class Clip:
            config = SimpleNamespace(_commit_hash="clip-fixture")
            def __init__(self, output_object): self.output_object = output_object
            def get_image_features(self, **batch):
                features = torch.tensor([[1., 0.], [.6, .8]])
                return SimpleNamespace(pooler_output=features) if self.output_object else features
        for output_object in (False, True):
            with self.subTest(model_output=output_object), \
                 patch("metrics.dinov2.metric.cached_encoder", return_value=(Processor(), Dino())), \
                 patch("metrics.clip.metric.cached_encoder", return_value=(Processor(), Clip(output_object))):
                result = evaluate_input_view(self.reference, self.prediction, ["CLIP", "DINOv2"], {"device": "cpu"})
            self.assertEqual(result["DINOv2"], 0.)
            self.assertAlmostEqual(result["CLIP"], .6, places=6)
        self.assertEqual(RUNTIME_INFO["DINOv2"]["feature"], "CLS token")
        self.assertEqual(RUNTIME_INFO["CLIP"]["feature"], "projected image embedding")
        self.assertEqual(RUNTIME_INFO["CLIP"]["resolved_revision"], "clip-fixture")

    def test_individual_configs_compose_without_overwriting_the_metric_list(self):
        repository = Path(__file__).resolve().parents[1]
        paths = [repository/"metrics"/name/"config.json" for name in ("lpips", "dinov2", "clip")]
        for path in paths:
            self.assertEqual(set(json.loads(path.read_text())["metrics"]["input_view"]), {"options"})
        names = ["LPIPS", "DINOv2", "CLIP"]
        config = self.root/"config.json"
        config.write_text(json.dumps({"extends": [str(path) for path in paths],
                                      "metrics": {"input_view": {"enabled": True, "metrics": names}}}))
        merged = load_composed_config(config)["metrics"]["input_view"]
        self.assertEqual(merged["metrics"], names)
        self.assertEqual(merged["options"]["lpips_network"], "alex")
        self.assertEqual(merged["options"]["dinov2_revision"], "f9e44c814b77203eaa57a6bdbbd535f21ede1415")
        self.assertEqual(merged["options"]["clip_revision"], "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")


if __name__ == "__main__":
    unittest.main()
