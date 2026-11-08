"""Dispatch the independent input-view metric algorithms on one image pair."""
from metrics import _image_common
from metrics._image_common import validate_options


DEFAULT_OPTIONS = {
    "lpips_network": "alex",
    "dinov2_model": "facebook/dinov2-base",
    "clip_model": "openai/clip-vit-base-patch32",
}
RUNTIME_INFO = {}


def evaluate_input_view(reference, rendered, metrics, options=None):
    """Compute LPIPS distance and DINOv2/CLIP image cosine similarities.

    LPIPS requires equal dimensions unless image_size explicitly resizes both
    images. Encoders retain each checkpoint's original processor and crop.
    """
    options = options or {}
    validate_options(options)
    requested = list(metrics)
    unknown = set(requested) - {"LPIPS", "DINOv2", "CLIP"}
    if unknown:
        raise ValueError(f"Unknown input-view metrics: {sorted(unknown)}")
    np, torch, image_module = _image_common.imports()
    device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ref, pred = _image_common.load_images(reference, rendered, options, image_module)
    result = {}
    with torch.inference_mode():
        if "LPIPS" in requested:
            from metrics.lpips.metric import evaluate
            result["LPIPS"] = evaluate(ref, pred, options, device, np, torch, RUNTIME_INFO)
        if "DINOv2" in requested:
            from metrics.dinov2.metric import evaluate
            result["DINOv2"] = evaluate(ref, pred, options, device, np, torch, RUNTIME_INFO)
        if "CLIP" in requested:
            from metrics.clip.metric import evaluate
            result["CLIP"] = evaluate(ref, pred, options, device, np, torch, RUNTIME_INFO)
    return result
