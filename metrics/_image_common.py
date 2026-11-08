"""Shared image loading, validation, encoder cache and runtime metadata."""
from functools import lru_cache
from pathlib import Path


def validate_options(options):
    allowed = {"lpips_network", "dinov2_model", "clip_model", "device", "image_size",
               "dinov2_revision", "clip_revision"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown input-view options: {sorted(unknown)}")
    if not isinstance(options.get("lpips_network", "alex"), str) or options.get("lpips_network", "alex") not in {"alex", "vgg", "squeeze"}:
        raise ValueError("lpips_network must be alex, vgg, or squeeze")
    size = options.get("image_size")
    if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 32 for x in size)):
        raise ValueError("image_size must be [width, height] with integer dimensions >= 32")
    for key in ("device", "dinov2_model", "clip_model", "dinov2_revision", "clip_revision"):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f"{key} must be a non-empty string")


def imports():
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Input-view metrics require numpy, torch and Pillow") from exc
    return np, torch, Image


def load_images(reference, rendered, options, image_module):
    with image_module.open(Path(reference)) as image:
        ref = image.convert("RGB")
    with image_module.open(Path(rendered)) as image:
        pred = image.convert("RGB")
    if options.get("image_size"):
        size = tuple(options["image_size"])
        ref = ref.resize(size, image_module.Resampling.BICUBIC)
        pred = pred.resize(size, image_module.Resampling.BICUBIC)
    return ref, pred


@lru_cache(maxsize=16)
def cached_encoder(loader, name, device, revision=None):
    """Share the former 16-entry DINOv2/CLIP cache across both algorithms."""
    return loader(name, device, revision)


def encoder_info(name, revision, device, processor, model, feature):
    image_processor = getattr(processor, "image_processor", processor)
    return {"model": name, "requested_revision": revision,
            "resolved_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "device": device, "feature": feature, "similarity": "cosine",
            "processor": image_processor.to_dict() if hasattr(image_processor, "to_dict") else None}
