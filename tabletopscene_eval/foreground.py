"""Explicit, shared-canvas foreground compositing without fitting image layouts."""
from pathlib import Path


DEFAULT_FOREGROUND = {
    "enabled": False,
    "include_table": True,
    "background_rgb": [255, 255, 255],
    "mask_threshold": 127,
}


def validate_foreground_options(options):
    if not isinstance(options, dict):
        raise ValueError("foreground must be an object")
    if set(options) - set(DEFAULT_FOREGROUND):
        raise ValueError(f"Unknown foreground options: {sorted(set(options) - set(DEFAULT_FOREGROUND))}")
    for key in ("enabled", "include_table"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"foreground.{key} must be a boolean")
    color = options.get("background_rgb", DEFAULT_FOREGROUND["background_rgb"])
    if (not isinstance(color, list) or len(color) != 3 or
            any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in color)):
        raise ValueError("foreground.background_rgb must contain three integer RGB values in [0,255]")
    threshold = options.get("mask_threshold", 127)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 254:
        raise ValueError("foreground.mask_threshold must be an integer in [0,254]")
    return {**DEFAULT_FOREGROUND, **options}


def composite_foreground(image_path, mask_path, output_path, options, *, allow_empty=False, expected_size=None):
    """Mask values above threshold are foreground, with no crop/resize/alignment.

    Reference masks must be nonempty. Empty predictions remain valid failures.
    Each image must use its OWN visible foreground mask in the same pixel grid.
    """
    from PIL import Image, ImageChops
    options = validate_foreground_options(options)
    image_path, mask_path, output_path = map(Path, (image_path, mask_path, output_path))
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if expected_size is not None and list(image.size) != list(expected_size):
        raise ValueError(f"Foreground render canvas differs from reference: {image_path} {image.size} vs {tuple(expected_size)}; render the same pixel dimensions")
    with Image.open(mask_path) as source:
        if source.size != image.size:
            raise ValueError(f"Foreground mask dimensions differ: {mask_path} {source.size} vs image {image.size}; masks are never resized")
        if source.mode in {"1", "L"}:
            mask = source.convert("L")
        elif source.mode in {"RGB", "RGBA", "LA", "P"}:
            if ((source.mode in {"RGBA", "LA"} and source.getchannel("A").getextrema() != (255, 255)) or
                    (source.mode == "P" and "transparency" in source.info and source.convert("RGBA").getchannel("A").getextrema() != (255, 255))):
                raise ValueError(f"Foreground mask must encode foreground in grayscale, not transparency: {mask_path}")
            rgb = source.convert("RGB")
            red, green, blue = rgb.split()
            if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
                raise ValueError(f"Foreground mask must be grayscale, not a colored segmentation map: {mask_path}")
            mask = red
        else:
            raise ValueError(f"Foreground mask must be 8-bit grayscale or grayscale RGB: {mask_path} ({source.mode})")
    mask = mask.point(lambda pixel: 255 if pixel > options["mask_threshold"] else 0)
    count = mask.histogram()[255]
    if count == 0 and not allow_empty:
        raise ValueError(f"Reference foreground mask is empty: {mask_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background = Image.new("RGB", image.size, tuple(options["background_rgb"]))
    Image.composite(image, background, mask).save(output_path, format="PNG")
    return {
        "source_image": str(image_path.resolve()), "mask": str(mask_path.resolve()),
        "image": str(output_path.resolve()), "size": list(image.size),
        "foreground_pixels": count, "foreground_fraction": count / (image.width * image.height),
        "mask_rule": f"grayscale > {options['mask_threshold']}",
        "background_rgb": options["background_rgb"], "crop": "none", "alignment": "unchanged pixel canvas",
    }
