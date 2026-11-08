"""Learned perceptual image patch similarity on equal-size RGB images."""
from functools import lru_cache


@lru_cache(maxsize=16)
def load_model(network, device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS requires the optional 'lpips' package") from exc
    return lpips.LPIPS(net=network, verbose=False).to(device).eval()


def evaluate(ref, pred, options, device, np, torch, runtime_info):
    if ref.size != pred.size:
        raise ValueError(f"LPIPS image dimensions differ: {ref.size} vs {pred.size}; render matching dimensions or explicitly set image_size for both images")
    network = options.get("lpips_network", "alex")
    model = load_model(network, device)
    runtime_info["LPIPS"] = {"network": network, "device": device,
                             "normalization": "RGB [-1,1]", "image_size": list(ref.size)}
    def to_tensor(image):
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
    return float(model(to_tensor(ref), to_tensor(pred)).item())
