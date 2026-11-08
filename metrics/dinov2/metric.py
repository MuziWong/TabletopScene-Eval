"""Cosine similarity of the final DINOv2 CLS token embeddings."""
from metrics._image_common import cached_encoder, encoder_info


def load_model(name, device, revision=None):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError("DINOv2 requires the optional 'transformers' package") from exc
    kwargs = {"revision": revision} if revision else {}
    processor = AutoImageProcessor.from_pretrained(name, use_fast=False, **kwargs)
    model = AutoModel.from_pretrained(name, **kwargs)
    return processor, model.to(device).eval()


def evaluate(ref, pred, options, device, np, torch, runtime_info):
    name = options.get("dinov2_model", "facebook/dinov2-base")
    revision = options.get("dinov2_revision")
    processor, model = cached_encoder(load_model, name, device, revision) if revision else cached_encoder(load_model, name, device)
    runtime_info["DINOv2"] = encoder_info(name, revision, device, processor, model, "CLS token")
    batch = processor(images=[ref, pred], return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    features = model(**batch).last_hidden_state[:, 0, :]
    return float(torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item())
