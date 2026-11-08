"""Cosine similarity of CLIP projected image embeddings."""
from metrics._image_common import cached_encoder, encoder_info


def load_model(name, device, revision=None):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("CLIP requires the optional 'transformers' package") from exc
    kwargs = {"revision": revision} if revision else {}
    processor = CLIPProcessor.from_pretrained(name, use_fast=False, **kwargs)
    model = CLIPModel.from_pretrained(name, **kwargs)
    return processor, model.to(device).eval()


def evaluate(ref, pred, options, device, np, torch, runtime_info):
    name = options.get("clip_model", "openai/clip-vit-base-patch32")
    revision = options.get("clip_revision")
    processor, model = cached_encoder(load_model, name, device, revision) if revision else cached_encoder(load_model, name, device)
    runtime_info["CLIP"] = encoder_info(name, revision, device, processor, model, "projected image embedding")
    batch = processor(images=[ref, pred], return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    features = model.get_image_features(**batch)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return float(torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item())
