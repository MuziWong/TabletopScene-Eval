"""Blind visual judging of one or more methods for a shared input image."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import random
import sys
from pathlib import Path
from uuid import uuid4

from tabletopscene_eval.api import create_api_client

RUNTIME_JUDGMENTS = []
DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.txt")
if not DEFAULT_PROMPT_FILE.is_file():
    DEFAULT_PROMPT_FILE = (Path(sys.prefix) / "share" / "tabletopscene-eval" / "metrics"
                           / "gpt_evaluation" / "prompt.txt")


def validate_options(options):
    unknown = set(options) - {"model", "order_seed", "seed", "image_detail", "audit_dir", "base_url", "env_file", "prompt_file"}
    if unknown:
        raise ValueError(f"Unknown GPT options: {sorted(unknown)}")
    if "seed" in options and "order_seed" in options and options["seed"] != options["order_seed"]:
        raise ValueError("Use order_seed; conflicting legacy seed was also supplied")
    order_seed = options.get("order_seed", options.get("seed", 0))
    if isinstance(order_seed, bool) or not isinstance(order_seed, int):
        raise ValueError("GPT order_seed must be an integer")
    if not isinstance(options.get("image_detail", "high"), str) or options.get("image_detail", "high") not in {"auto", "low", "high"}:
        raise ValueError("GPT image_detail must be 'auto', 'low' or 'high'")
    for key in ("model", "base_url", "audit_dir", "env_file", "prompt_file"):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f"GPT {key} must be a non-empty string")


def _image_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image format: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _image_record(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load_prompt(options, *, has_context, candidate_count):
    source = Path(options.get("prompt_file", DEFAULT_PROMPT_FILE)).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"GPT prompt_file does not exist: {source}")
    data = source.read_bytes()
    try:
        rubric = data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("GPT prompt_file must be UTF-8 text") from exc
    if not rubric:
        raise ValueError("GPT prompt_file must not be empty")
    if has_context:
        presentation = (
            "IMAGE PRESENTATION FOR THIS REQUEST\n"
            "Group A contains foreground images: one reference and one image per candidate. "
            "Use Group A for VF and IA. Ignore the uniform replacement background and only "
            "minor edge jaggies caused by masking; do not reward the removal of background "
            "content. Real differences in object silhouettes, count and placement must still "
            "affect the evaluation.\n"
            "Group B contains the corresponding original/support-context images for the same "
            "reference and the same candidates. Use Group B only for PP, including visible "
            "table support and contact. Unreconstructed room/background content is outside "
            "the evaluated scene. Group B does not add candidates or viewpoints; its labels "
            "refer to exactly the candidates in Group A.\n"
            "Rank each candidate once using its VF, IA and PP evidence together."
        )
    else:
        presentation = (
            "IMAGE PRESENTATION FOR THIS REQUEST\n"
            "One reference image and one image per candidate follow. Use this single image "
            "group for VF, IA and PP. No separate support-context images are provided."
        )
    if candidate_count == 1:
        presentation += (
            "\nSINGLE-CANDIDATE REQUEST\n"
            "Exactly one candidate is supplied. Evaluate its VF, IA and PP against "
            "the reference with the same absolute rubric. Set OR to the integer 1 "
            "by convention; this is not a comparative ranking or evidence of "
            "superiority over any other method. Do not invent another candidate "
            "or a baseline. Return only candidate_1."
        )
    prompt = rubric + "\n\n" + presentation
    return prompt, {
        "prompt_source": str(source),
        "prompt_source_sha256": hashlib.sha256(data).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def _validate_context(context_images, renders):
    if context_images is None:
        return
    if not isinstance(context_images, dict) or set(context_images) != {"reference", "renders"}:
        raise ValueError("GPT context_images must contain exactly reference and renders")
    if not isinstance(context_images["renders"], dict) or set(context_images["renders"]) != set(renders):
        raise ValueError("GPT context renders must contain exactly the same methods as renders")
    if not isinstance(context_images["reference"], (str, Path)) or not all(
            isinstance(path, (str, Path)) for path in context_images["renders"].values()):
        raise ValueError("GPT context images must be image file paths")


def _image_records(reference, renders, labels):
    return {"reference": _image_record(reference),
            "candidates": {label: {"method": method, **_image_record(renders[method])}
                           for label, method in labels.items()}}


def _append_images(content, reference, renders, labels, *, detail, title):
    content.extend([{"type": "text", "text": f"{title} — reference image:"},
                    {"type": "image_url", "image_url": {"url": _image_url(reference), "detail": detail}}])
    for label, name in labels.items():
        content.extend([{"type": "text", "text": f"{title} — {label}:"},
                        {"type": "image_url", "image_url": {"url": _image_url(renders[name]), "detail": detail}}])


def _write_audit(directory, response, *, images, context, prompt, prompt_record, model, order_seed, detail):
    """Keep enough evidence to inspect a judgment without duplicating image bytes."""
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if hasattr(response, "model_dump"):
        raw = response.model_dump(mode="json")
    else:
        # Also supports lightweight compatible clients and offline tests.
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        elif hasattr(usage, "__dict__"):
            usage = vars(usage)
        raw = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": usage,
            "choices": [{"message": {"content": choice.message.content},
                         "finish_reason": getattr(choice, "finish_reason", None)}
                        for choice in response.choices],
        }
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": {"model": model, "order_seed": order_seed, "image_detail": detail,
                    "candidate_count": len(images["candidates"]),
                    "ranking_comparative": len(images["candidates"]) > 1,
                    "prompt": prompt, **prompt_record,
                    "response_format": {"type": "json_object"}},
        **images,
        "context_images": context,
        "response": raw,
    }
    # Preserve retries/repeats rather than overwrite a prior judgment of a scene.
    path = root / f"gpt_{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return str(path.resolve())


def evaluate_gpt(reference, renders, options=None, *, context_images=None):
    """Return per-method scores and a directly judged rank for one input scene.

    Scores range from 1 to 7. With multiple methods OR is directly judged,
    not derived from the scores. With one method OR=1 is a noncomparative
    convention. Renders are anonymized and ordered by order_seed
    (legacy alias: seed). This seed does not make model inference deterministic.
    The external project rubric is configurable with prompt_file. Optional
    context_images supplies original images for PP when reference/renders are
    foreground images for VF/IA. Optional audit_dir preserves raw evidence.
    """
    options = options or {}
    validate_options(options)
    if (not isinstance(renders, dict) or not renders or
            any(not isinstance(name, str) or not name.strip() for name in renders)):
        raise ValueError("GPT renders must contain at least one named method")
    _validate_context(context_images, renders)
    prompt, prompt_record = _load_prompt(options, has_context=context_images is not None,
                                        candidate_count=len(renders))
    order_seed = options.get("order_seed", options.get("seed", 0))
    detail = options.get("image_detail", "high")
    audit_dir = options.get("audit_dir")
    names = sorted(renders)
    random.Random(order_seed).shuffle(names)
    labels = {f"candidate_{i + 1}": name for i, name in enumerate(names)}
    images = _image_records(reference, renders, labels)
    context = (_image_records(context_images["reference"], context_images["renders"], labels)
               if context_images is not None else None)
    content = [{"type": "text", "text": prompt}]
    _append_images(content, reference, renders, labels, detail=detail,
                   title="Group A / foreground" if context is not None else "Evaluation images")
    if context is not None:
        _append_images(content, context_images["reference"], context_images["renders"], labels,
                       detail=detail, title="Group B / original support context")
    client, settings = create_api_client(options)
    model = settings.model
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    # Save before validation: malformed output is evidence that needs inspection.
    audit_path = None
    if audit_dir is not None:
        audit_path = _write_audit(audit_dir, response, images=images, context=context,
                                  prompt=prompt, prompt_record=prompt_record, model=model,
                                  order_seed=order_seed, detail=detail)
    RUNTIME_JUDGMENTS.append({"reference": str(reference), "requested_model": model,
                              "response_model": getattr(response, "model", None),
                              "response_id": getattr(response, "id", None),
                              "system_fingerprint": getattr(response, "system_fingerprint", None),
                              "candidate_order": list(labels.values()), "order_seed": order_seed,
                              "candidate_count": len(names), "ranking_comparative": len(names) > 1,
                              "image_detail": detail, "audit_path": audit_path,
                              **prompt_record, "reference_image": images["reference"],
                              "candidate_images": images["candidates"], "context_images": context})
    text = response.choices[0].message.content
    try:
        judged = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("GPT returned invalid JSON") from exc
    if not isinstance(judged, dict) or set(judged) != set(labels):
        raise ValueError("GPT response must contain exactly all candidate labels")
    result = {}
    ranks = []
    for label, method in labels.items():
        entry = judged[label]
        if not isinstance(entry, dict) or set(entry) != {"VF", "IA", "PP", "OR"}:
            raise ValueError(f"Invalid GPT response for {label}")
        scores = []
        for key in ("VF", "IA", "PP"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 7:
                raise ValueError(f"GPT returned invalid {key} for {label}: {value}")
            scores.append(float(value))
        rank = entry.get("OR")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(f"GPT returned invalid OR for {label}: {rank}")
        ranks.append(rank)
        result[method] = {"VF": scores[0], "IA": scores[1], "PP": scores[2],
                          "Avg.": sum(scores) / 3, "OR": rank}
    if sorted(ranks) != list(range(1, len(names) + 1)):
        raise ValueError("GPT ranks must be unique and consecutive starting at 1")
    return result
