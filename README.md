# TabletopScene-Eval

Evaluate AgenTable `scene_export` packages, individually or in batches.

- **Input-view consistency:** LPIPS, DINOv2, CLIP.
- **GPT evaluation:** VF, IA, PP, average score, overall rank.
- **Physical validity:** ColO and ColS at 0, 0.1, 0.3, 0.5, and 1 mm.

## Setup

```bash
conda env create -f environment.yml  # Skip if the environment already exists.
conda activate tabletopscene-eval
python -m pip install -e . --no-deps --no-build-isolation
```

Install Blender separately and set its executable and device in `configs/render.json`.
For GPT evaluation, configure `.env` using `.env.example` and set `OPENAI_API_KEY`.
`OPENAI_BASE_URL` and `OPENAI_MODEL` are optional.

## Usage

Single scene:

```bash
tabletopscene-eval /path/to/scene_export --auto-mask --output runs/single
```

Batch:

```bash
tabletopscene-eval /path/to/outputs --batch --auto-mask --output runs/batch
```

`--auto-mask` unions existing AgenTable input-analysis masks; it does not segment new images.
These masks are not human ground truth. Alternatively, provide `--reference` and `--mask`,
or `--reference-root` and `--mask-root` for batches (matched by relative scene ID).
Masks must match the image dimensions, with white foreground including the table and objects.

Use `--no-gpt` to skip API calls, or `--physics-only` to evaluate collisions without rendering.
Run `tabletopscene-eval --help` for all options.

## Configuration and results

Each metric's implementation and configuration live in `metrics/<metric>/`.
Shared camera and rendering settings live in `configs/`; `configs/default.json` combines them.
Pass `--config` to override the default protocol.

Use a fresh output directory. Results are saved to `summary.json`, with per-scene results,
renders, and audit records retained for inspection.

See [Environment](docs/ENVIRONMENT.md) for installation details and [Metrics](docs/METRICS.md)
for scoring rules, normalization, view selection, and collision definitions.
