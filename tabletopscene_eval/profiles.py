"""Compose experiment JSON fragments while preserving each file's path base.

``extends`` is an ordered list of other JSON files, relative to the file that
declares it. Dictionaries merge recursively; lists and scalar values replace
the inherited value. Consequently a small experiment can reuse camera,
lighting, metric and API profiles without duplicating their settings.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path


class ConfigCompositionError(ValueError):
    """A configuration fragment or its inheritance graph is invalid."""


def _merge(parent: dict, child: dict) -> dict:
    result = deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _absolute(value, directory: Path, *, executable=False):
    # Leave malformed values to the domain-specific configuration validator.
    if not isinstance(value, str) or not value.strip():
        return value
    # A bare interpreter command, e.g. "python", is resolved through PATH by
    # subprocess. Relative interpreter paths such as ./env/bin/python have the
    # same origin-relative semantics as other configuration paths.
    if executable and "/" not in value and "\\" not in value and not value.startswith("~"):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    if executable:
        # Keep dispatch symlinks such as /snap/bin/blender intact: executing
        # their resolved /usr/bin/snap target changes argv[0] and the command.
        return os.path.abspath(path)
    return str(path.resolve())


def _paths_at_origin(raw: dict, directory: Path) -> dict:
    """Resolve only schema-defined paths, before inherited fragments merge."""
    result = deepcopy(raw)

    def paths(mapping, names, executables=()):
        if isinstance(mapping, dict):
            for name in names:
                if name in mapping:
                    mapping[name] = _absolute(mapping[name], directory, executable=name in executables)

    samples = result.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            paths(sample, ("reference", "reference_mask"))
            methods = sample.get("methods")
            if not isinstance(methods, dict):
                continue
            for method in methods.values():
                paths(method, ("scene_export",))
    rendering = result.get("rendering")
    paths(rendering, ("executable", "output_dir"), ("executable",))
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        gpt = metrics.get("gpt_evaluation")
        if isinstance(gpt, dict):
            paths(gpt.get("options"), ("audit_dir", "env_file", "prompt_file"))
        physics = metrics.get("physical_validity")
        if isinstance(physics, dict):
            paths(physics.get("options"), ("cache_dir",))
    return result


def load_composed_config(filename: str | Path) -> dict:
    """Read a JSON experiment and its profiles; return a fresh merged mapping.

    ``_config_sources`` is reserved for the ordered list of unique, absolute
    source paths read by this function. Path values need not exist yet because
    render output directories are valid configuration paths as well as inputs.
    Domain validators check the existence of actual input files afterward.
    """
    sources: list[str] = []
    active: list[Path] = []

    def visit(path: Path) -> dict:
        path = path.expanduser().resolve()
        if path in active:
            cycle = " -> ".join(str(p) for p in [*active, path])
            raise ConfigCompositionError(f"Configuration extends cycle: {cycle}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigCompositionError(f"Cannot read configuration {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigCompositionError(f"Configuration {path} must contain an object")
        if "_config_sources" in raw:
            raise ConfigCompositionError(f"Configuration {path}: _config_sources is reserved")
        parents = raw.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list) or any(not isinstance(p, str) or not p.strip() for p in parents):
            raise ConfigCompositionError(f"Configuration {path}: extends must be a path or a list of non-empty paths")
        active.append(path)
        composed = {}
        try:
            for parent in parents:
                candidate = Path(parent).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                composed = _merge(composed, visit(candidate))
            composed = _merge(composed, _paths_at_origin(raw, path.parent))
        finally:
            active.pop()
        if str(path) not in sources:
            sources.append(str(path))
        return composed

    result = visit(Path(filename))
    result["_config_sources"] = sources
    return result
