"""OpenAI-compatible connection settings shared by the evaluator and API check.

Credentials stay in the SDK/environment. Only an explicitly selected dotenv file
is read; existing shell exports always win over that file.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class APISettings:
    model: str
    # A gateway URL can contain credentials. Do not expose it in repr/log output.
    base_url: str | None = field(default=None, repr=False)


def resolve_api_settings(options=None):
    """Resolve explicit options > shell > explicit dotenv > built-in defaults."""
    options = options or {}
    if options.get("env_file"):
        source = Path(options["env_file"]).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"GPT env_file does not exist: {source}")
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError("Explicit GPT env_file loading requires python-dotenv") from exc
        load_dotenv(dotenv_path=source, override=False)
    return APISettings(
        model=options.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-5",
        base_url=options.get("base_url") or os.environ.get("OPENAI_BASE_URL") or None,
    )


def create_api_client(options=None, **client_options):
    """Create a Chat Completions client without recording keys or gateway URLs."""
    settings = resolve_api_settings(options)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("GPT evaluation requires the optional 'openai' package") from exc
    if settings.base_url:
        client_options["base_url"] = settings.base_url
    return OpenAI(**client_options), settings


def check_api_connection(options=None):
    """Make one bounded text/JSON probe; return sanitized status, never secrets.

    This checks authentication, model access, Chat Completions and JSON mode.
    It deliberately sends no scene images and does not validate visual judging.
    No automatic SDK retries are used for this optional diagnostic request.
    """
    requested_model = None
    try:
        client, settings = create_api_client(options, timeout=30.0, max_retries=0)
        requested_model = settings.model
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": 'Return only this JSON object: {"ok": true}'}],
            response_format={"type": "json_object"},
            max_completion_tokens=128,
        )
        choice = response.choices[0]
        try:
            payload = json.loads(choice.message.content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        valid_json = isinstance(payload, dict) and payload.get("ok") is True
        usage = getattr(response, "usage", None)
        return {
            "ok": valid_json,
            "requested_model": settings.model,
            "response_model": getattr(response, "model", None),
            "json_mode": valid_json,
            "finish_reason": getattr(choice, "finish_reason", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "scope": "text_chat_completions_json_only",
        }
    except Exception as exc:
        # SDK error messages may contain gateway URLs or response bodies. Report
        # only the exception class and HTTP status, never str(exc) or repr(exc).
        result = {"ok": False, "error_type": type(exc).__name__,
                  "requested_model": requested_model,
                  "scope": "text_chat_completions_json_only"}
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            result["status_code"] = status
        return result
