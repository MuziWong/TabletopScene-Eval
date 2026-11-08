"""Offline API configuration precedence and safe connectivity diagnostics."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tabletopscene_eval.api import check_api_connection, create_api_client, resolve_api_settings


class APIConfigurationTests(unittest.TestCase):
    def test_default_and_shell_model(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_api_settings().model, "gpt-5")
            self.assertIsNone(resolve_api_settings().base_url)
        with patch.dict(os.environ, {"OPENAI_MODEL": "shell-model",
                                    "OPENAI_BASE_URL": "https://shell.example/v1"}, clear=True):
            settings = resolve_api_settings()
            self.assertEqual(settings.model, "shell-model")
            self.assertEqual(settings.base_url, "https://shell.example/v1")
            self.assertNotIn("shell.example", repr(settings))

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "requires python-dotenv")
    def test_explicit_file_shell_and_option_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.env"
            source.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n"
                              "OPENAI_BASE_URL=https://file.example/v1\n")
            with patch.dict(os.environ, {}, clear=True):
                settings = resolve_api_settings({"env_file": str(source)})
                self.assertEqual(settings.model, "file-model")
                self.assertEqual(settings.base_url, "https://file.example/v1")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "file-key")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key", "OPENAI_MODEL": "shell-model",
                                        "OPENAI_BASE_URL": "https://shell.example/v1"}, clear=True):
                settings = resolve_api_settings({"env_file": str(source)})
                self.assertEqual(settings.model, "shell-model")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")
                settings = resolve_api_settings({"env_file": str(source), "model": "explicit-model",
                                                 "base_url": "https://explicit.example/v1"})
                self.assertEqual(settings.model, "explicit-model")
                self.assertEqual(settings.base_url, "https://explicit.example/v1")

    def test_missing_explicit_env_does_not_search_elsewhere(self):
        with self.assertRaisesRegex(ValueError, "env_file does not exist"):
            resolve_api_settings({"env_file": "/nonexistent-tabletopscene-eval-private.env"})

    def test_client_receives_gateway_but_does_not_export_key(self):
        calls = []
        class Client:
            def __init__(self, **kwargs):
                calls.append(kwargs)
        with patch.dict(os.environ, {"OPENAI_MODEL": "configured-model", "OPENAI_API_KEY": "secret-test-key",
                                    "OPENAI_BASE_URL": "https://gateway.example/v1"}, clear=True), \
                patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}):
            _, settings = create_api_client(timeout=30.0, max_retries=0)
        self.assertEqual(settings.model, "configured-model")
        self.assertEqual(calls, [{"base_url": "https://gateway.example/v1", "timeout": 30.0,
                                  "max_retries": 0}])
        self.assertNotIn("secret-test-key", repr(settings))

    def test_probe_is_bounded_and_only_reports_status(self):
        requests, clients = [], []
        class Client:
            def __init__(self, **kwargs):
                clients.append(kwargs)
                self.chat = types.SimpleNamespace(completions=self)
            def create(self, **kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(model="served-model", usage=types.SimpleNamespace(total_tokens=12),
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok":true}'),
                                                   finish_reason="stop")])
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}):
            result = check_api_connection({"model": "requested-model"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["response_model"], "served-model")
        self.assertEqual(result["total_tokens"], 12)
        self.assertEqual(clients[0], {"timeout": 30.0, "max_retries": 0})
        self.assertEqual(requests[0]["max_completion_tokens"], 128)
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertIsInstance(requests[0]["messages"][0]["content"], str)

    def test_probe_suppresses_remote_exception_body_and_url(self):
        class GatewayError(Exception):
            status_code = 401
        class Client:
            def __init__(self, **kwargs):
                raise GatewayError("secret-token at https://private-gateway.example/?key=secret-token")
        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}):
            result = check_api_connection()
        encoded = json.dumps(result)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status_code"], 401)
        self.assertEqual(result["error_type"], "GatewayError")
        self.assertNotIn("secret-token", encoded)
        self.assertNotIn("private-gateway", encoded)


if __name__ == "__main__":
    unittest.main()
