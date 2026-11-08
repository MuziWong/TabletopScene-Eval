"""Offline tests for judgment provenance and image presentation controls."""
import hashlib
import base64
from copy import deepcopy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics.gpt_evaluation import DEFAULT_PROMPT_FILE, RUNTIME_JUDGMENTS, evaluate_gpt, validate_options


class GPTProtocolTests(unittest.TestCase):
    @staticmethod
    def client(answer, requests):
        class Client:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=self)

            def create(self, **kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(answer)))])
        return Client

    def paired_fixture(self, root, methods=("private_method_A", "private_method_B")):
        reference = root/"reference.png"
        reference.write_bytes(b"reference-fixture")
        renders, views = {}, {}
        for method in methods:
            fixed, best = root/f"{method}_fixed.png", root/f"{method}_best.png"
            fixed.write_bytes(f"fixed-{method}".encode())
            best.write_bytes(f"best-{method}".encode())
            renders[method] = fixed
            views[method] = {
                "fixed": {"id": "e01_a00_d00", "render": fixed,
                          "camera": {"position_y_up_m": [0, 1, 2], "target_y_up_m": [0, 0, 0]}},
                "best": {"id": "e00_a02_d00", "render": best,
                         "camera": {"position_y_up_m": [2, 1, 0], "target_y_up_m": [0, 0, 0]},
                         "selection_metric": "LPIPS", "selection_score": .123456789,
                         "candidate_count": 16}}
        return reference, renders, views

    def test_paired_views_send_one_group_and_preserve_anonymous_selection_audit(self):
        answer = {f"candidate_{i}": {"VF": 4, "IA": 5, "PP": 6, "OR": i} for i in (1, 2)}
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, renders, views = self.paired_fixture(root)
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                scores = evaluate_gpt(reference, renders,
                    {"view_selection": "fixed_and_best", "best_view_metric": "LPIPS",
                     "order_seed": 9, "audit_dir": str(root/"audit")}, candidate_views=views)
            self.assertEqual(set(scores), set(renders))
            content = requests[0]["messages"][0]["content"]
            all_text = "\n".join(block["text"] for block in content if block["type"] == "text")
            self.assertIn("Use the BEST input-consistency view primarily for IA", all_text)
            self.assertIn("Use FIXED and BEST together for VF and PP", all_text)
            self.assertNotIn("private_method", all_text)
            self.assertNotIn("0.123456789", all_text)
            self.assertNotIn("LPIPS", all_text)
            self.assertNotIn("Group A / foreground", all_text)
            self.assertNotIn("Group B / original support context", all_text)
            images = [base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
                      for block in content if block["type"] == "image_url"]
            self.assertEqual(len(images), 5)
            runtime = RUNTIME_JUDGMENTS[-1]
            audit = json.loads(next((root/"audit").glob("*.json")).read_text())
            expected = [reference.read_bytes()]
            for label, candidate in runtime["candidate_views"].items():
                method = candidate["method"]
                for role in ("fixed", "best"):
                    expected.append(Path(views[method][role]["render"]).read_bytes())
                    self.assertEqual(candidate[role]["id"], views[method][role]["id"])
                    self.assertEqual(candidate[role]["camera"], views[method][role]["camera"])
                    self.assertEqual(candidate[role]["sha256"], hashlib.sha256(expected[-1]).hexdigest())
                self.assertEqual(candidate["best"]["selection_metric"], "LPIPS")
                self.assertEqual(candidate["best"]["selection_score"], .123456789)
                self.assertEqual(candidate["best"]["candidate_count"], 16)
                self.assertEqual(runtime["candidate_images"][label]["path"], candidate["fixed"]["path"])
            self.assertEqual(images, expected)
            for record in (runtime, audit):
                self.assertEqual(record["presentation"], "single_group_fixed_and_best")
                self.assertEqual(record["image_count"], 5)
                self.assertIsNone(record["context_images"])
            self.assertEqual(runtime["candidate_views"], audit["candidate_views"])
            self.assertEqual(audit["candidate_images"], runtime["candidate_images"])
            self.assertEqual(audit["request"]["image_count"], 5)

    def test_paired_same_view_ids_are_one_candidate_and_repeat_evidence(self):
        requests = []
        answer = {"candidate_1": {"VF": 4, "IA": 5, "PP": 6, "OR": 1}}
        with tempfile.TemporaryDirectory() as directory:
            reference, renders, views = self.paired_fixture(Path(directory), ("only_method",))
            fixed = views["only_method"]["fixed"]
            views["only_method"]["best"].update(id=fixed["id"], render=fixed["render"], camera=deepcopy(fixed["camera"]))
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                result = evaluate_gpt(reference, renders, {"view_selection": "fixed_and_best"}, candidate_views=views)
        self.assertEqual(set(result), {"only_method"})
        self.assertEqual(result["only_method"]["OR"], 1)
        self.assertEqual(RUNTIME_JUDGMENTS[-1]["candidate_count"], 1)
        self.assertEqual(RUNTIME_JUDGMENTS[-1]["image_count"], 3)
        captions = "\n".join(x["text"] for x in requests[0]["messages"][0]["content"] if x["type"] == "text")
        self.assertIn("same view as FIXED, repeated evidence", captions)
        self.assertIn("must not be counted as independent evidence", captions)

    def test_paired_metadata_errors_fail_before_api(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, renders, good = self.paired_fixture(Path(directory))
            cases = []
            bad = deepcopy(good); bad.pop("private_method_A"); cases.append(bad)
            for role, key, value in (("fixed", "id", ""), ("fixed", "render", renders["private_method_B"]),
                    ("best", "selection_score", True), ("best", "selection_score", float("nan")),
                    ("best", "selection_metric", "CLIP"), ("best", "candidate_count", 0),
                    ("best", "candidate_count", True), ("best", "camera", {"bad": float("nan")}),
                    ("best", "camera", None), ("best", "render", Path(directory)/"missing.png"),
                    ("best", "id", good["private_method_A"]["fixed"]["id"])):
                bad = deepcopy(good); bad["private_method_A"][role][key] = value; cases.append(bad)
            bad = deepcopy(good); bad["private_method_A"]["best"].pop("selection_metric"); cases.append(bad)
            with patch("metrics.gpt_evaluation.metric.create_api_client", side_effect=AssertionError("API must not be called")):
                for bad in cases:
                    with self.subTest(metadata=bad), self.assertRaisesRegex(ValueError, "GPT candidate_views"):
                        evaluate_gpt(reference, renders, {"view_selection": "fixed_and_best"}, candidate_views=bad)
                with self.assertRaisesRegex(ValueError, "requires candidate_views"):
                    evaluate_gpt(reference, renders, {"view_selection": "fixed_and_best"})
                with self.assertRaisesRegex(ValueError, "requires view_selection"):
                    evaluate_gpt(reference, renders, {"view_selection": "fixed"}, candidate_views=good)
                with self.assertRaisesRegex(ValueError, "forbids context_images"):
                    evaluate_gpt(reference, renders, {"view_selection": "fixed_and_best"}, candidate_views=good,
                                 context_images={"reference": reference, "renders": renders})

    def test_default_config_enables_paired_views_and_selection_metric_validation(self):
        config = json.loads(DEFAULT_PROMPT_FILE.with_name("config.json").read_text())
        options = config["metrics"]["gpt_evaluation"]["options"]
        self.assertEqual(options["view_selection"], "fixed_and_best")
        self.assertEqual(options["best_view_metric"], "LPIPS")
        for metric in ("LPIPS", "DINOv2", "CLIP"):
            validate_options({"view_selection": "fixed_and_best", "best_view_metric": metric})
        for bad in ({"view_selection": "best"}, {"best_view_metric": "text-CLIP"}):
            with self.subTest(options=bad), self.assertRaises(ValueError): validate_options(bad)

    def test_single_method_uses_absolute_scores_and_noncomparative_rank_one(self):
        requests = []
        answer = {"candidate_1": {"VF": 5.5, "IA": 4, "PP": 6, "OR": 1}}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)/"input.png"
            image.write_bytes(b"fixture")
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                values = evaluate_gpt(image, {"only_method": image})
        self.assertEqual(len(requests), 1)
        self.assertEqual(values["only_method"]["OR"], 1)
        self.assertAlmostEqual(values["only_method"]["Avg."], 15.5/3)
        content = requests[0]["messages"][0]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 2)
        prompt = content[0]["text"]
        self.assertIn("SINGLE-CANDIDATE REQUEST", prompt)
        self.assertIn("not a comparative ranking", prompt)
        self.assertIn("Do not invent another candidate", prompt)
        self.assertNotIn("candidate_2", "\n".join(block["text"] for block in content if block["type"] == "text"))
        record = RUNTIME_JUDGMENTS[-1]
        self.assertEqual(record["candidate_count"], 1)
        self.assertFalse(record["ranking_comparative"])
        self.assertEqual(record["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest())

    def test_single_method_foreground_context_and_custom_prompt_are_audited(self):
        requests = []
        answer = {"candidate_1": {"VF": 4, "IA": 5, "PP": 6, "OR": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/"input.png"
            image.write_bytes(b"fixture")
            rubric = root/"custom.txt"
            rubric.write_text("Custom absolute rubric; respond with JSON.")
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                evaluate_gpt(image, {"only_method": image},
                             {"prompt_file": str(rubric), "audit_dir": str(root/"audit")},
                             context_images={"reference": image, "renders": {"only_method": image}})
            audit = json.loads(next((root/"audit").glob("*.json")).read_text())
        content = requests[0]["messages"][0]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 4)
        self.assertTrue(content[0]["text"].startswith("Custom absolute rubric"))
        self.assertIn("SINGLE-CANDIDATE REQUEST", content[0]["text"])
        self.assertEqual(set(audit["candidates"]), {"candidate_1"})
        self.assertEqual(set(audit["context_images"]["candidates"]), {"candidate_1"})
        self.assertEqual(audit["request"]["candidate_count"], 1)
        self.assertFalse(audit["request"]["ranking_comparative"])

    def test_singleton_still_rejects_invalid_rank_or_invented_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)/"input.png"
            image.write_bytes(b"fixture")
            for rank in (0, 2, 1.0, True):
                answer = {"candidate_1": {"VF": 4, "IA": 4, "PP": 4, "OR": rank}}
                with self.subTest(rank=rank), patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, []))}):
                    with self.assertRaisesRegex(ValueError, "OR|ranks"):
                        evaluate_gpt(image, {"only_method": image})
            answer = {f"candidate_{i}": {"VF": 4, "IA": 4, "PP": 4, "OR": i} for i in (1, 2)}
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, []))}):
                with self.assertRaisesRegex(ValueError, "exactly all candidate labels"):
                    evaluate_gpt(image, {"only_method": image})

    def test_empty_or_invalid_method_mapping_fails_before_api(self):
        for renders in (None, [], {}, {"": "input.png"}):
            with self.subTest(renders=renders), patch("metrics.gpt_evaluation.metric.create_api_client",
                                                    side_effect=AssertionError("API must not be called")):
                with self.assertRaisesRegex(ValueError, "at least one named method"):
                    evaluate_gpt("input.png", renders)

    def test_default_external_rubric_has_anchors_and_is_identified(self):
        answer = {f"candidate_{i}": {"VF": 4, "IA": 4, "PP": 4, "OR": i} for i in (1, 2)}
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.write_bytes(b"fixture")
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                evaluate_gpt(image, {"A": image, "B": image})
        prompt = requests[0]["messages"][0]["content"][0]["text"]
        for marker in ("VF — Visual Fidelity", "IA — Image Alignment", "PP — Physical Plausibility",
                       "1:", "4:", "7:", "not a physics simulation", "do not mechanically sort"):
            self.assertIn(marker, prompt)
        self.assertIn("No separate support-context images", prompt)
        record = RUNTIME_JUDGMENTS[-1]
        self.assertEqual(record["prompt_source"], str(DEFAULT_PROMPT_FILE))
        self.assertEqual(record["prompt_source_sha256"], hashlib.sha256(DEFAULT_PROMPT_FILE.read_bytes()).hexdigest())
        self.assertEqual(record["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest())
        self.assertIsNone(record["context_images"])

    def test_custom_rubric_and_context_keep_same_anonymous_mapping(self):
        answer = {f"candidate_{i}": {"VF": 4, "IA": 4, "PP": 4, "OR": i} for i in (1, 2)}
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, raw_reference = root / "foreground_ref.png", root / "raw_ref.png"
            reference.write_bytes(b"foreground-reference")
            raw_reference.write_bytes(b"original-reference")
            renders, raw_renders = {}, {}
            for method in ("private_method_A", "private_method_B"):
                renders[method], raw_renders[method] = root / f"fg_{method}.png", root / f"raw_{method}.png"
                renders[method].write_bytes(f"foreground-{method}".encode())
                raw_renders[method].write_bytes(f"original-{method}".encode())
            rubric = root / "custom.txt"
            rubric.write_text("Custom project rubric: use JSON.\n", encoding="utf-8")
            context = {"reference": raw_reference, "renders": raw_renders}
            options = {"prompt_file": str(rubric), "order_seed": 9, "audit_dir": str(root / "audit")}
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                evaluate_gpt(reference, renders, options, context_images=context)
            content = requests[0]["messages"][0]["content"]
            prompt = content[0]["text"]
            self.assertTrue(prompt.startswith("Custom project rubric: use JSON.\n\n"))
            self.assertIn("Use Group A for VF and IA", prompt)
            self.assertIn("minor edge jaggies caused by masking", prompt)
            self.assertIn("object silhouettes, count and placement must still", prompt)
            self.assertIn("Use Group B only for PP", prompt)
            self.assertIn("does not add candidates or viewpoints", prompt)
            captions = [block["text"] for block in content[1:] if block["type"] == "text"]
            self.assertEqual(captions, [
                "Group A / foreground — reference image:", "Group A / foreground — candidate_1:",
                "Group A / foreground — candidate_2:", "Group B / original support context — reference image:",
                "Group B / original support context — candidate_1:", "Group B / original support context — candidate_2:"])
            all_text = "\n".join(block["text"] for block in content if block["type"] == "text")
            self.assertNotIn("private_method", all_text)
            images = [base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
                      for block in content if block["type"] == "image_url"]
            record = json.loads(next((root / "audit").glob("*.json")).read_text())
            order = [record["candidates"][f"candidate_{i}"]["method"] for i in (1, 2)]
            expected = [reference.read_bytes(), *(renders[name].read_bytes() for name in order),
                        raw_reference.read_bytes(), *(raw_renders[name].read_bytes() for name in order)]
            self.assertEqual(images, expected)
            self.assertEqual(record["request"]["prompt_source"], str(rubric.resolve()))
            self.assertEqual(record["request"]["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest())
            for label, primary in record["candidates"].items():
                raw = record["context_images"]["candidates"][label]
                self.assertEqual(raw["method"], primary["method"])
                self.assertEqual(raw["sha256"], hashlib.sha256(raw_renders[raw["method"]].read_bytes()).hexdigest())
            self.assertEqual(RUNTIME_JUDGMENTS[-1]["context_images"], record["context_images"])

    def test_context_contract_fails_before_api_call(self):
        bad_contexts = ({}, {"reference": "ref.png"},
                        {"reference": "ref.png", "renders": {"A": "a.png"}},
                        {"reference": None, "renders": {"A": "a.png", "B": "b.png"}},
                        {"reference": "ref.png", "renders": {"A": "a.png", "B": "b.png"}, "extra": True})
        for context in bad_contexts:
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "GPT context"):
                evaluate_gpt("ref.png", {"A": "a.png", "B": "b.png"}, context_images=context)

    def test_prompt_file_must_be_nonempty_utf8_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.txt").write_text(" \n")
            (root / "binary.txt").write_bytes(b"\xff")
            for name in ("missing.txt", "empty.txt", "binary.txt"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "GPT prompt_file"):
                    evaluate_gpt("ref.png", {"A": "a.png", "B": "b.png"}, {"prompt_file": str(root / name)})

    def test_decimal_scores_and_direct_rank_are_preserved(self):
        answer = {"candidate_1": {"VF": 6.5, "IA": 7, "PP": 6, "OR": 2},
                  "candidate_2": {"VF": 4, "IA": 4, "PP": 4, "OR": 1}}
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "ref.png"
            image.write_bytes(b"fixture")
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, requests))}):
                values = evaluate_gpt(image, {"A": image, "B": image})
        higher = next(value for value in values.values() if value["VF"] == 6.5)
        self.assertEqual(higher["Avg."], 6.5)
        self.assertEqual(higher["OR"], 2)

    def test_invalid_scores_ranks_and_fields_are_rejected(self):
        mutations = ({"VF": True}, {"IA": float("nan")}, {"PP": 8}, {"OR": 1},
                     {"OR": 2.0}, {"Avg.": 4})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "ref.png"
            image.write_bytes(b"fixture")
            for mutation in mutations:
                answer = {"candidate_1": {"VF": 4, "IA": 4, "PP": 4, "OR": 1},
                          "candidate_2": {"VF": 4, "IA": 4, "PP": 4, "OR": 2}}
                answer["candidate_2"].update(mutation)
                with self.subTest(mutation=mutation), patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=self.client(answer, []))}), self.assertRaises(ValueError):
                    evaluate_gpt(image, {"A": image, "B": image})

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "requires python-dotenv")
    def test_explicit_env_file_preserves_shell_credentials(self):
        observed = []
        class Client:
            def __init__(self, **kwargs):
                observed.append(os.environ.get("OPENAI_API_KEY"))
                self.chat = types.SimpleNamespace(completions=self)
            def create(self, **kwargs):
                answer = {f"candidate_{i}": {"VF": 4, "IA": 4, "PP": 4, "OR": i} for i in (1, 2)}
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(answer)))])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "private.env").write_text("OPENAI_API_KEY=fake-key-from-file\n")
            (root / "ref.png").write_bytes(b"fixture")
            options = {"env_file": str(root / "private.env"), "audit_dir": str(root / "audit")}
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}), patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key-from-shell"}):
                evaluate_gpt(root / "ref.png", {"A": root / "ref.png", "B": root / "ref.png"}, options)
                self.assertEqual(observed, ["fake-key-from-shell"])
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}), patch.dict(os.environ, {}, clear=True):
                evaluate_gpt(root / "ref.png", {"A": root / "ref.png", "B": root / "ref.png"}, options)
                self.assertEqual(observed[-1], "fake-key-from-file")
            for record in (root / "audit").glob("*.json"):
                self.assertNotIn("fake-key", record.read_text())

    def test_audit_preserves_mapping_response_and_high_detail(self):
        requests = []
        answer = {"candidate_1": {"VF": 7, "IA": 6, "PP": 5, "OR": 1},
                  "candidate_2": {"VF": 4, "IA": 3, "PP": 2, "OR": 2}}
        class Client:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=self)
            def create(self, **kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(
                    id="offline-response", model="judge-snapshot", system_fingerprint="test",
                    usage=types.SimpleNamespace(total_tokens=123),
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(answer)),
                                                   finish_reason="stop")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ref.png"
            reference.write_bytes(b"reference-fixture")
            renders = {}
            for name in ("A", "B"):
                renders[name] = root / f"{name}.png"
                renders[name].write_bytes(name.encode())
            options = {"model": "requested-snapshot", "audit_dir": str(root / "audit"),
                       "order_seed": 9}
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}):
                judged = evaluate_gpt(reference, renders, options)
                # Invalid judgments must also leave evidence behind.
                answer["candidate_1"]["VF"] = 8
                with self.assertRaisesRegex(ValueError, "invalid VF"):
                    evaluate_gpt(reference, renders, options)
            records = [json.loads(p.read_text()) for p in (root / "audit").glob("*.json")]
            self.assertEqual(len(records), 2)
            valid = next(r for r in records if json.loads(r["response"]["choices"][0]["message"]["content"])["candidate_1"]["VF"] == 7)
            self.assertEqual(valid["request"]["model"], "requested-snapshot")
            self.assertEqual(valid["request"]["order_seed"], 9)
            self.assertEqual(valid["response"]["model"], "judge-snapshot")
            self.assertEqual(valid["response"]["usage"]["total_tokens"], 123)
            self.assertEqual(valid["reference"]["sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
            candidate = valid["candidates"]["candidate_1"]
            self.assertEqual(judged[candidate["method"]]["VF"], 7)
            self.assertIn("1–7", valid["request"]["prompt"])
            image_blocks = [b for b in requests[0]["messages"][0]["content"] if b["type"] == "image_url"]
            self.assertEqual(len(image_blocks), 3)
            self.assertTrue(all(b["image_url"]["detail"] == "high" for b in image_blocks))
            self.assertNotIn("seed", requests[0])

    def test_invalid_presentation_options_fail_before_api_call(self):
        for options in ({"image_detail": "huge"}, {"order_seed": True}, {"audit_dir": ""}, {"prompt_file": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_gpt("ref.png", {"A": "a.png", "B": "b.png"}, options)


if __name__ == "__main__":
    unittest.main()
