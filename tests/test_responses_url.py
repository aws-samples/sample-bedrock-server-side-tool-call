"""Unit tests for the per-model Bedrock Mantle Responses API URL resolution.

Background: the GPT-5.6 model family (Sol/Terra/Luna) — and 5.4/5.5 — is
served on the `openai/v1/responses` path instead of the standard
`v1/responses` path used by every other model on the bedrock-mantle
Responses endpoint. Source of truth:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html

Using the wrong path for a GPT-5.6-family model produces a live HTTP 400:
    {"error": {"code": "validation_error",
               "message": "The model 'openai.gpt-5.6-sol' does not support
               the '/v1/responses' API"}}

This test imports the `_responses_url` helper directly from both modified
modules (rather than re-implementing the logic) to prove the fix.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.serverside_agent import _responses_url as serverside_responses_url  # noqa: E402


class ServersideAgentResponsesUrlTest(unittest.TestCase):
    """Tests for agent/serverside_agent.py::_responses_url"""

    def test_gpt_5_6_family_uses_openai_v1_responses_path(self):
        for model_id in (
            "openai.gpt-5.6-sol",
            "openai.gpt-5.6-terra",
            "openai.gpt-5.6-luna",
        ):
            with self.subTest(model_id=model_id):
                url = serverside_responses_url("us-west-2", model_id)
                self.assertIn("/openai/v1/responses", url)

    def test_non_gpt_5_x_models_use_plain_v1_responses_path(self):
        for model_id in ("openai.gpt-oss-120b", "anthropic.claude-3-sonnet"):
            with self.subTest(model_id=model_id):
                url = serverside_responses_url("us-west-2", model_id)
                self.assertIn("/v1/responses", url)
                self.assertNotIn("/openai/v1/responses", url)

    def test_region_is_interpolated(self):
        url = serverside_responses_url("ap-southeast-1", "openai.gpt-5.6-sol")
        self.assertEqual(
            url, "https://bedrock-mantle.ap-southeast-1.api.aws/openai/v1/responses"
        )

        url = serverside_responses_url("eu-west-1", "openai.gpt-oss-120b")
        self.assertEqual(url, "https://bedrock-mantle.eu-west-1.api.aws/v1/responses")


class ShopassistRuntimeResponsesUrlTest(unittest.TestCase):
    """Tests for agent/shopassist_runtime.py::_responses_url

    This module requires `bedrock_agentcore` (Python >= 3.10 per its
    packaging metadata), which may not be importable on every interpreter
    on a given machine. Skip gracefully rather than failing the whole suite
    when that dependency isn't available under the interpreter running the
    tests.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from agent.shopassist_runtime import _responses_url

            cls._responses_url = staticmethod(_responses_url)
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(
                f"agent.shopassist_runtime not importable on this interpreter: {exc}"
            )

    def test_gpt_5_6_family_uses_openai_v1_responses_path(self):
        for model_id in (
            "openai.gpt-5.6-sol",
            "openai.gpt-5.6-terra",
            "openai.gpt-5.6-luna",
        ):
            with self.subTest(model_id=model_id):
                url = self._responses_url("us-west-2", model_id)
                self.assertIn("/openai/v1/responses", url)

    def test_non_gpt_5_x_models_use_plain_v1_responses_path(self):
        for model_id in ("openai.gpt-oss-120b", "anthropic.claude-3-sonnet"):
            with self.subTest(model_id=model_id):
                url = self._responses_url("us-west-2", model_id)
                self.assertIn("/v1/responses", url)
                self.assertNotIn("/openai/v1/responses", url)

    def test_region_is_interpolated(self):
        url = self._responses_url("ap-southeast-1", "openai.gpt-5.6-sol")
        self.assertEqual(
            url, "https://bedrock-mantle.ap-southeast-1.api.aws/openai/v1/responses"
        )


if __name__ == "__main__":
    unittest.main()
