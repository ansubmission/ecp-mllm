from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.base import InferenceResponse
from ecp_mllm.qwen.gemini_api import GeminiApiClient
from ecp_mllm.types import ClipRecord, InputVariant, PromptRevision


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class GeminiProviderTests(unittest.TestCase):
    def _make_clip(self, asset_path: Path) -> ClipRecord:
        return ClipRecord(
            dataset="private",
            domain="haida",
            clip_id="clip_001",
            asset_paths={InputVariant.RAW: asset_path},
            width=640,
            height=480,
            framerate=5.0,
            duration_seconds=6.0,
        )

    def test_builds_inline_video_payload_with_thinking_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4-payload")
            clip = self._make_clip(mp4_path)
            client = GeminiApiClient(
                model="gemini-3.1-pro-preview",
                api_key="test-key",
                thinking_level="medium",
            )
            payload = client._build_request_payload(
                clip,
                InputVariant.RAW,
                "Return JSON only.",
            )
            parts = payload["contents"][0]["parts"]
            self.assertIn("inline_data", parts[0])
            self.assertEqual(parts[0]["inline_data"]["mime_type"], "video/mp4")
            self.assertEqual(parts[0]["video_metadata"]["fps"], 5.0)
            self.assertEqual(parts[1]["text"], "Return JSON only.")
            self.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinkingLevel"], "medium")

    def test_builds_sampled_frame_payload_for_directory_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                (root / f"frame_{index:03d}.jpg").write_bytes(b"jpeg-bytes")
            clip = self._make_clip(root)
            client = GeminiApiClient(
                model="gemini-3-flash-preview",
                api_key="test-key",
                max_frames=3,
            )
            payload = client._build_request_payload(
                clip,
                InputVariant.RAW,
                "Return JSON only.",
            )
            parts = payload["contents"][0]["parts"]
            self.assertEqual(len(parts), 4)
            self.assertIn("inline_data", parts[0])
            self.assertEqual(parts[-1]["text"], "Return JSON only.")

    def test_infer_parses_generate_content_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            clip = self._make_clip(image_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            client = GeminiApiClient(
                model="gemini-3.1-pro-preview",
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
            response_payload = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"left_count": 0, "right_count": 2, "confidence": 0.8, "events": []}'
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 200,
                    "thoughtsTokenCount": 50,
                    "totalTokenCount": 1250,
                },
            }
            with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)) as patched:
                response = client.infer(clip, InputVariant.RAW, prompt)
            self.assertIsInstance(response, InferenceResponse)
            self.assertIn('"right_count": 2', response.text)
            self.assertEqual(response.usage_metadata["promptTokenCount"], 1000)
            self.assertAlmostEqual(response.estimated_cost_usd or 0.0, 0.0050, places=6)
            sent_request = patched.call_args.args[0]
            self.assertIn(
                "/models/gemini-3.1-pro-preview:generateContent",
                sent_request.full_url,
            )
            sent_body = json.loads(sent_request.data.decode("utf-8"))
            self.assertIn(prompt.prompt_text, sent_body["contents"][0]["parts"][1]["text"])

    def test_rejects_invalid_thinking_level(self) -> None:
        with self.assertRaises(ValueError):
            GeminiApiClient(model="gemini-3.1-pro-preview", thinking_level="fast")


if __name__ == "__main__":
    unittest.main()

