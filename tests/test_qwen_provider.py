from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib import error

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.base import InferenceResponse
from ecp_mllm.qwen.dashscope_openai import DashScopeOpenAIClient, _estimated_data_url_bytes
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


class _FakeRequestsResponse:
    def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


class DashScopeProviderTests(unittest.TestCase):
    def _make_clip(self, asset_path: Path) -> ClipRecord:
        return ClipRecord(
            dataset="private",
            domain="haida",
            clip_id="clip_001",
            asset_paths={InputVariant.RAW: asset_path},
            width=640,
            height=480,
            framerate=12.0,
            duration_seconds=4.0,
        )

    def test_builds_frame_directory_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(10):
                (root / f"frame_{index:03d}.jpg").write_bytes(b"jpeg-bytes")
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct", max_frames=4)
            payload = client._build_request_payload(
                self._make_clip(root),
                InputVariant.RAW,
                PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only."),
            )
            content = payload["messages"][0]["content"]
            media = content[0]
            prompt = content[1]
            self.assertEqual(payload["model"], "qwen2.5-vl-7b-instruct")
            self.assertEqual(media["type"], "video")
            self.assertEqual(len(media["video"]), 4)
            self.assertAlmostEqual(media["fps"], 4.0)
            self.assertEqual(prompt["type"], "text")
            self.assertIn("Return JSON only.", prompt["text"])

    def test_infer_parses_openai_compatible_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct")
            clip = self._make_clip(image_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"left_count": 1, "right_count": 2, "confidence": 0.75, "events": []}'
                        }
                    }
                ]
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)) as patched:
                    response = client.infer(clip, InputVariant.RAW, prompt)
            self.assertIsInstance(response, InferenceResponse)
            self.assertIn('"left_count": 1', response.text)
            sent_body = json.loads(patched.call_args.args[0].data.decode("utf-8"))
            self.assertEqual(sent_body["model"], "qwen2.5-vl-7b-instruct")
            self.assertEqual(sent_body["messages"][0]["content"][0]["type"], "image_url")

    def test_infer_retries_transient_ssl_eof(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct")
            clip = self._make_clip(image_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"left_count": 0, "right_count": 1, "confidence": 0.6, "events": []}'
                        }
                    }
                ]
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch(
                    "urllib.request.urlopen",
                    side_effect=[
                        error.URLError("EOF occurred in violation of protocol"),
                        _FakeHTTPResponse(response_payload),
                    ],
                ) as patched:
                    response = client.infer(clip, InputVariant.RAW, prompt)
            self.assertIn('"right_count": 1', response.text)
            self.assertEqual(patched.call_count, 2)

    def test_infer_uses_windows_user_env_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct")
            clip = self._make_clip(image_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"left_count": 0, "right_count": 1, "confidence": 0.6, "events": []}'
                        }
                    }
                ]
            }
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai._load_windows_user_env",
                    return_value="test-key",
                ):
                    with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)):
                        response = client.infer(clip, InputVariant.RAW, prompt)
            self.assertIn('"right_count": 1', response.text)

    def test_direct_api_key_skips_env_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct", api_key="sk-direct")
            clip = self._make_clip(image_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"left_count": 0, "right_count": 1, "confidence": 0.6, "events": []}'
                        }
                    }
                ]
            }
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)) as patched:
                    response = client.infer(clip, InputVariant.RAW, prompt)
            self.assertIn('"right_count": 1', response.text)
            sent_request = patched.call_args.args[0]
            header_map = {key.lower(): value for key, value in sent_request.header_items()}
            self.assertEqual(header_map["authorization"], "Bearer sk-direct")

    def test_uses_metadata_frame_paths_for_flat_clip_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_paths = []
            for index in range(6):
                frame_path = root / f"flat_clip_{index:03d}.jpg"
                frame_path.write_bytes(f"jpeg-{index}".encode("utf-8"))
                frame_paths.append(frame_path)
            clip = ClipRecord(
                dataset="cfc",
                domain="kenai-val",
                clip_id="flat_clip",
                asset_paths={InputVariant.RAW: frame_paths[0]},
                framerate=9.0,
                metadata={"frame_paths_by_variant": {"raw": [str(path) for path in frame_paths]}},
            )
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct", max_frames=3)
            payload = client._build_request_payload(
                clip,
                InputVariant.RAW,
                PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only."),
            )
            media = payload["messages"][0]["content"][0]
            self.assertEqual(media["type"], "video")
            self.assertEqual(len(media["video"]), 3)
            self.assertAlmostEqual(media["fps"], 4.5)

    def test_prefers_mp4_asset_over_metadata_frame_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4")
            frame_path = root / "frame_000.jpg"
            frame_path.write_bytes(b"jpeg-bytes")
            clip = ClipRecord(
                dataset="cfc",
                domain="kenai-val",
                clip_id="flat_clip",
                asset_paths={InputVariant.RAW: mp4_path},
                framerate=9.0,
                metadata={"frame_paths_by_variant": {"raw": [str(frame_path)]}},
            )
            client = DashScopeOpenAIClient(model="qwen2.5-vl-7b-instruct")
            media = client._build_media_item(clip, InputVariant.RAW, mp4_path)
            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("data:video/"))

    def test_estimated_data_url_bytes_accounts_for_base64_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"0123456789")
            estimated = _estimated_data_url_bytes(mp4_path)
            self.assertGreater(estimated, mp4_path.stat().st_size)

    def test_uses_oss_when_data_url_estimate_exceeds_dashscope_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"0123456789")
            clip = self._make_clip(mp4_path)
            client = DashScopeOpenAIClient(model="qwen3.5-plus")
            with mock.patch("ecp_mllm.qwen.dashscope_openai.DEFAULT_DASHSCOPE_MAX_DATA_URI_BYTES", 32):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai._upload_file_to_temporary_oss",
                    return_value="oss://dashscope-instant/test/clip.mp4",
                ) as patched_upload:
                    media = client._build_media_item(clip, InputVariant.RAW, mp4_path, api_key="test-key")
            self.assertEqual(media["type"], "video_url")
            self.assertEqual(media["video_url"]["url"], "oss://dashscope-instant/test/clip.mp4")
            self.assertEqual(patched_upload.call_count, 1)

    def test_infer_uploads_local_mp4_to_temporary_oss_and_sets_resolve_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4-payload")
            clip = self._make_clip(mp4_path)
            prompt = PromptRevision(version=0, prompt_id="base", prompt_text="Return JSON only.")
            client = DashScopeOpenAIClient(model="qwen3.5-plus")
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"left_count": 0, "right_count": 1, "confidence": 0.6, "events": []}'
                        }
                    }
                ]
            }
            policy_payload = {
                "data": {
                    "upload_dir": "dashscope-instant/test/2026-03-20/upload",
                    "upload_host": "https://oss-upload.example.com",
                    "oss_access_key_id": "oss-key",
                    "signature": "sig",
                    "policy": "policy",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai.requests.get",
                    return_value=_FakeRequestsResponse(status_code=200, payload=policy_payload),
                ) as patched_get:
                    with mock.patch(
                        "ecp_mllm.qwen.dashscope_openai.requests.post",
                        return_value=_FakeRequestsResponse(status_code=200),
                    ) as patched_post:
                        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(response_payload)) as patched_urlopen:
                            response = client.infer(clip, InputVariant.RAW, prompt)

            self.assertIn('"right_count": 1', response.text)
            self.assertEqual(patched_get.call_count, 1)
            self.assertEqual(patched_post.call_count, 1)
            sent_request = patched_urlopen.call_args.args[0]
            sent_body = json.loads(sent_request.data.decode("utf-8"))
            media = sent_body["messages"][0]["content"][0]
            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("oss://dashscope-instant/test/2026-03-20/upload/"))
            header_map = {key.lower(): value for key, value in sent_request.header_items()}
            self.assertEqual(header_map["x-dashscope-ossresourceresolve"], "enable")
            self.assertTrue((mp4_path.parent / f"{mp4_path.name}.dashscope_upload.json").exists())

    def test_prepare_media_item_uploads_local_mp4_and_returns_oss_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4-payload")
            clip = self._make_clip(mp4_path)
            client = DashScopeOpenAIClient(model="qwen3.5-plus")
            policy_payload = {
                "data": {
                    "upload_dir": "dashscope-instant/test/2026-03-20/upload",
                    "upload_host": "https://oss-upload.example.com",
                    "oss_access_key_id": "oss-key",
                    "signature": "sig",
                    "policy": "policy",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai.requests.get",
                    return_value=_FakeRequestsResponse(status_code=200, payload=policy_payload),
                ) as patched_get:
                    with mock.patch(
                        "ecp_mllm.qwen.dashscope_openai.requests.post",
                        return_value=_FakeRequestsResponse(status_code=200),
                    ) as patched_post:
                        media = client.prepare_media_item(clip, InputVariant.RAW)

            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("oss://dashscope-instant/test/2026-03-20/upload/"))
            self.assertEqual(patched_get.call_count, 1)
            self.assertEqual(patched_post.call_count, 1)
            self.assertTrue((mp4_path.parent / f"{mp4_path.name}.dashscope_upload.json").exists())

    def test_stale_cached_upload_is_ignored_and_reuploaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4-payload")
            cache_path = mp4_path.parent / f"{mp4_path.name}.dashscope_upload.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "oss_url": "oss://dashscope-instant/test/stale/clip.mp4",
                        "sha256": "400a0a0de940446f841c000bca9dd97065337e407fe8d06a28dcd79f6f21978f",
                        "size_bytes": len(b"fake-mp4-payload"),
                        "model": "qwen3.5-plus",
                        "uploads_url": "https://dashscope.aliyuncs.com/api/v1/uploads",
                    }
                ),
                encoding="utf-8",
            )
            stale_time = cache_path.stat().st_mtime - (7 * 60 * 60)
            import os
            os.utime(cache_path, (stale_time, stale_time))

            clip = self._make_clip(mp4_path)
            client = DashScopeOpenAIClient(model="qwen3.5-plus")
            policy_payload = {
                "data": {
                    "upload_dir": "dashscope-instant/test/2026-03-20/upload",
                    "upload_host": "https://oss-upload.example.com",
                    "oss_access_key_id": "oss-key",
                    "signature": "sig",
                    "policy": "policy",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai.requests.get",
                    return_value=_FakeRequestsResponse(status_code=200, payload=policy_payload),
                ) as patched_get:
                    with mock.patch(
                        "ecp_mllm.qwen.dashscope_openai.requests.post",
                        return_value=_FakeRequestsResponse(status_code=200),
                    ) as patched_post:
                        media = client.prepare_media_item(clip, InputVariant.RAW)

            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("oss://dashscope-instant/test/2026-03-20/upload/"))
            self.assertEqual(patched_get.call_count, 1)
            self.assertEqual(patched_post.call_count, 1)

    def test_large_dashscope_mp4_falls_back_to_oss_even_when_prefer_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            with mp4_path.open("wb") as handle:
                handle.seek(10 * 1024 * 1024)
                handle.write(b"\0")
            clip = self._make_clip(mp4_path)
            client = DashScopeOpenAIClient(
                model="qwen3.5-plus",
                prefer_temporary_oss_upload=False,
            )
            policy_payload = {
                "data": {
                    "upload_dir": "dashscope-instant/test/2026-03-20/upload",
                    "upload_host": "https://oss-upload.example.com",
                    "oss_access_key_id": "oss-key",
                    "signature": "sig",
                    "policy": "policy",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            }
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
                with mock.patch(
                    "ecp_mllm.qwen.dashscope_openai.requests.get",
                    return_value=_FakeRequestsResponse(status_code=200, payload=policy_payload),
                ) as patched_get:
                    with mock.patch(
                        "ecp_mllm.qwen.dashscope_openai.requests.post",
                        return_value=_FakeRequestsResponse(status_code=200),
                    ) as patched_post:
                        media = client.prepare_media_item(clip, InputVariant.RAW)

            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("oss://dashscope-instant/test/2026-03-20/upload/"))
            self.assertEqual(patched_get.call_count, 1)
            self.assertEqual(patched_post.call_count, 1)

    def test_non_dashscope_base_url_defaults_to_inline_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "clip.mp4"
            mp4_path.write_bytes(b"fake-mp4-payload")
            clip = self._make_clip(mp4_path)
            client = DashScopeOpenAIClient(
                model="qwen-3.5",
                api_key="sk-direct",
                base_url="https://once.novai.su/v1",
            )
            media = client.prepare_media_item(clip, InputVariant.RAW)
            self.assertEqual(media["type"], "video_url")
            self.assertTrue(media["video_url"]["url"].startswith("data:video/"))


if __name__ == "__main__":
    unittest.main()

