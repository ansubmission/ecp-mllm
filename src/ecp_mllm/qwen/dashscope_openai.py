from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib import parse
from urllib import error, request

import requests

from ..types import ClipRecord, InputVariant, PromptRevision
from .base import InferenceResponse, QwenClient
from .prompting import build_prompt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_FRAMES = 8
DEFAULT_VIDEO_INLINE_BYTES = 24 * 1024 * 1024
DEFAULT_DASHSCOPE_MAX_DATA_URI_BYTES = 10 * 1024 * 1024
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 1800
DEFAULT_UPLOAD_CACHE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_INFER_TRANSIENT_RETRY_ATTEMPTS = 2
OSS_RESOLVE_HEADER = "X-DashScope-OssResourceResolve"
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class _CachedUpload:
    oss_url: str
    sha256: str
    size_bytes: int
    model: str
    uploads_url: str


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def _estimated_data_url_bytes(path: Path) -> int:
    size_bytes = path.stat().st_size
    base64_bytes = ((size_bytes + 2) // 3) * 4
    prefix = f"data:{_guess_mime_type(path)};base64,"
    return len(prefix.encode("utf-8")) + base64_bytes


def _completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _uploads_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized.endswith("/api/v1/uploads"):
        return normalized
    if normalized.endswith("/api/v1"):
        return f"{normalized}/uploads"
    if normalized.endswith("/compatible-mode/v1"):
        return f"{normalized[: -len('/compatible-mode/v1')]}/api/v1/uploads"
    parsed = parse.urlsplit(normalized)
    if parsed.path.endswith("/compatible-mode/v1"):
        prefix = parsed.path[: -len("/compatible-mode/v1")]
        return parse.urlunsplit((parsed.scheme, parsed.netloc, f"{prefix}/api/v1/uploads", "", ""))
    return f"{normalized}/api/v1/uploads"


def _sorted_frame_paths(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _metadata_frame_paths(clip: ClipRecord, variant: InputVariant) -> list[Path]:
    frame_paths_by_variant = clip.metadata.get("frame_paths_by_variant")
    if not isinstance(frame_paths_by_variant, dict):
        return []
    value = frame_paths_by_variant.get(variant.value)
    if not isinstance(value, list):
        return []
    return [Path(item) for item in value]


def _sample_uniform(paths: list[Path], max_items: int) -> tuple[list[Path], int]:
    if not paths:
        return [], 1
    if len(paths) <= max_items:
        return paths, 1
    stride = max(1, (len(paths) + max_items - 1) // max_items)
    sampled = paths[::stride][:max_items]
    return sampled, stride


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text") is not None:
                text_parts.append(str(item["text"]))
        return "\n".join(part for part in text_parts if part.strip())
    raise ValueError("model response did not include text content")


def _load_windows_user_env(var_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, var_name)
    except OSError:
        return None
    return str(value).strip() or None


def _resolve_api_key(var_name: str) -> str | None:
    return os.getenv(var_name) or _load_windows_user_env(var_name)


def _supports_dashscope_upload(base_url: str) -> bool:
    host = parse.urlsplit(base_url.rstrip("/")).netloc.lower()
    return "dashscope" in host or host.endswith("aliyuncs.com")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_cache_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.dashscope_upload.json")


def _load_cached_upload(path: Path) -> _CachedUpload | None:
    cache_path = _upload_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        age_seconds = max(0.0, time.time() - cache_path.stat().st_mtime)
    except OSError:
        return None
    if age_seconds > DEFAULT_UPLOAD_CACHE_TTL_SECONDS:
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _CachedUpload(
            oss_url=str(payload["oss_url"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            model=str(payload["model"]),
            uploads_url=str(payload["uploads_url"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_cached_upload(path: Path, cached: _CachedUpload) -> None:
    cache_path = _upload_cache_path(path)
    payload = {
        "oss_url": cached.oss_url,
        "sha256": cached.sha256,
        "size_bytes": cached.size_bytes,
        "model": cached.model,
        "uploads_url": cached.uploads_url,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_cached_upload(path: Path) -> None:
    cache_path = _upload_cache_path(path)
    try:
        cache_path.unlink()
    except FileNotFoundError:
        return


def _emit_progress(callback: ProgressCallback | None, stage: str, **details: object) -> None:
    if callback is None:
        return
    callback(stage, details)


def _is_transient_infer_error(detail: str) -> bool:
    normalized = detail.lower()
    transient_needles = (
        "unexpected eof while reading",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "remote end closed connection",
        "ssl eof",
        "eof occurred in violation of protocol",
    )
    return any(needle in normalized for needle in transient_needles)


def _get_upload_policy(
    api_key: str,
    uploads_url: str,
    model: str,
    timeout_seconds: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    _emit_progress(
        progress_callback,
        "upload_policy_start",
        uploads_url=uploads_url,
        model=model,
    )
    response = requests.get(
        uploads_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        params={"action": "getPolicy", "model": model},
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        raise RuntimeError(f"DashScope upload policy request failed with HTTP {response.status_code}: {response.text}")
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"DashScope upload policy response did not include data: {payload}")
    _emit_progress(
        progress_callback,
        "upload_policy_done",
        uploads_url=uploads_url,
        model=model,
        elapsed_sec=time.perf_counter() - start,
    )
    return data


def _upload_file_to_temporary_oss(
    *,
    api_key: str,
    uploads_url: str,
    model: str,
    asset_path: Path,
    timeout_seconds: int,
    progress_callback: ProgressCallback | None = None,
    force_reupload: bool = False,
) -> str:
    upload_start = time.perf_counter()
    file_sha256 = _sha256_path(asset_path)
    size_bytes = asset_path.stat().st_size
    _emit_progress(
        progress_callback,
        "upload_prepare",
        asset_path=str(asset_path),
        size_bytes=size_bytes,
        model=model,
    )
    cached = None if force_reupload else _load_cached_upload(asset_path)
    if cached is not None:
        if (
            cached.sha256 == file_sha256
            and cached.size_bytes == size_bytes
            and cached.model == model
            and cached.uploads_url == uploads_url
            and cached.oss_url.startswith("oss://")
        ):
            _emit_progress(
                progress_callback,
                "upload_cache_hit",
                asset_path=str(asset_path),
                size_bytes=size_bytes,
                model=model,
                uploads_url=uploads_url,
                oss_url=cached.oss_url,
                elapsed_sec=time.perf_counter() - upload_start,
            )
            return cached.oss_url

    policy = _get_upload_policy(
        api_key,
        uploads_url,
        model,
        timeout_seconds,
        progress_callback=progress_callback,
    )
    required_keys = (
        "upload_dir",
        "upload_host",
        "oss_access_key_id",
        "signature",
        "policy",
        "x_oss_object_acl",
        "x_oss_forbid_overwrite",
    )
    missing = [key for key in required_keys if key not in policy]
    if missing:
        raise RuntimeError(f"DashScope upload policy response is missing fields: {', '.join(missing)}")

    key = f"{policy['upload_dir']}/{asset_path.name}"
    post_start = time.perf_counter()
    _emit_progress(
        progress_callback,
        "upload_start",
        asset_path=str(asset_path),
        size_bytes=size_bytes,
        upload_host=str(policy["upload_host"]),
        oss_key=key,
    )
    with asset_path.open("rb") as handle:
        files = {
            "OSSAccessKeyId": (None, str(policy["oss_access_key_id"])),
            "Signature": (None, str(policy["signature"])),
            "policy": (None, str(policy["policy"])),
            "x-oss-object-acl": (None, str(policy["x_oss_object_acl"])),
            "x-oss-forbid-overwrite": (None, str(policy["x_oss_forbid_overwrite"])),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (asset_path.name, handle, _guess_mime_type(asset_path)),
        }
        response = requests.post(str(policy["upload_host"]), files=files, timeout=timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"DashScope temporary OSS upload failed with HTTP {response.status_code}: {response.text}")

    oss_url = f"oss://{key}"
    _write_cached_upload(
        asset_path,
        _CachedUpload(
            oss_url=oss_url,
            sha256=file_sha256,
            size_bytes=size_bytes,
            model=model,
            uploads_url=uploads_url,
        ),
    )
    _emit_progress(
        progress_callback,
        "upload_done",
        asset_path=str(asset_path),
        size_bytes=size_bytes,
        uploads_url=uploads_url,
        oss_url=oss_url,
        post_elapsed_sec=time.perf_counter() - post_start,
        total_elapsed_sec=time.perf_counter() - upload_start,
    )
    return oss_url


def _payload_uses_oss_resources(payload: dict[str, object]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video_url":
                video_url = item.get("video_url")
                if isinstance(video_url, dict) and str(video_url.get("url", "")).startswith("oss://"):
                    return True
            if item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and str(image_url.get("url", "")).startswith("oss://"):
                    return True
    return False


class DashScopeOpenAIClient(QwenClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
        timeout_seconds: int = 120,
        upload_timeout_seconds: int = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
        max_frames: int = DEFAULT_MAX_FRAMES,
        temperature: float = 0.1,
        max_inline_video_bytes: int = DEFAULT_VIDEO_INLINE_BYTES,
        prefer_temporary_oss_upload: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.api_key_env = api_key_env
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.upload_timeout_seconds = upload_timeout_seconds
        self.max_frames = max_frames
        self.temperature = temperature
        self.max_inline_video_bytes = max_inline_video_bytes
        self.prefer_temporary_oss_upload = (
            _supports_dashscope_upload(self.base_url)
            if prefer_temporary_oss_upload is None
            else prefer_temporary_oss_upload
        )
        self.progress_callback: ProgressCallback | None = None

    def resolve_api_key(self) -> str | None:
        return self.api_key or _resolve_api_key(self.api_key_env)

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self.progress_callback = callback

    def prepare_media_item(self, clip: ClipRecord, variant: InputVariant) -> dict[str, object]:
        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"API key is not configured. Set qwen.api_key or make {self.api_key_env} available "
                "in either the current process environment or the Windows user environment."
            )
        asset_path = clip.get_asset_path(variant)
        if asset_path is None:
            raise ValueError(f"Clip {clip.key.value} is missing asset path for {variant.value}")
        return self._build_media_item(clip, variant, asset_path, api_key=api_key)

    def infer(self, clip: ClipRecord, variant: InputVariant, prompt: PromptRevision) -> InferenceResponse:
        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"API key is not configured. Set qwen.api_key or make {self.api_key_env} available "
                "in either the current process environment or the Windows user environment."
            )

        start = time.perf_counter()
        asset_path = clip.get_asset_path(variant)
        _emit_progress(
            self.progress_callback,
            "infer_start",
            clip_key=clip.key.value,
            variant=variant.value,
            model=self.model,
            asset_path=str(asset_path) if asset_path is not None else None,
        )
        response_payload: dict[str, object] | None = None
        last_detail = ""
        for attempt, force_reupload in enumerate((False, True), start=1):
            payload = self._build_request_payload(clip, variant, prompt, api_key=api_key, force_reupload=force_reupload)
            body = json.dumps(payload).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload_uses_oss = _payload_uses_oss_resources(payload)
            if payload_uses_oss:
                headers[OSS_RESOLVE_HEADER] = "enable"
            req = request.Request(
                url=_completions_url(self.base_url),
                data=body,
                method="POST",
                headers=headers,
            )
            _emit_progress(
                self.progress_callback,
                "infer_request_start",
                clip_key=clip.key.value,
                variant=variant.value,
                model=self.model,
                completions_url=_completions_url(self.base_url),
                payload_uses_oss=payload_uses_oss,
                attempt=attempt,
                force_reupload=force_reupload,
                elapsed_sec=time.perf_counter() - start,
            )
            transient_attempts = max(1, DEFAULT_INFER_TRANSIENT_RETRY_ATTEMPTS)
            for transient_attempt in range(1, transient_attempts + 1):
                try:
                    with request.urlopen(req, timeout=self.timeout_seconds) as response:
                        response_payload = json.loads(response.read().decode("utf-8"))
                    break
                except error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    last_detail = detail
                    if (
                        exc.code == 403
                        and "Resource.AccessDenied" in detail
                        and payload_uses_oss
                        and not force_reupload
                        and asset_path is not None
                    ):
                        _emit_progress(
                            self.progress_callback,
                            "infer_retry_force_reupload",
                            clip_key=clip.key.value,
                            variant=variant.value,
                            model=self.model,
                            attempt=attempt,
                        )
                        _clear_cached_upload(asset_path)
                        break
                    if exc.code >= 500 and transient_attempt < transient_attempts:
                        _emit_progress(
                            self.progress_callback,
                            "infer_retry_transient_http",
                            clip_key=clip.key.value,
                            variant=variant.value,
                            model=self.model,
                            attempt=attempt,
                            transient_attempt=transient_attempt,
                            http_status=exc.code,
                        )
                        continue
                    raise RuntimeError(f"DashScope request failed with HTTP {exc.code}: {detail}") from exc
                except error.URLError as exc:
                    detail = str(exc.reason)
                    last_detail = detail
                    if _is_transient_infer_error(detail) and transient_attempt < transient_attempts:
                        _emit_progress(
                            self.progress_callback,
                            "infer_retry_transient_network",
                            clip_key=clip.key.value,
                            variant=variant.value,
                            model=self.model,
                            attempt=attempt,
                            transient_attempt=transient_attempt,
                            reason=detail,
                        )
                        continue
                    raise RuntimeError(f"DashScope request failed: {exc.reason}") from exc
                except OSError as exc:
                    detail = str(exc)
                    last_detail = detail
                    if _is_transient_infer_error(detail) and transient_attempt < transient_attempts:
                        _emit_progress(
                            self.progress_callback,
                            "infer_retry_transient_network",
                            clip_key=clip.key.value,
                            variant=variant.value,
                            model=self.model,
                            attempt=attempt,
                            transient_attempt=transient_attempt,
                            reason=detail,
                        )
                        continue
                    raise RuntimeError(f"DashScope request failed: {detail}") from exc
            if response_payload is not None:
                break
            if (
                force_reupload
                or asset_path is None
                or last_detail
                and "Resource.AccessDenied" not in last_detail
            ):
                if last_detail:
                    raise RuntimeError(f"DashScope request failed after retry: {last_detail}")
                break
        if response_payload is None:
            raise RuntimeError(f"DashScope request failed after retry: {last_detail}")

        message = response_payload["choices"][0]["message"]
        text = _coerce_message_text(message.get("content"))
        latency_sec = time.perf_counter() - start
        _emit_progress(
            self.progress_callback,
            "infer_done",
            clip_key=clip.key.value,
            variant=variant.value,
            model=self.model,
            latency_sec=latency_sec,
        )
        return InferenceResponse(text=text, latency_sec=latency_sec)

    def _build_request_payload(
        self,
        clip: ClipRecord,
        variant: InputVariant,
        prompt: PromptRevision,
        api_key: str | None = None,
        force_reupload: bool = False,
    ) -> dict[str, object]:
        asset_path = clip.get_asset_path(variant)
        if asset_path is None:
            raise ValueError(f"Clip {clip.key.value} is missing asset path for {variant.value}")
        _emit_progress(
            self.progress_callback,
            "payload_build_start",
            clip_key=clip.key.value,
            variant=variant.value,
            asset_path=str(asset_path),
        )
        prompt_text = prompt.prompt_text if prompt.assistant_prefill else build_prompt(clip, variant, prompt)
        content = [
            self._build_media_item(clip, variant, asset_path, api_key=api_key, force_reupload=force_reupload),
            {"type": "text", "text": prompt_text},
        ]
        messages: list[dict[str, object]] = [{"role": "user", "content": content}]
        if prompt.assistant_prefill:
            messages.append({"role": "assistant", "content": str(prompt.assistant_prefill)})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

    def _build_media_item(
        self,
        clip: ClipRecord,
        variant: InputVariant,
        asset_path: Path,
        api_key: str | None = None,
        force_reupload: bool = False,
    ) -> dict[str, object]:
        extension = asset_path.suffix.lower()
        if extension in VIDEO_EXTENSIONS and asset_path.is_file():
            size_bytes = asset_path.stat().st_size
            estimated_data_url_bytes = _estimated_data_url_bytes(asset_path)
            _emit_progress(
                self.progress_callback,
                "media_prepare_video",
                clip_key=clip.key.value,
                variant=variant.value,
                asset_path=str(asset_path),
                size_bytes=size_bytes,
                estimated_data_url_bytes=estimated_data_url_bytes,
            )
            inline_limit = self.max_inline_video_bytes
            if _supports_dashscope_upload(self.base_url):
                inline_limit = min(inline_limit, DEFAULT_DASHSCOPE_MAX_DATA_URI_BYTES)
            should_use_temporary_oss = self.prefer_temporary_oss_upload or (
                _supports_dashscope_upload(self.base_url) and estimated_data_url_bytes > inline_limit
            )
            if should_use_temporary_oss and api_key:
                video_url = _upload_file_to_temporary_oss(
                    api_key=api_key,
                    uploads_url=_uploads_url(self.base_url),
                    model=self.model,
                    asset_path=asset_path,
                    timeout_seconds=self.upload_timeout_seconds,
                    progress_callback=self.progress_callback,
                    force_reupload=force_reupload,
                )
            else:
                if estimated_data_url_bytes > inline_limit:
                    raise RuntimeError(
                        f"Video file is too large to inline as Base64 ({estimated_data_url_bytes} encoded bytes): {asset_path}. "
                        "Provide an API key for temporary OSS upload or reduce stitch resolution/quality."
                    )
                video_url = _data_url(asset_path)
            payload = {"type": "video_url", "video_url": {"url": video_url}}
            if clip.framerate:
                payload["fps"] = clip.framerate
            return payload

        frame_paths = _metadata_frame_paths(clip, variant)
        if not frame_paths and asset_path.is_dir():
            frame_paths = _sorted_frame_paths(asset_path)
        if frame_paths:
            if not frame_paths:
                raise RuntimeError(f"No image frames found in directory: {asset_path}")
            sampled_frames, stride = _sample_uniform(frame_paths, self.max_frames)
            _emit_progress(
                self.progress_callback,
                "media_prepare_frames",
                clip_key=clip.key.value,
                variant=variant.value,
                asset_path=str(asset_path),
                frame_count=len(frame_paths),
                sampled_count=len(sampled_frames),
                stride=stride,
            )
            payload: dict[str, object] = {
                "type": "video",
                "video": [_data_url(frame) for frame in sampled_frames],
            }
            if clip.framerate:
                payload["fps"] = max(0.1, clip.framerate / stride)
            return payload

        if extension in IMAGE_EXTENSIONS:
            return {"type": "image_url", "image_url": {"url": _data_url(asset_path)}}
        raise RuntimeError(f"Unsupported asset path for DashScope inference: {asset_path}")
