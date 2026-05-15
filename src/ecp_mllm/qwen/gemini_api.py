from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import time
from typing import Any, Callable
from urllib import error, request

from ..types import ClipRecord, InputVariant, PromptRevision
from .base import InferenceResponse, QwenClient
from .dashscope_openai import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _metadata_frame_paths,
    _resolve_api_key,
    _sample_uniform,
    _sorted_frame_paths,
)
from .prompting import build_prompt


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MAX_FRAMES = 8
DEFAULT_MAX_INLINE_FILE_BYTES = 20 * 1024 * 1024
VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
ProgressCallback = Callable[[str, dict[str, object]], None]
PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0, "input_long": 4.0, "output_long": 18.0},
    "gemini-3-flash-preview": {"input": 0.5, "output": 3.0},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.5},
}


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _inline_blob(path: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "mime_type": _guess_mime_type(path),
        "data": encoded,
    }


def _generate_content_url(base_url: str, model: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith(":generateContent"):
        return normalized
    model_name = model if model.startswith("models/") else f"models/{model}"
    return f"{normalized}/{model_name}:generateContent"


def _emit_progress(callback: ProgressCallback | None, stage: str, **details: object) -> None:
    if callback is None:
        return
    callback(stage, details)


def _coerce_response_text(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Gemini response did not include candidates: {payload}")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError(f"Gemini response candidate is not an object: {payload}")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError(f"Gemini response candidate did not include content: {payload}")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError(f"Gemini response content did not include parts: {payload}")
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("text") is not None:
            text_parts.append(str(part["text"]))
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError(f"Gemini response did not include text parts: {payload}")


def _extract_usage_metadata(payload: dict[str, object]) -> dict[str, Any]:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return {}
    return dict(usage)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def estimate_gemini_cost_usd(model: str, usage_metadata: dict[str, Any]) -> float | None:
    pricing = PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None or not usage_metadata:
        return None
    prompt_tokens = _coerce_int(usage_metadata.get("promptTokenCount"))
    candidate_tokens = _coerce_int(usage_metadata.get("candidatesTokenCount"))
    thought_tokens = _coerce_int(usage_metadata.get("thoughtsTokenCount"))
    total_tokens = _coerce_int(usage_metadata.get("totalTokenCount"))
    output_tokens = candidate_tokens + thought_tokens
    if output_tokens <= 0 and total_tokens > prompt_tokens:
        output_tokens = total_tokens - prompt_tokens
    input_rate = pricing["input"]
    output_rate = pricing["output"]
    if model == "gemini-3.1-pro-preview" and prompt_tokens > 200_000:
        input_rate = pricing["input_long"]
        output_rate = pricing["output_long"]
    input_cost = prompt_tokens / 1_000_000 * input_rate
    output_cost = output_tokens / 1_000_000 * output_rate
    return input_cost + output_cost


class GeminiApiClient(QwenClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str | None = None,
        timeout_seconds: int = 120,
        max_frames: int = DEFAULT_MAX_FRAMES,
        temperature: float = 0.1,
        thinking_level: str | None = None,
        max_inline_file_bytes: int = DEFAULT_MAX_INLINE_FILE_BYTES,
    ) -> None:
        normalized_thinking_level = None
        if thinking_level is not None and thinking_level.strip():
            normalized_thinking_level = thinking_level.strip().lower()
            if normalized_thinking_level not in VALID_THINKING_LEVELS:
                raise ValueError(
                    f"Unsupported Gemini thinking level: {thinking_level}. "
                    f"Expected one of: {', '.join(sorted(VALID_THINKING_LEVELS))}."
                )
        self.model = model
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.api_key_env = api_key_env
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_frames = max_frames
        self.temperature = temperature
        self.thinking_level = normalized_thinking_level
        self.max_inline_file_bytes = max_inline_file_bytes
        self.progress_callback: ProgressCallback | None = None

    def resolve_api_key(self) -> str | None:
        return self.api_key or _resolve_api_key(self.api_key_env)

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self.progress_callback = callback

    def prepare_media_item(self, clip: ClipRecord, variant: InputVariant) -> dict[str, object]:
        asset_path = clip.get_asset_path(variant)
        if asset_path is None:
            raise ValueError(f"Clip {clip.key.value} is missing asset path for {variant.value}")
        parts = self._build_media_parts(clip, variant, asset_path)
        if len(parts) == 1:
            return parts[0]
        return {"type": "multi_part", "parts": parts}

    def infer(self, clip: ClipRecord, variant: InputVariant, prompt: PromptRevision) -> InferenceResponse:
        return self.infer_text_prompt(clip, variant, build_prompt(clip, variant, prompt))

    def infer_text_prompt(self, clip: ClipRecord, variant: InputVariant, prompt_text: str) -> InferenceResponse:
        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"API key is not configured. Set qwen.api_key or make {self.api_key_env} available "
                "in either the current process environment or the Windows user environment."
            )

        asset_path = clip.get_asset_path(variant)
        if asset_path is None:
            raise ValueError(f"Clip {clip.key.value} is missing asset path for {variant.value}")

        start = time.perf_counter()
        _emit_progress(
            self.progress_callback,
            "infer_start",
            clip_key=clip.key.value,
            variant=variant.value,
            model=self.model,
            asset_path=str(asset_path),
        )
        payload = self._build_request_payload(clip, variant, prompt_text)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=_generate_content_url(self.base_url, self.model),
            data=body,
            method="POST",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc
        text = _coerce_response_text(response_payload)
        usage_metadata = _extract_usage_metadata(response_payload)
        estimated_cost_usd = estimate_gemini_cost_usd(self.model, usage_metadata)
        latency_sec = time.perf_counter() - start
        _emit_progress(
            self.progress_callback,
            "infer_done",
            clip_key=clip.key.value,
            variant=variant.value,
            model=self.model,
            latency_sec=latency_sec,
            prompt_tokens=_coerce_int(usage_metadata.get("promptTokenCount")),
            candidate_tokens=_coerce_int(usage_metadata.get("candidatesTokenCount")),
            thought_tokens=_coerce_int(usage_metadata.get("thoughtsTokenCount")),
            total_tokens=_coerce_int(usage_metadata.get("totalTokenCount")),
            estimated_cost_usd=estimated_cost_usd,
        )
        return InferenceResponse(
            text=text,
            latency_sec=latency_sec,
            usage_metadata=usage_metadata,
            estimated_cost_usd=estimated_cost_usd,
        )

    def _build_request_payload(self, clip: ClipRecord, variant: InputVariant, prompt_text: str) -> dict[str, object]:
        asset_path = clip.get_asset_path(variant)
        if asset_path is None:
            raise ValueError(f"Clip {clip.key.value} is missing asset path for {variant.value}")
        parts = self._build_media_parts(clip, variant, asset_path)
        parts.append({"text": prompt_text})
        payload: dict[str, object] = {
            "contents": [
                {
                    "parts": parts,
                }
            ]
        }
        generation_config: dict[str, object] = {"temperature": self.temperature}
        if self.thinking_level is not None:
            generation_config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        payload["generationConfig"] = generation_config
        return payload

    def _build_media_parts(self, clip: ClipRecord, variant: InputVariant, asset_path: Path) -> list[dict[str, object]]:
        extension = asset_path.suffix.lower()
        if extension in VIDEO_EXTENSIONS and asset_path.is_file():
            size_bytes = asset_path.stat().st_size
            _emit_progress(
                self.progress_callback,
                "media_prepare_video",
                clip_key=clip.key.value,
                variant=variant.value,
                asset_path=str(asset_path),
                size_bytes=size_bytes,
            )
            if size_bytes > self.max_inline_file_bytes:
                raise RuntimeError(
                    f"Video file exceeds the current Gemini inline limit ({size_bytes} bytes): {asset_path}. "
                    "Reduce stitch size/quality or add native Files API upload support."
                )
            part: dict[str, object] = {
                "inline_data": _inline_blob(asset_path),
            }
            if clip.framerate:
                part["video_metadata"] = {"fps": clip.framerate}
            return [part]

        frame_paths = _metadata_frame_paths(clip, variant)
        if not frame_paths and asset_path.is_dir():
            frame_paths = _sorted_frame_paths(asset_path)
        if frame_paths:
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
            return [{"inline_data": _inline_blob(frame)} for frame in sampled_frames]

        if extension in IMAGE_EXTENSIONS and asset_path.is_file():
            return [{"inline_data": _inline_blob(asset_path)}]
        raise RuntimeError(f"Unsupported asset path for Gemini inference: {asset_path}")
