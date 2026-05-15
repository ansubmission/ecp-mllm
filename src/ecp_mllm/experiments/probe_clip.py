from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace
from urllib import request

from ..config import load_local_settings
from ..data.cfc_adapter import CFCAdapter
from ..preprocess.sff3c import params_cache_token, process_frame_paths
from ..qwen.base import QwenClient
from ..qwen.gemini_api import GeminiApiClient
from ..qwen.prompt_archive import write_prompt_artifact
from ..qwen.dashscope_openai import DashScopeOpenAIClient, _completions_url, _resolve_api_key
from ..qwen.parsing import parse_passage_prediction
from ..qwen.prompting import build_prompt
from ..types import InputVariant, PromptRevision
from ..video import stitch_frames_to_mp4, stitched_probe_mp4_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--domain", default="kenai-val")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mode", choices=["count", "describe"], default="count")
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--transport", choices=["sampled_frames", "stitched_mp4"], default="sampled_frames")
    parser.add_argument("--stitch-fps", type=float, default=10.0)
    parser.add_argument("--stitch-height", type=int, default=0)
    parser.add_argument("--stitch-crf", type=int, default=30)
    return parser.parse_args()


def _load_clip(settings_path: str, domain: str, clip_id: str):
    settings = load_local_settings(settings_path)
    adapter = CFCAdapter(settings.paths)
    clips = adapter.load_clip_records([domain])
    matches = [clip for clip in clips if clip.clip_id == clip_id]
    if not matches:
        raise RuntimeError(f"Clip not found: {domain}/{clip_id}")
    return settings, matches[0]


def _resolve_effective_provider(configured_provider: str, requested_model: str | None) -> str:
    provider = str(configured_provider or "").strip().lower()
    if provider in {"mock", "scripted"} or requested_model is None:
        return provider
    model_name = requested_model.strip().lower()
    if provider in {"gemini", "gemini_api"} and model_name.startswith("qwen"):
        return "dashscope_openai"
    if provider in {"dashscope", "dashscope_openai"} and model_name.startswith("gemini"):
        return "gemini_api"
    return provider


def _runtime_provider_settings(settings, requested_model: str | None) -> tuple[str, str | None, str, str | None]:
    configured_provider = str(settings.qwen.provider or "").strip().lower()
    effective_provider = _resolve_effective_provider(configured_provider, requested_model)
    provider_switched = effective_provider != configured_provider

    if effective_provider in {"gemini", "gemini_api"}:
        return (
            effective_provider,
            None if provider_switched else settings.qwen.api_key,
            "GEMINI_API_KEY" if provider_switched else settings.qwen.api_key_env,
            None if provider_switched else settings.qwen.base_url,
        )

    if effective_provider in {"dashscope", "dashscope_openai"}:
        return (
            effective_provider,
            None if provider_switched else settings.qwen.api_key,
            "DASHSCOPE_API_KEY" if provider_switched else settings.qwen.api_key_env,
            None if provider_switched else settings.qwen.base_url,
        )

    return (
        effective_provider,
        settings.qwen.api_key,
        settings.qwen.api_key_env,
        settings.qwen.base_url,
    )


def _build_client(settings, model: str | None, max_frames: int | None) -> QwenClient:
    provider, api_key, api_key_env, base_url = _runtime_provider_settings(settings, model or settings.qwen.model)
    if provider in {"gemini", "gemini_api"}:
        return GeminiApiClient(
            model=model or settings.qwen.model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=settings.qwen.timeout_seconds,
            max_frames=max_frames or settings.qwen.max_frames,
            temperature=settings.qwen.temperature,
            thinking_level=settings.qwen.thinking_level,
        )
    return DashScopeOpenAIClient(
        model=model or settings.qwen.model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_seconds=settings.qwen.timeout_seconds,
        max_frames=max_frames or settings.qwen.max_frames,
        temperature=settings.qwen.temperature,
        prefer_temporary_oss_upload=settings.qwen.prefer_temporary_oss_upload,
    )


def _frame_paths_for_variant(clip, variant: InputVariant) -> list[Path]:
    frame_paths_by_variant = clip.metadata.get("frame_paths_by_variant", {})
    if isinstance(frame_paths_by_variant, dict):
        value = frame_paths_by_variant.get(variant.value)
        if isinstance(value, list):
            return [Path(item) for item in value]
    asset_path = clip.get_asset_path(variant)
    if asset_path and asset_path.is_dir():
        return sorted(asset_path.iterdir())
    return []


def _sample_uniform_paths(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    stride = max(1, (len(paths) + max_items - 1) // max_items)
    return paths[::stride][:max_items]


def _sample_uniform_indices(total_items: int, max_items: int) -> list[int]:
    if total_items <= max_items:
        return list(range(total_items))
    stride = max(1, (total_items + max_items - 1) // max_items)
    return list(range(0, total_items, stride))[:max_items]


def _prepare_sff3c_variant(settings, clip, transport: str, max_frames: int | None):
    if clip.get_asset_path(InputVariant.SFF3C) is not None and _frame_paths_for_variant(clip, InputVariant.SFF3C):
        return clip

    raw_frame_paths = _frame_paths_for_variant(clip, InputVariant.RAW)
    if not raw_frame_paths:
        raise RuntimeError(
            f"Clip {clip.key.value} does not have raw frame paths available, so SFF3C cannot be derived on demand."
        )

    selected_raw_paths = raw_frame_paths
    prepared_root = settings.paths.output_root / "prepared" / "sff3c" / clip.domain
    cache_token = params_cache_token(clip.domain)
    full_output_dir = prepared_root / f"{clip.clip_id}__full__{cache_token}"
    prepared_dir, full_prepared_paths = process_frame_paths(raw_frame_paths, full_output_dir, clip.domain)
    if transport == "sampled_frames":
        limit = max_frames or settings.qwen.max_frames
        selected_indices = _sample_uniform_indices(len(raw_frame_paths), limit)
        prepared_paths = [full_prepared_paths[index] for index in selected_indices]
    else:
        prepared_paths = full_prepared_paths
    asset_paths = dict(clip.asset_paths)
    asset_paths[InputVariant.SFF3C] = prepared_dir
    metadata = dict(clip.metadata)
    frame_paths_by_variant = metadata.get("frame_paths_by_variant", {})
    if isinstance(frame_paths_by_variant, dict):
        updated_frame_paths = dict(frame_paths_by_variant)
    else:
        updated_frame_paths = {}
    updated_frame_paths[InputVariant.SFF3C.value] = [str(path) for path in prepared_paths]
    metadata["frame_paths_by_variant"] = updated_frame_paths
    return replace(clip, asset_paths=asset_paths, metadata=metadata)


def _materialize_clip_transport(args: argparse.Namespace, settings, clip, variant: InputVariant):
    if args.transport != "stitched_mp4":
        return clip, None
    frame_paths = _frame_paths_for_variant(clip, variant)
    if not frame_paths:
        raise RuntimeError(f"No frame paths available for stitched_mp4 transport on {clip.domain}/{clip.clip_id}")
    output_path = stitched_probe_mp4_path(
        settings.paths.output_root,
        clip.key.value,
        variant.value,
        args.stitch_fps,
        args.stitch_height,
        args.stitch_crf,
    )
    video_path = stitch_frames_to_mp4(
        frame_paths=frame_paths,
        output_path=output_path,
        fps=args.stitch_fps,
        target_height=args.stitch_height if args.stitch_height > 0 else None,
        crf=args.stitch_crf,
    )
    asset_paths = dict(clip.asset_paths)
    asset_paths[variant] = video_path
    duration_seconds = len(frame_paths) / args.stitch_fps if args.stitch_fps > 0 else clip.duration_seconds
    metadata = dict(clip.metadata)
    metadata["stitched_transport"] = {
        "source_variant": variant.value,
        "video_path": str(video_path),
        "fps": args.stitch_fps,
        "frame_count": len(frame_paths),
        "resized_height": args.stitch_height if args.stitch_height > 0 else None,
    }
    stitched_clip = replace(
        clip,
        asset_paths=asset_paths,
        framerate=args.stitch_fps,
        duration_seconds=duration_seconds,
        metadata=metadata,
    )
    return stitched_clip, video_path


def _run_describe(client: QwenClient, clip, variant: InputVariant, prompt_text: str) -> str:
    if isinstance(client, GeminiApiClient):
        return client.infer_text_prompt(clip, variant, prompt_text).text
    if not isinstance(client, DashScopeOpenAIClient):
        raise RuntimeError(f"Unsupported describe-mode client: {type(client).__name__}")
    api_key = client.resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"API key is not configured. Set qwen.api_key or make {client.api_key_env} available "
            "in either the current process environment or the Windows user environment."
        )
    media = client._build_media_item(clip, variant, clip.get_asset_path(variant), api_key=api_key)
    payload = {
        "model": client.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    media,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "temperature": client.temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=_completions_url(client.base_url),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(
                {"X-DashScope-OssResourceResolve": "enable"}
                if isinstance(media, dict)
                and media.get("type") == "video_url"
                and isinstance(media.get("video_url"), dict)
                and str(media["video_url"].get("url", "")).startswith("oss://")
                else {}
            ),
        },
    )
    with request.urlopen(req, timeout=client.timeout_seconds) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return content if isinstance(content, str) else json.dumps(content)


def _archive_probe_prompt(
    settings,
    clip,
    variant: InputVariant,
    prompt: PromptRevision,
    model: str,
    mode: str,
    transport: str,
    rendered_prompt: str | None,
) -> dict[str, str]:
    return write_prompt_artifact(
        output_root=settings.paths.output_root,
        scope="probes",
        variant=variant.value,
        prompt_id=prompt.prompt_id,
        version=prompt.version,
        prompt_text=prompt.prompt_text,
        critique=prompt.critique,
        metrics=dict(prompt.metrics),
        rendered_prompt=rendered_prompt,
        metadata={
            "domain": clip.domain,
            "clip_id": clip.clip_id,
            "mode": mode,
            "model": model,
            "transport": transport,
        },
    )


def _safe_name(value: str) -> str:
    return (
        "".join(ch if ((ch.isascii() and ch.isalnum()) or ch in {"-", "_", "."}) else "-" for ch in value).strip("-")
        or "probe"
    )


def _response_base_path(
    settings,
    clip,
    variant: InputVariant,
    model: str,
    mode: str,
    transport: str,
) -> Path:
    model_safe = _safe_name(model)
    file_stem = _safe_name(f"{clip.key.value}__{variant.value}__{transport}__{mode}")
    output_dir = settings.paths.output_root / "probes" / "responses" / model_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / file_stem
    if not candidate.with_suffix(".json").exists() and not candidate.with_suffix(".txt").exists():
        return candidate
    index = 2
    while True:
        numbered = output_dir / f"{file_stem}__run{index:02d}"
        if not numbered.with_suffix(".json").exists() and not numbered.with_suffix(".txt").exists():
            return numbered
        index += 1


def main() -> int:
    args = _parse_args()
    settings, clip = _load_clip(args.settings, args.domain, args.clip_id)
    variant = InputVariant.from_value(args.variant)
    if variant == InputVariant.SFF3C:
        clip = _prepare_sff3c_variant(settings, clip, args.transport, args.max_frames)
    client = _build_client(settings, args.model, args.max_frames)
    effective_clip, video_path = _materialize_clip_transport(args, settings, clip, variant)

    if args.mode == "describe":
        prompt_text = args.prompt_text or (
            "Describe what is visible in these chronological frames. "
            "Mention whether they look like sonar imagery, whether fish-like moving objects are visible, "
            "and whether the scene is noisy or low contrast."
        )
        describe_prompt = PromptRevision(version=0, prompt_id="describe", prompt_text=prompt_text)
        prompt_artifact = _archive_probe_prompt(
            settings,
            effective_clip,
            variant,
            describe_prompt,
            client.model,
            args.mode,
            args.transport,
            prompt_text,
        )
        response_text = _run_describe(client, effective_clip, variant, prompt_text)
        response_base = _response_base_path(
            settings,
            effective_clip,
            variant,
            client.model,
            args.mode,
            args.transport,
        )
        response_txt_path = response_base.with_suffix(".txt")
        response_json_path = response_base.with_suffix(".json")
        response_txt_path.write_text(response_text.rstrip() + "\n", encoding="utf-8")
        response_json_path.write_text(
            json.dumps(
                {
                    "domain": effective_clip.domain,
                    "clip_id": effective_clip.clip_id,
                    "variant": variant.value,
                    "transport": args.transport,
                    "mode": args.mode,
                    "model": client.model,
                    "video_path": str(video_path) if video_path is not None else None,
                    "prompt_path": prompt_artifact["json_path"],
                    "response_text": response_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if video_path is not None:
            print(f"[stitched_mp4] {video_path}")
        print(f"[prompt] {prompt_artifact['json_path']}")
        print(f"[response] {response_json_path}")
        print(response_text)
        return 0

    prompt_text = args.prompt_text or (
        "Analyze this sonar clip for fish passage. "
        "Identify each candidate passage episode, estimate how many distinct fish traverse during that episode, "
        "and then sum across episodes. "
        "Do not anchor only on the clearest 2 to 3 simultaneous tracks if the same school continues streaming "
        "through the corridor for many seconds. "
        "If a long stream contains multiple waves or mini-bursts over time, split them into separate passage episodes instead of one broad school summary. "
        "Use best-effort throughput estimates for high-passage situations, and estimate total distinct fish passage, not instantaneous occupancy. "
        "Return JSON only."
    )
    prompt = PromptRevision(version=0, prompt_id="probe", prompt_text=prompt_text)
    rendered_prompt = build_prompt(effective_clip, variant, prompt)
    prompt_artifact = _archive_probe_prompt(
        settings,
        effective_clip,
        variant,
        prompt,
        client.model,
        args.mode,
        args.transport,
        rendered_prompt,
    )
    response = client.infer(effective_clip, variant, prompt)
    prediction = parse_passage_prediction(
        raw_text=response.text,
        domain=effective_clip.domain,
        clip_id=effective_clip.clip_id,
        prompt_id=prompt.prompt_id,
        latency_sec=response.latency_sec,
        usage_metadata=response.usage_metadata,
        estimated_cost_usd=response.estimated_cost_usd,
        model_name=client.model,
    )
    response_base = _response_base_path(
        settings,
        effective_clip,
        variant,
        client.model,
        args.mode,
        args.transport,
    )
    response_txt_path = response_base.with_suffix(".txt")
    response_json_path = response_base.with_suffix(".json")
    summary_payload = {
        "parse_success": prediction.parse_success,
        "left_count": prediction.left_count,
        "right_count": prediction.right_count,
        "commentary": prediction.commentary,
        "evidence_summary": prediction.evidence_summary,
        "latency_sec": prediction.latency_sec,
        "estimated_cost_usd": prediction.estimated_cost_usd,
        "usage_metadata": prediction.usage_metadata,
    }
    response_txt_path.write_text(response.text.rstrip() + "\n", encoding="utf-8")
    response_json_path.write_text(
        json.dumps(
            {
                "domain": effective_clip.domain,
                "clip_id": effective_clip.clip_id,
                "variant": variant.value,
                "transport": args.transport,
                "mode": args.mode,
                "model": client.model,
                "video_path": str(video_path) if video_path is not None else None,
                "prompt_path": prompt_artifact["json_path"],
                "response_text": response.text,
                "summary": summary_payload,
                "usage_metadata": response.usage_metadata,
                "estimated_cost_usd": response.estimated_cost_usd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if video_path is not None:
        print(f"[stitched_mp4] {video_path}")
    print(f"[prompt] {prompt_artifact['json_path']}")
    print(f"[response] {response_json_path}")
    print(response.text)
    print()
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
