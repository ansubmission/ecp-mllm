from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
import json
import mimetypes
from pathlib import Path
import shutil
import tempfile
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from peft import PeftModel
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
import uvicorn
from qwen_vl_utils import process_vision_info


def _guess_suffix_from_data_url(url: str, fallback: str) -> str:
    if not url.startswith("data:"):
        return fallback
    header = url.split(",", 1)[0]
    mime_type = header[5:].split(";", 1)[0]
    guessed = mimetypes.guess_extension(mime_type)
    return guessed or fallback


def _write_data_url_to_temp(url: str, temp_dir: Path, fallback_suffix: str) -> Path:
    try:
        _, encoded = url.split("base64,", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Only base64 data URLs are supported.") from exc
    suffix = _guess_suffix_from_data_url(url, fallback_suffix)
    payload = base64.b64decode(encoded)
    target = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(payload)
    return target


def _coerce_media_url(raw: Any) -> str:
    if isinstance(raw, dict):
        value = raw.get("url")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    raise HTTPException(status_code=400, detail="Media item is missing a usable URL.")


def _normalize_video_value(raw: Any, temp_dir: Path) -> str | list[str]:
    if isinstance(raw, list):
        normalized_frames: list[str] = []
        for frame in raw:
            if not isinstance(frame, str) or not frame.strip():
                raise HTTPException(status_code=400, detail="Video frame entries must be non-empty strings.")
            normalized_frames.append(frame.strip())
        return normalized_frames
    url = _coerce_media_url(raw)
    if url.startswith("data:video"):
        return str(_write_data_url_to_temp(url, temp_dir, ".mp4"))
    return url


def _normalize_message_content(messages: list[dict[str, Any]], temp_dir: Path) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        normalized_items: list[dict[str, Any]] = []
        if isinstance(content, str):
            normalized_items.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).strip()
                if item_type == "text":
                    normalized_items.append({"type": "text", "text": str(item.get("text", ""))})
                    continue
                if item_type in {"image", "image_url"}:
                    image_value = item.get("image") if item_type == "image" else item.get("image_url")
                    normalized_item: dict[str, Any] = {
                        "type": "image",
                        "image": _coerce_media_url(image_value),
                    }
                    for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                        if key in item:
                            normalized_item[key] = item[key]
                    normalized_items.append(normalized_item)
                    continue
                if item_type in {"video", "video_url"}:
                    video_value = item.get("video") if item_type == "video" else item.get("video_url")
                    normalized_item = {
                        "type": "video",
                        "video": _normalize_video_value(video_value, temp_dir),
                    }
                    for key in (
                        "fps",
                        "sample_fps",
                        "min_pixels",
                        "max_pixels",
                        "resized_height",
                        "resized_width",
                        "total_pixels",
                        "raw_fps",
                    ):
                        if key in item:
                            normalized_item[key] = item[key]
                    normalized_items.append(normalized_item)
                    continue
        normalized_messages.append({"role": role, "content": normalized_items})
    return normalized_messages


def _decode_assistant_text(processor: AutoProcessor, prompt_inputs: Any, generated_ids: torch.Tensor) -> str:
    input_ids = prompt_inputs["input_ids"]
    trimmed_ids = []
    for prompt_ids, output_ids in zip(input_ids, generated_ids):
        trimmed_ids.append(output_ids[len(prompt_ids) :])
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not decoded:
        return ""
    text = decoded[0].strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def _assistant_prefill_text(normalized_messages: list[dict[str, Any]]) -> str:
    if not normalized_messages:
        return ""
    final_message = normalized_messages[-1]
    if str(final_message.get("role", "")) != "assistant":
        return ""
    parts: list[str] = []
    for item in final_message.get("content", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")) != "text":
            continue
        text = str(item.get("text", ""))
        if text:
            parts.append(text)
    return "".join(parts)


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype | None:
    normalized = dtype_name.strip().lower()
    if normalized in {"auto", ""}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def _summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        item_summary: list[dict[str, Any]] = []
        for item in message.get("content", []):
            item_type = str(item.get("type", "unknown"))
            entry: dict[str, Any] = {"type": item_type}
            if item_type == "text":
                entry["text_chars"] = len(str(item.get("text", "")))
            elif item_type == "image":
                entry["source_kind"] = "data_url" if str(item.get("image", "")).startswith("data:") else "url_or_path"
            elif item_type == "video":
                video_value = item.get("video")
                if isinstance(video_value, list):
                    entry["source_kind"] = "frame_list"
                    entry["frame_count"] = len(video_value)
                else:
                    entry["source_kind"] = "data_url" if str(video_value).startswith("data:") else "url_or_path"
                for key in ("fps", "sample_fps", "min_pixels", "max_pixels", "resized_height", "resized_width"):
                    if key in item:
                        entry[key] = item[key]
            item_summary.append(entry)
        summary.append({"role": message.get("role", "user"), "items": item_summary})
    return summary


def _summarize_vision_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return {
            "container": "list",
            "length": len(payload),
            "sample_type": type(payload[0]).__name__ if payload else None,
        }
    if isinstance(payload, tuple):
        return {
            "container": "tuple",
            "length": len(payload),
            "sample_type": type(payload[0]).__name__ if payload else None,
        }
    if hasattr(payload, "shape"):
        return {
            "container": type(payload).__name__,
            "shape": list(payload.shape),
        }
    return {"container": type(payload).__name__}


def _summarize_processor_inputs(prompt_inputs: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if hasattr(value, "shape"):
            summary[key] = {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", "unknown"))}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


class QwenLocalServer:
    def __init__(
        self,
        *,
        model_id: str,
        served_model_name: str,
        max_new_tokens: int,
        default_temperature: float,
        quantization: str,
        torch_dtype_name: str,
        debug_jsonl: str | None,
    ) -> None:
        self.model_id = model_id
        self.served_model_name = served_model_name
        self.max_new_tokens = max_new_tokens
        self.default_temperature = default_temperature
        self.quantization = quantization
        self.torch_dtype_name = torch_dtype_name
        self.debug_jsonl = Path(debug_jsonl).expanduser() if debug_jsonl else None
        self.processor: AutoProcessor | None = None
        self.model: Qwen3_5ForConditionalGeneration | None = None
        self.device: torch.device | None = None
        self.base_model_id: str | None = None
        self.adapter_path: str | None = None

    def _append_debug(self, record: dict[str, Any]) -> None:
        if self.debug_jsonl is None:
            return
        self.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        resolved_model_id = Path(self.model_id).expanduser()
        adapter_config_path = resolved_model_id / "adapter_config.json"
        processor_model_id = self.model_id
        base_model_id = self.model_id
        adapter_path: str | None = None
        if resolved_model_id.is_dir() and adapter_config_path.exists():
            adapter_payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_id = str(adapter_payload.get("base_model_name_or_path") or "").strip()
            if not base_model_id:
                raise ValueError(f"Adapter config missing base_model_name_or_path: {adapter_config_path}")
            processor_model_id = base_model_id
            adapter_path = str(resolved_model_id)

        print(
            f"[local_qwen] loading model={self.model_id} served_name={self.served_model_name} "
            f"quantization={self.quantization} torch_dtype={self.torch_dtype_name}",
            flush=True,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "none":
            resolved_dtype = _torch_dtype_from_name(self.torch_dtype_name)
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype
        else:
            raise ValueError(f"Unsupported quantization mode: {self.quantization}")

        self.processor = AutoProcessor.from_pretrained(processor_model_id, trust_remote_code=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(base_model_id, **model_kwargs)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        print(f"[local_qwen] model loaded on device={self.device}", flush=True)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.processor is None or self.device is None:
            raise RuntimeError("Model server is not loaded yet.")

        request_id = uuid.uuid4().hex
        debug_record: dict[str, Any] = {
            "request_id": request_id,
            "model": self.served_model_name,
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "adapter_path": self.adapter_path,
            "quantization": self.quantization,
            "torch_dtype": self.torch_dtype_name,
            "created_unix": int(time.time()),
        }

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise HTTPException(status_code=400, detail="Request must include a non-empty messages list.")

        temperature = float(payload.get("temperature", self.default_temperature))
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or self.max_new_tokens)
        do_sample = temperature > 0.05
        debug_record["request"] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "message_count": len(raw_messages),
        }

        temp_dir = Path(tempfile.mkdtemp(prefix="qwen_local_req_"))
        try:
            normalized_messages = _normalize_message_content(raw_messages, temp_dir)
            debug_record["messages"] = _summarize_messages(normalized_messages)

            continue_final_message = bool(normalized_messages) and normalized_messages[-1].get("role") == "assistant"
            prompt_text = self.processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                enable_thinking=False,
            )
            assistant_prefill = _assistant_prefill_text(normalized_messages) if continue_final_message else ""
            debug_record["prompt_text_chars"] = len(prompt_text)
            debug_record["continue_final_message"] = continue_final_message
            if assistant_prefill:
                debug_record["assistant_prefill_chars"] = len(assistant_prefill)

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                normalized_messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            debug_record["vision_inputs"] = {
                "images": _summarize_vision_payload(image_inputs),
                "videos": _summarize_vision_payload(video_inputs),
                "video_kwargs": video_kwargs,
            }

            processor_kwargs: dict[str, Any] = {
                "text": [prompt_text],
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                if isinstance(video_inputs, list) and video_inputs and isinstance(video_inputs[0], tuple):
                    video_tensors = []
                    video_metadata = []
                    for item in video_inputs:
                        video_tensor, metadata = item
                        video_tensors.append(video_tensor)
                        video_metadata.append(metadata)
                    processor_kwargs["videos"] = video_tensors
                    processor_kwargs["video_metadata"] = video_metadata
                else:
                    processor_kwargs["videos"] = video_inputs
            if video_kwargs:
                fps_value = video_kwargs.get("fps")
                if isinstance(fps_value, list):
                    unique_fps = {float(value) for value in fps_value}
                    if len(unique_fps) == 1:
                        video_kwargs["fps"] = float(next(iter(unique_fps)))
                processor_kwargs.update(video_kwargs)

            prompt_inputs = self.processor(**processor_kwargs)
            debug_record["processor_inputs"] = _summarize_processor_inputs(prompt_inputs)
            prompt_inputs = prompt_inputs.to(self.device)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
            }
            if do_sample:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = max(temperature, 0.1)
                generation_kwargs["top_p"] = 0.95
            else:
                generation_kwargs["do_sample"] = False

            start = time.perf_counter()
            with torch.inference_mode():
                generated_ids = self.model.generate(**prompt_inputs, **generation_kwargs)
            text = _decode_assistant_text(self.processor, prompt_inputs, generated_ids)
            if assistant_prefill:
                text = f"{assistant_prefill}{text}"
            elapsed = time.perf_counter() - start

            prompt_tokens = int(prompt_inputs["input_ids"].shape[-1])
            completion_tokens = int(generated_ids.shape[-1] - prompt_tokens)
            finish_reason = "length" if completion_tokens >= max_tokens else "stop"
            debug_record["generation"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "elapsed_sec": round(elapsed, 3),
                "response_preview": text[:200],
            }
            self._append_debug(debug_record)

            return {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "local_debug": {
                    "request_id": request_id,
                    "elapsed_sec": round(elapsed, 3),
                    "quantization": self.quantization,
                    "torch_dtype": self.torch_dtype_name,
                    "debug_jsonl": str(self.debug_jsonl) if self.debug_jsonl else None,
                },
            }
        except Exception as exc:
            debug_record["error"] = repr(exc)
            self._append_debug(debug_record)
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_app(server: QwenLocalServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        server.load()
        yield

    app = FastAPI(title="Local Qwen OpenAI Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "model": server.served_model_name,
            "model_id": server.model_id,
            "base_model_id": server.base_model_id,
            "adapter_path": server.adapter_path,
            "device": str(server.device) if server.device is not None else None,
            "quantization": server.quantization,
            "torch_dtype": server.torch_dtype_name,
            "debug_jsonl": str(server.debug_jsonl) if server.debug_jsonl else None,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object payload required.")
        requested_model = payload.get("model")
        if requested_model and str(requested_model) != server.served_model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model {requested_model!r} does not match served model {server.served_model_name!r}.",
            )
        return server.infer(payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--served-model-name", default="qwen3.5-4b-local")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--debug-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        QwenLocalServer(
            model_id=args.model,
            served_model_name=args.served_model_name,
            max_new_tokens=args.max_new_tokens,
            default_temperature=args.temperature,
            quantization=args.quantization,
            torch_dtype_name=args.torch_dtype,
            debug_jsonl=args.debug_jsonl or None,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
