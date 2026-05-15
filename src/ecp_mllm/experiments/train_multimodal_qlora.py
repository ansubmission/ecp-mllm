from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return payload


def _torch_dtype_from_name(value: str | None) -> torch.dtype | None:
    if not value:
        return None
    normalized = value.strip().lower()
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
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def _assistant_text(example: dict[str, Any]) -> str:
    if isinstance(example.get("assistant_text"), str) and example["assistant_text"].strip():
        return example["assistant_text"].strip()
    target_json = example.get("target_json")
    if target_json is None:
        raise ValueError("Example is missing assistant_text and target_json")
    return json.dumps(target_json, ensure_ascii=False)


def _user_text(example: dict[str, Any]) -> str:
    training_prompt = example.get("training_prompt")
    if isinstance(training_prompt, str) and training_prompt.strip():
        return training_prompt.strip()
    prompt_text = example.get("prompt_text")
    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()
    rendered_prompt = example.get("rendered_prompt")
    if isinstance(rendered_prompt, str) and rendered_prompt.strip():
        return rendered_prompt.strip()
    raise ValueError("Example is missing training_prompt/prompt_text/rendered_prompt")


def _video_path(example: dict[str, Any]) -> str:
    video_path = example.get("overlay_video_path") or example.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("Example is missing overlay_video_path/video_path")
    resolved = Path(video_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Video path does not exist: {resolved}")
    return str(resolved)


def _video_fps(example: dict[str, Any], default_fps: float | None = None) -> float | None:
    raw = example.get("video_fps", default_fps)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _build_messages(
    example: dict[str, Any],
    *,
    default_video_fps: float | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_item: dict[str, Any] = {"type": "video", "video": _video_path(example)}
    fps = _video_fps(example, default_fps=default_video_fps)
    if fps is not None:
        video_item["fps"] = fps
    if min_pixels is not None:
        video_item["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        video_item["max_pixels"] = int(max_pixels)
    user_content: list[dict[str, Any]] = [video_item]
    user_content.append({"type": "text", "text": _user_text(example)})
    user_message = {"role": "user", "content": user_content}
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": _assistant_text(example)}],
    }
    return [user_message], [user_message, assistant_message]


class MultimodalSFTCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        max_seq_length: int | None = None,
        *,
        default_video_fps: float | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.default_video_fps = default_video_fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_messages: list[list[dict[str, Any]]] = []
        full_messages: list[list[dict[str, Any]]] = []
        for feature in features:
            prompt_only, full = _build_messages(
                feature,
                default_video_fps=self.default_video_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            prompt_messages.append(prompt_only)
            full_messages.append(full)

        prompt_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages
        ]
        full_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]

        images, videos, video_kwargs = process_vision_info(
            full_messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        processor_kwargs: dict[str, Any] = {
            "padding": True,
            "return_tensors": "pt",
        }
        if videos is not None and videos and isinstance(videos[0], tuple):
            video_tensors = []
            video_metadata = []
            for item in videos:
                video_tensor, metadata = item
                video_tensors.append(video_tensor)
                video_metadata.append(metadata)
            videos = video_tensors
            processor_kwargs["video_metadata"] = video_metadata
        if video_kwargs:
            fps_value = video_kwargs.get("fps")
            if isinstance(fps_value, list):
                unique_fps = {float(value) for value in fps_value}
                if len(unique_fps) == 1:
                    video_kwargs["fps"] = float(next(iter(unique_fps)))
            processor_kwargs.update(video_kwargs)
        if self.max_seq_length is not None and self.max_seq_length > 0:
            processor_kwargs["truncation"] = True
            processor_kwargs["max_length"] = self.max_seq_length

        batch = self.processor(
            text=full_texts,
            images=images,
            videos=videos,
            **processor_kwargs,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            videos=videos,
            **processor_kwargs,
        )

        labels = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        labels[attention_mask == 0] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100

        if self.pad_token_id is not None:
            labels[batch["input_ids"] == self.pad_token_id] = -100

        valid_counts = (labels != -100).sum(dim=1)
        if int(valid_counts.min().item()) <= 0:
            counts = [int(value) for value in valid_counts.tolist()]
            raise ValueError(
                "All assistant tokens were truncated or masked for at least one sample. "
                f"valid_label_counts={counts}. Increase max_seq_length or shorten the training prompt."
            )

        batch["labels"] = labels
        return batch


def _load_datasets(config: dict[str, Any], args: argparse.Namespace):
    data_files: dict[str, str] = {"train": str(config["dataset_path"])}
    val_dataset_path = config.get("val_dataset_path")
    if val_dataset_path:
        val_path = Path(str(val_dataset_path)).expanduser()
        if val_path.exists():
            data_files["validation"] = str(val_path)
    dataset_dict = load_dataset("json", data_files=data_files)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation")
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    return train_dataset, eval_dataset


def _build_model(config: dict[str, Any]):
    quantization_config = None
    if bool(config.get("load_in_4bit", False)):
        compute_dtype = _torch_dtype_from_name(str(config.get("bnb_4bit_compute_dtype") or config.get("torch_dtype") or "bfloat16"))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type") or "nf4"),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype or torch.bfloat16,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    torch_dtype = _torch_dtype_from_name(config.get("torch_dtype"))
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = AutoModelForImageTextToText.from_pretrained(
        str(config["model_name_or_path"]),
        **model_kwargs,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = config.get("lora") or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg.get("target_modules") or []),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _build_training_args(config: dict[str, Any], args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    training_cfg = config.get("training") or {}
    output_dir = str(config["output_dir"])
    eval_strategy = "steps" if has_eval else "no"
    if args.smoke:
        eval_strategy = "no"
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", training_cfg.get("per_device_train_batch_size", 1))),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", 2 if args.smoke else -1)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_steps=int(training_cfg.get("save_steps", 200)),
        eval_steps=int(training_cfg.get("eval_steps", 200)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        bf16=bool(training_cfg.get("bf16", True)),
        fp16=bool(training_cfg.get("fp16", False)),
        remove_unused_columns=False,
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        report_to=[],
        optim=str(training_cfg.get("optim", "paged_adamw_8bit")),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=eval_strategy,
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=False,
        seed=int(training_cfg.get("seed", 42)),
        data_seed=int(training_cfg.get("data_seed", training_cfg.get("seed", 42))),
    )


def main() -> int:
    args = _parse_args()
    config = _read_yaml(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset = _load_datasets(config, args)
    processor = AutoProcessor.from_pretrained(str(config["model_name_or_path"]), trust_remote_code=True)
    collator = MultimodalSFTCollator(
        processor=processor,
        max_seq_length=int((config.get("training") or {}).get("max_seq_length", 4096)),
        default_video_fps=(float(config["default_video_fps"]) if config.get("default_video_fps") is not None else None),
        min_pixels=(int(config["min_pixels"]) if config.get("min_pixels") is not None else None),
        max_pixels=(int(config["max_pixels"]) if config.get("max_pixels") is not None else None),
    )
    model = _build_model(config)
    training_args = _build_training_args(config, args, has_eval=eval_dataset is not None and len(eval_dataset) > 0)

    summary = {
        "model_name_or_path": str(config["model_name_or_path"]),
        "dataset_path": str(config["dataset_path"]),
        "val_dataset_path": str(config.get("val_dataset_path") or ""),
        "output_dir": str(output_dir),
        "train_examples": len(train_dataset),
        "eval_examples": 0 if eval_dataset is None else len(eval_dataset),
        "smoke": args.smoke,
    }
    (output_dir / "train_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    metrics["train_examples"] = len(train_dataset)
    metrics["eval_examples"] = 0 if eval_dataset is None else len(eval_dataset)
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
