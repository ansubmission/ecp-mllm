from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ..types import ClipRecord, InputVariant, PromptRevision


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    latency_sec: float
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None


class QwenClient(ABC):
    @abstractmethod
    def infer(self, clip: ClipRecord, variant: InputVariant, prompt: PromptRevision) -> InferenceResponse:
        raise NotImplementedError


class MockQwenClient(QwenClient):
    def infer(self, clip: ClipRecord, variant: InputVariant, prompt: PromptRevision) -> InferenceResponse:
        start = time.perf_counter()
        digest = hashlib.sha256(f"{clip.key.value}|{variant.value}|{prompt.prompt_id}".encode("utf-8")).hexdigest()
        left = int(digest[:2], 16) % 3
        right = int(digest[2:4], 16) % 3
        if variant == InputVariant.SFF3C:
            right += 1
        if prompt.version > 0:
            left = max(0, left - 1)
        payload = {"left_count": left, "right_count": right, "confidence": 0.5, "events": []}
        return InferenceResponse(text=json.dumps(payload), latency_sec=time.perf_counter() - start)


class ScriptedQwenClient(QwenClient):
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses

    @classmethod
    def from_path(cls, path: str | Path) -> "ScriptedQwenClient":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def infer(self, clip: ClipRecord, variant: InputVariant, prompt: PromptRevision) -> InferenceResponse:
        start = time.perf_counter()
        clip_entry = self.responses.get(clip.key.value) or {}
        variant_entry = clip_entry.get(variant.value, {}) if isinstance(clip_entry, dict) else {}
        response = None
        if isinstance(variant_entry, dict):
            response = variant_entry.get(prompt.prompt_id)
            if response is None:
                response = variant_entry.get("default")
        if response is None:
            raise KeyError(f"No scripted response for {clip.key.value} / {variant.value} / {prompt.prompt_id}")
        text = response if isinstance(response, str) else json.dumps(response)
        return InferenceResponse(text=text, latency_sec=time.perf_counter() - start)
