from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def prompt_fingerprint(
    *,
    prompt_id: str,
    version: int,
    prompt_text: str,
    critique: str = "",
    metrics: dict[str, float | None] | None = None,
    rendered_prompt: str | None = None,
) -> str:
    payload = {
        "prompt_id": prompt_id,
        "version": version,
        "prompt_text": prompt_text,
        "critique": critique,
        "metrics": metrics or {},
        "rendered_prompt": rendered_prompt,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _safe_name(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return collapsed.strip("-") or "prompt"


def write_prompt_artifact(
    *,
    output_root: str | Path,
    scope: str,
    variant: str,
    prompt_id: str,
    version: int,
    prompt_text: str,
    critique: str = "",
    metrics: dict[str, float | None] | None = None,
    rendered_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    fingerprint = prompt_fingerprint(
        prompt_id=prompt_id,
        version=version,
        prompt_text=prompt_text,
        critique=critique,
        metrics=metrics,
        rendered_prompt=rendered_prompt,
    )
    safe_prompt_id = _safe_name(prompt_id)
    prompt_root = Path(output_root) / "prompts" / scope / variant
    prompt_root.mkdir(parents=True, exist_ok=True)
    stem = f"v{version:02d}__{safe_prompt_id}__{fingerprint}"
    json_path = prompt_root / f"{stem}.json"
    text_path = prompt_root / f"{stem}.txt"
    rendered_path = prompt_root / f"{stem}.rendered.txt"

    payload = {
        "scope": scope,
        "variant": variant,
        "prompt_id": prompt_id,
        "version": version,
        "fingerprint": fingerprint,
        "prompt_text": prompt_text,
        "critique": critique,
        "metrics": metrics or {},
        "rendered_prompt": rendered_prompt,
        "metadata": metadata or {},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text(prompt_text.rstrip() + "\n", encoding="utf-8")
    if rendered_prompt is not None:
        rendered_path.write_text(rendered_prompt.rstrip() + "\n", encoding="utf-8")
    return {
        "fingerprint": fingerprint,
        "json_path": str(json_path),
        "text_path": str(text_path),
        "rendered_path": str(rendered_path) if rendered_prompt is not None else "",
    }
