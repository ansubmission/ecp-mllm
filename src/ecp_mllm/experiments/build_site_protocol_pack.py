from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import load_local_settings
from ..data.cfc_adapter import CFCAdapter
from ..eval.counting import count_tracks_like_cfc, read_mot_tracks
from ..eval.site_profiles import infer_clip_hints, resolve_site_profile
from ..types import ClipRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapt-size", type=int, default=10)
    parser.add_argument("--selection-size", type=int, default=8)
    parser.add_argument("--holdout-size", type=int, default=8)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "site-pack"


def _ground_truth_total(adapter: CFCAdapter, clip: ClipRecord) -> int | None:
    if clip.width is None or clip.height is None:
        return None
    gt_path = adapter.gt_path(clip.domain, clip.clip_id)
    if not gt_path.exists():
        return None
    counts = count_tracks_like_cfc(read_mot_tracks(gt_path), clip.width, clip.height, filter_dist=0.05)
    return counts.left + counts.right


def _count_bucket(total_count: int) -> str:
    if total_count <= 0:
        return "count0"
    if total_count == 1:
        return "count1"
    if total_count == 2:
        return "count2"
    if total_count <= 5:
        return "count3to5"
    return "count6plus"


def _clip_bucket(clip: ClipRecord, total_count: int) -> str:
    hints = infer_clip_hints(f"{clip.domain}/{clip.clip_id}")
    prefix = "far" if "far_view" in hints else "near"
    return f"{prefix}_{_count_bucket(total_count)}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['name']}",
        "",
        f"- domain: `{payload['domain']}`",
        f"- site profile: `{payload['site_profile_id']}`",
        "",
        "| Split | Target | Actual |",
        "|---|---:|---:|",
        f"| `adapt` | {payload['targets']['adapt']} | {payload['counts']['adapt']} |",
        f"| `selection` | {payload['targets']['selection']} | {payload['counts']['selection']} |",
        f"| `holdout` | {payload['targets']['holdout']} | {payload['counts']['holdout']} |",
        "",
        "| Bucket | Clips |",
        "|---|---:|",
    ]
    for bucket, count in sorted(payload["bucket_counts"].items()):
        lines.append(f"| `{bucket}` | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    settings = load_local_settings(args.settings)
    adapter = CFCAdapter(settings.paths)
    clips = adapter.load_clip_records([args.domain])
    items: list[dict[str, Any]] = []
    for clip in clips:
        total_count = _ground_truth_total(adapter, clip)
        if total_count is None:
            continue
        items.append(
            {
                "clip_id": clip.clip_id,
                "bucket": _clip_bucket(clip, total_count),
                "truth_total": total_count,
            }
        )
    if not items:
        raise RuntimeError(f"no labeled clips available for domain={args.domain}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda row: (row["bucket"], row["truth_total"], row["clip_id"])):
        grouped[str(item["bucket"])].append(item)
    original_bucket_counts = {bucket: len(entries) for bucket, entries in grouped.items()}

    targets = {
        "adapt": max(0, int(args.adapt_size)),
        "selection": max(0, int(args.selection_size)),
        "holdout": max(0, int(args.holdout_size)),
    }
    remaining = dict(targets)
    splits: dict[str, list[dict[str, str]]] = {"adapt": [], "selection": [], "holdout": []}
    order = ("adapt", "selection", "holdout")
    bucket_names = sorted(grouped.keys())
    cursor = 0
    while bucket_names and any(value > 0 for value in remaining.values()):
        bucket = bucket_names[cursor % len(bucket_names)]
        entries = grouped[bucket]
        if not entries:
            bucket_names.remove(bucket)
            continue
        assigned = False
        for split in order:
            if remaining[split] <= 0:
                continue
            item = entries.pop(0)
            splits[split].append({"bucket": bucket, "clip_id": str(item["clip_id"])})
            remaining[split] -= 1
            assigned = True
            break
        if not assigned:
            break
        if not entries:
            bucket_names.remove(bucket)
        cursor += 1

    outdir = Path("outputs") / "site_protocol_packs" / _safe_name(args.name)
    profile = resolve_site_profile(args.domain)
    _write_json(outdir / "adapt.json", splits["adapt"])
    _write_json(outdir / "selection.json", splits["selection"])
    _write_json(outdir / "holdout.json", splits["holdout"])
    summary = {
        "name": args.name,
        "domain": args.domain,
        "site_profile_id": profile.site_id,
        "targets": targets,
        "counts": {split: len(items) for split, items in splits.items()},
        "bucket_counts": original_bucket_counts,
        "unassigned_count": sum(len(entries) for entries in grouped.values()),
        "outputs": {
            "adapt": str((outdir / "adapt.json").resolve()),
            "selection": str((outdir / "selection.json").resolve()),
            "holdout": str((outdir / "holdout.json").resolve()),
        },
    }
    _write_json(outdir / "summary.json", summary)
    _write_markdown(outdir / "summary.md", summary)
    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
