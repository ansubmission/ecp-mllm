from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfc-analysis", default="outputs/analysis/cfc_single_event_bias_v1/clip_rows.csv")
    parser.add_argument(
        "--thermal-assignments",
        default="outputs/analysis/thermal_inventory32_difficulty_v1/clip_assignments.csv",
    )
    parser.add_argument("--cfc-output", default="config/prompt_dev/cfc_capability_probe8_v1.json")
    parser.add_argument("--thermal-output", default="config/prompt_dev/thermal_capability_probe8_v1.json")
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_clip_key(clip_key: str) -> tuple[str, str]:
    domain, clip_id = clip_key.split("/", 1)
    return domain, clip_id


def build_cfc_probe(rows: list[dict[str, str]]) -> list[dict]:
    single_salient = sorted(
        (row for row in rows if row.get("gt_total") == "1" and row.get("exact") == "1"),
        key=lambda row: (row.get("site") or "", row.get("subset") or "", row.get("clip_key") or ""),
    )[:2]
    fragmented_two = sorted(
        (row for row in rows if row.get("gt_total") == "2"),
        key=lambda row: (-float(row.get("total_abs_error") or 0.0), row.get("clip_key") or ""),
    )[:3]
    ambiguous_multiwave = sorted(
        (
            row
            for row in rows
            if float(row.get("peak_simultaneous_count") or 0.0) >= 3.0
            and float(row.get("gt_total") or 0.0) >= 4.0
        ),
        key=lambda row: (
            -float(row.get("peak_simultaneous_count") or 0.0),
            -(float(row.get("gt_total") or 0.0)),
            row.get("clip_key") or "",
        ),
    )[:3]
    selected = single_salient + fragmented_two + ambiguous_multiwave
    seen: set[str] = set()
    frozen: list[dict] = []
    for row in selected:
        clip_key = str(row.get("clip_key") or "")
        if clip_key in seen:
            continue
        seen.add(clip_key)
        domain, clip_id = _split_clip_key(clip_key)
        if row in single_salient:
            bucket = "single_salient"
        elif row in fragmented_two:
            bucket = "fragmented_two_event"
        else:
            bucket = "ambiguous_multiwave"
        frozen.append(
            {
                "dataset": "cfc",
                "bucket": bucket,
                "domain": domain,
                "clip_id": clip_id,
            }
        )
    return frozen[:8]


def build_thermal_probe(rows: list[dict[str, str]]) -> list[dict]:
    buckets = [
        "animal_brief",
        "animal_persistent",
        "false_positive_easy",
        "false_positive_hard",
    ]
    frozen: list[dict] = []
    for bucket in buckets:
        selected = sorted(
            (row for row in rows if row.get("difficulty_group") == bucket),
            key=lambda row: row.get("clip_key") or "",
        )[:2]
        for row in selected:
            domain, clip_id = _split_clip_key(str(row.get("clip_key") or ""))
            frozen.append(
                {
                    "dataset": "thermal",
                    "bucket": bucket,
                    "domain": domain,
                    "clip_id": clip_id,
                }
            )
    return frozen[:8]


def main() -> int:
    args = _parse_args()
    cfc_probe = build_cfc_probe(_load_csv(Path(args.cfc_analysis)))
    thermal_probe = build_thermal_probe(_load_csv(Path(args.thermal_assignments)))
    _write_json(Path(args.cfc_output), cfc_probe)
    _write_json(Path(args.thermal_output), thermal_probe)
    print(Path(args.cfc_output))
    print(Path(args.thermal_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
