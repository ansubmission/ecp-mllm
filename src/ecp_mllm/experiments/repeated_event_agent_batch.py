from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from ..types import ExperimentProtocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variant", default="sff3c")
    parser.add_argument("--model", default=None)
    parser.add_argument("--transport", choices=["sampled_frames", "stitched_mp4"], default="stitched_mp4")
    parser.add_argument("--stitch-fps", type=float, default=5.0)
    parser.add_argument("--stitch-height", type=int, default=0)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--clips-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--proposal-min-duration-sec", type=float, default=1.0)
    parser.add_argument("--proposal-merge-gap-sec", type=float, default=1.0)
    parser.add_argument("--proposal-min-video-sec", type=float, default=8.0)
    parser.add_argument("--flagged-repeat-runs", type=int, default=1)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "repeated"


def _run_summary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    completed = [item for item in payload.get("results", []) if item.get("status") == "completed"]
    if not completed:
        return {
            "left_abs_error": 0.0,
            "right_abs_error": 0.0,
            "total_abs_error": 0.0,
            "mae": 0.0,
            "nmae": 0.0,
            "parse_rate": 0.0,
            "mean_latency_sec": 0.0,
            "proposal_selected_rate": 0.0,
            "global_selected_rate": 0.0,
            "custom_selected_rate": 0.0,
            "custom_high_throughput_selected_rate": 0.0,
            "custom_low_count_selected_rate": 0.0,
        }
    left_abs = sum(float(item.get("abs_error_left", 0.0)) for item in completed)
    right_abs = sum(float(item.get("abs_error_right", 0.0)) for item in completed)
    total_abs = left_abs + right_abs
    gt_total = sum(float((item.get("ground_truth") or {}).get("total_count", 0.0)) for item in completed)
    parse_successes = sum(1 for item in completed if (item.get("prediction") or {}).get("parse_success", False))
    mean_latency = sum(float((item.get("prediction") or {}).get("latency_sec", 0.0)) for item in completed) / len(completed)
    proposal_selected = sum(1 for item in completed if str(item.get("selected_action", "")).endswith(":proposal_guided"))
    global_selected = sum(1 for item in completed if str(item.get("selected_action", "")).endswith(":global"))
    custom_selected = sum(1 for item in completed if ":custom_" in str(item.get("selected_action", "")))
    custom_high_selected = sum(1 for item in completed if str(item.get("selected_action", "")).endswith(":custom_high_throughput"))
    custom_low_selected = sum(1 for item in completed if str(item.get("selected_action", "")).endswith(":custom_low_count"))
    return {
        "left_abs_error": left_abs,
        "right_abs_error": right_abs,
        "total_abs_error": total_abs,
        "mae": total_abs / len(completed),
        "nmae": (total_abs / gt_total) if gt_total > 0 else 0.0,
        "parse_rate": parse_successes / len(completed),
        "mean_latency_sec": mean_latency,
        "proposal_selected_rate": proposal_selected / len(completed),
        "global_selected_rate": global_selected / len(completed),
        "custom_selected_rate": custom_selected / len(completed),
        "custom_high_throughput_selected_rate": custom_high_selected / len(completed),
        "custom_low_count_selected_rate": custom_low_selected / len(completed),
    }


def _aggregate(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(metrics_list[0].keys()) if metrics_list else []
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [item[key] for item in metrics_list]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['name']}",
        "",
        f"- domain: `{payload['domain']}`",
        f"- variant: `{payload['variant']}`",
        f"- model: `{payload['model']}`",
        f"- repeats: `{payload['protocol']['repeats']}`",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, stats in payload["aggregate"].items():
        lines.append(
            f"| `{metric}` | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "| Run | Summary |", "|---|---|"])
    for run in payload["runs"]:
        lines.append(f"| `{run['run_name']}` | `{run['summary_json']}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_event_agent_command(args: argparse.Namespace, run_name: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ecp_mllm.experiments.event_agent_batch",
        "--settings",
        args.settings,
        "--domain",
        args.domain,
        "--variant",
        args.variant,
        "--transport",
        args.transport,
        "--stitch-fps",
        str(args.stitch_fps),
        "--stitch-height",
        str(args.stitch_height),
        "--stitch-crf",
        str(args.stitch_crf),
        "--clips-path",
        args.clips_path,
        "--name",
        run_name,
        "--max-proposals",
        str(args.max_proposals),
        "--proposal-min-duration-sec",
        str(args.proposal_min_duration_sec),
        "--proposal-merge-gap-sec",
        str(args.proposal_merge_gap_sec),
        "--proposal-min-video-sec",
        str(args.proposal_min_video_sec),
        "--flagged-repeat-runs",
        str(args.flagged_repeat_runs),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.prompt_text:
        command.extend(["--prompt-text", args.prompt_text])
    if args.max_clips is not None:
        command.extend(["--max-clips", str(args.max_clips)])
    if args.stop_on_error:
        command.append("--stop-on-error")
    return command


def main() -> int:
    args = _parse_args()
    protocol = ExperimentProtocol(
        name=args.name,
        repeats=max(1, args.repeats),
        paired=True,
        split_names=(args.domain,),
        notes="repeated event_agent_batch protocol for variance-aware evaluation",
    )
    run_entries: list[dict[str, Any]] = []
    metrics_list: list[dict[str, float]] = []

    for run_index in range(1, protocol.repeats + 1):
        run_name = f"{args.name}__run{run_index:02d}"
        command = _build_event_agent_command(args, run_name)
        completed = subprocess.run(command, cwd=str(Path.cwd()), capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"repeat {run_index} failed with exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
        summary_json = Path("outputs") / "event_agent" / _safe_name(run_name) / "summary.json"
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        metrics = _run_summary_metrics(payload)
        metrics_list.append(metrics)
        run_entries.append(
            {
                "run_name": run_name,
                "summary_json": str(summary_json).replace("/", "\\"),
                "metrics": metrics,
            }
        )

    output = {
        "name": args.name,
        "domain": args.domain,
        "variant": args.variant,
        "model": args.model,
        "protocol": protocol.__dict__,
        "runs": run_entries,
        "aggregate": _aggregate(metrics_list),
    }
    outdir = Path("outputs") / "repro_protocol" / _safe_name(args.name)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_markdown(outdir / "summary.md", output)
    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

