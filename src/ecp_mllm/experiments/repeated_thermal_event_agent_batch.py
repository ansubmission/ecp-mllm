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
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default="thermal_filtered")
    parser.add_argument("--model", default=None)
    parser.add_argument("--clips-path", default=None)
    parser.add_argument("--pack", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--stitch-height", type=int, default=360)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--proposal-min-duration-sec", type=float, default=2.0)
    parser.add_argument("--proposal-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--proposal-source", choices=["raw_video", "track_oracle"], default="raw_video")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "thermal"


def _run_summary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    return {
        "binary_accuracy": float(metrics.get("binary_accuracy", 0.0) or 0.0),
        "binary_f1": float(metrics.get("binary_f1", 0.0) or 0.0),
        "binary_balanced_accuracy": float(metrics.get("binary_balanced_accuracy", 0.0) or 0.0),
        "coarse_macro_f1": float(metrics.get("coarse_macro_f1", 0.0) or 0.0),
        "event_window_recall": float(metrics.get("event_window_recall", 0.0) or 0.0),
        "event_window_mean_tiou": float(metrics.get("event_window_mean_tiou", 0.0) or 0.0),
        "center_zone_entry_f1": float(metrics.get("center_zone_entry_f1", 0.0) or 0.0),
        "center_zone_first_entry_mae_sec": float(metrics.get("center_zone_first_entry_mae_sec", 0.0) or 0.0),
        "abstention_rate": float(metrics.get("abstention_rate", 0.0) or 0.0),
        "mean_latency_sec": float(metrics.get("mean_latency_sec", 0.0) or 0.0),
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
        f"- split: `{payload['split']}`",
        f"- variant: `{payload['variant']}`",
        f"- model: `{payload['model']}`",
        f"- repeats: `{payload['protocol']['repeats']}`",
        f"- proposal_source: `{payload['proposal_source']}`",
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


def _build_command(args: argparse.Namespace, run_name: str) -> list[str]:
    split_value = "all" if str(args.split).strip().lower() in {"", "all", "*", "none"} else args.split
    command = [
        sys.executable,
        "-m",
        "ecp_mllm.experiments.thermal_event_agent_batch",
        "--settings",
        args.settings,
        "--split",
        split_value,
        "--variant",
        args.variant,
        "--name",
        run_name,
        "--stitch-height",
        str(args.stitch_height),
        "--stitch-crf",
        str(args.stitch_crf),
        "--max-proposals",
        str(args.max_proposals),
        "--proposal-min-duration-sec",
        str(args.proposal_min_duration_sec),
        "--proposal-overlap-ratio",
        str(args.proposal_overlap_ratio),
        "--proposal-source",
        args.proposal_source,
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.clips_path:
        command.extend(["--clips-path", args.clips_path])
    if args.pack:
        command.extend(["--pack", args.pack])
    if args.prompt_text:
        command.extend(["--prompt-text", args.prompt_text])
    if args.max_clips is not None:
        command.extend(["--max-clips", str(args.max_clips)])
    if args.include_calibration:
        command.append("--include-calibration")
    if args.stop_on_error:
        command.append("--stop-on-error")
    return command


def main() -> int:
    args = _parse_args()
    split_value = "all" if str(args.split).strip().lower() in {"", "all", "*", "none"} else args.split
    protocol = ExperimentProtocol(
        name=args.name,
        repeats=max(1, args.repeats),
        paired=True,
        split_names=(split_value,),
        notes="repeated thermal_event_agent_batch protocol for thermal consistency evaluation",
    )
    metrics_list: list[dict[str, float]] = []
    runs: list[dict[str, Any]] = []
    for run_index in range(1, protocol.repeats + 1):
        run_name = f"{args.name}__run{run_index:02d}"
        command = _build_command(args, run_name)
        completed = subprocess.run(command, cwd=str(Path.cwd()), capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"repeat {run_index} failed with exit code {completed.returncode}\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
        summary_json = Path("outputs") / "thermal_event_agent" / _safe_name(run_name) / "summary.json"
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        metrics = _run_summary_metrics(payload)
        metrics_list.append(metrics)
        runs.append({"run_name": run_name, "summary_json": str(summary_json).replace("/", "\\"), "metrics": metrics})
    output = {
        "name": args.name,
        "split": split_value,
        "variant": args.variant,
        "model": args.model,
        "proposal_source": args.proposal_source,
        "protocol": protocol.__dict__,
        "runs": runs,
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
