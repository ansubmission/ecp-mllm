from __future__ import annotations

from csv import DictWriter
import json
from pathlib import Path
from typing import Iterable

from ..types import EvalReport, PromptRevision


def flatten_report_rows(experiment_name: str, variant: str, prompt_mode: str, report: EvalReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain, summary in report.per_domain.items():
        rows.append(
            {
                "experiment": experiment_name,
                "variant": variant,
                "prompt_mode": prompt_mode,
                "domain": domain,
                "clips": summary.clips,
                "mae": round(summary.mae, 6),
                "rmse": round(summary.rmse, 6),
                "nmae": round(summary.nmae, 6) if summary.nmae is not None else "",
                "parse_rate": round(summary.parse_rate, 6),
                "mean_latency_sec": round(summary.mean_latency_sec, 6),
            }
        )
    rows.append(
        {
            "experiment": experiment_name,
            "variant": variant,
            "prompt_mode": prompt_mode,
            "domain": "OVERALL",
            "clips": len(report.clip_results),
            "mae": round(report.overall_mae, 6),
            "rmse": round(report.overall_rmse, 6),
            "nmae": round(report.overall_nmae, 6) if report.overall_nmae is not None else "",
            "parse_rate": round(report.parse_rate, 6),
            "mean_latency_sec": round(report.mean_latency_sec, 6),
        }
    )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if hasattr(value, "__dict__"):
        return value.__dict__
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_markdown_summary(
    path: str | Path,
    experiment_name: str,
    variants: Iterable[str],
    base_reports: dict[str, EvalReport],
    refined_reports: dict[str, EvalReport],
    best_prompts: dict[str, PromptRevision],
    observation_samples: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> None:
    lines = [f"# {experiment_name}", "", "| Variant | Prompt | Overall nMAE | Overall MAE | Parse Rate |", "|---|---|---:|---:|---:|"]
    for variant in variants:
        for mode, reports in [("base", base_reports), ("refined", refined_reports)]:
            report = reports[variant]
            lines.append(
                f"| {variant} | {mode} | {report.overall_nmae if report.overall_nmae is not None else ''} | {report.overall_mae:.4f} | {report.parse_rate:.4f} |"
            )
        lines.append(f"| {variant} | best_prompt_id | `{best_prompts[variant].prompt_id}` |  |  |")
        if best_prompts[variant].critique:
            lines.extend(["", f"## {variant} Critique", "", best_prompts[variant].critique])
        if observation_samples and variant in observation_samples:
            refined_samples = observation_samples[variant].get("refined", [])
            if refined_samples:
                lines.extend(["", f"## {variant} Observation Samples", ""])
                for sample in refined_samples:
                    lines.append(
                        f"- `{sample['domain']}/{sample['clip_id']}` truth=({sample['truth_left']},{sample['truth_right']}) "
                        f"pred=({sample['pred_left']},{sample['pred_right']}) commentary={sample['commentary']!r}"
                    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
