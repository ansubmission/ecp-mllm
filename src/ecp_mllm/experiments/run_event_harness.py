from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-train-dataset", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--selection-dataset", required=True)
    parser.add_argument("--holdout-dataset", required=True)
    parser.add_argument("--transfer-holdout-dataset", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--search-dev-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--mode-set", choices=["broad", "focused"], default="broad")
    parser.add_argument("--combo-set", choices=["broad", "focused"], default="broad")
    parser.add_argument("--preset", choices=["grid", "paper_lora_6h"], default="grid")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--eval-quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--eval-torch-dtype", default="float16")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(cmd)}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _event_present(row: dict[str, Any]) -> bool:
    target = dict(row.get("target_json") or {})
    return int(target.get("left_count") or 0) + int(target.get("right_count") or 0) > 0


def _full_strict_prompt(row: dict[str, Any]) -> str:
    window_start = float(row.get("window_start_sec") or 0.0)
    window_end = float(row.get("window_end_sec") or 0.0)
    window_duration = max(0.05, float(row.get("window_duration_sec") or (window_end - window_start) or 0.0))
    return (
        "Analyze fish passage in this sonar video window.\n"
        f"Window start in parent clip: {window_start:.1f}s.\n"
        f"Window end in parent clip: {window_end:.1f}s.\n"
        f"Window duration: {window_duration:.1f}s.\n"
        "Use only evidence visible inside this window.\n"
        "Do not infer events outside this window.\n"
        'Return exactly one JSON object and nothing else. '
        'The first character of your answer must be "{". '
        'Use exactly this schema: {"left_count": int, "right_count": int, "candidate_passages": ['
        '{"timestamp_start_sec": number, "timestamp_end_sec": number, "direction": "left|right|uncertain", '
        '"estimated_count": int, "peak_simultaneous_count": int, "episode_duration_sec": number, '
        '"wave_count": int, "throughput_best_count": int, "evidence_note": string}'
        ']}. '
        'Do not include scene_assessment, commentary, evidence_summary, markdown, bullets, or prose. '
        'If no fish passage is visible, return {"left_count": 0, "right_count": 0, "candidate_passages": []}.'
    )


def _counts_only_strict_prompt(row: dict[str, Any]) -> str:
    window_start = float(row.get("window_start_sec") or 0.0)
    window_end = float(row.get("window_end_sec") or 0.0)
    window_duration = max(0.05, float(row.get("window_duration_sec") or (window_end - window_start) or 0.0))
    return (
        "Count fish passage in this sonar video window.\n"
        f"Window start in parent clip: {window_start:.1f}s.\n"
        f"Window end in parent clip: {window_end:.1f}s.\n"
        f"Window duration: {window_duration:.1f}s.\n"
        "Use only evidence visible inside this window.\n"
        "Do not infer events outside this window.\n"
        'Return exactly one JSON object and nothing else. '
        'The first character of your answer must be "{". '
        'Use exactly this schema: {"left_count": int, "right_count": int}. '
        'Do not include candidate_passages, scene_assessment, commentary, markdown, bullets, or prose. '
        'If no fish passage is visible, return {"left_count": 0, "right_count": 0}.'
    )


def _transform_row(row: dict[str, Any], *, schema: str) -> dict[str, Any]:
    item = deepcopy(row)
    target = dict(item.get("target_json") or {})
    if schema == "counts_only":
        item["target_json"] = {
            "left_count": int(target.get("left_count") or 0),
            "right_count": int(target.get("right_count") or 0),
        }
        item["output_schema"] = "counts_only"
        item["training_prompt"] = _counts_only_strict_prompt(item)
    else:
        item["target_json"] = {
            "left_count": int(target.get("left_count") or 0),
            "right_count": int(target.get("right_count") or 0),
            "candidate_passages": deepcopy(list(target.get("candidate_passages") or [])),
        }
        item["output_schema"] = "full"
        item["training_prompt"] = _full_strict_prompt(item)
    return item


def _duplicate_rows(
    rows: list[dict[str, Any]],
    *,
    pos_mult: int = 1,
    neg_mult: int = 1,
    count2plus_mult: int = 1,
) -> list[dict[str, Any]]:
    positive = [row for row in rows if _event_present(row)]
    negative = [row for row in rows if not _event_present(row)]
    count2plus = [
        row
        for row in positive
        if int((row.get("target_json") or {}).get("left_count") or 0)
        + int((row.get("target_json") or {}).get("right_count") or 0)
        >= 2
    ]
    return positive * pos_mult + negative * neg_mult + count2plus * max(0, count2plus_mult - 1)


def _group_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = str(row.get("parent_clip_id") or row.get("clip_id") or "")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in order:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _build_search_dev(rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    positive = [row for row in rows if _event_present(row)]
    negative = [row for row in rows if not _event_present(row)]
    pos_take = min(len(positive), max_rows // 2)
    neg_take = min(len(negative), max_rows - pos_take)
    if pos_take + neg_take < max_rows:
        pos_take = min(len(positive), max_rows - neg_take)
    return _group_rows(positive, limit=pos_take) + _group_rows(negative, limit=neg_take)


def _build_dataset_variants(
    *,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    outdir: Path,
    search_dev_size: int,
    mode_set: str,
) -> dict[str, dict[str, Path]]:
    if mode_set == "focused":
        modes = [
            {"name": "counts_only_pos2x_strict", "schema": "counts_only", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 1},
            {"name": "counts_only_pos2x_neg2x_strict", "schema": "counts_only", "pos_mult": 2, "neg_mult": 2, "count2plus_mult": 1},
            {"name": "counts_only_pos2x_count2x_strict", "schema": "counts_only", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 2},
            {"name": "full_pos2x_strict", "schema": "full", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 1},
            {"name": "full_pos2x_count2x_strict", "schema": "full", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 2},
        ]
    else:
        modes = [
            {"name": "counts_only_base_strict", "schema": "counts_only", "pos_mult": 1, "neg_mult": 1, "count2plus_mult": 1},
            {"name": "counts_only_pos2x_strict", "schema": "counts_only", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 1},
            {"name": "counts_only_neg2x_strict", "schema": "counts_only", "pos_mult": 1, "neg_mult": 2, "count2plus_mult": 1},
            {"name": "full_base_strict", "schema": "full", "pos_mult": 1, "neg_mult": 1, "count2plus_mult": 1},
            {"name": "full_pos2x_strict", "schema": "full", "pos_mult": 2, "neg_mult": 1, "count2plus_mult": 1},
        ]

    outdir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, dict[str, Path]] = {}
    manifest: dict[str, Any] = {}
    for mode in modes:
        schema = str(mode["schema"])
        train_payload = [
            _transform_row(row, schema=schema)
            for row in _duplicate_rows(
                train_rows,
                pos_mult=int(mode["pos_mult"]),
                neg_mult=int(mode["neg_mult"]),
                count2plus_mult=int(mode["count2plus_mult"]),
            )
        ]
        dev_payload = [_transform_row(row, schema=schema) for row in dev_rows]
        search_dev_payload = _build_search_dev(dev_payload, max_rows=max(2, int(search_dev_size)))
        selection_payload = [_transform_row(row, schema=schema) for row in selection_rows]
        holdout_payload = [_transform_row(row, schema=schema) for row in holdout_rows]

        mode_dir = outdir / mode["name"]
        mode_dir.mkdir(parents=True, exist_ok=True)
        train_path = mode_dir / "train.jsonl"
        search_dev_path = mode_dir / "search_dev.jsonl"
        selection_path = mode_dir / "selection.jsonl"
        holdout_path = mode_dir / "holdout.jsonl"
        _dump_rows(train_path, train_payload)
        _dump_rows(search_dev_path, search_dev_payload)
        _dump_rows(selection_path, selection_payload)
        _dump_rows(holdout_path, holdout_payload)

        resolved[mode["name"]] = {
            "train": train_path,
            "search_dev": search_dev_path,
            "selection": selection_path,
            "holdout": holdout_path,
        }
        manifest[mode["name"]] = {
            "schema": schema,
            "train_rows": len(train_payload),
            "search_dev_rows": len(search_dev_payload),
            "selection_rows": len(selection_payload),
            "holdout_rows": len(holdout_payload),
            "pos_mult": int(mode["pos_mult"]),
            "neg_mult": int(mode["neg_mult"]),
            "count2plus_mult": int(mode["count2plus_mult"]),
        }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in str(value or "42").split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds or [42]


def _build_configs(
    iterations: int,
    *,
    mode_set: str,
    combo_set: str,
    preset: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    if preset == "paper_lora_6h":
        combo_best_selected = {"rank": 32, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.0}
        combo_best_holdout = {"rank": 32, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.05}
        combo_no_dropout_short = {"rank": 32, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.0}
        combo_rank16 = {"rank": 16, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.0}
        families = [
            ("counts_only_pos2x_strict", combo_best_selected),
            ("counts_only_pos2x_strict", combo_best_holdout),
            ("counts_only_pos2x_strict", combo_no_dropout_short),
            ("counts_only_pos2x_neg2x_strict", combo_best_selected),
            ("counts_only_pos2x_neg2x_strict", combo_best_holdout),
            ("counts_only_pos2x_neg2x_strict", combo_rank16),
            ("counts_only_pos2x_count2x_strict", combo_best_selected),
            ("counts_only_pos2x_count2x_strict", combo_best_holdout),
            ("full_pos2x_strict", combo_best_selected),
        ]
        items = []
        for (dataset_mode, combo), seed in itertools.product(families, seeds):
            items.append(
                {
                    "dataset_mode": dataset_mode,
                    "rank": combo["rank"],
                    "learning_rate": combo["learning_rate"],
                    "max_steps": combo["max_steps"],
                    "dropout": combo["dropout"],
                    "seed": int(seed),
                }
            )
        return items[:iterations]

    if mode_set == "focused":
        dataset_modes = [
            "counts_only_pos2x_strict",
            "counts_only_pos2x_neg2x_strict",
            "counts_only_pos2x_count2x_strict",
            "full_pos2x_strict",
            "full_pos2x_count2x_strict",
        ]
    else:
        dataset_modes = [
            "counts_only_base_strict",
            "counts_only_pos2x_strict",
            "counts_only_neg2x_strict",
            "full_base_strict",
            "full_pos2x_strict",
        ]
    if combo_set == "focused":
        combos = [
            {"rank": 16, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.0},
            {"rank": 16, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.0},
            {"rank": 16, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.0},
            {"rank": 16, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 16, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 32, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.0},
            {"rank": 32, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.0},
            {"rank": 32, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.0},
            {"rank": 32, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 32, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.05},
        ]
    else:
        combos = [
            {"rank": 16, "learning_rate": 1e-4, "max_steps": 16, "dropout": 0.05},
            {"rank": 16, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 16, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 16, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.05},
            {"rank": 16, "learning_rate": 3e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 32, "learning_rate": 1e-4, "max_steps": 16, "dropout": 0.05},
            {"rank": 32, "learning_rate": 1e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 32, "learning_rate": 2e-4, "max_steps": 24, "dropout": 0.05},
            {"rank": 32, "learning_rate": 2e-4, "max_steps": 32, "dropout": 0.05},
            {"rank": 32, "learning_rate": 3e-4, "max_steps": 24, "dropout": 0.05},
        ]
    items = []
    for dataset_mode, combo in itertools.product(dataset_modes, combos):
        for seed in seeds:
            items.append(
                {
                    "dataset_mode": dataset_mode,
                    "rank": combo["rank"],
                    "learning_rate": combo["learning_rate"],
                    "max_steps": combo["max_steps"],
                    "dropout": combo["dropout"],
                    "seed": int(seed),
                }
            )
    return items[:iterations]


def _score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    positive_recall = float(metrics.get("positive_recall") or 0.0)
    negative_specificity = float(metrics.get("negative_specificity") or 0.0)
    balanced_accuracy = 0.5 * (positive_recall + negative_specificity)
    return (
        balanced_accuracy,
        -float(metrics.get("count_nmae") or 0.0),
        positive_recall,
        float(metrics.get("presence_accuracy") or 0.0),
    )


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("base config must be a mapping")

    train_rows = _load_rows(Path(args.base_train_dataset).expanduser())
    dev_rows = _load_rows(Path(args.dev_dataset).expanduser())
    selection_rows = _load_rows(Path(args.selection_dataset).expanduser())
    holdout_rows = _load_rows(Path(args.holdout_dataset).expanduser())
    transfer_holdout_path = Path(args.transfer_holdout_dataset).expanduser() if args.transfer_holdout_dataset else None

    dataset_variants = _build_dataset_variants(
        train_rows=train_rows,
        dev_rows=dev_rows,
        selection_rows=selection_rows,
        holdout_rows=holdout_rows,
        outdir=output_root / "datasets",
        search_dev_size=int(args.search_dev_size),
        mode_set=str(args.mode_set),
    )
    seeds = _parse_seeds(str(args.seeds))
    experiments = _build_configs(
        args.iterations,
        mode_set=str(args.mode_set),
        combo_set=str(args.combo_set),
        preset=str(args.preset),
        seeds=seeds,
    )
    results: list[dict[str, Any]] = []
    results_jsonl = output_root / "search_results.jsonl"

    for index, spec in enumerate(experiments, start=1):
        exp_name = (
            f"{index:03d}_{spec['dataset_mode']}_r{spec['rank']}_"
            f"lr{spec['learning_rate']}_s{spec['max_steps']}"
        )
        exp_root = output_root / "experiments" / exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
        config = deepcopy(base_config)
        config["dataset_path"] = str(dataset_variants[spec["dataset_mode"]]["train"])
        config["output_dir"] = str(exp_root / "adapter")
        config["lora"]["r"] = int(spec["rank"])
        config["lora"]["alpha"] = int(spec["rank"]) * 2
        config["lora"]["dropout"] = float(spec["dropout"])
        config["training"]["learning_rate"] = float(spec["learning_rate"])
        config["training"]["max_steps"] = int(spec["max_steps"])
        config["training"]["seed"] = int(spec["seed"])
        config["training"]["data_seed"] = int(spec["seed"])
        config["training"]["save_steps"] = 100
        config["training"]["eval_steps"] = 100
        config_path = exp_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        train_log = exp_root / "train.log"
        eval_json = exp_root / "search_dev_eval.json"
        eval_log = exp_root / "search_dev_eval.log"

        record: dict[str, Any] = {
            "index": index,
            "name": exp_name,
            "spec": spec,
            "config_path": str(config_path),
            "status": "running",
            "updated_at": _now_iso(),
        }
        try:
            _run(
                [
                    "python",
                    "-m",
                    "ecp_mllm.experiments.train_multimodal_qlora",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                log_path=train_log,
            )
            _run(
                [
                    "python",
                    "-m",
                    "ecp_mllm.experiments.eval_event_window_local",
                    "--model-path",
                    str(exp_root / "adapter"),
                    "--dataset-path",
                    str(dataset_variants[spec["dataset_mode"]]["search_dev"]),
                "--output-json",
                str(eval_json),
                "--quantization",
                str(args.eval_quantization),
                "--torch-dtype",
                str(args.eval_torch_dtype),
                "--max-new-tokens",
                str(args.max_new_tokens),
                ],
                cwd=repo_root,
                log_path=eval_log,
            )
            eval_payload = _read_json(eval_json)
            record["status"] = "completed"
            record["metrics"] = eval_payload.get("metrics", {})
            record["score"] = list(_score(record["metrics"]))
            record["updated_at"] = _now_iso()
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record["updated_at"] = _now_iso()
        results.append(record)
        with results_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed = [row for row in results if row.get("status") == "completed" and row.get("metrics")]
    completed.sort(key=lambda row: _score(dict(row["metrics"])), reverse=True)
    leaderboard = []
    for rank_index, row in enumerate(completed[: int(args.topk)], start=1):
        spec = dict(row["spec"])
        exp_root = output_root / "experiments" / str(row["name"])
        selection_eval_json = exp_root / "selection_eval.json"
        selection_eval_log = exp_root / "selection_eval.log"
        holdout_eval_json = exp_root / "holdout_eval.json"
        holdout_eval_log = exp_root / "holdout_eval.log"
        transfer_eval_json = exp_root / "transfer_holdout_eval.json"
        transfer_eval_log = exp_root / "transfer_holdout_eval.log"

        _run(
            [
                "python",
                "-m",
                "ecp_mllm.experiments.eval_event_window_local",
                "--model-path",
                str(exp_root / "adapter"),
                "--dataset-path",
                str(dataset_variants[spec["dataset_mode"]]["selection"]),
                "--output-json",
                str(selection_eval_json),
                "--quantization",
                str(args.eval_quantization),
                "--torch-dtype",
                str(args.eval_torch_dtype),
                "--max-new-tokens",
                str(args.max_new_tokens),
            ],
            cwd=repo_root,
            log_path=selection_eval_log,
        )
        _run(
            [
                "python",
                "-m",
                "ecp_mllm.experiments.eval_event_window_local",
                "--model-path",
                str(exp_root / "adapter"),
                "--dataset-path",
                str(dataset_variants[spec["dataset_mode"]]["holdout"]),
                "--output-json",
                str(holdout_eval_json),
                "--quantization",
                str(args.eval_quantization),
                "--torch-dtype",
                str(args.eval_torch_dtype),
                "--max-new-tokens",
                str(args.max_new_tokens),
            ],
            cwd=repo_root,
            log_path=holdout_eval_log,
        )

        transfer_payload = None
        if transfer_holdout_path is not None:
            _run(
                [
                    "python",
                    "-m",
                    "ecp_mllm.experiments.eval_event_window_local",
                    "--model-path",
                    str(exp_root / "adapter"),
                    "--dataset-path",
                    str(transfer_holdout_path),
                    "--output-json",
                    str(transfer_eval_json),
                    "--quantization",
                    str(args.eval_quantization),
                    "--torch-dtype",
                    str(args.eval_torch_dtype),
                    "--max-new-tokens",
                    str(max(128, int(args.max_new_tokens))),
                ],
                cwd=repo_root,
                log_path=transfer_eval_log,
            )
            transfer_payload = _read_json(transfer_eval_json)

        selection_payload = _read_json(selection_eval_json)
        holdout_payload = _read_json(holdout_eval_json)
        leaderboard.append(
            {
                "rank": rank_index,
                "name": row["name"],
                "spec": spec,
                "search_dev_metrics": row["metrics"],
                "selection_metrics": selection_payload.get("metrics", {}),
                "holdout_metrics": holdout_payload.get("metrics", {}),
                "transfer_holdout_metrics": None if transfer_payload is None else transfer_payload.get("metrics", {}),
            }
        )

    summary = {
        "generated_at": _now_iso(),
        "output_root": str(output_root),
        "iterations": int(args.iterations),
        "topk": int(args.topk),
        "preset": str(args.preset),
        "seeds": seeds,
        "eval_quantization": str(args.eval_quantization),
        "eval_torch_dtype": str(args.eval_torch_dtype),
        "leaderboard": leaderboard,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
