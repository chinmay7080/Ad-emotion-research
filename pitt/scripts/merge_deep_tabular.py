#!/usr/bin/env python3
"""Merge neural tabular folds and run paired, ad-level model comparisons."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional, Sequence

import numpy as np

import evaluate_models_bc as common
from evaluate_deep_tabular import (
    CLASSIFICATION_TARGETS,
    MODELS,
    REGRESSION_TARGETS,
    classification_metrics,
)


SCRIPT_VERSION = "1.1"
TARGETS = tuple([*REGRESSION_TARGETS, *CLASSIFICATION_TARGETS])
COMPARISONS = (
    ("gated_mlp_vs_content_mlp", "gated_mlp", "content_mlp"),
    ("combined_ft_vs_content_ft", "combined_ft", "content_ft"),
)
REGRESSION_METRICS = common.METRICS
CLASSIFICATION_METRICS = ("macro_f1", "macro_recall", "accuracy", "top3_accuracy")


class MergeError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise MergeError(f"missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return common.calculate_metrics(target.astype(np.float64), prediction.astype(np.float64))


def paired_bootstrap(
    target: np.ndarray,
    challenger: np.ndarray,
    baseline: np.ndarray,
    task: str,
    classes: Optional[np.ndarray],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if task == "regression":
        calculate = regression_metrics
        metrics = REGRESSION_METRICS
    else:
        assert classes is not None
        calculate = lambda y, p: classification_metrics(y, p, classes)
        metrics = CLASSIFICATION_METRICS
    challenger_point = calculate(target, challenger)
    baseline_point = calculate(target, baseline)
    draws = {metric: np.empty(repetitions, dtype=np.float64) for metric in metrics}
    for repetition in range(repetitions):
        index = rng.integers(0, target.size, target.size)
        challenger_metrics = calculate(target[index], challenger[index])
        baseline_metrics = calculate(target[index], baseline[index])
        for metric in metrics:
            if metric in {"mae", "rmse"}:
                improvement = baseline_metrics[metric] - challenger_metrics[metric]
            else:
                improvement = challenger_metrics[metric] - baseline_metrics[metric]
            draws[metric][repetition] = improvement
    rows = []
    for metric in metrics:
        if metric in {"mae", "rmse"}:
            point_improvement = baseline_point[metric] - challenger_point[metric]
        else:
            point_improvement = challenger_point[metric] - baseline_point[metric]
        low95, high95 = np.percentile(draws[metric], [2.5, 97.5])
        low_adjusted, high_adjusted = np.percentile(draws[metric], [0.25, 99.75])
        rows.append(
            {
                "metric": metric,
                "baseline_value": baseline_point[metric],
                "challenger_value": challenger_point[metric],
                "positive_improvement": float(point_improvement),
                "ci95_low": float(low95),
                "ci95_high": float(high95),
                "ci99_5_low_ten_test_bonferroni": float(low_adjusted),
                "ci99_5_high_ten_test_bonferroni": float(high_adjusted),
            }
        )
    return rows


def baseline_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = []
    specifications = (
        (Path(args.effectiveness_baseline_metrics), {"effectiveness"}),
        (Path(args.auxiliary_baseline_metrics), {"funny", "exciting"}),
        (Path(args.categorical_baseline_metrics), {"sentiment", "topic"}),
    )
    for path, expected_targets in specifications:
        rows = read_csv(path.expanduser().resolve())
        found = set()
        for row in rows:
            target = row.get("target", "effectiveness")
            if target in expected_targets and row.get("model") == "content_ridge":
                metric_names = (
                    REGRESSION_METRICS if target in REGRESSION_TARGETS else CLASSIFICATION_METRICS
                )
                result.append(
                    {
                        "target": target,
                        "task": (
                            "regression" if target in REGRESSION_TARGETS else "classification"
                        ),
                        "model": "content_ridge",
                        **{metric: float(row[metric]) for metric in metric_names},
                    }
                )
                found.add(target)
        if found != expected_targets:
            raise MergeError(
                f"baseline metrics {path} supplied {sorted(found)}, expected {sorted(expected_targets)}"
            )
    return result


def run(args: argparse.Namespace) -> Path:
    fold_root = Path(args.fold_root).expanduser().resolve()
    final = Path(args.output_dir).expanduser().resolve()
    if final.exists():
        raise MergeError(f"refusing to replace existing output: {final}")
    summary_rows: list[dict[str, Any]] = baseline_rows(args)
    comparison_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    probability_payload: dict[str, np.ndarray] = {}
    for target_offset, target_name in enumerate(TARGETS):
        task = "classification" if target_name in CLASSIFICATION_TARGETS else "regression"
        target_rows: list[dict[str, str]] = []
        probability_blocks: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
        classes: Optional[np.ndarray] = None
        for fold in range(5):
            fold_dir = fold_root / target_name / f"fold_{fold}"
            rows = read_csv(fold_dir / "predictions.csv")
            if len(rows) != 373:
                raise MergeError(
                    f"{target_name} fold {fold} has {len(rows)} rows; expected 373"
                )
            if any(row["target"] != target_name or int(row["fold"]) != fold for row in rows):
                raise MergeError(f"target/fold labels are invalid in {fold_dir}")
            metadata = read_csv(fold_dir / "model_metrics.csv")
            if len(metadata) != len(MODELS):
                raise MergeError(f"model metrics are incomplete in {fold_dir}")
            fold_rows.extend(
                {"target": target_name, "fold": fold, **row} for row in metadata
            )
            if task == "classification":
                with np.load(fold_dir / "probabilities.npz", allow_pickle=False) as bundle:
                    fold_classes = bundle["classes"].astype(np.int64)
                    if classes is None:
                        classes = fold_classes
                    elif not np.array_equal(classes, fold_classes):
                        raise MergeError(f"class universe changed in {fold_dir}")
                    bundle_ids = bundle["video_ids"].astype(str)
                    if list(bundle_ids) != [row["video_id"] for row in rows]:
                        raise MergeError(f"probability IDs do not match in {fold_dir}")
                    for model in MODELS:
                        values = bundle[model].astype(np.float64)
                        if values.shape != (373, fold_classes.size):
                            raise MergeError(f"invalid {model} probabilities in {fold_dir}")
                        if not np.isfinite(values).all() or np.any(
                            np.abs(values.sum(axis=1) - 1.0) > 1e-5
                        ):
                            raise MergeError(f"invalid {model} probability values in {fold_dir}")
                        probability_blocks[model].append(values)
            target_rows.extend(rows)
        ids = [row["video_id"] for row in target_rows]
        if len(ids) != 1865 or len(set(ids)) != 1865:
            raise MergeError(f"{target_name} does not contain 1,865 unique ads")
        observed = np.asarray(
            [float(row["observed"]) for row in target_rows],
            dtype=np.float64 if task == "regression" else np.int64,
        )
        if task == "regression":
            predictions = {
                model: np.asarray([float(row[model]) for row in target_rows], dtype=np.float64)
                for model in MODELS
            }
            for model in MODELS:
                summary_rows.append(
                    {
                        "target": target_name,
                        "task": task,
                        "model": model,
                        **regression_metrics(observed, predictions[model]),
                    }
                )
        else:
            assert classes is not None
            predictions = {
                model: np.concatenate(probability_blocks[model], axis=0) for model in MODELS
            }
            probability_payload[f"{target_name}__classes"] = classes
            probability_payload[f"{target_name}__video_ids"] = np.asarray(ids)
            probability_payload[f"{target_name}__observed"] = observed
            for model in MODELS:
                probability_payload[f"{target_name}__{model}"] = predictions[model].astype(
                    np.float32
                )
                summary_rows.append(
                    {
                        "target": target_name,
                        "task": task,
                        "model": model,
                        **classification_metrics(observed, predictions[model], classes),
                    }
                )
        for comparison_offset, (comparison, challenger, baseline) in enumerate(COMPARISONS):
            bootstrapped = paired_bootstrap(
                observed,
                predictions[challenger],
                predictions[baseline],
                task,
                classes,
                args.bootstrap_repetitions,
                args.seed + target_offset * 100 + comparison_offset * 10,
            )
            comparison_rows.extend(
                {
                    "target": target_name,
                    "task": task,
                    "comparison": comparison,
                    "challenger": challenger,
                    "baseline": baseline,
                    **row,
                }
                for row in bootstrapped
            )
        for row_index, row in enumerate(target_rows):
            if task == "regression":
                predictions_for_row = {
                    model: float(predictions[model][row_index]) for model in MODELS
                }
            else:
                assert classes is not None
                predictions_for_row = {
                    model: int(classes[np.argmax(predictions[model][row_index])])
                    for model in MODELS
                }
            prediction_rows.append(
                {
                    "target": target_name,
                    "task": task,
                    "fold": int(row["fold"]),
                    "video_id": row["video_id"],
                    "observed": row["observed"],
                    **predictions_for_row,
                }
            )

    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        metric_fields = sorted(
            set(REGRESSION_METRICS).union(CLASSIFICATION_METRICS)
        )
        normalized_summary = [
            {field: row.get(field, "") for field in ["target", "task", "model", *metric_fields]}
            for row in summary_rows
        ]
        common.write_csv(
            staging / "overall_metrics.csv",
            ["target", "task", "model", *metric_fields],
            normalized_summary,
        )
        common.write_csv(
            staging / "paired_comparisons.csv",
            [
                "target", "task", "comparison", "challenger", "baseline", "metric",
                "baseline_value", "challenger_value", "positive_improvement",
                "ci95_low", "ci95_high", "ci99_5_low_ten_test_bonferroni",
                "ci99_5_high_ten_test_bonferroni",
            ],
            comparison_rows,
        )
        common.write_csv(
            staging / "out_of_fold_predictions.csv",
            ["target", "task", "fold", "video_id", "observed", *MODELS],
            prediction_rows,
        )
        common.write_csv(
            staging / "fold_metrics.csv",
            ["target", "fold", "model", *metric_fields],
            [
                {field: row.get(field, "") for field in ["target", "fold", "model", *metric_fields]}
                for row in fold_rows
            ],
        )
        np.savez_compressed(staging / "classification_probabilities.npz", **probability_payload)
        baseline_paths = [
            Path(args.effectiveness_baseline_metrics).expanduser().resolve(),
            Path(args.auxiliary_baseline_metrics).expanduser().resolve(),
            Path(args.categorical_baseline_metrics).expanduser().resolve(),
        ]
        common.dump_json(
            staging / "metadata.json",
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "targets": list(TARGETS),
                "models": list(MODELS),
                "reference_model": "content_ridge from the prior frozen-fold benchmark",
                "ads_per_target": 1865,
                "outer_folds": 5,
                "fold_sizes": [373] * 5,
                "outer_group_overlap_every_fold": 0,
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "primary_matched_comparisons": [value[0] for value in COMPARISONS],
                "multiple_testing": (
                    "Bonferroni family alpha 0.05 across 10 target-by-primary-comparison "
                    "tests; 99.5% interval per primary metric test"
                ),
                "interpretation_boundary": (
                    "brain inputs are TRIBE-predicted cortical features, not measured viewer fMRI"
                ),
                "fold_root": str(fold_root),
                "baseline_metric_inputs": [
                    {"path": str(path), "sha256": common.sha256_file(path)}
                    for path in baseline_paths
                ],
            },
        )
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"MERGE_COMPLETE output={final}", flush=True)
    return final


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fold-root", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--effectiveness-baseline-metrics", required=True)
    result.add_argument("--auxiliary-baseline-metrics", required=True)
    result.add_argument("--categorical-baseline-metrics", required=True)
    result.add_argument("--bootstrap-repetitions", type=int, default=5000)
    result.add_argument("--seed", type=int, default=20260830)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    output = run(args)
    print(f"OUTPUT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
