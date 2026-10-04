#!/usr/bin/env python3
"""Paired bootstrap comparison of neural models with frozen content Ridge predictions."""

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
from merge_deep_tabular import paired_bootstrap


SCRIPT_VERSION = "1.0"
TARGETS = ("effectiveness", "funny", "exciting", "sentiment", "topic")
CHALLENGERS = ("content_mlp", "gated_mlp")


class ComparisonError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ComparisonError(f"missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def regression_baseline(
    target_name: str,
    ids: list[str],
    effectiveness_path: Path,
    auxiliary_path: Path,
) -> np.ndarray:
    if target_name == "effectiveness":
        rows = read_csv(effectiveness_path)
        mapping = {row["video_id"]: float(row["content_ridge"]) for row in rows}
    else:
        rows = [row for row in read_csv(auxiliary_path) if row["target"] == target_name]
        mapping = {row["video_id"]: float(row["content_ridge"]) for row in rows}
    if len(mapping) != 1865 or set(mapping) != set(ids):
        raise ComparisonError(f"invalid Ridge predictions for {target_name}")
    return np.asarray([mapping[value] for value in ids], dtype=np.float64)


def classification_baseline(
    target_name: str,
    ids: list[str],
    fold_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    classes: Optional[np.ndarray] = None
    for fold in range(5):
        fold_dir = fold_root / target_name / f"fold_{fold}"
        with np.load(fold_dir / "probabilities.npz", allow_pickle=False) as bundle:
            fold_classes = bundle["classes"].astype(np.int64)
            if classes is None:
                classes = fold_classes
            elif not np.array_equal(classes, fold_classes):
                raise ComparisonError(f"Ridge classes changed for {target_name}")
            id_key = "video_id" if "video_id" in bundle.files else "video_ids"
            fold_ids = bundle[id_key].astype(str)
            probabilities = bundle["content_ridge"].astype(np.float64)
            if probabilities.shape != (fold_ids.size, fold_classes.size):
                raise ComparisonError(f"invalid Ridge probability shape for {target_name}")
            for row, video_id in enumerate(fold_ids):
                if video_id in mapping:
                    raise ComparisonError(f"duplicate Ridge ID for {target_name}: {video_id}")
                mapping[video_id] = probabilities[row]
    if classes is None or len(mapping) != 1865 or set(mapping) != set(ids):
        raise ComparisonError(f"invalid Ridge probabilities for {target_name}")
    result = np.stack([mapping[value] for value in ids], axis=0)
    return result, classes


def run(args: argparse.Namespace) -> Path:
    deep_dir = Path(args.deep_results).expanduser().resolve()
    deep_rows = read_csv(deep_dir / "out_of_fold_predictions.csv")
    with np.load(
        deep_dir / "classification_probabilities.npz", allow_pickle=False
    ) as bundle:
        deep_probabilities = {key: bundle[key] for key in bundle.files}
    effectiveness_path = Path(args.effectiveness_predictions).expanduser().resolve()
    auxiliary_path = Path(args.auxiliary_predictions).expanduser().resolve()
    categorical_root = Path(args.categorical_fold_root).expanduser().resolve()
    result_rows: list[dict[str, Any]] = []
    for target_offset, target_name in enumerate(TARGETS):
        rows = [row for row in deep_rows if row["target"] == target_name]
        ids = [row["video_id"] for row in rows]
        if len(rows) != 1865 or len(set(ids)) != 1865:
            raise ComparisonError(f"invalid deep predictions for {target_name}")
        task = rows[0]["task"]
        observed = np.asarray(
            [float(row["observed"]) for row in rows],
            dtype=np.float64 if task == "regression" else np.int64,
        )
        if task == "regression":
            baseline = regression_baseline(
                target_name, ids, effectiveness_path, auxiliary_path
            )
            classes = None
            challenger_values = {
                name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
                for name in CHALLENGERS
            }
        else:
            baseline, classes = classification_baseline(
                target_name, ids, categorical_root
            )
            if not np.array_equal(
                classes, deep_probabilities[f"{target_name}__classes"].astype(np.int64)
            ):
                raise ComparisonError(f"deep and Ridge classes differ for {target_name}")
            if list(deep_probabilities[f"{target_name}__video_ids"].astype(str)) != ids:
                raise ComparisonError(f"deep probability IDs differ for {target_name}")
            challenger_values = {
                name: deep_probabilities[f"{target_name}__{name}"].astype(np.float64)
                for name in CHALLENGERS
            }
        for comparison_offset, challenger_name in enumerate(CHALLENGERS):
            results = paired_bootstrap(
                observed,
                challenger_values[challenger_name],
                baseline,
                task,
                classes,
                args.bootstrap_repetitions,
                args.seed + 100 * target_offset + 10 * comparison_offset,
            )
            for row in results:
                result_rows.append(
                    {
                        "target": target_name,
                        "task": task,
                        "comparison": f"{challenger_name}_vs_content_ridge",
                        "challenger": challenger_name,
                        "baseline": "content_ridge",
                        **row,
                    }
                )
    final = Path(args.output_dir).expanduser().resolve()
    if final.exists():
        raise ComparisonError(f"refusing to replace existing output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        common.write_csv(
            staging / "paired_comparisons.csv",
            [
                "target", "task", "comparison", "challenger", "baseline", "metric",
                "baseline_value", "challenger_value", "positive_improvement",
                "ci95_low", "ci95_high", "ci99_5_low_ten_test_bonferroni",
                "ci99_5_high_ten_test_bonferroni",
            ],
            result_rows,
        )
        inputs = [
            deep_dir / "out_of_fold_predictions.csv",
            deep_dir / "classification_probabilities.npz",
            effectiveness_path,
            auxiliary_path,
        ] + sorted(categorical_root.glob("*/fold_*/probabilities.npz"))
        common.dump_json(
            staging / "metadata.json",
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "targets": list(TARGETS),
                "challengers": list(CHALLENGERS),
                "baseline": "content_ridge",
                "ads_per_target": 1865,
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "primary_metrics": {"regression": "spearman", "classification": "macro_f1"},
                "multiple_testing": (
                    "Bonferroni family alpha 0.05 across 10 target-by-challenger primary tests; "
                    "99.5% interval per primary test"
                ),
                "inputs": [
                    {"path": str(path), "sha256": common.sha256_file(path)} for path in inputs
                ],
            },
        )
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"RIDGE_COMPARISON_COMPLETE output={final}", flush=True)
    return final


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--deep-results", required=True)
    result.add_argument("--effectiveness-predictions", required=True)
    result.add_argument("--auxiliary-predictions", required=True)
    result.add_argument("--categorical-fold-root", required=True)
    result.add_argument("--output-dir", required=True)
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
