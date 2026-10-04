#!/usr/bin/env python3
"""Evaluate an explicitly named numeric ad target with leakage-safe nested CV."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy as np


SCRIPT_VERSION = "1.0"
METRIC_NAMES = ("pearson", "spearman", "mae", "rmse")


class EvaluationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def delimiter_for(path: Path) -> str:
    name = path.name.lower()
    return "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter_for(path))
            if reader.fieldnames is None:
                raise EvaluationError(f"table has no header: {path}")
            headers = [str(value) for value in reader.fieldnames]
            if len(headers) != len(set(headers)):
                raise EvaluationError(f"duplicate column names in {path}")
            rows = [{key: ("" if value is None else value) for key, value in row.items()} for row in reader]
    except (OSError, csv.Error) as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    return headers, rows


@contextlib.contextmanager
def gzip_text_writer(path: Path) -> Iterator[io.TextIOWrapper]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with gzip_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def index_unique(
    rows: Sequence[dict[str, str]], headers: Sequence[str], id_column: str, label: str
) -> dict[str, dict[str, str]]:
    if id_column not in headers:
        raise EvaluationError(f"{label} is missing ID column {id_column!r}")
    result: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        identifier = row[id_column]
        if not identifier or not identifier.strip():
            raise EvaluationError(f"{label} has blank ID at row {row_number}")
        if identifier != identifier.strip():
            raise EvaluationError(f"{label} has surrounding whitespace in ID at row {row_number}")
        if identifier in result:
            raise EvaluationError(f"{label} has duplicate ID {identifier!r}")
        result[identifier] = row
    return result


def parse_finite_number(value: str, context: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise EvaluationError(f"non-numeric value in {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise EvaluationError(f"non-finite value in {context}: {value!r}")
    return number


def load_target(
    path: Path,
    id_column: str,
    target_column: str,
    group_column: str,
    missing_policy: str,
) -> tuple[list[str], np.ndarray, list[str], dict[str, Any]]:
    headers, rows = read_table(path)
    if target_column not in headers:
        raise EvaluationError(f"target column {target_column!r} is absent from {path}")
    if group_column not in headers:
        raise EvaluationError(f"group column {group_column!r} is absent from {path}")
    index_unique(rows, headers, id_column, "cohort")
    identifiers: list[str] = []
    values: list[float] = []
    groups: list[str] = []
    missing_ids: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        identifier = row[id_column]
        group = row[group_column]
        if not group or not group.strip():
            raise EvaluationError(f"cohort has blank group {group_column!r} at row {row_number}")
        if group != group.strip():
            raise EvaluationError(
                f"cohort has surrounding whitespace in group {group_column!r} at row {row_number}"
            )
        raw = row[target_column]
        if not raw.strip():
            missing_ids.append(identifier)
            continue
        identifiers.append(identifier)
        values.append(parse_finite_number(raw, f"target {target_column} for ID {identifier}"))
        groups.append(group)
    if missing_ids and missing_policy == "error":
        raise EvaluationError(
            f"target {target_column!r} is missing for {len(missing_ids)} ads; first IDs: "
            f"{missing_ids[:10]}. Use --missing-target-policy drop only after review."
        )
    if not values:
        raise EvaluationError("no finite target values remain")
    target = np.asarray(values, dtype=np.float64)
    if np.unique(target).size < 2:
        raise EvaluationError("target has no variation")
    return identifiers, target, groups, {
        "cohort_rows": len(rows),
        "eligible_target_rows": len(identifiers),
        "missing_target_count": len(missing_ids),
        "missing_target_ids": missing_ids,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_integer_like_count": int(np.count_nonzero(target == np.floor(target))),
        "group_column": group_column,
        "eligible_group_count": len(set(groups)),
    }


def load_features(
    path: Path,
    id_column: str,
    feature_regex: str,
    label: str,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    headers, rows = read_table(path)
    indexed = index_unique(rows, headers, id_column, label)
    try:
        pattern = re.compile(feature_regex)
    except re.error as exc:
        raise EvaluationError(f"invalid {label} feature regex {feature_regex!r}: {exc}") from exc
    feature_columns = [
        column for column in headers if column != id_column and pattern.search(column) is not None
    ]
    if not feature_columns:
        raise EvaluationError(f"{label} regex selected no feature columns: {feature_regex!r}")

    result: dict[str, np.ndarray] = {}
    missing_counts = np.zeros(len(feature_columns), dtype=np.int64)
    for identifier, row in indexed.items():
        values = np.empty(len(feature_columns), dtype=np.float64)
        for column_index, column in enumerate(feature_columns):
            raw = row[column]
            if not raw.strip():
                values[column_index] = np.nan
                missing_counts[column_index] += 1
            else:
                values[column_index] = parse_finite_number(
                    raw, f"{label} feature {column} for ID {identifier}"
                )
        result[identifier] = values
    all_missing = [
        column for column, count in zip(feature_columns, missing_counts) if int(count) == len(rows)
    ]
    if all_missing:
        raise EvaluationError(f"{label} has all-missing feature columns: {all_missing[:10]}")
    return result, feature_columns, {
        "rows": len(rows),
        "feature_count": len(feature_columns),
        "feature_regex": feature_regex,
        "missing_value_count": int(np.sum(missing_counts)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    return {
        "pearson": correlation(y_true, y_pred),
        "spearman": correlation(average_ranks(y_true), average_ranks(y_pred)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def deterministic_group_folds(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled, sample-balanced folds while keeping every group intact."""

    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or groups.size == 0:
        raise EvaluationError("group vector must be a nonempty one-dimensional array")
    unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if unique_groups.size < n_splits:
        raise EvaluationError(
            f"requested {n_splits} group folds but only {unique_groups.size} distinct groups remain"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups.size)
    # Stable sorting retains the seeded random order for equal-sized groups.
    ordered = shuffled[np.argsort(-counts[shuffled], kind="mergesort")]
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = np.full(unique_groups.size, -1, dtype=np.int64)
    for group_index in ordered:
        candidates = np.flatnonzero(fold_sizes == np.min(fold_sizes))
        chosen_fold = int(candidates[int(rng.integers(0, candidates.size))])
        group_to_fold[group_index] = chosen_fold
        fold_sizes[chosen_fold] += counts[group_index]

    result: list[tuple[np.ndarray, np.ndarray]] = []
    sample_folds = group_to_fold[inverse]
    for fold in range(n_splits):
        test = np.flatnonzero(sample_folds == fold)
        train = np.flatnonzero(sample_folds != fold)
        if train.size == 0 or test.size == 0:
            raise EvaluationError("group fold construction produced an empty train or test split")
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise EvaluationError(f"internal error: group leakage in constructed fold: {sorted(overlap)}")
        result.append((train, test))
    return result


def make_outer_splits(
    groups: np.ndarray, outer_splits: int, outer_repeats: int, seed: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    result: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(outer_repeats):
        for fold, (train, test) in enumerate(
            deterministic_group_folds(groups, outer_splits, seed + repeat)
        ):
            result.append((repeat, fold, train, test))
    return result


def group_cv_audit(
    groups: np.ndarray,
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    inner_splits: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    outer_checks: list[dict[str, Any]] = []
    inner_checks: list[dict[str, Any]] = []
    for repeat, outer_fold, outer_train, outer_test in splits:
        train_groups = set(groups[outer_train])
        test_groups = set(groups[outer_test])
        overlap = train_groups.intersection(test_groups)
        outer_checks.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "train_group_count": len(train_groups),
                "test_group_count": len(test_groups),
                "overlap_count": len(overlap),
            }
        )
        for role, indices in (("train", outer_train), ("test", outer_test)):
            for group in sorted(set(groups[indices])):
                assignments.append(
                    {
                        "level": "outer",
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": "",
                        "split_role": role,
                        "group_id": group,
                        "ad_count": int(np.count_nonzero(groups[indices] == group)),
                    }
                )

        inner = deterministic_group_folds(
            groups[outer_train], inner_splits, seed + 10_000 * repeat + outer_fold
        )
        for inner_fold, (relative_train, relative_validation) in enumerate(inner):
            inner_train = outer_train[relative_train]
            inner_validation = outer_train[relative_validation]
            train_inner_groups = set(groups[inner_train])
            validation_groups = set(groups[inner_validation])
            inner_overlap = train_inner_groups.intersection(validation_groups)
            inner_checks.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_group_count": len(train_inner_groups),
                    "validation_group_count": len(validation_groups),
                    "overlap_count": len(inner_overlap),
                }
            )
            for role, indices in (("train", inner_train), ("validation", inner_validation)):
                for group in sorted(set(groups[indices])):
                    assignments.append(
                        {
                            "level": "inner",
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "split_role": role,
                            "group_id": group,
                            "ad_count": int(np.count_nonzero(groups[indices] == group)),
                        }
                    )

    max_outer_overlap = max(check["overlap_count"] for check in outer_checks)
    max_inner_overlap = max(check["overlap_count"] for check in inner_checks)
    if max_outer_overlap or max_inner_overlap:
        raise EvaluationError("group leakage detected while auditing CV folds")
    return assignments, {
        "outer_checks": outer_checks,
        "inner_checks": inner_checks,
        "maximum_outer_group_overlap": max_outer_overlap,
        "maximum_inner_group_overlap": max_inner_overlap,
    }


def nested_predictions(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    outer_repeats: int,
    inner_splits: int,
    alphas: Sequence[float],
    seed: int,
    identifiers: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise EvaluationError("scikit-learn is required for evaluation") from exc

    repeat_predictions = [np.full(target.size, np.nan, dtype=np.float64) for _ in range(outer_repeats)]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for repeat, fold, train, test in splits:
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        outer_overlap = train_groups.intersection(test_groups)
        if outer_overlap:
            raise EvaluationError(f"outer group leakage detected: {sorted(outer_overlap)}")
        if len(train_groups) < inner_splits:
            raise EvaluationError(
                f"outer training fold has {len(train_groups)} groups, fewer than "
                f"inner_splits={inner_splits}"
            )
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge()),
            ]
        )
        inner = deterministic_group_folds(
            groups[train], inner_splits, seed + 10_000 * repeat + fold
        )
        inner_max_overlap = 0
        for inner_train, inner_validation in inner:
            overlap = set(groups[train][inner_train]).intersection(
                groups[train][inner_validation]
            )
            inner_max_overlap = max(inner_max_overlap, len(overlap))
        if inner_max_overlap:
            raise EvaluationError("inner group leakage detected")
        search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": list(alphas)},
            scoring="neg_mean_squared_error",
            cv=inner,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(matrix[train], target[train])
        predicted = np.asarray(search.predict(matrix[test]), dtype=np.float64)
        if not np.isfinite(predicted).all():
            raise EvaluationError("model produced a non-finite prediction")
        repeat_predictions[repeat][test] = predicted
        fold_metric = calculate_metrics(target[test], predicted)
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "train_n": int(train.size),
                "test_n": int(test.size),
                "train_group_n": len(train_groups),
                "test_group_n": len(test_groups),
                "outer_group_overlap_n": len(outer_overlap),
                "maximum_inner_group_overlap_n": inner_max_overlap,
                "best_alpha": float(search.best_params_["ridge__alpha"]),
                **fold_metric,
            }
        )
        if identifiers is not None:
            for index, prediction in zip(test, predicted):
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "ad_id": identifiers[int(index)],
                        "group_id": groups[int(index)],
                        "y_true": format(float(target[int(index)]), ".17g"),
                        "y_pred": format(float(prediction), ".17g"),
                    }
                )

    repeat_rows: list[dict[str, Any]] = []
    for repeat, predicted in enumerate(repeat_predictions):
        if np.isnan(predicted).any():
            raise EvaluationError(f"outer CV did not predict every ad in repeat {repeat}")
        repeat_rows.append({"repeat": repeat, **calculate_metrics(target, predicted)})
    return repeat_rows, fold_rows, prediction_rows


def parse_alphas(raw: str) -> list[float]:
    try:
        alphas = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise EvaluationError("--alphas must be a comma-separated list of numbers") from exc
    if not alphas or any(not math.isfinite(value) or value <= 0 for value in alphas):
        raise EvaluationError("all --alphas must be finite and positive")
    return alphas


def aggregate_observed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": "" if finite.size == 0 else format(float(np.mean(finite)), ".17g"),
                    "std": ""
                    if finite.size == 0
                    else format(float(np.std(finite, ddof=0)), ".17g"),
                    "finite_repeat_count": int(finite.size),
                }
            )
    return output


def paired_incremental_rows(
    rows: Sequence[Mapping[str, Any]], level: str
) -> list[dict[str, Any]]:
    """Return paired combined-vs-content deltas with positive always meaning improvement."""

    key_fields = ("repeat",) if level == "repeat" else ("repeat", "fold")
    lookup = {
        (str(row["model"]), *(int(row[field]) for field in key_fields)): row for row in rows
    }
    keys = sorted(
        {
            tuple(int(row[field]) for field in key_fields)
            for row in rows
            if row["model"] in {"content_only", "combined"}
        }
    )
    output: list[dict[str, Any]] = []
    for key in keys:
        content = lookup.get(("content_only", *key))
        combined = lookup.get(("combined", *key))
        if content is None or combined is None:
            raise EvaluationError(f"missing paired model result at {level} key {key}")
        for metric in METRIC_NAMES:
            content_value = float(content[metric])
            combined_value = float(combined[metric])
            raw_delta = combined_value - content_value
            improvement = raw_delta if metric in {"pearson", "spearman"} else -raw_delta
            output.append(
                {
                    "level": level,
                    "repeat": key[0],
                    "fold": "" if level == "repeat" else key[1],
                    "metric": metric,
                    "content_only_value": content_value,
                    "combined_value": combined_value,
                    "raw_combined_minus_content": raw_delta,
                    "improvement_positive": improvement,
                }
            )
    return output


def incremental_repeat_summary(
    rows: Sequence[Mapping[str, Any]], lower_percentile: float = 2.5, upper_percentile: float = 97.5
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        values = np.asarray(
            [
                float(row["improvement_positive"])
                for row in rows
                if row["level"] == "repeat" and row["metric"] == metric
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        output.append(
            {
                "metric": metric,
                "direction": "positive_is_combined_better_than_content_only",
                "repeat_count": int(values.size),
                "mean_improvement": ""
                if values.size == 0
                else format(float(np.mean(values)), ".17g"),
                "median_improvement": ""
                if values.size == 0
                else format(float(np.median(values)), ".17g"),
                "descriptive_percentile_lower": ""
                if values.size == 0
                else format(float(np.percentile(values, lower_percentile)), ".17g"),
                "descriptive_percentile_upper": ""
                if values.size == 0
                else format(float(np.percentile(values, upper_percentile)), ".17g"),
                "percentile_interval_label": (
                    "descriptive_across_repeats_not_an_independence_based_confidence_interval_or_p_value"
                ),
                "lower_percentile": lower_percentile,
                "upper_percentile": upper_percentile,
            }
        )
    return output


def control_summary(
    observed: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    observed_lookup: dict[tuple[str, str], float] = {}
    for model in sorted({str(row["model"]) for row in observed}):
        model_rows = [row for row in observed if row["model"] == model]
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            observed_lookup[(model, metric)] = float(np.mean(values)) if values.size else float("nan")

    pairs = sorted({(str(row["control"]), str(row["model"])) for row in controls})
    for control, model in pairs:
        subset = [row for row in controls if row["control"] == control and row["model"] == model]
        control_indices = sorted({int(row["control_index"]) for row in subset})
        for metric in METRIC_NAMES:
            null_values: list[float] = []
            for control_index in control_indices:
                repeat_values = np.asarray(
                    [
                        float(row[metric])
                        for row in subset
                        if int(row["control_index"]) == control_index
                    ],
                    dtype=np.float64,
                )
                repeat_values = repeat_values[np.isfinite(repeat_values)]
                if repeat_values.size:
                    null_values.append(float(np.mean(repeat_values)))
            null = np.asarray(null_values, dtype=np.float64)
            observed_value = observed_lookup.get((model, metric), float("nan"))
            if null.size == 0 or not math.isfinite(observed_value):
                p_value: Union[float, str] = ""
            elif metric in {"pearson", "spearman"}:
                p_value = float((1 + np.count_nonzero(null >= observed_value)) / (1 + null.size))
            else:
                p_value = float((1 + np.count_nonzero(null <= observed_value)) / (1 + null.size))
            output.append(
                {
                    "control": control,
                    "model": model,
                    "metric": metric,
                    "observed_mean": ""
                    if not math.isfinite(observed_value)
                    else format(observed_value, ".17g"),
                    "null_mean": "" if null.size == 0 else format(float(np.mean(null)), ".17g"),
                    "null_std": "" if null.size == 0 else format(float(np.std(null)), ".17g"),
                    "empirical_one_sided_p": "" if p_value == "" else format(float(p_value), ".17g"),
                    "control_count": int(null.size),
                }
            )
    return output


def run(args: argparse.Namespace) -> Path:
    if not args.acknowledge_target_lineage:
        raise EvaluationError(
            "refusing to evaluate without --acknowledge-target-lineage; this flag confirms "
            "that the selected target's source and semantics were reviewed outside this script"
        )
    for name in ("outer_splits", "outer_repeats", "inner_splits"):
        if getattr(args, name) < 2:
            raise EvaluationError(f"--{name.replace('_', '-')} must be at least 2")
    if args.target_permutations <= 0 or args.shuffled_brain_repeats <= 0:
        raise EvaluationError("both control counts must be positive")
    alphas = parse_alphas(args.alphas)

    cohort_path = Path(args.cohort).expanduser().resolve()
    brain_path = Path(args.brain_features).expanduser().resolve()
    content_path = Path(args.content_features).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (cohort_path, brain_path, content_path):
        if not path.is_file():
            raise EvaluationError(f"input is not a file: {path}")
    if output_dir.exists():
        raise EvaluationError(f"refusing to replace existing output directory: {output_dir}")

    target_ids, target, target_groups, target_profile = load_target(
        cohort_path,
        args.id_column,
        args.target_column,
        args.group_column,
        args.missing_target_policy,
    )
    brain_by_id, brain_columns, brain_profile = load_features(
        brain_path, args.brain_id_column, args.brain_feature_regex, "brain features"
    )
    content_by_id, content_columns, content_profile = load_features(
        content_path, args.content_id_column, args.content_feature_regex, "content features"
    )
    target_lookup = {identifier: value for identifier, value in zip(target_ids, target)}
    group_lookup = {identifier: group for identifier, group in zip(target_ids, target_groups)}
    missing_brain = [identifier for identifier in target_ids if identifier not in brain_by_id]
    missing_content = [identifier for identifier in target_ids if identifier not in content_by_id]
    if (missing_brain or missing_content) and not args.allow_id_intersection:
        raise EvaluationError(
            f"feature tables do not cover the eligible cohort: missing brain={len(missing_brain)}, "
            f"missing content={len(missing_content)}. Use --allow-id-intersection only after review."
        )
    analysis_ids = [
        identifier
        for identifier in target_ids
        if identifier in brain_by_id and identifier in content_by_id
    ]
    if not analysis_ids:
        raise EvaluationError("brain/content/cohort ID intersection is empty")
    target = np.asarray([target_lookup[identifier] for identifier in analysis_ids], dtype=np.float64)
    groups = np.asarray([group_lookup[identifier] for identifier in analysis_ids], dtype=str)
    brain = np.vstack([brain_by_id[identifier] for identifier in analysis_ids])
    content = np.vstack([content_by_id[identifier] for identifier in analysis_ids])
    distinct_group_count = np.unique(groups).size
    if distinct_group_count < args.outer_splits:
        raise EvaluationError("fewer distinct groups than outer CV splits")

    splits = make_outer_splits(
        groups, args.outer_splits, args.outer_repeats, args.seed
    )
    group_assignments, group_audit = group_cv_audit(
        groups, splits, args.inner_splits, args.seed
    )
    model_matrices = {
        "content_only": content,
        "brain_only": brain,
        "combined": np.concatenate([content, brain], axis=1),
    }
    observed_repeat: list[dict[str, Any]] = []
    observed_fold: list[dict[str, Any]] = []
    observed_predictions: list[dict[str, Any]] = []
    for model, matrix in model_matrices.items():
        repeat_rows, fold_rows, prediction_rows = nested_predictions(
            matrix,
            target,
            groups,
            splits,
            args.outer_repeats,
            args.inner_splits,
            alphas,
            args.seed,
            identifiers=analysis_ids,
        )
        observed_repeat.extend({"model": model, **row} for row in repeat_rows)
        observed_fold.extend({"model": model, **row} for row in fold_rows)
        observed_predictions.extend({"model": model, **row} for row in prediction_rows)

    control_rows: list[dict[str, Any]] = []
    for control_index in range(args.target_permutations):
        rng = np.random.default_rng(args.seed + 1_000_000 + control_index)
        permuted_target = target[rng.permutation(target.size)]
        for model, matrix in model_matrices.items():
            repeat_rows, _, _ = nested_predictions(
                matrix,
                permuted_target,
                groups,
                splits,
                args.outer_repeats,
                args.inner_splits,
                alphas,
                args.seed,
            )
            control_rows.extend(
                {
                    "control": "target_permutation",
                    "control_index": control_index,
                    "model": model,
                    **row,
                }
                for row in repeat_rows
            )

    for control_index in range(args.shuffled_brain_repeats):
        rng = np.random.default_rng(args.seed + 2_000_000 + control_index)
        shuffled_brain = brain[rng.permutation(brain.shape[0])]
        shuffled_models = {
            "brain_only": shuffled_brain,
            "combined": np.concatenate([content, shuffled_brain], axis=1),
        }
        for model, matrix in shuffled_models.items():
            repeat_rows, _, _ = nested_predictions(
                matrix,
                target,
                groups,
                splits,
                args.outer_repeats,
                args.inner_splits,
                alphas,
                args.seed,
            )
            control_rows.extend(
                {
                    "control": "shuffled_brain_rows",
                    "control_index": control_index,
                    "model": model,
                    **row,
                }
                for row in repeat_rows
            )

    aggregate_rows = aggregate_observed(observed_repeat)
    summary_rows = control_summary(observed_repeat, control_rows)
    paired_repeat_rows = paired_incremental_rows(observed_repeat, "repeat")
    paired_fold_rows = paired_incremental_rows(observed_fold, "fold")
    incremental_summary_rows = incremental_repeat_summary(paired_repeat_rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        repeat_fields = ["model", "repeat", *METRIC_NAMES]
        fold_fields = [
            "model",
            "repeat",
            "fold",
            "train_n",
            "test_n",
            "train_group_n",
            "test_group_n",
            "outer_group_overlap_n",
            "maximum_inner_group_overlap_n",
            "best_alpha",
            *METRIC_NAMES,
        ]
        prediction_fields = [
            "model",
            "repeat",
            "fold",
            "ad_id",
            "group_id",
            "y_true",
            "y_pred",
        ]
        control_fields = ["control", "control_index", "model", "repeat", *METRIC_NAMES]
        write_csv(staging / "observed_repeat_metrics.csv", repeat_fields, observed_repeat)
        write_csv(staging / "observed_fold_metrics.csv", fold_fields, observed_fold)
        write_csv_gz(
            staging / "observed_predictions.csv.gz", prediction_fields, observed_predictions
        )
        write_csv_gz(staging / "control_metrics.csv.gz", control_fields, control_rows)
        write_csv_gz(
            staging / "group_fold_assignments.csv.gz",
            [
                "level",
                "repeat",
                "outer_fold",
                "inner_fold",
                "split_role",
                "group_id",
                "ad_count",
            ],
            group_assignments,
        )
        write_csv(
            staging / "observed_metric_summary.csv",
            ["model", "metric", "mean", "std", "finite_repeat_count"],
            aggregate_rows,
        )
        write_csv(
            staging / "control_summary.csv",
            [
                "control",
                "model",
                "metric",
                "observed_mean",
                "null_mean",
                "null_std",
                "empirical_one_sided_p",
                "control_count",
            ],
            summary_rows,
        )
        paired_fields = [
            "level",
            "repeat",
            "fold",
            "metric",
            "content_only_value",
            "combined_value",
            "raw_combined_minus_content",
            "improvement_positive",
        ]
        write_csv(staging / "paired_incremental_repeat.csv", paired_fields, paired_repeat_rows)
        write_csv(staging / "paired_incremental_fold.csv", paired_fields, paired_fold_rows)
        write_csv(
            staging / "paired_incremental_repeat_summary.csv",
            [
                "metric",
                "direction",
                "repeat_count",
                "mean_improvement",
                "median_improvement",
                "descriptive_percentile_lower",
                "descriptive_percentile_upper",
                "percentile_interval_label",
                "lower_percentile",
                "upper_percentile",
            ],
            incremental_summary_rows,
        )
        feature_manifest = [
            {"feature_set": "brain", "feature_index": index, "source_column": column}
            for index, column in enumerate(brain_columns)
        ] + [
            {"feature_set": "content", "feature_index": index, "source_column": column}
            for index, column in enumerate(content_columns)
        ]
        write_csv(
            staging / "feature_manifest.csv",
            ["feature_set", "feature_index", "source_column"],
            feature_manifest,
        )

        try:
            import sklearn

            sklearn_version = sklearn.__version__
        except ImportError:
            sklearn_version = "unknown"
        script_path = Path(__file__).resolve()
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "target": {
                "column": args.target_column,
                "lineage_acknowledged": True,
                "lineage_note": args.target_lineage_note,
                "semantics_inferred_by_script": False,
                **target_profile,
            },
            "inputs": {
                "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
                "brain_features": {
                    "path": str(brain_path),
                    "sha256": sha256_file(brain_path),
                    **brain_profile,
                },
                "content_features": {
                    "path": str(content_path),
                    "sha256": sha256_file(content_path),
                    **content_profile,
                },
            },
            "analysis_cohort": {
                "ad_count": len(analysis_ids),
                "ad_ids": analysis_ids,
                "group_column": args.group_column,
                "distinct_group_count": int(distinct_group_count),
                "missing_brain_ids": missing_brain,
                "missing_content_ids": missing_content,
                "id_intersection_allowed": args.allow_id_intersection,
            },
            "model": {
                "estimator": "median imputation + standard scaling + ridge regression",
                "alphas": alphas,
                "outer_splits": args.outer_splits,
                "outer_repeats": args.outer_repeats,
                "inner_splits": args.inner_splits,
                "seed": args.seed,
                "selection_metric": "negative_mean_squared_error",
                "n_jobs": 1,
                "models": list(model_matrices),
                "ad_level_split": True,
                "group_aware_split": True,
                "group_split_algorithm": "seeded shuffled greedy sample-balanced whole-group folds",
                "ordinary_kfold_fallback": False,
            },
            "group_cv_audit": group_audit,
            "controls": {
                "target_permutations": args.target_permutations,
                "shuffled_brain_repeats": args.shuffled_brain_repeats,
            },
            "incremental_value": {
                "comparison": "combined_minus_content_only",
                "improvement_sign": {
                    "pearson": "combined - content_only",
                    "spearman": "combined - content_only",
                    "mae": "content_only - combined",
                    "rmse": "content_only - combined",
                },
                "interval": (
                    "2.5th to 97.5th descriptive percentiles across repeats; repeats are not "
                    "treated as independent for a p-value"
                ),
            },
            "software": {"numpy": np.__version__, "scikit_learn": sklearn_version},
        }
        dump_json(staging / "metadata.json", metadata)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cohort", required=True)
    result.add_argument("--brain-features", required=True)
    result.add_argument("--content-features", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--id-column", default="video_id")
    result.add_argument("--brain-id-column", default="video_id")
    result.add_argument("--content-id-column", default="video_id")
    result.add_argument("--target-column", required=True)
    result.add_argument(
        "--group-column",
        required=True,
        help="Cohort column defining related ads that must never cross a CV boundary",
    )
    result.add_argument("--brain-feature-regex", default=r"^brain__")
    result.add_argument("--content-feature-regex", default=r"^content__")
    result.add_argument("--missing-target-policy", choices=("error", "drop"), default="error")
    result.add_argument("--allow-id-intersection", action="store_true")
    result.add_argument("--acknowledge-target-lineage", action="store_true")
    result.add_argument("--target-lineage-note")
    result.add_argument("--outer-splits", type=int, default=5)
    result.add_argument("--outer-repeats", type=int, default=5)
    result.add_argument("--inner-splits", type=int, default=5)
    result.add_argument("--alphas", default="0.01,0.1,1,10,100")
    result.add_argument("--seed", type=int, default=20260822)
    result.add_argument("--target-permutations", type=int, required=True)
    result.add_argument("--shuffled-brain-repeats", type=int, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = run(args)
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
