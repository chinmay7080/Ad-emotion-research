#!/usr/bin/env python3
"""Evaluate brain-only and content+brain ad-effectiveness models without ad leakage."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


SCRIPT_VERSION = "1.0"
METRICS = ("pearson", "spearman", "mae", "rmse")
FAMILIES = ("ridge", "elastic_net", "gradient_boosting")
PRIMARY_FAMILY = "ridge"


class EvaluationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    return {
        "pearson": correlation(target, prediction),
        "spearman": correlation(average_ranks(target), average_ranks(prediction)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def positive_improvement(metric: str, challenger: float, reference: float) -> float:
    if metric in {"pearson", "spearman"}:
        return challenger - reference
    if metric in {"mae", "rmse"}:
        return reference - challenger
    raise EvaluationError(f"unsupported metric: {metric}")


def deterministic_group_folds(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=str)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if groups.ndim != 1 or groups.size == 0 or unique.size < n_splits:
        raise EvaluationError("insufficient nonempty groups for requested folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique.size)
    ordered = shuffled[np.argsort(-counts[shuffled], kind="mergesort")]
    sizes = np.zeros(n_splits, dtype=np.int64)
    assignment = np.full(unique.size, -1, dtype=np.int64)
    for group_index in ordered:
        candidates = np.flatnonzero(sizes == np.min(sizes))
        chosen = int(candidates[int(rng.integers(0, candidates.size))])
        assignment[group_index] = chosen
        sizes[chosen] += counts[group_index]
    sample_folds = assignment[inverse]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        test = np.flatnonzero(sample_folds == fold)
        train = np.flatnonzero(sample_folds != fold)
        if not train.size or not test.size:
            raise EvaluationError("fold construction produced an empty split")
        if set(groups[train]).intersection(groups[test]):
            raise EvaluationError("group leakage in deterministic fold construction")
        result.append((train, test))
    return result


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_reference_folds(path: Path) -> dict[str, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise EvaluationError(f"cannot read Model A predictions: {exc}") from exc
    required = {"fold", "video_id", "group_id", "effective", "prediction"}
    if not rows or not required.issubset(rows[0]):
        raise EvaluationError("Model A prediction schema is incomplete")
    result: dict[str, int] = {}
    for row in rows:
        video_id = row["video_id"]
        if not video_id or video_id in result:
            raise EvaluationError("Model A video IDs must be nonblank and unique")
        if row["group_id"] != video_id:
            raise EvaluationError("Model A group_id must equal video_id for ad-level CV")
        try:
            fold = int(row["fold"])
        except ValueError as exc:
            raise EvaluationError("Model A fold must be an integer") from exc
        result[video_id] = fold
    folds = sorted(set(result.values()))
    if folds != [0, 1, 2, 3, 4]:
        raise EvaluationError(f"expected Model A folds 0..4, found {folds}")
    counts = [sum(value == fold for value in result.values()) for fold in folds]
    if counts != [373, 373, 373, 373, 373]:
        raise EvaluationError(f"unexpected Model A fold sizes: {counts}")
    return result


def load_frames(args: argparse.Namespace):
    try:
        import pandas as pd
    except ImportError as exc:
        raise EvaluationError("pandas is required") from exc

    cohort_path = Path(args.cohort).expanduser().resolve()
    brain_path = Path(args.brain_features).expanduser().resolve()
    content_path = Path(args.content_features).expanduser().resolve()
    model_a_path = Path(args.model_a_predictions).expanduser().resolve()
    for path in (cohort_path, brain_path, content_path, model_a_path):
        if not path.is_file():
            raise EvaluationError(f"input does not exist: {path}")

    reference_folds = read_reference_folds(model_a_path)
    cohort = pd.read_csv(
        cohort_path,
        usecols=[args.id_column, args.target_column],
        dtype={args.id_column: str},
    )
    if cohort[args.id_column].isna().any() or cohort[args.id_column].duplicated().any():
        raise EvaluationError("cohort IDs must be complete and unique")
    if set(cohort[args.id_column]) != set(reference_folds):
        raise EvaluationError("cohort IDs do not exactly match Model A fold IDs")
    target = np.asarray(cohort[args.target_column], dtype=np.float64)
    if not np.isfinite(target).all() or np.any(target != np.floor(target)):
        raise EvaluationError("target must be finite integer values")
    if np.any((target < 1) | (target > 5)):
        raise EvaluationError("target must be in [1,5]")

    ids = cohort[args.id_column].astype(str).to_numpy()
    fold_vector = np.asarray([reference_folds[value] for value in ids], dtype=np.int64)
    groups = ids.copy()

    brain_header = pd.read_csv(brain_path, nrows=0).columns.tolist()
    brain_columns = [str(value) for value in brain_header if str(value).startswith("brain__")]
    if not brain_columns:
        raise EvaluationError("no brain__ columns found")
    brain_frame = pd.read_csv(
        brain_path,
        usecols=[args.brain_id_column, *brain_columns],
        dtype={args.brain_id_column: str},
    ).set_index(args.brain_id_column)
    if brain_frame.index.has_duplicates or set(brain_frame.index) != set(ids):
        raise EvaluationError("brain feature IDs do not exactly match the cohort")
    brain = brain_frame.loc[ids, brain_columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(brain).all():
        raise EvaluationError("brain features contain NaN or infinity")

    content_columns: list[str] = []
    content: Optional[np.ndarray] = None
    if args.feature_set == "combined":
        content_header = pd.read_csv(content_path, nrows=0).columns.tolist()
        content_columns = [
            str(value) for value in content_header if str(value).startswith("content__")
        ]
        if not content_columns:
            raise EvaluationError("no content__ columns found")
        content_frame = pd.read_csv(
            content_path,
            usecols=[args.content_id_column, *content_columns],
            dtype={args.content_id_column: str},
        ).set_index(args.content_id_column)
        if content_frame.index.has_duplicates or set(content_frame.index) != set(ids):
            raise EvaluationError("content feature IDs do not exactly match the cohort")
        content = content_frame.loc[ids, content_columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(content).all():
            raise EvaluationError("content features contain NaN or infinity")
        matrix = np.concatenate([content, brain], axis=1)
        feature_columns = [f"content:{value}" for value in content_columns] + [
            f"brain:{value}" for value in brain_columns
        ]
    else:
        matrix = brain
        feature_columns = [f"brain:{value}" for value in brain_columns]

    outer_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(5):
        test = np.flatnonzero(fold_vector == fold)
        train = np.flatnonzero(fold_vector != fold)
        if test.size != 373 or train.size != 1492:
            raise EvaluationError("reference outer fold sizes changed")
        if set(groups[train]).intersection(groups[test]):
            raise EvaluationError("outer ad-group leakage detected")
        outer_splits.append((train, test))

    paths = {
        "cohort": cohort_path,
        "brain_features": brain_path,
        "content_features": content_path,
        "model_a_predictions": model_a_path,
    }
    return ids, target, groups, fold_vector, matrix, feature_columns, outer_splits, paths


def make_search(
    family: str,
    inner: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    search_jobs: int,
    feature_count: int,
):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise EvaluationError("scikit-learn is required") from exc

    if family == "ridge":
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(solver="lsqr", tol=1e-3)),
            ]
        )
        grid: Any = {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif family == "elastic_net":
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        max_iter=3000,
                        tol=3e-3,
                        selection="cyclic",
                        random_state=seed,
                    ),
                ),
            ]
        )
        grid = {
            "model__alpha": [0.03, 0.3, 3.0],
            "model__l1_ratio": [0.25, 0.75],
        }
    elif family == "gradient_boosting":
        selected = min(256, feature_count)
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_regression, k=selected)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        subsample=0.8,
                        random_state=seed,
                        loss="squared_error",
                    ),
                ),
            ]
        )
        grid = {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2],
            "model__min_samples_leaf": [10],
        }
    else:
        raise EvaluationError(f"unknown family: {family}")

    return GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="neg_mean_squared_error",
        cv=list(inner),
        n_jobs=search_jobs,
        pre_dispatch=search_jobs,
        refit=True,
        error_score="raise",
        verbose=1,
    )


def evaluate(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise EvaluationError(f"refusing to replace existing output directory: {output_dir}")
    (
        ids,
        target,
        groups,
        fold_vector,
        matrix,
        feature_columns,
        outer_splits,
        paths,
    ) = load_frames(args)

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        predicted = np.full(target.size, np.nan, dtype=np.float64)
        for fold, (train, test) in enumerate(outer_splits):
            inner = deterministic_group_folds(
                groups[train], args.inner_splits, args.seed + 10_000 + fold
            )
            maximum_inner_overlap = max(
                len(
                    set(groups[train][inner_train]).intersection(
                        groups[train][inner_validation]
                    )
                )
                for inner_train, inner_validation in inner
            )
            if maximum_inner_overlap:
                raise EvaluationError("inner ad-group leakage detected")
            print(
                f"START family={family} fold={fold} train={train.size} "
                f"test={test.size} features={matrix.shape[1]}",
                flush=True,
            )
            search = make_search(
                family,
                inner,
                args.seed + fold,
                args.search_jobs,
                matrix.shape[1],
            )
            search.fit(matrix[train], target[train])
            fold_prediction = np.asarray(search.predict(matrix[test]), dtype=np.float64)
            if not np.isfinite(fold_prediction).all():
                raise EvaluationError("model produced non-finite predictions")
            predicted[test] = fold_prediction
            metrics = calculate_metrics(target[test], fold_prediction)
            fold_rows.append(
                {
                    "family": family,
                    "fold": fold,
                    "train_n": int(train.size),
                    "test_n": int(test.size),
                    "train_groups": len(set(groups[train])),
                    "test_groups": len(set(groups[test])),
                    "outer_group_overlap": 0,
                    "maximum_inner_group_overlap": maximum_inner_overlap,
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                    **metrics,
                }
            )
            prediction_rows.extend(
                {
                    "family": family,
                    "fold": fold,
                    "video_id": ids[index],
                    "group_id": groups[index],
                    "effective": float(target[index]),
                    "prediction": float(value),
                }
                for index, value in zip(test, fold_prediction)
            )
            print(
                f"DONE family={family} fold={fold} "
                f"spearman={metrics['spearman']:.6f} rmse={metrics['rmse']:.6f}",
                flush=True,
            )
        if np.isnan(predicted).any():
            raise EvaluationError(f"{family} failed to predict every ad")
        overall_rows.append({"family": family, **calculate_metrics(target, predicted)})

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        metric_fields = list(METRICS)
        write_csv(
            staging / "fold_metrics.csv",
            [
                "family",
                "fold",
                "train_n",
                "test_n",
                "train_groups",
                "test_groups",
                "outer_group_overlap",
                "maximum_inner_group_overlap",
                "best_params_json",
                *metric_fields,
            ],
            fold_rows,
        )
        write_csv(
            staging / "out_of_fold_predictions.csv",
            ["family", "fold", "video_id", "group_id", "effective", "prediction"],
            prediction_rows,
        )
        write_csv(staging / "overall_metrics.csv", ["family", *metric_fields], overall_rows)
        write_csv(
            staging / "split_assignments.csv",
            ["video_id", "group_id", "fold"],
            (
                {"video_id": video_id, "group_id": group, "fold": int(fold)}
                for video_id, group, fold in zip(ids, groups, fold_vector)
            ),
        )
        write_csv(
            staging / "feature_manifest.csv",
            ["feature_index", "feature_set", "source_column"],
            (
                {
                    "feature_index": index,
                    "feature_set": value.split(":", 1)[0],
                    "source_column": value.split(":", 1)[1],
                }
                for index, value in enumerate(feature_columns)
            ),
        )
        try:
            import pandas as pd
            import sklearn

            software = {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            }
        except ImportError:
            software = {"numpy": np.__version__}
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_name": (
                "Model B — Brain only"
                if args.feature_set == "brain"
                else "Model C — Content + Brain"
            ),
            "feature_set": args.feature_set,
            "analysis": {
                "ads": int(target.size),
                "features": int(matrix.shape[1]),
                "families": list(FAMILIES),
                "primary_family": PRIMARY_FAMILY,
            },
            "target": {
                "column": args.target_column,
                "contract": "integer_ordinal_1_to_5",
                "lineage_acknowledged": True,
                "lineage_note": args.target_lineage_note,
            },
            "splitting": {
                "outer_splits": 5,
                "inner_splits": args.inner_splits,
                "seed": args.seed,
                "ad_level": True,
                "group_column": args.id_column,
                "distinct_groups": int(np.unique(groups).size),
                "reference": str(paths["model_a_predictions"]),
                "reference_sha256": sha256_file(paths["model_a_predictions"]),
                "outer_group_overlap_every_fold": 0,
                "maximum_inner_group_overlap": 0,
                "brand_campaign_grouping": (
                    "not applied: cohort contains no reliable brand or campaign identifier; "
                    "title/topic are not treated as validated campaign labels"
                ),
            },
            "models": {
                "ridge": "median imputation + standard scaling + Ridge(lsqr)",
                "elastic_net": "median imputation + standard scaling + ElasticNet",
                "gradient_boosting": (
                    "median imputation + fold-local SelectKBest(f_regression,k<=256) + "
                    "GradientBoostingRegressor"
                ),
                "selection_metric": "inner-fold negative mean squared error",
                "search_jobs": args.search_jobs,
            },
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "software": software,
        }
        dump_json(staging / "metadata.json", metadata)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def read_prediction_table(path: Path, family: Optional[str] = None) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if family is not None:
        rows = [row for row in rows if row.get("family") == family]
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        video_id = row["video_id"]
        if video_id in result:
            raise EvaluationError(f"duplicate prediction for {video_id} in {path}")
        result[video_id] = row
    if len(result) != 1865:
        raise EvaluationError(f"expected 1865 predictions in {path}, found {len(result)}")
    return result


def paired_bootstrap(
    target: np.ndarray,
    reference: np.ndarray,
    challenger: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        sample = rng.integers(0, target.size, size=target.size)
        reference_metrics = calculate_metrics(target[sample], reference[sample])
        challenger_metrics = calculate_metrics(target[sample], challenger[sample])
        for metric in METRICS:
            improvement = positive_improvement(
                metric, challenger_metrics[metric], reference_metrics[metric]
            )
            if math.isfinite(improvement):
                values[metric].append(improvement)
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=np.float64)
        if array.size == 0:
            raise EvaluationError(
                f"bootstrap produced no finite {metric} improvements; "
                "check for constant targets or predictions"
            )
        rows.append(
            {
                "metric": metric,
                "direction": "positive_is_model_c_better_than_model_a",
                "repetitions_requested": repetitions,
                "finite_repetitions": int(array.size),
                "mean_improvement": float(np.mean(array)),
                "median_improvement": float(np.median(array)),
                "percentile_2_5": float(np.percentile(array, 2.5)),
                "percentile_97_5": float(np.percentile(array, 97.5)),
                "fraction_improvement_le_zero": float(np.mean(array <= 0)),
            }
        )
    return rows


def compare(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise EvaluationError(f"refusing to replace existing output directory: {output_dir}")
    model_a_path = Path(args.model_a_predictions).expanduser().resolve()
    model_b_path = Path(args.model_b_predictions).expanduser().resolve()
    model_c_path = Path(args.model_c_predictions).expanduser().resolve()
    for path in (model_a_path, model_b_path, model_c_path):
        if not path.is_file():
            raise EvaluationError(f"prediction input does not exist: {path}")
    model_a = read_prediction_table(model_a_path)
    model_b = read_prediction_table(model_b_path, PRIMARY_FAMILY)
    model_c = read_prediction_table(model_c_path, PRIMARY_FAMILY)
    ids = sorted(model_a)
    if set(ids) != set(model_b) or set(ids) != set(model_c):
        raise EvaluationError("Model A/B/C prediction IDs differ")

    targets: list[float] = []
    predictions: dict[str, list[float]] = {"model_a": [], "model_b": [], "model_c": []}
    comparison_rows: list[dict[str, Any]] = []
    for video_id in ids:
        rows = (model_a[video_id], model_b[video_id], model_c[video_id])
        folds = [int(row["fold"]) for row in rows]
        groups = [row["group_id"] for row in rows]
        target_values = [float(row["effective"]) for row in rows]
        if len(set(folds)) != 1 or folds[0] not in range(5):
            raise EvaluationError(f"fold mismatch for {video_id}")
        if groups != [video_id, video_id, video_id]:
            raise EvaluationError(f"group mismatch for {video_id}")
        if max(target_values) != min(target_values):
            raise EvaluationError(f"target mismatch for {video_id}")
        target = target_values[0]
        target_values_pred = [
            float(model_a[video_id]["prediction"]),
            float(model_b[video_id]["prediction"]),
            float(model_c[video_id]["prediction"]),
        ]
        if not np.isfinite(target_values_pred).all():
            raise EvaluationError("non-finite prediction in comparison")
        targets.append(target)
        for name, value in zip(predictions, target_values_pred):
            predictions[name].append(value)
        comparison_rows.append(
            {
                "video_id": video_id,
                "group_id": video_id,
                "fold": folds[0],
                "effective": target,
                "model_a_prediction": target_values_pred[0],
                "model_b_prediction": target_values_pred[1],
                "model_c_prediction": target_values_pred[2],
            }
        )

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in predictions.items()
    }
    overall_rows = [
        {"model": name, **calculate_metrics(target_array, value)}
        for name, value in prediction_arrays.items()
    ]
    fold_rows: list[dict[str, Any]] = []
    fold_vector = np.asarray([int(row["fold"]) for row in comparison_rows], dtype=np.int64)
    for fold in range(5):
        selected = fold_vector == fold
        for name, value in prediction_arrays.items():
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "test_n": int(np.count_nonzero(selected)),
                    "group_overlap": 0,
                    **calculate_metrics(target_array[selected], value[selected]),
                }
            )
    lookup = {row["model"]: row for row in overall_rows}
    incremental_rows = []
    for metric in METRICS:
        model_a_value = float(lookup["model_a"][metric])
        model_c_value = float(lookup["model_c"][metric])
        incremental_rows.append(
            {
                "metric": metric,
                "model_a": model_a_value,
                "model_c": model_c_value,
                "raw_model_c_minus_model_a": model_c_value - model_a_value,
                "improvement_positive": positive_improvement(
                    metric, model_c_value, model_a_value
                ),
            }
        )
    bootstrap_rows = paired_bootstrap(
        target_array,
        prediction_arrays["model_a"],
        prediction_arrays["model_c"],
        args.bootstrap_repetitions,
        args.seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        write_csv(staging / "overall_metrics.csv", ["model", *METRICS], overall_rows)
        write_csv(
            staging / "fold_metrics.csv",
            ["model", "fold", "test_n", "group_overlap", *METRICS],
            fold_rows,
        )
        write_csv(
            staging / "incremental_model_c_vs_a.csv",
            [
                "metric",
                "model_a",
                "model_c",
                "raw_model_c_minus_model_a",
                "improvement_positive",
            ],
            incremental_rows,
        )
        write_csv(
            staging / "paired_bootstrap_model_c_vs_a.csv",
            [
                "metric",
                "direction",
                "repetitions_requested",
                "finite_repetitions",
                "mean_improvement",
                "median_improvement",
                "percentile_2_5",
                "percentile_97_5",
                "fraction_improvement_le_zero",
            ],
            bootstrap_rows,
        )
        write_csv(
            staging / "prediction_comparison.csv",
            [
                "video_id",
                "group_id",
                "fold",
                "effective",
                "model_a_prediction",
                "model_b_prediction",
                "model_c_prediction",
            ],
            comparison_rows,
        )
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "research_question": "Does Model C outperform Model A on the same unseen ads?",
            "primary_family": PRIMARY_FAMILY,
            "ads": len(ids),
            "outer_folds": 5,
            "fold_sizes": [int(np.count_nonzero(fold_vector == fold)) for fold in range(5)],
            "group_overlap_every_fold": 0,
            "grouping": (
                "whole-ad video_id; no reliable brand/campaign field exists in the cohort, "
                "so brand-disjoint testing was not performed"
            ),
            "bootstrap": {
                "repetitions": args.bootstrap_repetitions,
                "seed": args.seed,
                "interpretation": (
                    "paired ad-level percentile bootstrap of fixed out-of-fold predictions; "
                    "descriptive uncertainty, not a retraining-based generalization p-value"
                ),
            },
            "inputs": {
                "model_a_predictions": {
                    "path": str(model_a_path),
                    "sha256": sha256_file(model_a_path),
                },
                "model_b_predictions": {
                    "path": str(model_b_path),
                    "sha256": sha256_file(model_b_path),
                },
                "model_c_predictions": {
                    "path": str(model_c_path),
                    "sha256": sha256_file(model_c_path),
                },
            },
        }
        dump_json(staging / "metadata.json", metadata)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--feature-set", choices=("brain", "combined"), required=True)
    evaluate_parser.add_argument("--cohort", required=True)
    evaluate_parser.add_argument("--brain-features", required=True)
    evaluate_parser.add_argument("--content-features", required=True)
    evaluate_parser.add_argument("--model-a-predictions", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--id-column", default="video_id")
    evaluate_parser.add_argument("--brain-id-column", default="video_id")
    evaluate_parser.add_argument("--content-id-column", default="video_id")
    evaluate_parser.add_argument("--target-column", default="master__effective")
    evaluate_parser.add_argument("--inner-splits", type=int, default=5)
    evaluate_parser.add_argument("--seed", type=int, default=20260825)
    evaluate_parser.add_argument("--search-jobs", type=int, default=4)
    evaluate_parser.add_argument("--target-lineage-note", required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--model-a-predictions", required=True)
    compare_parser.add_argument("--model-b-predictions", required=True)
    compare_parser.add_argument("--model-c-predictions", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    compare_parser.add_argument("--seed", type=int, default=20260829)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = evaluate(args) if args.mode == "evaluate" else compare(args)
    except EvaluationError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
