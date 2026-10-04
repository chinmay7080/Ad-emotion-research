#!/usr/bin/env python3
"""Create cognitive map-similarity time series and temporal ROI/process features."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCRIPT_VERSION = "1.0"
STATISTICS = (
    "mean", "maximum", "minimum", "std", "auc", "time_to_peak",
    "first_third_mean", "middle_third_mean", "final_third_mean", "slope", "peak_count",
)


class FeatureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not token:
        raise FeatureError(f"cannot form feature token from {value!r}")
    return token


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    means = np.mean(matrix, axis=1, keepdims=True)
    stds = np.std(matrix, axis=1, ddof=0, keepdims=True)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0):
        raise FeatureError("cannot spatially standardize a constant or non-finite pattern")
    return (matrix - means) / stds


def temporal_features(values: np.ndarray, times: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or times.shape != values.shape or values.size == 0:
        raise FeatureError("temporal series and time axis must be equal nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise FeatureError("temporal series contains NaN or infinity")
    if np.any(np.diff(times) < 0):
        raise FeatureError("time axis must be nondecreasing")
    peak_index = int(np.argmax(values))
    thirds = np.array_split(values, 3)
    if any(part.size == 0 for part in thirds):
        raise FeatureError("at least three timepoints are required for thirds")
    centered_t = times - np.mean(times)
    denominator = float(np.dot(centered_t, centered_t))
    slope = 0.0 if denominator == 0.0 else float(np.dot(centered_t, values - np.mean(values)) / denominator)
    std = float(np.std(values, ddof=0))
    peak_count = 0
    if values.size >= 3 and std > 0:
        try:
            from scipy.signal import find_peaks
        except ImportError as exc:
            raise FeatureError("scipy is required for reproducible peak counting") from exc
        positive_steps = np.diff(times)
        positive_steps = positive_steps[positive_steps > 0]
        distance = 1 if positive_steps.size == 0 else max(1, int(math.ceil(2.0 / float(np.median(positive_steps)))))
        peaks, _ = find_peaks(values, prominence=0.5 * std, distance=distance)
        peak_count = int(peaks.size)
    return {
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "minimum": float(np.min(values)),
        "std": std,
        "auc": 0.0 if values.size == 1 else float(np.trapz(values, x=times)),
        "time_to_peak": float(times[peak_index] - times[0]),
        "first_third_mean": float(np.mean(thirds[0])),
        "middle_third_mean": float(np.mean(thirds[1])),
        "final_third_mean": float(np.mean(thirds[2])),
        "slope": slope,
        "peak_count": float(peak_count),
    }


def load_parcellation_module() -> Any:
    path = Path(__file__).with_name("parcellate_hcp_mmp1.py")
    spec = importlib.util.spec_from_file_location("tribe_parcellation", path)
    if spec is None or spec.loader is None:
        raise FeatureError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_maps(
    component_parcels: np.ndarray,
    component_names: Sequence[str],
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    component_z = zscore_rows(component_parcels)
    lookup = {name: component_z[index] for index, name in enumerate(component_names)}
    rows: list[np.ndarray] = []
    names: list[str] = []
    for item in manifest.get("processes", []):
        name = str(item["name"])
        components = [str(value) for value in item["components"]]
        weights = np.asarray(item["weights"], dtype=np.float64)
        if len(components) != weights.size or not np.isfinite(weights).all() or float(np.sum(weights)) == 0:
            raise FeatureError(f"invalid component weights for process {name}")
        try:
            matrix = np.stack([lookup[value] for value in components], axis=0)
        except KeyError as exc:
            raise FeatureError(f"unknown component in process {name}: {exc}") from exc
        composite = np.average(matrix, axis=0, weights=weights)
        rows.append(zscore_rows(composite[None, :])[0])
        names.append(name)
    if len(rows) != 9 or len(names) != len(set(names)):
        raise FeatureError("manifest must define exactly nine unique process maps")
    return np.stack(rows), names


def run(args: argparse.Namespace) -> Path:
    parcellation_run = Path(args.parcellation_run).expanduser().resolve()
    component_npz = Path(args.component_maps).expanduser().resolve()
    map_manifest_path = Path(args.map_manifest).expanduser().resolve()
    left_annot = Path(args.left_annot).expanduser().resolve()
    right_annot = Path(args.right_annot).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for path in (component_npz, map_manifest_path, left_annot, right_annot):
        if not path.is_file():
            raise FeatureError(f"required input is not a file: {path}")
    if not (parcellation_run / "ads").is_dir():
        raise FeatureError(f"parcellation run has no ads directory: {parcellation_run}")
    manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    with np.load(component_npz, allow_pickle=False) as payload:
        component_vertices = np.asarray(payload["component_maps"], dtype=np.float64)
        component_names = [str(value) for value in payload["component_names"].tolist()]
    if component_vertices.shape != (len(component_names), 20484):
        raise FeatureError(f"component map shape is invalid: {component_vertices.shape}")

    parcellation = load_parcellation_module()
    try:
        parcels, atlas_meta = parcellation.build_atlas(
            left_annot, right_annot, "exclude", args.expected_parcels
        )
        component_parcels = parcellation.parcellate_matrix(component_vertices, parcels)
    except Exception as exc:
        raise FeatureError(f"map parcellation failed: {exc}") from exc
    fixed_process_maps, process_names = process_maps(component_parcels, component_names, manifest)
    parcel_names = [str(item["parcel_name"]) for item in parcels]
    parcel_tokens = [safe_token(value) for value in parcel_names]
    process_tokens = [safe_token(value) for value in process_names]
    if len(parcel_tokens) != len(set(parcel_tokens)):
        raise FeatureError("parcel names collide after tokenization")

    candidates = sorted(path for path in (parcellation_run / "ads").iterdir() if path.is_dir())
    if args.video_id:
        lookup = {path.name: path for path in candidates}
        missing = [value for value in args.video_id if value not in lookup]
        if missing:
            raise FeatureError(f"unknown requested video IDs: {missing[:10]}")
        candidates = [lookup[value] for value in dict.fromkeys(args.video_id)]
    if not candidates:
        raise FeatureError("no ads selected")

    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("brain-features-%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / run_id
    if final.exists():
        raise FeatureError(f"refusing to replace existing output directory: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        ads_output = staging / "ads"
        ads_output.mkdir()
        fields = ["video_id", "n_timepoints", "first_segment_time", "last_segment_time"]
        fields += [f"brain__roi__{token}__{stat}" for token in parcel_tokens for stat in STATISTICS]
        fields += [f"brain__map_similarity__{token}__{stat}" for token in process_tokens for stat in STATISTICS]
        summary_path = staging / "brain_temporal_features.csv.gz"
        qc_rows: list[dict[str, Any]] = []
        with gzip.open(summary_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for ad_dir in candidates:
                video_id = ad_dir.name
                roi_path = ad_dir / "parcellated.npy"
                axis_path = ad_dir / "hcp_mmp1_timeseries.float16.npz"
                if not roi_path.is_file() or not axis_path.is_file():
                    raise FeatureError(f"missing ROI data or time axis for {video_id}")
                roi = np.asarray(np.load(roi_path, allow_pickle=False), dtype=np.float64)
                with np.load(axis_path, allow_pickle=False) as axis_payload:
                    times = np.asarray(axis_payload["segment_time"], dtype=np.float64)
                if roi.shape != (times.size, args.expected_parcels) or times.size < 3:
                    raise FeatureError(f"invalid ROI/time shape for {video_id}: {roi.shape}, T={times.size}")
                similarity = zscore_rows(roi) @ fixed_process_maps.T / float(args.expected_parcels)
                if not np.isfinite(similarity).all() or np.max(np.abs(similarity)) > 1.000001:
                    raise FeatureError(f"invalid spatial correlation for {video_id}")
                out_dir = ads_output / video_id
                out_dir.mkdir()
                np.save(out_dir / "functional_map_similarity.float32.npy", similarity.astype(np.float32), allow_pickle=False)
                np.savez_compressed(
                    out_dir / "functional_map_similarity.float16.npz",
                    scores=similarity.astype(np.float16),
                    segment_time=times,
                    process_names=np.asarray(process_names),
                )
                row: dict[str, Any] = {
                    "video_id": video_id,
                    "n_timepoints": times.size,
                    "first_segment_time": format(float(times[0]), ".17g"),
                    "last_segment_time": format(float(times[-1]), ".17g"),
                }
                for index, token in enumerate(parcel_tokens):
                    for statistic, value in temporal_features(roi[:, index], times).items():
                        row[f"brain__roi__{token}__{statistic}"] = format(value, ".17g")
                for index, token in enumerate(process_tokens):
                    for statistic, value in temporal_features(similarity[:, index], times).items():
                        row[f"brain__map_similarity__{token}__{statistic}"] = format(value, ".17g")
                writer.writerow(row)
                qc_rows.append(
                    {
                        "video_id": video_id,
                        "n_timepoints": int(times.size),
                        "similarity_min": float(np.min(similarity)),
                        "similarity_max": float(np.max(similarity)),
                        "similarity_float16_max_abs_error": float(np.max(np.abs(similarity - similarity.astype(np.float16).astype(np.float64)))),
                    }
                )

        np.savez_compressed(
            staging / "cognitive_maps_hcp_mmp1.float32.npz",
            component_maps=component_parcels.astype(np.float32),
            component_names=np.asarray(component_names),
            process_maps=fixed_process_maps.astype(np.float32),
            process_names=np.asarray(process_names),
            parcel_names=np.asarray(parcel_names),
        )
        with (staging / "qc.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
            writer.writeheader()
            writer.writerows(qc_rows)
        dictionary = {
            "schema_version": 1,
            "feature_count": len(fields) - 4,
            "roi_count": len(parcel_names),
            "process_count": len(process_names),
            "statistics": list(STATISTICS),
            "definitions": {
                "map_similarity": "Pearson spatial correlation across 360 HCP-MMP1 parcel means after within-pattern z-standardization",
                "std": "population standard deviation across model timepoints (ddof=0)",
                "auc": "signed trapezoidal area under the curve over the saved segment_time axis",
                "time_to_peak": "earliest maximum time minus the first saved segment time, in seconds",
                "thirds": "means of three chronological contiguous index chunks made by numpy.array_split",
                "slope": "ordinary least-squares slope of value against saved segment_time, units per second",
                "peak_count": "scipy.signal.find_peaks count with prominence 0.5 times within-ad population SD and minimum temporal separation 2 seconds",
            },
            "interpretation_guardrail": manifest.get("interpretation_guardrail"),
        }
        dump_json(staging / "feature_dictionary.json", dictionary)
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ads": len(candidates),
            "inputs": {
                "parcellation_run": str(parcellation_run),
                "component_maps": {"path": str(component_npz), "sha256": sha256_file(component_npz)},
                "map_manifest": {"path": str(map_manifest_path), "sha256": sha256_file(map_manifest_path)},
                "left_annotation": {"path": str(left_annot), "sha256": sha256_file(left_annot)},
                "right_annotation": {"path": str(right_annot), "sha256": sha256_file(right_annot)},
                "atlas": atlas_meta,
            },
            "outputs": {
                "feature_table": summary_path.name,
                "feature_table_sha256": sha256_file(summary_path),
                "per_ad_float32": "ads/<video_id>/functional_map_similarity.float32.npy",
                "per_ad_compressed_float16": "ads/<video_id>/functional_map_similarity.float16.npz",
            },
            **dictionary,
        }
        dump_json(staging / "metadata.json", metadata)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parcellation-run", required=True)
    result.add_argument("--component-maps", required=True)
    result.add_argument("--map-manifest", required=True)
    result.add_argument("--left-annot", required=True)
    result.add_argument("--right-annot", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--run-id")
    result.add_argument("--expected-parcels", type=int, default=360)
    result.add_argument("--video-id", action="append")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        output = run(parser().parse_args(argv))
    except FeatureError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
