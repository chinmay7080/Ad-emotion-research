#!/usr/bin/env python3
"""Parcellate verified fsaverage5 TRIBE predictions with explicit HCP-MMP1 annots.

The command is deliberately fail-closed.  It will not assume a vertex order, a
sampling interval, or a medial-wall convention.  A successful run is staged in
the destination filesystem and published with one atomic directory rename.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
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
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np


VERTICES_PER_HEMISPHERE = 10_242
TOTAL_VERTICES = 2 * VERTICES_PER_HEMISPHERE
SCRIPT_VERSION = "1.1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
MEDIAL_NAMES = {
    "",
    "?",
    "???",
    "unknown",
    "unassigned",
    "medialwall",
    "medial wall",
    "medial_wall",
}


class PipelineError(RuntimeError):
    """A validation failure that should stop the run without publishing output."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON {path}: {exc}") from exc


def json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


@contextlib.contextmanager
def gzip_text_writer(path: Path) -> Iterator[io.TextIOWrapper]:
    """Write reproducible gzip (mtime=0) text."""

    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def delimiter_for(path: Path) -> str:
    name = path.name.lower()
    return "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","


def read_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter_for(path))
            if reader.fieldnames is None:
                raise PipelineError(f"table has no header: {path}")
            headers = [str(name) for name in reader.fieldnames]
            if len(headers) != len(set(headers)):
                raise PipelineError(f"table has duplicate column names: {path}")
            rows = [{key: ("" if value is None else value) for key, value in row.items()} for row in reader]
    except (OSError, csv.Error) as exc:
        raise PipelineError(f"cannot read table {path}: {exc}") from exc
    return headers, rows


def normalize_hemi(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"left", "lh", "l"}:
        return "left"
    if normalized in {"right", "rh", "r"}:
        return "right"
    return normalized


def validate_atlas_contract(path: Path) -> dict[str, Any]:
    contract = json_load(path)
    if not isinstance(contract, dict):
        raise PipelineError("atlas contract must be a JSON object")
    if contract.get("affirmed") is not True:
        raise PipelineError("atlas contract must contain boolean affirmed=true")
    if str(contract.get("space", "")).strip().lower() != "fsaverage5":
        raise PipelineError("atlas contract must affirm space=fsaverage5")
    try:
        vertices = int(contract.get("vertices_per_hemisphere"))
    except (TypeError, ValueError) as exc:
        raise PipelineError("atlas contract vertices_per_hemisphere must be 10242") from exc
    if vertices != VERTICES_PER_HEMISPHERE:
        raise PipelineError("atlas contract vertices_per_hemisphere must be 10242")
    order = contract.get("hemisphere_order")
    if not isinstance(order, list) or [normalize_hemi(item) for item in order] != ["left", "right"]:
        raise PipelineError("atlas contract must explicitly affirm hemisphere_order=[left,right]")
    if str(contract.get("matrix_layout", "")).strip().lower() != "time_by_vertex":
        raise PipelineError("atlas contract must explicitly affirm matrix_layout=time_by_vertex")
    if (
        str(contract.get("parcel_aggregation", "")).strip().lower()
        != "unweighted_vertex_mean"
    ):
        raise PipelineError(
            "atlas contract must explicitly affirm parcel_aggregation=unweighted_vertex_mean"
        )

    snapshots = contract.get("source_snapshots")
    if snapshots is None and contract.get("source_snapshot") is not None:
        snapshots = [contract["source_snapshot"]]
    if not isinstance(snapshots, list) or not snapshots:
        raise PipelineError("atlas contract must list at least one hashed source_snapshot")

    verified: list[dict[str, str]] = []
    for index, item in enumerate(snapshots):
        if not isinstance(item, dict):
            raise PipelineError(f"source snapshot {index} must be an object")
        raw_path = item.get("path")
        expected = str(item.get("sha256", "")).strip().lower()
        if not raw_path or not HEX_SHA256.fullmatch(expected):
            raise PipelineError(f"source snapshot {index} needs path and 64-character sha256")
        snapshot_path = Path(str(raw_path)).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = path.parent / snapshot_path
        snapshot_path = snapshot_path.resolve()
        if not snapshot_path.is_file():
            raise PipelineError(f"source snapshot does not exist: {snapshot_path}")
        actual = sha256_file(snapshot_path)
        if actual != expected:
            raise PipelineError(
                f"source snapshot hash mismatch for {snapshot_path}: expected {expected}, got {actual}"
            )
        verified.append({"path": str(snapshot_path), "sha256": actual})

    return {"contract": contract, "verified_source_snapshots": verified}


def decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def normalized_label_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("-", " ").split())


def is_medial_name(name: str) -> bool:
    normalized = normalized_label_name(name)
    compact = normalized.replace(" ", "")
    return normalized in MEDIAL_NAMES or compact in {"medialwall", "unknown", "unassigned"}


def load_annotation(
    path: Path,
    hemisphere: str,
    medial_wall_policy: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    try:
        from nibabel.freesurfer.io import read_annot
        import nibabel
    except ImportError as exc:
        raise PipelineError("nibabel is required to read FreeSurfer .annot files") from exc

    try:
        labels, color_table, names = read_annot(str(path), orig_ids=False)
    except Exception as exc:  # nibabel raises several format-specific exception types
        raise PipelineError(f"cannot read annotation {path}: {exc}") from exc

    labels = np.asarray(labels)
    color_table = np.asarray(color_table)
    if labels.ndim != 1 or labels.shape[0] != VERTICES_PER_HEMISPHERE:
        raise PipelineError(
            f"{hemisphere} annotation has {labels.shape} labels; expected exactly "
            f"({VERTICES_PER_HEMISPHERE},)"
        )
    decoded_names = [decode_name(name).strip() for name in names]
    if len(decoded_names) != color_table.shape[0]:
        raise PipelineError(f"annotation name/color-table length mismatch: {path}")
    if len(decoded_names) != len(set(decoded_names)):
        raise PipelineError(f"annotation contains duplicate raw parcel names: {path}")

    hemi_prefix = "lh" if hemisphere == "left" else "rh"
    observed_labels = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    out: list[dict[str, Any]] = []
    medial_vertices = int(np.count_nonzero(labels < 0))

    for local_index in observed_labels:
        if local_index >= len(decoded_names):
            raise PipelineError(
                f"{path} contains label index {local_index}, beyond {len(decoded_names)} names"
            )
        raw_name = decoded_names[local_index]
        vertex_count = int(np.count_nonzero(labels == local_index))
        medial = is_medial_name(raw_name)
        if medial:
            medial_vertices += vertex_count
        if medial and medial_wall_policy == "exclude":
            continue
        if medial and medial_wall_policy == "error":
            continue
        out.append(
            {
                "hemisphere": hemisphere,
                "hemisphere_prefix": hemi_prefix,
                "local_label_index": local_index,
                "raw_name": raw_name,
                "parcel_name": f"{hemi_prefix}:{raw_name}",
                "vertex_count": vertex_count,
                "is_medial_wall": medial,
                "color_table_row": [int(value) for value in color_table[local_index].tolist()],
                "vertex_indices": np.flatnonzero(labels == local_index).astype(np.int32),
            }
        )

    if medial_wall_policy == "include" and np.any(labels < 0):
        indices = np.flatnonzero(labels < 0).astype(np.int32)
        out.append(
            {
                "hemisphere": hemisphere,
                "hemisphere_prefix": hemi_prefix,
                "local_label_index": -1,
                "raw_name": "__unassigned__",
                "parcel_name": f"{hemi_prefix}:__unassigned__",
                "vertex_count": int(indices.size),
                "is_medial_wall": True,
                "color_table_row": [],
                "vertex_indices": indices,
            }
        )
    if medial_wall_policy == "error" and medial_vertices:
        raise PipelineError(
            f"{path} has {medial_vertices} medial-wall/unassigned vertices but policy=error"
        )
    if not out:
        raise PipelineError(f"annotation yields no parcels under policy={medial_wall_policy}: {path}")
    if any(item["vertex_count"] <= 0 for item in out):
        raise PipelineError(f"annotation produced an empty parcel: {path}")

    return labels.astype(np.int32, copy=False), out, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "nibabel_version": getattr(nibabel, "__version__", "unknown"),
        "vertex_count": int(labels.size),
        "medial_wall_vertex_count": medial_vertices,
        "nonempty_output_parcels": len(out),
    }


def build_atlas(
    left_annot: Path,
    right_annot: Path,
    medial_wall_policy: str,
    expected_parcels: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, left, left_meta = load_annotation(left_annot, "left", medial_wall_policy)
    _, right, right_meta = load_annotation(right_annot, "right", medial_wall_policy)
    parcels = left + right
    names = [item["parcel_name"] for item in parcels]
    if len(names) != len(set(names)):
        raise PipelineError("hemisphere-aware parcel names are not globally unique")
    if len(parcels) != expected_parcels:
        raise PipelineError(
            f"atlas has {len(parcels)} nonempty parcels under policy={medial_wall_policy}; "
            f"expected {expected_parcels}"
        )
    for global_index, parcel in enumerate(parcels):
        parcel["parcel_index"] = global_index
        parcel["global_vertex_indices"] = parcel["vertex_indices"] + (
            0 if parcel["hemisphere"] == "left" else VERTICES_PER_HEMISPHERE
        )
    return parcels, {"left": left_meta, "right": right_meta}


def parse_clock(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("empty time")
    try:
        result = float(text)
    except ValueError:
        first = re.split(r"\s*(?:-->|–|—|\bto\b)\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        parts = first.strip().split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"unsupported timestamp {value!r}")
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"unsupported timestamp {value!r}") from exc
        result = numbers[-1] + 60.0 * numbers[-2]
        if len(numbers) == 3:
            result += 3600.0 * numbers[0]
    if not math.isfinite(result):
        raise ValueError(f"non-finite timestamp {value!r}")
    return result


def resolve_segment_axis(path: Path, expected_rows: int) -> dict[str, Any]:
    headers, rows = read_records(path)
    if len(rows) != expected_rows:
        raise PipelineError(
            f"segments row count {len(rows)} does not match prediction T={expected_rows}: {path}"
        )
    lowered = {name.strip().lower(): name for name in headers}
    attempts: list[str] = []
    selected: Optional[str] = None
    values: Optional[list[float]] = None
    for candidate in ("start", "offset", "timeline"):
        actual = lowered.get(candidate)
        if actual is None:
            continue
        try:
            parsed = [parse_clock(row[actual]) for row in rows]
        except ValueError as exc:
            attempts.append(f"{actual}: {exc}")
            continue
        selected, values = actual, parsed
        break
    if selected is None or values is None:
        detail = "; ".join(attempts) if attempts else "none of start/offset/timeline is present"
        raise PipelineError(
            f"cannot resolve an explicit time axis from segments.tsv ({detail}); "
            "duration alone is not treated as a sampling interval"
        )
    if any(values[index] > values[index + 1] for index in range(len(values) - 1)):
        raise PipelineError(f"resolved segment time axis is not nondecreasing: {path}")

    durations: list[Optional[float]] = [None] * len(rows)
    duration_column = lowered.get("duration")
    if duration_column is not None:
        try:
            parsed_durations = [float(row[duration_column].strip()) for row in rows]
        except ValueError as exc:
            raise PipelineError(f"segments duration column is not fully numeric: {path}") from exc
        if any(not math.isfinite(value) or value < 0 for value in parsed_durations):
            raise PipelineError(f"segments duration values must be finite and nonnegative: {path}")
        durations = parsed_durations

    return {
        "time_source_column": selected,
        "time_values": values,
        "duration_source_column": duration_column,
        "duration_values": durations,
        "headers": headers,
    }


def finite_matrix(matrix: np.ndarray, chunk_rows: int = 256) -> bool:
    for start in range(0, matrix.shape[0], chunk_rows):
        if not np.isfinite(matrix[start : start + chunk_rows]).all():
            return False
    return True


def parcellate_matrix(matrix: np.ndarray, parcels: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != TOTAL_VERTICES:
        raise PipelineError(f"prediction shape must be T x {TOTAL_VERTICES}; got {matrix.shape}")
    if matrix.shape[0] <= 0:
        raise PipelineError("prediction matrix has zero timepoints")
    if not np.issubdtype(matrix.dtype, np.number):
        raise PipelineError(f"prediction matrix must be numeric; got {matrix.dtype}")
    if not finite_matrix(matrix):
        raise PipelineError("prediction matrix contains NaN or infinity")
    output = np.empty((matrix.shape[0], len(parcels)), dtype=np.float32)
    for parcel_index, parcel in enumerate(parcels):
        indices = np.asarray(parcel["global_vertex_indices"], dtype=np.int64)
        if indices.size == 0:
            raise PipelineError(f"parcel {parcel['parcel_name']} has no vertices")
        output[:, parcel_index] = np.asarray(matrix[:, indices], dtype=np.float64).mean(axis=1)
    if not np.isfinite(output).all():
        raise PipelineError("parcellation produced NaN or infinity")
    return output


def safe_feature_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not token:
        raise PipelineError(f"parcel name cannot form a feature token: {name!r}")
    return token


def require_columns(headers: Sequence[str], required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in headers]
    if missing:
        raise PipelineError(f"{path} is missing required columns: {', '.join(missing)}")


def validate_manifest(path: Path) -> list[dict[str, str]]:
    headers, rows = read_records(path)
    require_columns(headers, ("production_index", "video_id", "video_path"), path)
    if not rows:
        raise PipelineError("production manifest is empty")
    for key in ("production_index", "video_id"):
        values = [row[key].strip() for row in rows]
        if any(not value for value in values):
            raise PipelineError(f"production manifest has a blank {key}")
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise PipelineError(f"production manifest has duplicate {key}: {duplicates[:5]}")
    for row in rows:
        video_id = row["video_id"].strip()
        if not SAFE_VIDEO_ID.fullmatch(video_id) or video_id in {".", ".."}:
            raise PipelineError(f"unsafe video_id for production directory lookup: {video_id!r}")
    return rows


def save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def save_compressed_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ without pickle-capable object arrays."""

    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def resolve_prediction_source(
    predictions_roots: Sequence[Path],
    video_id: str,
    ad_directory_suffix: str,
) -> tuple[Path, list[str]]:
    """Select the first complete source in explicit root-priority order."""

    candidates = [root / f"{video_id}{ad_directory_suffix}" for root in predictions_roots]
    complete = [
        candidate
        for candidate in candidates
        if (candidate / "predictions.npy").is_file()
        and (candidate / "segments.tsv").is_file()
    ]
    if not complete:
        checked = ", ".join(str(path) for path in candidates)
        raise PipelineError(
            f"missing predictions.npy or segments.tsv for video_id={video_id}; checked: {checked}"
        )
    return complete[0], [str(path) for path in complete[1:]]


def parcel_public_record(parcel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parcel.items()
        if key not in {"vertex_indices", "global_vertex_indices"}
    }


def run(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.production_manifest).expanduser().resolve()
    predictions_roots = [
        Path(value).expanduser().resolve() for value in args.predictions_root
    ]
    left_annot = Path(args.left_annot).expanduser().resolve()
    right_annot = Path(args.right_annot).expanduser().resolve()
    contract_path = Path(args.atlas_contract).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for required in (manifest_path, left_annot, right_annot, contract_path):
        if not required.is_file():
            raise PipelineError(f"required input is not a file: {required}")
    for predictions_root in predictions_roots:
        if not predictions_root.is_dir():
            raise PipelineError(f"predictions root is not a directory: {predictions_root}")
    if args.expected_parcels <= 0:
        raise PipelineError("--expected-parcels must be positive")

    contract_info = validate_atlas_contract(contract_path)
    parcels, annot_meta = build_atlas(
        left_annot,
        right_annot,
        args.medial_wall_policy,
        args.expected_parcels,
    )
    feature_tokens = [safe_feature_token(parcel["parcel_name"]) for parcel in parcels]
    if len(feature_tokens) != len(set(feature_tokens)):
        raise PipelineError("parcel names collide after conversion to summary feature names")
    rows = validate_manifest(manifest_path)
    if args.video_id:
        requested = list(dict.fromkeys(args.video_id))
        by_video_id = {row["video_id"].strip(): row for row in rows}
        missing = [video_id for video_id in requested if video_id not in by_video_id]
        if missing:
            raise PipelineError(
                f"--video-id values absent from production manifest: {missing[:10]}"
            )
        rows = [by_video_id[video_id] for video_id in requested]

    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    if not SAFE_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise PipelineError("--run-id must contain only letters, digits, dot, underscore, or hyphen")
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists():
        raise PipelineError(f"refusing to replace existing output directory: {final_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))

    long_name = f"{run_id}_parcel_timeseries_long.csv.gz"
    summary_name = f"{run_id}_parcel_summary.csv.gz"
    ad_metadata: list[dict[str, Any]] = []
    try:
        ads_dir = staging / "ads"
        ads_dir.mkdir()
        public_parcels = [parcel_public_record(parcel) for parcel in parcels]
        json_dump(staging / "parcel_dictionary.json", public_parcels)
        with (staging / "parcel_dictionary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "parcel_index",
                "parcel_name",
                "hemisphere",
                "raw_name",
                "local_label_index",
                "vertex_count",
                "is_medial_wall",
                "summary_feature_token",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for parcel, token in zip(public_parcels, feature_tokens):
                writer.writerow({**{key: parcel[key] for key in fields[:-1]}, "summary_feature_token": token})

        long_fields = [
            "production_index",
            "video_id",
            "timepoint_index",
            "segment_time",
            "segment_duration",
            "time_source_column",
            "parcel_index",
            "parcel_name",
            "value",
        ]
        statistic_names = ("mean", "std", "min", "max", "median")
        summary_fields = [
            "production_index",
            "video_id",
            "n_timepoints",
            "time_source_column",
            "first_segment_time",
            "last_segment_time",
        ] + [
            f"brain__{token}__{statistic}"
            for token in feature_tokens
            for statistic in statistic_names
        ]

        with contextlib.ExitStack() as stack:
            summary_handle = stack.enter_context(gzip_text_writer(staging / summary_name))
            long_writer = None
            if args.write_long_timeseries:
                long_handle = stack.enter_context(gzip_text_writer(staging / long_name))
                long_writer = csv.DictWriter(long_handle, fieldnames=long_fields)
                long_writer.writeheader()
            summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
            summary_writer.writeheader()

            for row in rows:
                video_id = row["video_id"].strip()
                production_index = row["production_index"].strip()
                source_dir, alternate_complete_sources = resolve_prediction_source(
                    predictions_roots,
                    video_id,
                    args.ad_directory_suffix,
                )
                prediction_path = source_dir / "predictions.npy"
                segments_path = source_dir / "segments.tsv"
                source_metadata_path = source_dir / "metadata.json"
                try:
                    matrix = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
                except Exception as exc:
                    raise PipelineError(f"cannot load {prediction_path}: {exc}") from exc
                if matrix.ndim != 2 or matrix.shape[1] != TOTAL_VERTICES:
                    raise PipelineError(
                        f"{prediction_path} shape must be T x {TOTAL_VERTICES}; got {matrix.shape}"
                    )
                segment_axis = resolve_segment_axis(segments_path, int(matrix.shape[0]))
                reduced = parcellate_matrix(matrix, parcels)

                ad_dir = ads_dir / video_id
                ad_dir.mkdir()
                output_npy = ad_dir / "parcellated.npy"
                save_npy(output_npy, reduced)
                raw_float16_npz = ad_dir / "raw_vertices.float16.npz"
                roi_float16_npz = ad_dir / "hcp_mmp1_timeseries.float16.npz"
                time_values = np.asarray(segment_axis["time_values"], dtype=np.float64)
                duration_values = np.asarray(
                    [
                        np.nan if value is None else value
                        for value in segment_axis["duration_values"]
                    ],
                    dtype=np.float64,
                )
                save_compressed_npz(
                    raw_float16_npz,
                    predictions=np.asarray(matrix, dtype=np.float16),
                    segment_time=time_values,
                    segment_duration=duration_values,
                )
                save_compressed_npz(
                    roi_float16_npz,
                    roi_timeseries=reduced.astype(np.float16),
                    segment_time=time_values,
                    segment_duration=duration_values,
                    parcel_index=np.arange(len(parcels), dtype=np.int16),
                )
                prediction_hash = sha256_file(prediction_path)
                segments_hash = sha256_file(segments_path)
                source_metadata_hash = (
                    sha256_file(source_metadata_path) if source_metadata_path.is_file() else None
                )
                per_ad = {
                    "schema_version": 1,
                    "video_id": video_id,
                    "production_index": production_index,
                    "source": {
                        "predictions_path": str(prediction_path),
                        "predictions_sha256": prediction_hash,
                        "segments_path": str(segments_path),
                        "segments_sha256": segments_hash,
                        "metadata_path": str(source_metadata_path) if source_metadata_path.is_file() else None,
                        "metadata_sha256": source_metadata_hash,
                        "alternate_complete_sources": alternate_complete_sources,
                    },
                    "input_shape": [int(value) for value in matrix.shape],
                    "output_shape": [int(value) for value in reduced.shape],
                    "output_dtype": str(reduced.dtype),
                    "output_sha256": sha256_file(output_npy),
                    "raw_float16_npz": {
                        "path": raw_float16_npz.name,
                        "shape": [int(value) for value in matrix.shape],
                        "dtype": "float16",
                        "sha256": sha256_file(raw_float16_npz),
                    },
                    "roi_float16_npz": {
                        "path": roi_float16_npz.name,
                        "shape": [int(value) for value in reduced.shape],
                        "dtype": "float16",
                        "sha256": sha256_file(roi_float16_npz),
                    },
                    "parcel_aggregation": "unweighted_vertex_mean",
                    "temporal_summary": "unweighted_over_model_timepoints",
                    "time_source_column": segment_axis["time_source_column"],
                    "duration_source_column": segment_axis["duration_source_column"],
                    "first_segment_time": segment_axis["time_values"][0],
                    "last_segment_time": segment_axis["time_values"][-1],
                }
                json_dump(ad_dir / "metadata.json", per_ad)
                ad_metadata.append(per_ad)

                if long_writer is not None:
                    for timepoint_index, (time_value, duration) in enumerate(
                        zip(segment_axis["time_values"], segment_axis["duration_values"])
                    ):
                        for parcel_index, parcel in enumerate(parcels):
                            long_writer.writerow(
                                {
                                    "production_index": production_index,
                                    "video_id": video_id,
                                    "timepoint_index": timepoint_index,
                                    "segment_time": format(time_value, ".17g"),
                                    "segment_duration": ""
                                    if duration is None
                                    else format(duration, ".17g"),
                                    "time_source_column": segment_axis["time_source_column"],
                                    "parcel_index": parcel_index,
                                    "parcel_name": parcel["parcel_name"],
                                    "value": format(
                                        float(reduced[timepoint_index, parcel_index]), ".9g"
                                    ),
                                }
                            )

                summary_row: dict[str, Any] = {
                    "production_index": production_index,
                    "video_id": video_id,
                    "n_timepoints": reduced.shape[0],
                    "time_source_column": segment_axis["time_source_column"],
                    "first_segment_time": format(segment_axis["time_values"][0], ".17g"),
                    "last_segment_time": format(segment_axis["time_values"][-1], ".17g"),
                }
                for parcel_index, token in enumerate(feature_tokens):
                    values = reduced[:, parcel_index].astype(np.float64)
                    statistics = (
                        float(np.mean(values)),
                        float(np.std(values, ddof=0)),
                        float(np.min(values)),
                        float(np.max(values)),
                        float(np.median(values)),
                    )
                    for statistic_name, statistic_value in zip(statistic_names, statistics):
                        summary_row[f"brain__{token}__{statistic_name}"] = format(
                            statistic_value, ".17g"
                        )
                summary_writer.writerow(summary_row)

        script_path = Path(__file__).resolve()
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "parameters": {
                "expected_parcels": args.expected_parcels,
                "medial_wall_policy": args.medial_wall_policy,
                "ad_directory_suffix": args.ad_directory_suffix,
                "matrix_layout": "time_by_vertex",
                "hemisphere_order": ["left", "right"],
                "vertices_per_hemisphere": VERTICES_PER_HEMISPHERE,
                "parcel_aggregation": "unweighted_vertex_mean",
                "temporal_summary": "unweighted_over_model_timepoints",
                "write_long_timeseries": args.write_long_timeseries,
                "selected_video_ids": args.video_id,
            },
            "inputs": {
                "production_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "rows": len(rows),
                },
                "predictions_roots_priority_order": [
                    str(path) for path in predictions_roots
                ],
                "atlas_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_file(contract_path),
                    **contract_info,
                },
                "annotations": annot_meta,
            },
            "outputs": {
                "long_csv_gz": long_name if args.write_long_timeseries else None,
                "long_timeseries_written": args.write_long_timeseries,
                "summary_csv_gz": summary_name,
                "parcel_dictionary_json": "parcel_dictionary.json",
                "parcel_dictionary_csv": "parcel_dictionary.csv",
                "ads_directory": "ads",
                "per_ad_raw_vertices_float16_npz": "ads/<video_id>/raw_vertices.float16.npz",
                "per_ad_roi_float16_npz": "ads/<video_id>/hcp_mmp1_timeseries.float16.npz",
            },
            "ads": ad_metadata,
        }
        json_dump(staging / "metadata.json", metadata)
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--production-manifest", required=True)
    result.add_argument(
        "--predictions-root",
        required=True,
        action="append",
        help=(
            "Prediction root in priority order; repeat for recovery/fallback roots. "
            "The first complete per-ad directory is selected."
        ),
    )
    result.add_argument("--left-annot", required=True)
    result.add_argument("--right-annot", required=True)
    result.add_argument("--atlas-contract", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--run-id", help="Reproducible output directory name; defaults to UTC timestamp")
    result.add_argument("--expected-parcels", type=int, default=360)
    result.add_argument(
        "--medial-wall-policy",
        required=True,
        choices=("exclude", "error", "include"),
        help="Explicit handling of unknown/medial-wall annotation vertices",
    )
    result.add_argument("--ad-directory-suffix", default="_TRIBE")
    result.add_argument(
        "--video-id",
        action="append",
        help="Process only this manifest video_id; repeat for a pilot subset.",
    )
    result.add_argument(
        "--write-long-timeseries",
        action="store_true",
        help="Opt in to the potentially very large run-level T x parcel long CSV.gz",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = run(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
