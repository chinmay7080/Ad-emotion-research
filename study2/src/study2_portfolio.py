"""Small, synthetic demonstration of Study 2's data integrity gates.

This module is illustrative. The restricted source data and production model
weights are intentionally absent from this portfolio package.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MODALITY_DIMS = {"video": (2, 4), "audio": (2, 3), "text": (2, 5)}
SPLITS = ("train", "validation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_label_blind(csv_path: Path, counts: dict[str, int], seed: str) -> list[dict[str, str]]:
    """Select exact counts using only ID, parent, split, and eligibility fields."""
    if set(counts) != set(SPLITS) or any(n < 0 for n in counts.values()):
        raise ValueError("Expected nonnegative train and validation targets")
    pools: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    seen_ids: set[str] = set()
    parent_splits: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"clip_id", "parent_ad_id", "split", "eligible"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Required identity columns are missing")
        for source in reader:
            # Intentionally project four fields. Any label column is ignored.
            row = {key: source[key] for key in required}
            clip_id, parent, split = row["clip_id"], row["parent_ad_id"], row["split"]
            if not clip_id or not parent or clip_id in seen_ids or split not in SPLITS:
                raise ValueError("Invalid or duplicate clip identity")
            seen_ids.add(clip_id)
            if parent in parent_splits and parent_splits[parent] != split:
                raise ValueError("A parent advertisement crosses splits")
            parent_splits[parent] = split
            if row["eligible"] not in {"0", "1"}:
                raise ValueError("Eligibility must be an explicit 0 or 1")
            if row["eligible"] == "1":
                pools[split].append({"clip_id": clip_id, "parent_ad_id": parent,
                                     "split": split})
    selected = []
    for split in SPLITS:
        ranked = sorted(pools[split], key=lambda row: (
            hashlib.sha256(f"{seed}:{row['clip_id']}".encode()).digest(), row["clip_id"]))
        if len(ranked) < counts[split]:
            raise ValueError(f"Insufficient eligible {split} clips")
        selected.extend(ranked[:counts[split]])
    selected.sort(key=lambda row: row["clip_id"])
    return selected


def write_synthetic_pair(root: Path, row: dict[str, str]) -> None:
    """Write one toy 2-Hz feature file and its checksum receipt."""
    (root / "clips").mkdir(parents=True, exist_ok=True)
    (root / "receipts").mkdir(parents=True, exist_ok=True)
    clip_id = row["clip_id"]
    npz_path = root / "clips" / f"{clip_id}.content.npz"
    receipt_path = root / "receipts" / f"{clip_id}.json"
    if npz_path.exists() or receipt_path.exists():
        raise FileExistsError(f"Existing artifact for {clip_id}")
    arrays: dict[str, np.ndarray] = {"times": np.arange(10, dtype=np.float32) / 2}
    for modality, (channels, width) in MODALITY_DIMS.items():
        values = np.arange(channels * width * 10, dtype=np.float32).reshape(channels, width, 10)
        values = values / 100 + (len(clip_id) % 7) / 10
        arrays[f"{modality}_features"] = values
        arrays[f"{modality}_mean"] = values.mean(axis=-1)
        arrays[f"{modality}_std"] = values.std(axis=-1)
    np.savez_compressed(npz_path, **arrays)
    receipt = {"clip_id": clip_id, "parent_ad_id": row["parent_ad_id"],
               "split": row["split"], "frequency_hz": 2.0,
               "window_policy": "half_open_5s", "npz_sha256": sha256_file(npz_path)}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


def validate_pair(npz_path: Path, receipt_path: Path, row: dict[str, str]) -> None:
    """Check identity, checksum, dimensions, pooling, and half-open timing."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("clip_id") != row["clip_id"]
            or receipt.get("parent_ad_id") != row["parent_ad_id"]
            or receipt.get("split") != row["split"]
            or receipt.get("frequency_hz") != 2.0
            or receipt.get("window_policy") != "half_open_5s"
            or receipt.get("npz_sha256") != sha256_file(npz_path)):
        raise ValueError(f"Identity or checksum mismatch: {row['clip_id']}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        expected_keys = {"times"}
        expected_keys.update(f"{modality}_{kind}" for modality in MODALITY_DIMS
                             for kind in ("features", "mean", "std"))
        if set(arrays.files) != expected_keys:
            raise ValueError("Unexpected feature schema")
        times = arrays["times"]
        if (times.shape != (10,) or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(10) / 2, atol=1e-6, rtol=0)
                or times[-1] >= 5.0):
            raise ValueError("Invalid 2-Hz half-open time grid")
        for modality, shape in MODALITY_DIMS.items():
            values = arrays[f"{modality}_features"]
            mean = arrays[f"{modality}_mean"]
            std = arrays[f"{modality}_std"]
            if (values.shape != (*shape, 10) or mean.shape != shape or std.shape != shape
                    or any(not np.isfinite(item).all() for item in (values, mean, std))
                    or not np.allclose(mean, values.mean(axis=-1), atol=1e-5, rtol=0)
                    or not np.allclose(std, values.std(axis=-1), atol=1e-5, rtol=0)):
                raise ValueError(f"Invalid {modality} feature or pooling")


def validate_exact_inventory(root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    """Require one valid feature and receipt pair for every selected ID."""
    expected = {row["clip_id"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate selected clip")
    clips = list((root / "clips").iterdir())
    receipts = list((root / "receipts").iterdir())
    clip_ids = {path.name.removesuffix(".content.npz") for path in clips
                if path.is_file() and path.name.endswith(".content.npz")}
    receipt_ids = {path.stem for path in receipts if path.is_file() and path.suffix == ".json"}
    if (len(clips) != len(expected) or len(receipts) != len(expected)
            or clip_ids != expected or receipt_ids != expected):
        raise ValueError("Missing, extra, or unexpected artifact")
    for row in rows:
        cid = row["clip_id"]
        validate_pair(root / "clips" / f"{cid}.content.npz",
                      root / "receipts" / f"{cid}.json", row)
    return dict(Counter(row["split"] for row in rows))
