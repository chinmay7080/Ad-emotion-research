#!/usr/bin/env python3
"""Run a self-contained toy version of the Study 2 integrity gates."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study2_portfolio import select_label_blind, validate_exact_inventory, write_synthetic_pair


def write_synthetic_source(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("clip_id", "parent_ad_id", "split",
                                                   "eligible", "label"))
        writer.writeheader()
        for split, parents in (("train", 5), ("validation", 3)):
            for parent in range(parents):
                for clip in range(2):
                    writer.writerow({"clip_id": f"toy_{split}_{parent}_{clip}",
                                     "parent_ad_id": f"toy_parent_{split}_{parent}",
                                     "split": split, "eligible": "1",
                                     "label": (parent + clip) % 8})


def main() -> None:
    with TemporaryDirectory(prefix="study2-synthetic-") as directory:
        root = Path(directory)
        source = root / "toy_source.csv"
        write_synthetic_source(source)
        selected = select_label_blind(source, {"train": 8, "validation": 4},
                                      seed="portfolio-demo-v1")
        artifacts = root / "artifacts"
        for row in selected:
            write_synthetic_pair(artifacts, row)
        counts = validate_exact_inventory(artifacts, selected)
        print(f"SYNTHETIC_DEMO_PASS clips={len(selected)} "
              f"train={counts['train']} validation={counts['validation']}")


if __name__ == "__main__":
    main()
