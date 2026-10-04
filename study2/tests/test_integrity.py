from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study2_portfolio import select_label_blind, validate_exact_inventory, write_synthetic_pair


FIELDS = ("clip_id", "parent_ad_id", "split", "eligible", "label")


def source_file(path: Path, labels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> Path:
    rows = [
        ("t0", "p0", "train", "1"), ("t1", "p0", "train", "1"),
        ("t2", "p1", "train", "1"), ("t3", "p1", "train", "1"),
        ("v0", "p2", "validation", "1"), ("v1", "p3", "validation", "1"),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for row, label in zip(rows, labels):
            writer.writerow((*row, label))
    return path


class IntegrityTests(unittest.TestCase):
    def test_label_changes_cannot_change_cohort(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = source_file(root / "first.csv")
            second = source_file(root / "second.csv", labels=(7, 6, 5, 4, 3, 2))
            targets = {"train": 3, "validation": 1}
            selected = select_label_blind(first, targets, seed="fixed")
            self.assertEqual(selected, select_label_blind(second, targets, seed="fixed"))
            self.assertEqual(len(selected), 4)

    def test_parent_crossing_splits_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = source_file(Path(directory) / "source.csv")
            text = path.read_text(encoding="utf-8").replace("v0,p2,validation", "v0,p0,validation")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                select_label_blind(path, {"train": 3, "validation": 1}, seed="fixed")

    def test_corrupt_or_extra_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = select_label_blind(source_file(root / "source.csv"),
                                          {"train": 2, "validation": 1}, seed="fixed")
            artifacts = root / "artifacts"
            for row in selected:
                write_synthetic_pair(artifacts, row)
            self.assertEqual(validate_exact_inventory(artifacts, selected),
                             {"train": 2, "validation": 1})
            one = selected[0]["clip_id"]
            npz_path = artifacts / "clips" / f"{one}.content.npz"
            with npz_path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_exact_inventory(artifacts, selected)
            npz_path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing, extra"):
                validate_exact_inventory(artifacts, selected)


if __name__ == "__main__":
    unittest.main()
