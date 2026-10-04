from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_models_bc.py"
SPEC = importlib.util.spec_from_file_location("evaluate_models_bc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODELS)


def write_reference(path: Path, bad_group: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "video_id", "group_id", "effective", "prediction"],
        )
        writer.writeheader()
        for index in range(1865):
            video_id = f"ad_{index:04d}"
            writer.writerow(
                {
                    "fold": index // 373,
                    "video_id": video_id,
                    "group_id": "wrong" if bad_group and index == 0 else video_id,
                    "effective": 1 + index % 5,
                    "prediction": 3.0,
                }
            )


class ModelsBCTests(unittest.TestCase):
    def test_improvement_direction(self):
        self.assertAlmostEqual(MODELS.positive_improvement("spearman", 0.3, 0.2), 0.1)
        self.assertAlmostEqual(MODELS.positive_improvement("pearson", 0.3, 0.2), 0.1)
        self.assertAlmostEqual(MODELS.positive_improvement("mae", 1.0, 1.2), 0.2)
        self.assertAlmostEqual(MODELS.positive_improvement("rmse", 1.0, 1.2), 0.2)

    def test_reference_folds_require_exact_leakage_free_model_a_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "model_a.csv"
            write_reference(reference)
            folds = MODELS.read_reference_folds(reference)
            self.assertEqual(len(folds), 1865)
            self.assertEqual([sum(value == fold for value in folds.values()) for fold in range(5)], [373] * 5)

            bad_reference = Path(tmp) / "bad_model_a.csv"
            write_reference(bad_reference, bad_group=True)
            with self.assertRaisesRegex(MODELS.EvaluationError, "group_id"):
                MODELS.read_reference_folds(bad_reference)

    def test_group_folds_are_deterministic_and_disjoint(self):
        groups = np.asarray([f"ad_{index:03d}" for index in range(100)])
        first = MODELS.deterministic_group_folds(groups, 5, 17)
        second = MODELS.deterministic_group_folds(groups, 5, 17)
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
            self.assertFalse(set(groups[train_a]).intersection(groups[test_a]))

    def test_paired_bootstrap_reports_positive_gains_for_better_model(self):
        target = np.tile(np.arange(1.0, 6.0), 80)
        reference = 6.0 - target + np.cos(np.arange(target.size)) * 0.01
        challenger = target + np.sin(np.arange(target.size)) * 0.01
        rows = MODELS.paired_bootstrap(target, reference, challenger, 100, 23)
        self.assertEqual({row["metric"] for row in rows}, set(MODELS.METRICS))
        for row in rows:
            self.assertEqual(row["finite_repetitions"], 100)
            self.assertGreater(row["mean_improvement"], 0)


if __name__ == "__main__":
    unittest.main()
