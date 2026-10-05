"""Tests for Phase 14 leakage boundaries, OOF coverage, and subgroup summaries."""
import unittest
import numpy as np
import pandas as pd

from src.ablation import run_ablation_experiment
from src.fairness import subgroup_fairness_tables


class Phase14Tests(unittest.TestCase):
    @staticmethod
    def sample_data(n=72):
        rng = np.random.default_rng(7)
        y = np.array([0, 1] * (n // 2))
        rng.shuffle(y)
        frame = pd.DataFrame({
            "age": rng.integers(35, 78, n), "trtbps": rng.normal(125, 15, n),
            "chol": rng.normal(230, 40, n), "thalachh": rng.normal(145, 20, n),
            "oldpeak": rng.normal(1, 0.8, n), "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n), "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n), "exng": rng.integers(0, 2, n),
            "slp": rng.integers(0, 3, n), "caa": rng.integers(0, 4, n),
            "thall": rng.integers(0, 4, n), "output": y,
        })
        # Add a modest signal so every synthetic fold has useful variation.
        frame["oldpeak"] += y * 0.5
        return frame

    def test_ablation_outputs_complete_oof_and_stages(self):
        frame = self.sample_data()
        summary, folds, oof = run_ablation_experiment(
            frame.drop(columns="output"), frame.output, n_splits=2, inner_splits=2,
            selection_size=8, rf_trees=15,
        )
        self.assertEqual(set(summary.stage), set("ABCDEF"))
        self.assertEqual(len(oof), len(frame))
        self.assertTrue(oof.filter(like="probability_").notna().all().all())
        self.assertEqual(len(folds), 12)
        self.assertTrue(oof.filter(like="probability_").apply(lambda s: s.between(0, 1).all()).all())

    def test_fairness_requires_exact_train_oof_rows_and_reports_groups(self):
        frame = self.sample_data()
        train = list(frame.index[:60])
        pred = pd.DataFrame({"row_index": train, "y_true": frame.loc[train, "output"],
                             "probability_F": np.linspace(0.1, 0.9, len(train))})
        metrics, disparities = subgroup_fairness_tables(frame, train, pred)
        self.assertEqual(set(metrics.grouping), {"sex_code", "age_group"})
        self.assertEqual(set(disparities.metric), {"accuracy", "recall_class_1", "specificity_class_0", "brier"})
        with self.assertRaises(ValueError):
            subgroup_fairness_tables(frame, train, pred.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
