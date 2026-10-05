import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.evaluation import binary_classification_metrics, cross_validate_baselines
from src.preprocessing import FEATURES, TARGET_COLUMN


class ModelEvaluationTests(unittest.TestCase):
    def test_binary_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_score = np.array([.1, .8, .6, .4, .9])
        metrics = binary_classification_metrics(y_true, y_pred, y_score)
        self.assertAlmostEqual(metrics["accuracy"], .6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["sensitivity"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], .5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertGreater(metrics["roc_auc"], .5)
        self.assertGreater(metrics["pr_auc"], 0)

    def test_cv_returns_stratified_metric_summary_using_pipeline(self):
        rows = 60
        frame = pd.DataFrame({feature: np.arange(rows, dtype=float) + i for i, feature in enumerate(FEATURES)})
        frame[TARGET_COLUMN] = np.tile([0, 1], rows // 2)
        result = cross_validate_baselines(
            frame[FEATURES], frame[TARGET_COLUMN], {"Logistic Regression": LogisticRegression(max_iter=1000)},
            n_splits=3, random_state=9,
        )
        self.assertEqual(result.loc[0, "folds"], 3)
        for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            self.assertIn(f"{metric}_mean", result.columns)
            self.assertTrue(0 <= result.loc[0, f"{metric}_mean"] <= 1)

    def test_rejects_non_binary_target(self):
        with self.assertRaises(ValueError):
            binary_classification_metrics([0, 1, 2], [0, 1, 2], [.1, .9, .8])


if __name__ == "__main__":
    unittest.main()
