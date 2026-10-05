import unittest

import numpy as np
import pandas as pd

from src.feature_selection import run_feature_selection_experiment
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


class FeatureSelectionTests(unittest.TestCase):
    def synthetic_frame(self, rows=80):
        rng = np.random.default_rng(19)
        frame = pd.DataFrame({feature: rng.normal(size=rows) for feature in FEATURES})
        frame[TARGET_COLUMN] = np.tile([0, 1], rows // 2)
        frame["age"] += frame[TARGET_COLUMN] * 1.5
        for feature in ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]:
            frame[feature] = rng.integers(0, 2, size=rows)
        return frame

    def test_selection_comparison_uses_fixed_budget_and_returns_stability(self):
        frame = self.synthetic_frame()
        comparison, features = run_feature_selection_experiment(
            frame, list(range(len(frame))), n_splits=3, random_state=7, selection_size=5
        )
        self.assertEqual(len(comparison), 10)
        self.assertIn("All features", comparison["selection_method"].tolist())
        encoded_count = len(NUMERIC_FEATURES) + sum(frame[column].nunique() for column in CATEGORICAL_FEATURES)
        self.assertTrue((comparison.loc[comparison["selection_method"] == "All features", "selected_feature_count"] == encoded_count).all())
        selected = comparison.loc[comparison["selection_method"] != "All features"]
        self.assertTrue((selected["selected_feature_count"] == 5).all())
        self.assertEqual(features["selection_method"].nunique(), 4)
        self.assertTrue(features["fold_selection_rate"].between(0, 1).all())
        self.assertTrue(comparison["roc_auc_mean"].between(0, 1).all())

    def test_rejects_invalid_selection_budget(self):
        frame = self.synthetic_frame()
        with self.assertRaises(ValueError):
            run_feature_selection_experiment(frame, list(range(len(frame))), n_splits=3, selection_size=0)


if __name__ == "__main__":
    unittest.main()
