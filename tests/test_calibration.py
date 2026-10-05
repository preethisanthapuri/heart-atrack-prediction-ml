import unittest

import numpy as np
import pandas as pd

from src.calibration import _base_estimators, run_calibration_experiment
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class CalibrationTests(unittest.TestCase):
    def test_all_candidates_include_fold_local_preprocessing(self):
        candidates = _base_estimators()
        for name in ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]:
            self.assertEqual(candidates[name].steps[0][0], "preprocessor")

    def synthetic_data(self, rows=40):
        rng = np.random.default_rng(37)
        y = np.tile([0, 1], rows // 2)
        data = {column: rng.normal(size=rows) for column in NUMERIC_FEATURES}
        data["age"] += y
        for index, column in enumerate(CATEGORICAL_FEATURES):
            data[column] = (np.arange(rows) + index) % 2
        return pd.DataFrame(data)[FEATURES], pd.Series(y, name="output")

    def test_out_of_fold_probabilities_produce_metrics_and_curves(self):
        X, y = self.synthetic_data()
        summary, curves = run_calibration_experiment(
            X, y, n_splits=2, inner_folds=2, random_state=8, n_bins=4
        )
        self.assertEqual(len(summary), 12)
        self.assertEqual(set(summary["method"]), {"uncalibrated", "sigmoid", "isotonic"})
        self.assertTrue(summary["brier_oof_pooled"].between(0, 1).all())
        self.assertTrue(summary["roc_auc_oof_pooled"].between(0, 1).all())
        self.assertTrue(curves["observed_fraction_class1"].between(0, 1).all())

    def test_rejects_non_binary_target(self):
        X, y = self.synthetic_data()
        y.iloc[-1] = 2
        with self.assertRaises(ValueError):
            run_calibration_experiment(X, y, n_splits=2, inner_folds=2)


if __name__ == "__main__":
    unittest.main()
