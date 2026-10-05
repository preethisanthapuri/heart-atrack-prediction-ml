import unittest

import numpy as np
import pandas as pd

from src.ensemble import run_ensemble_experiment
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class EnsembleTests(unittest.TestCase):
    def synthetic_data(self, rows=40):
        rng = np.random.default_rng(31)
        y = np.tile([0, 1], rows // 2)
        data = {column: rng.normal(size=rows) for column in NUMERIC_FEATURES}
        data["age"] += y
        for index, column in enumerate(CATEGORICAL_FEATURES):
            data[column] = (np.arange(rows) + index) % 2
        return pd.DataFrame(data)[FEATURES], pd.Series(y, name="output")

    def test_cv_compares_individual_and_ensemble_estimators(self):
        X, y = self.synthetic_data()
        summary, folds = run_ensemble_experiment(X, y, n_splits=2, random_state=4)
        self.assertEqual(set(summary["model"]), {
            "Logistic Regression", "K-Nearest Neighbors", "Support Vector Classifier (RBF)",
            "Random Forest", "Soft Voting", "Stacking",
        })
        self.assertEqual(len(folds), 6 * 2 * 7)
        self.assertTrue(summary["roc_auc_mean"].between(0, 1).all())

    def test_rejects_non_binary_target(self):
        X, y = self.synthetic_data()
        y.iloc[-1] = 2
        with self.assertRaises(ValueError):
            run_ensemble_experiment(X, y, n_splits=2)


if __name__ == "__main__":
    unittest.main()
