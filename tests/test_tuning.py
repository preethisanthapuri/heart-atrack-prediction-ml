import unittest

import numpy as np
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.tuning import tune_training_partition


class TuningTests(unittest.TestCase):
    def synthetic_data(self, rows=60):
        rng = np.random.default_rng(23)
        y = np.tile([0, 1], rows // 2)
        data = {column: rng.normal(size=rows) for column in NUMERIC_FEATURES}
        data["age"] += y
        for index, column in enumerate(CATEGORICAL_FEATURES):
            data[column] = (np.arange(rows) + index) % 2
        return pd.DataFrame(data)[FEATURES], pd.Series(y, name="output")

    def test_runs_grid_and_random_search_with_cv_metrics(self):
        X, y = self.synthetic_data()
        summary, candidates, estimators = tune_training_partition(
            X, y, n_splits=2, random_state=5, n_iter=2
        )
        self.assertEqual(set(summary["search_type"]), {"GridSearchCV", "RandomizedSearchCV"})
        self.assertEqual(summary["model"].nunique(), 3)
        self.assertEqual(len(candidates), 52)  # 10 LR + 40 SVC + 2 sampled KNN settings.
        self.assertEqual(set(estimators), set(summary["model"]))
        self.assertTrue(summary["best_mean_roc_auc"].between(0, 1).all())
        self.assertTrue(summary["best_mean_accuracy"].between(0, 1).all())

    def test_rejects_invalid_targets(self):
        X, y = self.synthetic_data()
        y.iloc[-1] = 2
        with self.assertRaises(ValueError):
            tune_training_partition(X, y, n_splits=2, n_iter=1)


if __name__ == "__main__":
    unittest.main()
