import unittest

import numpy as np
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.uncertainty import (
    conformal_prediction_sets,
    conformal_quantile,
    run_uncertainty_experiment,
)


class UncertaintyTests(unittest.TestCase):
    def synthetic_data(self, rows=40):
        rng = np.random.default_rng(43)
        y = np.tile([0, 1], rows // 2)
        data = {column: rng.normal(size=rows) for column in NUMERIC_FEATURES}
        data["age"] += y
        for index, column in enumerate(CATEGORICAL_FEATURES):
            data[column] = (np.arange(rows) + index) % 2
        return pd.DataFrame(data)[FEATURES], pd.Series(y, name="output")

    def test_finite_sample_quantile_and_prediction_sets(self):
        self.assertAlmostEqual(conformal_quantile([.1, .2, .4, .7], alpha=.2), .7)
        self.assertEqual(conformal_quantile([.1, .2, .4, .7], alpha=.01), 1.0)
        sets = conformal_prediction_sets([[.9, .1], [.55, .45], [.2, .8]], quantile=.3)
        self.assertEqual(sets, [(0,), (), (1,)])

    def test_uncertainty_experiment_reports_coverage_and_risk(self):
        X, y = self.synthetic_data()
        summary, risk, predictions = run_uncertainty_experiment(
            X, y, n_splits=2, calibration_fraction=.25, alphas=(.1, .2), random_state=6
        )
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(risk), 6)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(summary["empirical_coverage_oof_pooled"].between(0, 1).all())
        self.assertTrue(summary["mean_set_size"].between(0, 2).all())
        self.assertTrue(risk["error_rate"].between(0, 1).all())

    def test_rejects_non_binary_targets(self):
        X, y = self.synthetic_data()
        y.iloc[-1] = 2
        with self.assertRaises(ValueError):
            run_uncertainty_experiment(X, y, n_splits=2)


if __name__ == "__main__":
    unittest.main()
