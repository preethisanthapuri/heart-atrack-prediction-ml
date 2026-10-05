import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.external_validation import UCI_COLUMNS, evaluate_external_site, load_and_map_uci_site
from src.preprocessing import FEATURES, build_preprocessing_pipeline


class ExternalValidationTests(unittest.TestCase):
    def test_maps_source_codes_and_target_without_losing_missingness(self):
        rows = [
            [50, 1, 1, 120, 200, 0, 0, 150, 0, 0.5, 1, 0, 3, 0],
            [60, 0, 4, 140, 240, 1, 2, 120, 1, 2.0, 3, 2, 7, 4],
            [55, 1, 2, "?", 220, 0, 1, 140, 0, 1.0, 2, "?", 6, 2],
            [45, 0, 3, 110, 210, 0, 0, 160, 0, 0.0, 1, 1, "?", 0],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "site.data"
            pd.DataFrame(rows, columns=UCI_COLUMNS).to_csv(path, index=False, header=False)
            X, y, details = load_and_map_uci_site(path)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(X["cp"].tolist(), [3, 0, 1, 2])
        self.assertEqual(X["restecg"].tolist(), [1, 0, 2, 1])
        self.assertEqual(X["slp"].tolist(), [2, 0, 1, 2])
        self.assertEqual(X["thall"].tolist(), [2, 3, 1, 0])
        self.assertEqual(X["caa"].tolist(), [0, 2, 4, 1])
        self.assertEqual(y.tolist(), [1, 0, 0, 1])
        self.assertEqual(details["source_feature_rows_with_any_missing"], 2)
        self.assertEqual(details["feature_rows_with_any_missing"], 1)
        self.assertEqual(details["binary_target_counts"], {"0": 2, "1": 2})

    def test_external_scoring_does_not_fit_and_reports_required_metrics(self):
        rng = np.random.default_rng(34)
        frame = pd.DataFrame({name: rng.normal(size=60) for name in FEATURES})
        for col in ("sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"):
            frame[col] = (np.arange(len(frame)) + len(col)) % 2
        y = pd.Series(np.tile([0, 1], 30), name="output")
        prep = build_preprocessing_pipeline().fit(frame.iloc[:40])
        transformed = prep.transform(frame.iloc[:40])
        model = LogisticRegression(random_state=42).fit(transformed, y.iloc[:40])
        with tempfile.TemporaryDirectory() as directory:
            prep_path, model_path = Path(directory) / "prep.joblib", Path(directory) / "model.joblib"
            joblib.dump(prep, prep_path)
            joblib.dump(model, model_path)
            table, metrics = evaluate_external_site(frame.iloc[40:], y.iloc[40:], prep_path, model_path)
        self.assertEqual(metrics["n"], 20)
        self.assertIn("roc_auc", table.columns)
        self.assertIn("brier_score", table.columns)
        self.assertEqual(sum(metrics["confusion_matrix"].values()), 20)


if __name__ == "__main__":
    unittest.main()
