"""Final-model selection, fit boundary, and serialized prediction tests."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.final_model import fit_final_model
from src.preprocessing import FEATURES, TARGET_COLUMN


class FinalModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        n = 80
        y = np.array([0, 1] * 40)
        rng.shuffle(y)
        self.frame = pd.DataFrame({
            "age": rng.integers(35, 80, n), "trtbps": rng.normal(125, 15, n),
            "chol": rng.normal(220, 35, n), "thalachh": rng.normal(145, 20, n),
            "oldpeak": rng.normal(1, 0.8, n), "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n), "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n), "exng": rng.integers(0, 2, n),
            "slp": rng.integers(0, 3, n), "caa": rng.integers(0, 4, n),
            "thall": rng.integers(0, 4, n), TARGET_COLUMN: y,
        })
        self.indices = {"train_indices": list(range(64)), "test_indices": list(range(64, 80)),
                        "fitted_on": "training split only"}

    def test_fit_and_reload_final_pipeline_on_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ablation_path = root / "ablation.csv"
            pd.DataFrame([{
                "stage": "B", "accuracy": .84, "precision": .85, "sensitivity": .88,
                "specificity": .80, "f1": .86, "roc_auc": .91, "pr_auc": .92,
                "brier": .11, "log_loss": .36,
            }]).to_csv(ablation_path, index=False)
            metadata = fit_final_model(
                self.frame, self.indices, model_path=root / "final.joblib",
                selector_path=root / "selector.joblib", metadata_path=root / "final.json",
                ablation_results_path=ablation_path,
            )
            loaded = joblib.load(root / "final.joblib")
            probabilities = loaded.predict_proba(self.frame.loc[:2, FEATURES])
            saved_metadata = json.loads((root / "final.json").read_text())
        self.assertEqual(metadata["selected_stage"], "B")
        self.assertEqual(metadata["training_rows"], 64)
        self.assertEqual(metadata["holdout_rows_not_used"], 16)
        self.assertEqual(len(metadata["selected_transformed_features"]), 15)
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertIsNone(saved_metadata["classification_cutoff"])

    def test_rejects_overlapping_saved_split(self):
        indices = {"train_indices": list(range(65)), "test_indices": list(range(64, 80)),
                   "fitted_on": "training split only"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ablation = root / "ablation.csv"
            pd.DataFrame([{"stage": "B"}]).to_csv(ablation, index=False)
            with self.assertRaises(ValueError):
                fit_final_model(self.frame, indices, model_path=root / "m.joblib",
                                selector_path=root / "s.joblib", metadata_path=root / "x.json",
                                ablation_results_path=ablation)


if __name__ == "__main__":
    unittest.main()
