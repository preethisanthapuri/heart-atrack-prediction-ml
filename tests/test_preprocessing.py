import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN,
    build_preprocessing_pipeline, fit_and_save_preprocessing,
)


def synthetic_frame(rows=60):
    data = {
        "age": np.arange(30, 30 + rows, dtype=float),
        "trtbps": np.linspace(100, 160, rows),
        "chol": np.linspace(150, 300, rows),
        "thalachh": np.linspace(80, 190, rows),
        "oldpeak": np.linspace(0, 3, rows),
    }
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        data[feature] = (np.arange(rows) + i) % (2 if feature in {"sex", "fbs", "exng"} else 3)
    data[TARGET_COLUMN] = np.tile([0, 1], rows // 2)
    frame = pd.DataFrame(data)
    frame.loc[0, "chol"] = np.nan
    frame.loc[1, "cp"] = np.nan
    return frame


class PreprocessingTests(unittest.TestCase):
    def test_pipeline_imputes_encodes_and_ignores_unknown_categories(self):
        frame = synthetic_frame()
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        transformed = pipeline.transform(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[0], len(frame))
        unknown = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].iloc[[0]].copy()
        unknown.loc[:, "cp"] = 99
        self.assertTrue(np.isfinite(pipeline.transform(unknown)).all())

    def test_split_is_stratified_and_saved_pipeline_fits_train_only(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "preprocessing_pipeline.joblib"
            metadata_path = Path(temp_dir) / "split.json"
            metadata = fit_and_save_preprocessing(frame, model_path, metadata_path, random_state=7)
            loaded = joblib.load(model_path)
            self.assertEqual(metadata["train_rows"] + metadata["test_rows"], len(frame))
            self.assertEqual(set(metadata["train_indices"]) & set(metadata["test_indices"]), set())
            self.assertEqual(metadata["train_class_counts"], {"0": 24, "1": 24})
            self.assertTrue(model_path.exists())
            train = frame.loc[metadata["train_indices"], NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            expected_mean = train["age"].mean()
            actual_mean = loaded.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"].mean_[0]
            self.assertAlmostEqual(actual_mean, expected_mean)
            self.assertEqual(len(metadata["transformed_feature_names"]), metadata["transformed_feature_count"])

    def test_rejects_target_missing_or_schema_mismatch(self):
        frame = synthetic_frame()
        frame.loc[0, TARGET_COLUMN] = np.nan
        with self.assertRaises(ValueError):
            fit_and_save_preprocessing(frame, "ignored.joblib", "ignored.json")
        with self.assertRaises(ValueError):
            fit_and_save_preprocessing(synthetic_frame().drop(columns=["sex"]), "ignored.joblib", "ignored.json")

if __name__ == "__main__":
    unittest.main()
