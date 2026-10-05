import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.evaluation import screen_baseline_models
from src.preprocessing import FEATURES, TARGET_COLUMN, build_preprocessing_pipeline


class BaselineModelTests(unittest.TestCase):
    def make_data(self):
        rows = 40
        frame = pd.DataFrame({column: np.arange(rows, dtype=float) + index for index, column in enumerate(FEATURES)})
        frame[TARGET_COLUMN] = np.tile([0, 1], rows // 2)
        train = list(range(30))
        test = list(range(30, rows))
        metadata = {"fitted_on": "training split only", "train_indices": train, "test_indices": test}
        prep = build_preprocessing_pipeline()
        prep.fit(frame.loc[train, FEATURES])
        return frame, metadata, prep

    def test_screening_fits_and_reports_holdout_accuracy(self):
        frame, metadata, prep = self.make_data()
        with tempfile.TemporaryDirectory() as output_dir:
            results = screen_baseline_models(
                frame, metadata, prep,
                {"Dummy": DummyClassifier(strategy="most_frequent")}, output_dir
            )
            self.assertEqual(results.iloc[0]["status"], "completed")
            self.assertGreaterEqual(results.iloc[0]["holdout_accuracy"], 0)
            self.assertLessEqual(results.iloc[0]["holdout_accuracy"], 1)
            self.assertTrue(results.iloc[0]["model_artifact"])

    def test_rejects_split_overlap(self):
        frame, metadata, prep = self.make_data()
        metadata["test_indices"] = [29, *metadata["test_indices"][1:]]
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(ValueError):
                screen_baseline_models(frame, metadata, prep, {"Dummy": DummyClassifier()}, output_dir)

    def test_rejects_non_training_fitted_preprocessor(self):
        frame, metadata, prep = self.make_data()
        metadata["fitted_on"] = "all rows"
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(ValueError):
                screen_baseline_models(frame, metadata, prep, {"Dummy": DummyClassifier()}, output_dir)


if __name__ == "__main__":
    unittest.main()
