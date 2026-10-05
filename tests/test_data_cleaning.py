import unittest
import pandas as pd
from src.data_cleaning import clean_dataset

class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "age": [50, 50, 51, 0], "sex": [1, 1, 1, 0], "cp": [2, 2, 2, 1],
            "trtbps": [120, 120, 120, 120], "chol": [200, 200, 200, 200],
            "fbs": [0, 0, 0, 0], "restecg": [1, 1, 1, 0], "thalachh": [150, 150, 150, 150],
            "exng": [0, 0, 0, 0], "oldpeak": [0.0, 0.0, 0.0, 0.0], "slp": [2, 2, 2, 1],
            "caa": [0, 0, 0, 0], "thall": [2, 2, 2, 2], "output": [1, 1, 1, 0],
        })

    def test_removes_duplicate_and_invalid_positive_measurement(self):
        cleaned, report = clean_dataset(self.frame)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)
        self.assertEqual(report["rows_with_any_invalid_value_removed"], 1)
        self.assertEqual(cleaned.iloc[0]["output"], 1)

    def test_retains_iqr_outliers_and_does_not_mutate_input(self):
        source = pd.read_csv("data/raw/heart_disease.csv")
        before = source.copy(deep=True)
        cleaned, report = clean_dataset(source)
        pd.testing.assert_frame_equal(source, before)
        self.assertEqual(len(cleaned), 302)
        self.assertGreater(report["outlier_investigation_iqr"]["chol"]["flag_count"], 0)
        self.assertEqual(report["rows_with_any_invalid_value_removed"], 0)

    def test_rejects_schema_drift(self):
        with self.assertRaises(ValueError):
            clean_dataset(self.frame.drop(columns=["output"]))

if __name__ == "__main__":
    unittest.main()



