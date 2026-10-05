"""Flask UI/API tests for the final-model research application."""
import unittest
import numpy as np
import pandas as pd

from app.app import create_app
from src.final_model import build_final_pipeline
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class ResearchAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(22)
        n = 64
        cls.y = np.tile([0, 1], n // 2)
        rng.shuffle(cls.y)
        cls.frame = pd.DataFrame({
            "age": rng.integers(35, 76, n), "trtbps": rng.normal(125, 12, n),
            "chol": rng.normal(225, 30, n), "thalachh": rng.normal(145, 18, n),
            "oldpeak": rng.normal(1, 0.7, n), "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n), "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n), "exng": rng.integers(0, 2, n),
            "slp": rng.integers(0, 3, n), "caa": rng.integers(0, 5, n),
            "thall": rng.integers(0, 4, n),
        })
        cls.model = build_final_pipeline(selection_size=6).fit(cls.frame, cls.y)
        cls.metadata = {
            "target_classes": [0, 1],
            "input_schema": {
                "numeric": {col: {"training_min": float(cls.frame[col].min()),
                                  "training_max": float(cls.frame[col].max())} for col in NUMERIC_FEATURES},
                "categorical_codes": {col: sorted(int(v) for v in cls.frame[col].unique())
                                       for col in CATEGORICAL_FEATURES},
            },
        }

    def setUp(self):
        self.client = create_app(model=self.model, metadata=self.metadata).test_client()
        self.payload = self.frame.loc[0, FEATURES].to_dict()

    def test_home_page_has_required_disclaimer_and_all_fields(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("not a medical diagnostic tool", page)
        for name in FEATURES:
            self.assertIn(f'name="{name}"', page)

    def test_prediction_endpoint_returns_probabilities_and_explanation(self):
        response = self.client.post("/api/predict", json=self.payload)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertAlmostEqual(result["probability_class_0"] + result["probability_class_1"], 1.0)
        self.assertIn(result["predicted_dataset_class"], [0, 1])
        self.assertEqual(result["uncertainty"], "Not estimated for the selected final model.")
        self.assertGreater(len(result["explanation"]), 0)

    def test_invalid_schema_code_and_nonfinite_number_are_rejected(self):
        missing = dict(self.payload)
        missing.pop("age")
        self.assertEqual(self.client.post("/api/predict", json=missing).status_code, 400)
        unknown = dict(self.payload, cp=99)
        self.assertEqual(self.client.post("/api/predict", json=unknown).status_code, 400)
        nonfinite = dict(self.payload, age=float("inf"))
        self.assertEqual(self.client.post("/api/predict", json=nonfinite).status_code, 400)

    def test_out_of_training_range_returns_extrapolation_warning(self):
        payload = dict(self.payload, age=-1)
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any("outside the observed training range" in item for item in response.get_json()["warnings"]))


if __name__ == "__main__":
    unittest.main()
