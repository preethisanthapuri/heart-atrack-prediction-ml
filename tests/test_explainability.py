import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.explainability import compute_shap_explanations, summarize_shap_values


class ExplainabilityTests(unittest.TestCase):
    def test_linear_shap_values_add_to_logit_and_summaries_align(self):
        rng = np.random.default_rng(29)
        X = rng.normal(size=(60, 5))
        y = (X[:, 0] - 0.8 * X[:, 2] + rng.normal(scale=0.4, size=60) > 0).astype(int)
        model = LogisticRegression(random_state=42).fit(X, y)
        explanation = compute_shap_explanations(model, X, [f"f{i}" for i in range(5)], background_size=30)
        reconstructed = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model.decision_function(X), atol=1e-6)
        global_table, local_table = summarize_shap_values(explanation, X, [f"f{i}" for i in range(5)], local_row=2)
        self.assertEqual(len(global_table), X.shape[1])
        self.assertEqual(len(local_table), X.shape[1])
        self.assertEqual(global_table.iloc[0]["feature"], "f0")

    def test_rejects_shape_mismatch(self):
        model = LogisticRegression().fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            compute_shap_explanations(model, np.ones((4, 2)), ["only-one"])


if __name__ == "__main__":
    unittest.main()
