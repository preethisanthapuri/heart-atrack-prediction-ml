"""Tests for final-results reconciliation with recorded ablation evidence."""
import unittest
import pandas as pd
from src.final_results import build_final_stage_table


class FinalResultsTests(unittest.TestCase):
    def test_marks_only_metadata_selected_stage_and_preserves_recorded_values(self):
        ablation = pd.DataFrame({"stage": list("ABCDEF"), "accuracy": [.8] * 6,
                                 "sensitivity": [.8] * 6, "specificity": [.8] * 6,
                                 "f1": [.8] * 6, "roc_auc": [.8] * 6,
                                 "pr_auc": [.8] * 6, "brier": [.2] * 6})
        table = build_final_stage_table(ablation, {"selected_stage": "B"})
        self.assertEqual(table.selected_final_candidate.sum(), 1)
        self.assertEqual(table.loc[table.selected_final_candidate, "stage"].item(), "B")
        self.assertEqual(table.loc[table.stage == "D", "roc_auc"].item(), .8)
        self.assertTrue(table.metric_basis.str.contains("training partition only").all())

    def test_rejects_incomplete_stages(self):
        ablation = pd.DataFrame({"stage": ["A"], "accuracy": [.8]})
        with self.assertRaises(ValueError):
            build_final_stage_table(ablation, {"selected_stage": "A"})


if __name__ == "__main__":
    unittest.main()
