"""Run the complete unittest suite and persist an actual machine-readable summary."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import unittest

ROOT = Path(__file__).resolve().parent


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_ids: list[str] = []

    def startTest(self, test):
        self.case_ids.append(test.id())
        super().startTest(test)


class RecordingRunner(unittest.TextTestRunner):
    resultclass = RecordingResult


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = RecordingRunner(verbosity=2).run(suite)
    output = {
        "project_name": "heart-atrack-prediction-ml",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "discovered_test_cases": result.testsRun,
        "passed_test_cases": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "success": result.wasSuccessful(),
        "test_case_ids": result.case_ids,
        "failures": [{"test": test.id(), "detail": detail} for test, detail in result.failures],
        "errors": [{"test": test.id(), "detail": detail} for test, detail in result.errors],
    }
    metrics = ROOT / "results/metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "test_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Test summary saved: {metrics / 'test_results.json'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
