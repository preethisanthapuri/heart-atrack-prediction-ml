"""Flask research interface for the saved Stage B class-probability model."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

FIELD_LABELS = {
    "age": "age", "trtbps": "trtbps", "chol": "chol",
    "thalachh": "thalachh", "oldpeak": "oldpeak", "sex": "sex code",
    "cp": "cp code", "fbs": "fbs code", "restecg": "restecg code",
    "exng": "exng code", "slp": "slp code", "caa": "caa code",
    "thall": "thall code",
}


class InputValidationError(ValueError):
    """Raised when an input payload is incomplete or outside the model schema."""


def _load_artifacts(model_path: Path, metadata_path: Path) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    if not model_path.is_file() or not metadata_path.is_file():
        return None, None, "Model artifacts are missing. Run `python run_final_model.py` from the project root first."
    try:
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"Could not load the saved research model: {exc}"
    if not hasattr(model, "predict_proba") or not isinstance(metadata.get("input_schema"), dict):
        return None, None, "The saved artifact does not match the expected probability-model format."
    if set(metadata.get("target_classes", [])) != {0, 1}:
        return None, None, "The saved model metadata does not describe the expected binary classes."
    schema = metadata["input_schema"]
    if set(schema.get("numeric", {})) != set(NUMERIC_FEATURES) or set(schema.get("categorical_codes", {})) != set(CATEGORICAL_FEATURES):
        return None, None, "The saved input schema does not match the application feature contract."
    return model, metadata, None


def validate_inputs(payload: Any, metadata: dict[str, Any], *, allow_form_strings: bool = False) -> tuple[dict[str, float | int], list[str]]:
    if not isinstance(payload, dict):
        raise InputValidationError("Submit a JSON object containing all model inputs.")
    expected = set(FEATURES)
    missing, extra = expected - set(payload), set(payload) - expected
    problems = []
    if missing:
        problems.append(f"Missing fields: {', '.join(sorted(missing))}.")
    if extra:
        problems.append(f"Unexpected fields: {', '.join(sorted(extra))}.")
    if problems:
        raise InputValidationError(" ".join(problems))

    parsed: dict[str, float | int] = {}
    numeric_ranges = metadata["input_schema"].get("numeric", {})
    category_codes = metadata["input_schema"].get("categorical_codes", {})
    for column in NUMERIC_FEATURES:
        value = payload[column]
        if isinstance(value, bool) or (not isinstance(value, (int, float)) and not allow_form_strings):
            raise InputValidationError(f"{column} must be a numeric value.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InputValidationError(f"{column} must be a numeric value.") from None
        if not math.isfinite(number):
            raise InputValidationError(f"{column} must be finite.")
        parsed[column] = number

    for column in CATEGORICAL_FEATURES:
        value = payload[column]
        if isinstance(value, bool) or (not isinstance(value, (int, float)) and not allow_form_strings):
            raise InputValidationError(f"{column} must be one of the available integer codes.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise InputValidationError(f"{column} must be one of the available integer codes.") from None
        allowed = category_codes.get(column, [])
        if not math.isfinite(numeric_value) or not numeric_value.is_integer() or int(numeric_value) not in allowed:
            raise InputValidationError(f"{column} must be one of these dataset codes: {allowed}.")
        parsed[column] = int(numeric_value)

    warnings = []
    for column in NUMERIC_FEATURES:
        bounds = numeric_ranges.get(column, {})
        low, high = bounds.get("training_min"), bounds.get("training_max")
        if low is not None and high is not None and not low <= parsed[column] <= high:
            warnings.append(
                f"{column} is outside the observed training range ({low:g} to {high:g}); "
                "the model is extrapolating beyond its training data."
            )
    return parsed, warnings


def explain_prediction(model: Any, inputs: dict[str, float | int]) -> list[dict[str, Any]]:
    """Return largest signed linear contributions in model log-odds units."""
    row = pd.DataFrame([{feature: inputs[feature] for feature in FEATURES}], columns=FEATURES)
    transformed = model.named_steps["preprocessor"].transform(row)
    selector = model.named_steps["selector"]
    selected_values = selector.transform(transformed)[0]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()[selector.get_support()]
    coefficients = model.named_steps["classifier"].coef_[0]
    contribution = selected_values * coefficients
    order = np.argsort(-np.abs(contribution), kind="stable")[:5]
    return [{
        "feature": str(feature_names[i]),
        "model_contribution_log_odds": float(contribution[i]),
        "direction": "toward dataset class 1" if contribution[i] >= 0 else "toward dataset class 0",
    } for i in order]


def create_app(*, model: Any | None = None, metadata: dict[str, Any] | None = None,
               model_path: str | Path | None = None,
               metadata_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    artifact, artifact_metadata, error = _load_artifacts(
        Path(model_path or ROOT / "models/final_model.joblib"),
        Path(metadata_path or ROOT / "results/metrics/final_model.json"),
    ) if model is None else (model, metadata, None)
    app.extensions["research_model"] = artifact
    app.extensions["research_model_metadata"] = artifact_metadata
    app.extensions["research_model_error"] = error

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Request is larger than the allowed 16 KB input limit."}), 413

    @app.get("/")
    def index():
        return render_template("index.html", fields=FEATURES,
                               numeric_fields=NUMERIC_FEATURES,
                               categorical_fields=CATEGORICAL_FEATURES,
                               labels=FIELD_LABELS,
                               metadata=app.extensions["research_model_metadata"],
                               model_error=app.extensions["research_model_error"])

    @app.post("/api/predict")
    def predict():
        loaded = app.extensions.get("research_model")
        info = app.extensions.get("research_model_metadata")
        if loaded is None or info is None:
            return jsonify({"error": app.extensions.get("research_model_error") or "Research model unavailable."}), 503
        payload = request.get_json(silent=True)
        try:
            inputs, warnings = validate_inputs(payload, info, allow_form_strings=True)
        except InputValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        row = pd.DataFrame([{feature: inputs[feature] for feature in FEATURES}], columns=FEATURES)
        try:
            probabilities = loaded.predict_proba(row)[0]
            class_order = list(loaded.classes_)
            probability_by_class = {int(label): float(probabilities[i]) for i, label in enumerate(class_order)}
            if set(probability_by_class) != {0, 1} or not np.isfinite(list(probability_by_class.values())).all():
                raise ValueError("Model returned unexpected or non-finite class probabilities")
            result = {
                "predicted_dataset_class": int(max(probability_by_class, key=probability_by_class.get)),
                "probability_class_0": probability_by_class[0],
                "probability_class_1": probability_by_class[1],
                "probability_label": "Estimated probability of dataset class 1 (label meaning unverified)",
                "uncertainty": "Not estimated for the selected final model.",
                "explanation": explain_prediction(loaded, inputs),
                "explanation_note": "Local Logistic Regression contributions in log-odds units; associations with the model output, not causal effects.",
                "warnings": warnings,
            }
        except Exception:
            app.logger.exception("Research model prediction failed")
            return jsonify({"error": "Prediction could not be generated by the saved model."}), 500
        return jsonify(result)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
