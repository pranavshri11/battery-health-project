"""One inference contract for the app and the validated training pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .data_preprocessing import FEATURES, ROOT
except ImportError:
    from data_preprocessing import FEATURES, ROOT

MODEL_PATH = ROOT / "models" / "random_forest_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


def validate_inputs(values):
    frame = pd.DataFrame([values]) if isinstance(values, dict) else pd.DataFrame(values).copy()
    if set(frame.columns) != set(FEATURES) or len(frame.columns) != len(FEATURES):
        raise ValueError("Provide exactly cycle, voltage, temperature, and capacity; target columns are not accepted.")
    frame = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Every input must be a finite number.")
    if ((frame["cycle"] < 0) | (frame["cycle"] != np.floor(frame["cycle"]))).any():
        raise ValueError("Cycle count must be a non-negative integer.")
    if (frame[["voltage", "capacity"]] <= 0).any().any():
        raise ValueError("Voltage and capacity must be greater than zero.")
    if (frame["temperature"] < -273.15).any():
        raise ValueError("Temperature cannot be below absolute zero in degrees Celsius.")
    return frame


def load_predictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    pipeline = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if list(pipeline.feature_names_in_) != FEATURES or metadata["features"] != FEATURES:
        raise ValueError("The model feature order does not match the input contract.")
    try:
        from .data_preprocessing import sha256
    except ImportError:
        from data_preprocessing import sha256
    if sha256(model_path) != metadata["artifact_sha256"]:
        raise ValueError("The model does not match its metadata; deploy the matching files together.")
    return pipeline, metadata


def predict_soh(pipeline, values):
    prediction = np.asarray(pipeline.predict(validate_inputs(values)), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("The model returned a non-finite prediction.")
    return prediction
