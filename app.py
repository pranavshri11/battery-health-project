"""Streamlit interface for the project's saved Random Forest and scaler."""

from datetime import datetime, timezone
import logging
from math import isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURES, load_dataset as read_validated_dataset
from src.prediction import MODEL_PATH, METADATA_PATH, load_predictor as read_predictor, predict_soh


ROOT = Path(__file__).resolve().parent
LABELS = {
    "cycle": "Cycle count", "voltage": "Voltage (V)",
    "temperature": "Temperature (°C)", "capacity": "Capacity (Ah)",
}
DEFAULTS = {"cycle": 42, "voltage": 3.5, "temperature": 32.0, "capacity": 1.75}

st.set_page_config(page_title="Battery Lab | Health Prediction", page_icon="🔋", layout="wide")
st.html((ROOT / "assets" / "app.css").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_predictor(model_stamp, metadata_stamp):
    return read_predictor()


@st.cache_data(show_spinner=False)
def load_dataset(dataset_stamp):
    return read_validated_dataset()


def set_inputs(values):
    for feature in FEATURES:
        st.session_state[feature] = int(values[feature]) if feature == "cycle" else float(values[feature])
    st.session_state.pop("prediction", None)


def gauge(value=None):
    # Only the visual arc is limited to 0–100. The model's value is never clipped.
    progress = 0 if value is None else min(100, max(0, value))
    label = "—" if value is None else f"{value:.1f}<span>%</span>"
    accessible = "No prediction yet" if value is None else f"Predicted state of health: {value:.1f} percent"
    return f"""<div class="gauge-wrap" role="img" aria-label="{accessible}">
      <div class="gauge" style="--progress:{progress}%">
        <div class="gauge-core"><div class="gauge-number">{label}</div>
        <div class="gauge-label">STATE OF HEALTH</div></div>
      </div>
      <div class="gauge-scale"><span>0%</span><span>100% reference</span></div>
    </div>"""


st.html("""<div class="brand-row"><div class="brand"><span class="brand-icon">ϟ</span>
    BATTERY<span class="brand-light">LAB</span></div><span class="brand-tag">EV BATTERY INSIGHTS</span></div>
    <div class="hero"><div><div class="eyebrow">BATTERY HEALTH PREDICTION</div>
    <h1>A clearer view of<br><span>battery health.</span></h1>
    <p>Turn battery measurements into an estimated State of Health.</p></div>
    <div class="hero-note"><span class="status-dot"></span> RANDOM FOREST<br>
    <small>Four inputs. One useful insight.</small></div></div>""")

try:
    model, metadata = load_predictor(MODEL_PATH.stat().st_mtime_ns, METADATA_PATH.stat().st_mtime_ns)
    dataset = load_dataset((ROOT / "data" / "dataset.csv").stat().st_mtime_ns)
except Exception:
    logging.exception("Could not load battery prediction resources")
    st.error("The prediction resources could not be loaded. Check the saved model, scaler, dataset, and installed package versions.")
    st.stop()

ranges = pd.DataFrame(metadata["training_ranges"])
sample_pool = dataset[dataset["battery_id"].isin(metadata["development_batteries"])]
sample = sample_pool.loc[(sample_pool["soh"] - sample_pool["soh"].median()).abs().idxmin()]
for feature, default in DEFAULTS.items():
    st.session_state.setdefault(feature, default)

st.html(f"""<div class="facts">
    <div><span>DATASET</span><strong>NASA battery ageing</strong></div>
    <div><span>CYCLE RECORDS</span><strong>{len(dataset):,}</strong></div>
    <div><span>BATTERIES</span><strong>{dataset['battery_id'].nunique()}</strong></div>
    <div><span>PREDICTION MODEL</span><strong>Random Forest</strong></div></div>""")

inputs_col, result_col = st.columns([1.16, 1], gap="large")
with inputs_col:
    with st.container(border=True, key="input_card"):
        st.html('<div class="panel-kicker">01 / BATTERY INPUTS</div><h2>Enter your measurements</h2><p class="panel-copy">Use readings from the same battery cycle.</p>')
        a, b = st.columns(2)
        with a:
            st.number_input("Cycle count", min_value=0, step=1, key="cycle", help="Number of completed charge/discharge cycles.")
            st.number_input("Temperature (°C)", min_value=-273.15, step=0.5, format="%.2f", key="temperature", help="Battery temperature in degrees Celsius.")
        with b:
            st.number_input("Voltage (V)", min_value=0.001, step=0.01, format="%.3f", key="voltage", help="Battery voltage measured in volts.")
            st.number_input("Capacity (Ah)", min_value=0.001, step=0.01, format="%.3f", key="capacity", help="Measured discharge capacity in ampere-hours, not the rated capacity on a label.")

        values = {feature: st.session_state[feature] for feature in FEATURES}
        valid = all(isfinite(float(value)) for value in values.values())
        outside = [feature for feature, value in values.items()
                   if not ranges.loc["min", feature] <= value <= ranges.loc["max", feature]]
        if not valid:
            st.error("Enter a finite number in every field.")
        elif outside:
            names = ", ".join(LABELS[feature] for feature in outside)
            st.warning(f"Outside the training range: {names}. This estimate may be less reliable.")
        else:
            st.html('<div class="range-check"><span class="status-dot"></span> All inputs are within the model training ranges</div>')

        predict_clicked = st.button("Predict battery health →", type="primary", width="stretch", disabled=not valid)
        sample_col, reset_col = st.columns(2)
        with sample_col:
            st.button("Load sample", on_click=set_inputs, args=(sample.to_dict(),), width="stretch", help="Load a real record from the project dataset.")
        with reset_col:
            st.button("Reset inputs", on_click=set_inputs, args=(DEFAULTS,), width="stretch")

        if predict_clicked:
            # Discard an old result even if this prediction fails.
            st.session_state.pop("prediction", None)
            try:
                prediction = float(predict_soh(model, values)[0])
                st.session_state["prediction"] = {
                    "inputs": values.copy(), "soh": prediction,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception:
                logging.exception("Battery prediction failed")
                st.error("The prediction could not be completed. Check the model and scaler compatibility, then try again.")

with result_col:
    with st.container(border=True, key="result_card"):
        result = st.session_state.get("prediction")
        stale = result is not None and result["inputs"] != values
        if stale:
            result = None
        st.html('<div class="panel-kicker">02 / YOUR RESULT</div><h2>Estimated battery health</h2>')
        st.html(gauge(None if result is None else result["soh"] * 100))
        if result is None:
            st.html('<div class="result-empty"><strong>' + ("Measurements updated" if stale else "Ready when you are") + '</strong><p>' + ("Run a new prediction to see the updated estimate." if stale else "Enter your measurements and select<br>Predict battery health to see your result.") + '</p></div>')
        else:
            percentage = result["soh"] * 100
            st.html(f'<div class="result-summary"><span class="result-badge">PREDICTION COMPLETE</span><p>Estimated capacity relative to first observed capacity:<br><strong>{percentage:.1f}%</strong></p></div>')
            if not 0 <= percentage <= 100:
                st.info("The model returned a value outside 0–100%. The value is shown as predicted; the gauge is capped at its visual limits.")
            export = pd.DataFrame([{
                **result["inputs"], "predicted_soh_ratio": result["soh"],
                "predicted_soh_percent": percentage, "model": "Random Forest",
                "outside_training_range": bool(outside), "predicted_at_utc": result["created_at"],
                "model_sha256": metadata["artifact_sha256"],
            }])
            st.download_button("Download prediction · CSV", export.to_csv(index=False).encode("utf-8"),
                               "battery_health_prediction.csv", "text/csv", width="stretch")

with st.expander("Understand the inputs & prediction"):
    about, reference = st.columns([1, 1.2], gap="large")
    with about:
        st.markdown("**What does SOH mean?**")
        st.write("In this dataset, State of Health is measured capacity divided by the battery's first observed capacity. An estimate of 85% refers to that reference capacity. It is different from the current charge level (SOC).")
        st.markdown("**How it works**")
        st.write("Your four measurements are scaled using the saved training scaler, then passed to the trained Random Forest. Its SOH ratio is converted to a percentage.")
        st.caption(f"Saved model: {metadata['rf_parameters']['n_estimators']} trees. Capacity also defines the SOH labels, so this is a capacity-informed estimate, not independent health forecasting. Sample inputs are a demonstration, not a validation test.")
        st.markdown("**Measured validation results**")
        for key, label in [("rf_unseen_batteries", "Unseen batteries"), ("rf_later_cycles", "Later cycles")]:
            scores = metadata["metrics"][key]
            st.write(f"{label}: MAE {scores['MAE'] * 100:.2f} percentage points · R² {scores['R2']:.3f} · {scores['records']} records from {scores['batteries']} batteries.")
        st.caption("Selection used only early development cycles. These are two different validation settings; the later-cycle check uses a separately fitted model. The app serves the model evaluated on unseen batteries. No calibrated confidence interval is available.")
    with reference:
        st.markdown("**Ranges observed in the training data**")
        reference_table = pd.DataFrame({
            "Measurement": [LABELS[feature] for feature in FEATURES],
            "Minimum": [f"{ranges.loc['min', feature]:.3f}" for feature in FEATURES],
            "Maximum": [f"{ranges.loc['max', feature]:.3f}" for feature in FEATURES],
        })
        st.dataframe(reference_table, hide_index=True, width="stretch")
        st.caption("These ranges come only from the model's training records. They are not operating limits or a guarantee of accuracy. Unseen-battery performance remains limited, and new combinations can still differ from the training data.")

st.html('<div class="footer"><span>BATTERY LAB <span class="footer-dot">/</span> EV State of Health</span><span>NASA battery ageing dataset · Random Forest</span></div>')
