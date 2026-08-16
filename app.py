import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Battery Health Prediction",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 Battery Health Prediction")

st.write(
    "Predict the **State of Health (SOH)** of an EV battery using a trained Random Forest model."
)

st.divider()

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

cycle = st.number_input(
    "Cycle Count",
    min_value=0,
    value=500
)

voltage = st.number_input(
    "Voltage (V)",
    value=3.7,
    format="%.3f"
)

temperature = st.number_input(
    "Temperature (°C)",
    value=25.0,
    format="%.2f"
)

capacity = st.number_input(
    "Capacity (Ah)",
    value=2.5,
    format="%.3f"
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Battery Health"):

    input_data = pd.DataFrame({
        "cycle":[cycle],
        "voltage":[voltage],
        "temperature":[temperature],
        "capacity":[capacity]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Battery SOH: **{prediction:.4f}**")