import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="💊",
    layout="centered"
)

# ---------------- TITLE ---------------- #
st.title("💊 Medical Insurance Charges Predictor")

st.write(
    "Predict medical insurance charges using a trained SVM Regression model."
)

# ---------------- LOAD ARTIFACT ---------------- #
MODEL_PATH = Path("svm_insurance_artifact.joblib")

if not MODEL_PATH.exists():
    st.error("❌ Model file not found!")
    st.info("Please add 'svm_insurance_artifact.joblib' in your project folder.")
    st.stop()

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]

# ---------------- USER INPUTS ---------------- #
st.subheader("Enter Patient Details")

age = st.slider(
    "Age",
    min_value=18,
    max_value=64,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.slider(
    "Number of Children",
    min_value=0,
    max_value=5,
    value=1
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# ---------------- CREATE INPUT DATA ---------------- #
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# ---------------- ENCODING ---------------- #
input_encoded = pd.get_dummies(input_data)

# Add missing columns
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Arrange columns in same order
input_encoded = input_encoded[feature_columns]

# ---------------- SCALING ---------------- #
input_scaled = scaler.transform(input_encoded)

# ---------------- PREDICTION ---------------- #
if st.button("Predict Insurance Charges"):

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"💰 Estimated Medical Insurance Charges: ${prediction:,.2f}"
    )

    st.subheader("Entered Details")
    st.dataframe(input_data)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with Streamlit and Scikit-Learn")