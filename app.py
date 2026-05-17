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

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = Path("svm_insurance_artifact.joblib")

if not MODEL_PATH.exists():
    st.error("❌ Model file not found!")
    st.stop()

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]

# ---------------- USER INPUTS ---------------- #
st.subheader("Enter Patient Details")

age = st.slider("Age", 18, 64, 30)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.slider(
    "Children",
    0,
    5,
    1
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
input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

# Create dataframe
input_df = pd.DataFrame([input_data])

# Ensure same column order
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ---------------- SCALE DATA ---------------- #
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ---------------- #
if st.button("Predict Insurance Charges"):

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"💰 Estimated Insurance Charges: ${prediction:,.2f}"
    )

    st.subheader("Entered Details")
    st.dataframe(input_df)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with Streamlit and Scikit-Learn")