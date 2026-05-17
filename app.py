import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title='SVM Insurance Charges Predictor', page_icon='💊', layout='centered')
st.title('Medical Insurance Charges Predictor')
st.write('Use the trained SVM regression pipeline to estimate medical insurance charges.')

MODEL_PATH = Path('svm_insurance_model.joblib')
model = joblib.load(MODEL_PATH)

age = st.slider('Age', min_value=18, max_value=64, value=30)
sex = st.selectbox('Sex', ['female', 'male'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=30.0, step=0.1)
children = st.slider('Children', min_value=0, max_value=5, value=1)
smoker = st.selectbox('Smoker', ['no', 'yes'])
region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

if st.button('Predict Charges'):
    prediction = model.predict(input_data)[0]
    st.success(f'Estimated insurance charges: ${prediction:,.2f}')
    st.dataframe(input_data)

st.caption('Run the app with: streamlit run app.py')
