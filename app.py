import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="🏠 House Price Prediction App", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter details below to estimate the house sale price.")

# Example inputs based on top important features
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=200, max_value=6000, value=1500)
GarageCars = st.slider("Garage Capacity (Cars)", 0, 4, 1)
TotalBsmtSF = st.number_input("Basement Square Feet", min_value=0, max_value=3000, value=800)
FullBath = st.slider("Full Bathrooms", 0, 4, 2)
YearBuilt = st.number_input("Year Built", min_value=1850, max_value=2025, value=2000)

# Convert input into dataframe format model expects
input_data = pd.DataFrame({
    "OverallQual": [OverallQual],
    "GrLivArea": [GrLivArea],
    "GarageCars": [GarageCars],
    "TotalBsmtSF": [TotalBsmtSF],
    "FullBath": [FullBath],
    "YearBuilt": [YearBuilt]
})

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
