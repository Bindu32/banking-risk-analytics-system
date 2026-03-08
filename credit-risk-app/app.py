import streamlit as st
import joblib
import pandas as pd

model = joblib.load("credit_risk_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Credit Risk Prediction")

st.write("Enter borrower details to predict default risk")

loan_amnt = st.number_input("Loan Amount", 1000, 50000, 10000)
annual_inc = st.number_input("Annual Income", 10000, 200000, 50000)
dti = st.number_input("Debt-to-Income Ratio", 0.0, 50.0, 10.0)
fico = st.number_input("FICO Score", 300, 850, 700)

input_data = pd.DataFrame(
[[loan_amnt, annual_inc, dti, fico]],
columns=["loan_amnt","annual_inc","dti","fico"]
)

if st.button("Predict"):

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.1:
        risk = "Low Risk"
    elif prob < 0.2:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.write("Default Probability:", round(prob,3))
    st.write("Risk Category:", risk)
