import streamlit as st
import pandas as pd
import joblib

st.title("What-If Scenario Analysis")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

model = joblib.load("credit-risk-app/credit_risk_model.pkl")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    borrower_index = st.slider(
        "Select Borrower",
        0,
        len(data)-1,
        0
    )

    borrower = data.iloc[[borrower_index]].copy()

    income_change = st.slider(
        "Income Change %",
        -50,
        50,
        0
    )

    rate_change = st.slider(
        "Interest Rate Change",
        -5,
        5,
        0
    )

    if "annual_inc" in borrower.columns:
        borrower["annual_inc"] *= (1 + income_change/100)

    if "int_rate" in borrower.columns:
        borrower["int_rate"] += rate_change

    new_pd = model.predict_proba(borrower)[:,1][0]

    st.metric("Scenario Default Probability",f"{new_pd:.2%}")
