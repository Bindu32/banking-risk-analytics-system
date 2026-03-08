import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Model Explainability")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

model = joblib.load("credit-risk-app/credit_risk_model.pkl")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    explainer = shap.Explainer(model)

    shap_values = explainer(data)

    st.subheader("Feature Importance")

    fig = plt.figure()

    shap.summary_plot(shap_values,data,show=False)

    st.pyplot(fig)
