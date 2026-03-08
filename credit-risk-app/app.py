import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction System")

st.write("Upload borrower dataset to predict default probability")

# load model
model = joblib.load("credit-risk-app/credit_risk_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        predictions = model.predict_proba(data)[:,1]

        data["Default_Probability"] = predictions

        st.success("Prediction Completed")

        st.dataframe(data.head())

        csv = data.to_csv(index=False).encode()

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="credit_risk_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Feature mismatch with training model")
        st.write(e)
