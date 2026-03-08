import streamlit as st
import pandas as pd
import joblib

st.title("Banking Risk Analytics Platform")

st.write("Upload borrower dataset to predict default probability")

# load model
model = joblib.load("credit-risk-app/credit_risk_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        # Predict probability of default
        predictions = model.predict_proba(data)[:, 1]

        data["PD"] = predictions

        # Risk band function
        def risk_band(pd):
            if pd < 0.05:
                return "Low Risk"
            elif pd < 0.15:
                return "Medium Risk"
            elif pd < 0.30:
                return "High Risk"
            else:
                return "Very High Risk"

        # Apply risk band
        data["Risk_Band"] = data["PD"].apply(risk_band)

        # Credit score transformation
        data["Credit_Score"] = (850 - (data["PD"] * 550)).astype(int)

        # Expected loss calculation
        LGD = 0.6
        EXPOSURE = 20000

        data["Expected_Loss"] = data["PD"] * LGD * EXPOSURE

        st.success("Prediction Completed")

        st.dataframe(data.head())

        # Download results
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
