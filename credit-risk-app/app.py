import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Banking Risk Analytics Platform")

st.write("Upload borrower dataset to analyze portfolio credit risk")

# load model
model = joblib.load("credit-risk-app/credit_risk_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    try:
        # Predict PD
        predictions = model.predict_proba(data)[:, 1]
        data["PD"] = predictions

        # Risk band
        def risk_band(pd):
            if pd < 0.05:
                return "Low Risk"
            elif pd < 0.15:
                return "Medium Risk"
            elif pd < 0.30:
                return "High Risk"
            else:
                return "Very High Risk"

        data["Risk_Band"] = data["PD"].apply(risk_band)

        # Credit score
        data["Credit_Score"] = (850 - (data["PD"] * 550)).astype(int)

        # Expected loss
        LGD = 0.6
        data["Expected_Loss"] = data["PD"] * LGD * data["loan_amnt"]

        st.success("Predictions completed")

        st.subheader("Prediction Results")
        st.dataframe(data.head())

        # ---------- PORTFOLIO FILTER ----------

        st.subheader("Portfolio Filter")

        selected_risk = st.multiselect(
            "Select Risk Bands",
            options=data["Risk_Band"].unique(),
            default=data["Risk_Band"].unique()
        )

        filtered_data = data[data["Risk_Band"].isin(selected_risk)]

        # ---------- PORTFOLIO METRICS ----------

        st.subheader("Portfolio Risk Summary")

        avg_pd = filtered_data["PD"].mean()
        total_el = filtered_data["Expected_Loss"].sum()
        high_risk_pct = (filtered_data["Risk_Band"] == "Very High Risk").mean() * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Average PD", f"{avg_pd:.2%}")
        col2.metric("Total Expected Loss", f"${total_el:,.0f}")
        col3.metric("Very High Risk %", f"{high_risk_pct:.1f}%")

        # ---------- TOP RISKY BORROWERS ----------

        st.subheader("Top 20 Riskiest Borrowers")

        top_risk = filtered_data.sort_values("PD", ascending=False).head(20)

        st.dataframe(top_risk)

        # ---------- EXPECTED LOSS BY RISK BAND ----------

        st.subheader("Expected Loss by Risk Band")

        loss_by_band = filtered_data.groupby("Risk_Band")["Expected_Loss"].sum()

        fig = plt.figure()
        loss_by_band.plot(kind="bar")
        plt.ylabel("Expected Loss")

        st.pyplot(fig)

        # ---------- PD DISTRIBUTION ----------

        st.subheader("PD Distribution")

        fig2 = plt.figure()
        filtered_data["PD"].hist(bins=30)
        plt.xlabel("Probability of Default")
        plt.ylabel("Borrower Count")

        st.pyplot(fig2)

        # ---------- STRESS TEST ----------

        st.subheader("Stress Scenario (PD increases by 50%)")

        stressed_pd = filtered_data["PD"] * 1.5
        stressed_el = (stressed_pd * LGD * filtered_data["loan_amnt"]).sum()

        st.metric(
            "Stressed Expected Loss",
            f"${stressed_el:,.0f}"
        )

        # ---------- DOWNLOAD ----------

        csv = filtered_data.to_csv(index=False).encode()

        st.download_button(
            label="Download Portfolio Results",
            data=csv,
            file_name="credit_risk_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Feature mismatch with training model")
        st.write(e)
