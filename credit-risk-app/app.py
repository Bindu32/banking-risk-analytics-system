import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Banking Risk Analytics Platform", layout="wide")

st.title("Banking Risk Analytics Platform")

# Sidebar
st.sidebar.header("Portfolio Controls")

uploaded_file = st.sidebar.file_uploader("Upload Borrower Dataset", type=["csv"])

model = joblib.load("credit-risk-app/credit_risk_model.pkl")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Predict PD
    predictions = model.predict_proba(data)[:,1]
    data["PD"] = predictions

    # Risk bands
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

    # Sidebar filter
    risk_filter = st.sidebar.multiselect(
        "Filter Risk Bands",
        data["Risk_Band"].unique(),
        default=data["Risk_Band"].unique()
    )

    filtered = data[data["Risk_Band"].isin(risk_filter)]

    # ---------- KPI DASHBOARD ----------

    avg_pd = filtered["PD"].mean()
    total_el = filtered["Expected_Loss"].sum()
    portfolio_size = len(filtered)
    high_risk = (filtered["Risk_Band"] == "Very High Risk").mean()*100

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Portfolio Size", portfolio_size)
    col2.metric("Average PD", f"{avg_pd:.2%}")
    col3.metric("Total Expected Loss", f"${total_el:,.0f}")
    col4.metric("Very High Risk %", f"{high_risk:.1f}%")

    st.divider()

    # ---------- CHARTS ----------

    colA,colB = st.columns(2)

    with colA:
        st.subheader("Risk Band Distribution")

        fig = plt.figure()
        filtered["Risk_Band"].value_counts().plot(kind="bar")
        plt.ylabel("Borrowers")

        st.pyplot(fig)

    with colB:
        st.subheader("PD Distribution")

        fig2 = plt.figure()
        filtered["PD"].hist(bins=30)
        plt.xlabel("Probability of Default")

        st.pyplot(fig2)

    st.divider()

    # ---------- EXPECTED LOSS BY BAND ----------

    st.subheader("Expected Loss by Risk Band")

    loss_band = filtered.groupby("Risk_Band")["Expected_Loss"].sum()

    fig3 = plt.figure()
    loss_band.plot(kind="bar")

    st.pyplot(fig3)

    st.divider()

    # ---------- TOP RISKY BORROWERS ----------

    st.subheader("Top 20 Riskiest Borrowers")

    risky = filtered.sort_values("PD",ascending=False).head(20)

    st.dataframe(risky)

    st.divider()

    # ---------- STRESS TEST ----------

    st.subheader("Stress Test Scenario")

    stressed_pd = filtered["PD"]*1.5
    stressed_loss = (stressed_pd*LGD*filtered["loan_amnt"]).sum()

    st.metric("Stressed Expected Loss",f"${stressed_loss:,.0f}")

    st.divider()

    # ---------- DOWNLOAD ----------

    csv = filtered.to_csv(index=False).encode()

    st.download_button(
        "Download Portfolio Results",
        csv,
        "credit_risk_results.csv",
        "text/csv"
    )
