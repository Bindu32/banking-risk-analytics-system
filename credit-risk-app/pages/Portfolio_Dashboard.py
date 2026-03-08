import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Portfolio Risk Dashboard")

uploaded_file = st.file_uploader("Upload Portfolio Dataset", type=["csv"])

model = joblib.load("credit-risk-app/credit_risk_model.pkl")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    pd_pred = model.predict_proba(data)[:,1]
    data["PD"] = pd_pred

    LGD = 0.6
    data["Expected_Loss"] = data["PD"] * LGD * data["loan_amnt"]

    col1,col2,col3 = st.columns(3)

    col1.metric("Portfolio Size", len(data))
    col2.metric("Average PD", f"{data['PD'].mean():.2%}")
    col3.metric("Total Expected Loss", f"${data['Expected_Loss'].sum():,.0f}")

    st.subheader("PD Distribution")

    fig = plt.figure()
    data["PD"].hist(bins=30)

    st.pyplot(fig)

    st.subheader("Top Risky Borrowers")

    st.dataframe(data.sort_values("PD",ascending=False).head(20))
