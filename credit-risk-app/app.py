import streamlit as st

st.set_page_config(page_title="Banking Risk Analytics Platform", layout="wide")

st.title("Banking Risk Analytics Platform")

st.write("Select a module")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Portfolio Dashboard"):
        st.switch_page("pages/Portfolio_Dashboard.py")

with col2:
    if st.button("🤖 Model Explainability"):
        st.switch_page("pages/Model_Explainability.py")

with col3:
    if st.button("⚙️ Scenario Analysis"):
        st.switch_page("pages/Scenario_Analysis.py")
