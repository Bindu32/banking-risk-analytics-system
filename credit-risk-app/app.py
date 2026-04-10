import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import shap
import os
from datetime import datetime

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Banking Risk Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark financial theme */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1526 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #38bdf8 !important;
    font-weight: 600;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem !important;
}

/* Divider */
hr {
    border-color: #1e3a5f !important;
    margin: 1.5rem 0;
}

/* Section headers */
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
    box-shadow: 0 0 20px rgba(56,189,248,0.3);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46 0%, #059669 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Sliders */
[data-testid="stSlider"] .stSlider {
    accent-color: #38bdf8;
}

/* Selectbox, multiselect */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
    color: #e2e8f0;
}

/* Alert boxes */
.stAlert {
    border-radius: 8px;
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: #0d1526;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
}

/* Badge style for risk bands */
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-low     { background:#064e3b; color:#34d399; }
.badge-medium  { background:#451a03; color:#fb923c; }
.badge-high    { background:#4a1d2e; color:#f472b6; }
.badge-vhigh   { background:#450a0a; color:#f87171; }

/* Top header banner */
.header-banner {
    background: linear-gradient(90deg, #0d1526 0%, #0f2040 50%, #0d1526 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: -0.02em;
}
.header-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.live-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1} 50%{opacity:0.3}
}
.status-live {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────── MATPLOTLIB DARK THEME ───────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a0e1a',
    'axes.facecolor':    '#0d1526',
    'axes.edgecolor':    '#1e3a5f',
    'axes.labelcolor':   '#94a3b8',
    'xtick.color':       '#64748b',
    'ytick.color':       '#64748b',
    'text.color':        '#e2e8f0',
    'grid.color':        '#1e3a5f',
    'grid.alpha':        0.6,
    'axes.grid':         True,
    'font.family':       'monospace',
})
CHART_COLORS = ['#38bdf8','#818cf8','#34d399','#fb923c','#f472b6','#fbbf24']

# ─────────────────────────── HELPERS ───────────────────────────────
def risk_band(pd_val):
    if pd_val < 0.05:   return "Low Risk"
    elif pd_val < 0.15: return "Medium Risk"
    elif pd_val < 0.30: return "High Risk"
    else:               return "Very High Risk"

BAND_COLORS = {
    "Low Risk":       "#34d399",
    "Medium Risk":    "#fb923c",
    "High Risk":      "#f472b6",
    "Very High Risk": "#f87171",
}
BAND_ORDER = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

def enrich_data(data, model):
    X = data.copy()
    for col in ["PD","Risk_Band","Credit_Score","Expected_Loss"]:
        X = X.drop(columns=[col], errors="ignore")
    predictions = model.predict_proba(X)[:, 1]
    data = data.copy()
    data["PD"] = predictions
    data["Risk_Band"] = data["PD"].apply(risk_band)
    data["Credit_Score"] = (850 - (data["PD"] * 550)).astype(int)
    LGD = 0.6
    data["Expected_Loss"] = data["PD"] * LGD * data["loan_amnt"]
    return data

# ─────────────────────── HEADER BANNER ─────────────────────────────
st.markdown(f"""
<div class="header-banner">
  <div>
    <div class="header-title">🏦 Banking Risk Analytics Platform</div>
    <div class="header-sub">Credit Portfolio Risk Management · Model-Driven Insights · Basel III Compliant</div>
  </div>
  <div>
    <span class="live-dot"></span>
    <span class="status-live">LIVE ANALYTICS · {datetime.now().strftime('%d %b %Y  %H:%M')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Controls")
    st.divider()

    # ── Model load ──
    MODEL_PATH = "credit-risk-app/credit_risk_model.pkl"
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.success("✅ Model loaded")
        except Exception as e:
            st.error(f"Model load error: {e}")
    else:
        st.warning("⚠️ Model file not found at `credit-risk-app/credit_risk_model.pkl`")

    st.divider()

    # ── Data source selection ──
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose input method",
        ["📄 Use Sample Dataset", "⬆️ Upload CSV"],
        index=0
    )

    raw_data = None

    if data_source == "📄 Use Sample Dataset":
        SAMPLE_PATH = "data/sample_credit_input.csv"
        if os.path.exists(SAMPLE_PATH):
            raw_data = pd.read_csv(SAMPLE_PATH)
            st.info(f"✅ Loaded `{SAMPLE_PATH}` — **{len(raw_data):,}** records")
        else:
            st.error(f"❌ `{SAMPLE_PATH}` not found in working directory.")
            st.caption("Place `sample_credit_input.csv` next to `app.py`.")

    else:  # Upload CSV
        uploaded_file = st.file_uploader(
            "Upload Borrower Dataset (.csv)",
            type=["csv"],
            help="CSV must contain the same feature columns the model was trained on."
        )
        if uploaded_file:
            raw_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded — **{len(raw_data):,}** records")

    st.divider()

    if raw_data is not None and model is not None:
        st.markdown("### 🔍 Risk Band Filter")
        all_bands = BAND_ORDER
        risk_filter = st.multiselect(
            "Show Risk Bands",
            all_bands,
            default=all_bands
        )

        st.divider()
        st.markdown("### ⚠️ Stress Parameters")
        stress_mult = st.slider("PD Stress Multiplier", 1.0, 3.0, 1.5, 0.1,
                                help="Multiply all PDs by this factor for stress test")
        lgd_override = st.slider("LGD Override (%)", 30, 90, 60, 5,
                                 help="Loss Given Default assumption")
        LGD = lgd_override / 100

# ────────────────────── MAIN ANALYTICS ────────────────────────────
if raw_data is None or model is None:
    st.info("👈 Configure your data source and model in the sidebar to begin analysis.")
    st.stop()

# Enrich
data = enrich_data(raw_data, model)
filtered = data[data["Risk_Band"].isin(risk_filter)] if risk_filter else data

if filtered.empty:
    st.warning("No records match the selected risk bands.")
    st.stop()

# ══════════════════════ KPI DASHBOARD ═════════════════════════════
st.markdown("## 📊 Portfolio Summary")

avg_pd    = filtered["PD"].mean()
total_el  = filtered["Expected_Loss"].sum()
port_size = len(filtered)
vhigh_pct = (filtered["Risk_Band"] == "Very High Risk").mean() * 100
avg_score = filtered["Credit_Score"].mean()
total_exp = filtered["loan_amnt"].sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Portfolio Size",      f"{port_size:,}")
c2.metric("Avg Credit Score",    f"{avg_score:.0f}")
c3.metric("Average PD",          f"{avg_pd:.2%}")
c4.metric("Total Exposure",      f"${total_exp:,.0f}")
c5.metric("Total Expected Loss", f"${total_el:,.0f}")
c6.metric("Very High Risk %",    f"{vhigh_pct:.1f}%")

st.divider()

# ══════════════════════ RISK DISTRIBUTION CHARTS ══════════════════
st.markdown("## 📈 Risk Distribution")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("#### Risk Band Distribution")
    counts = filtered["Risk_Band"].value_counts().reindex(BAND_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(counts.index, counts.values,
                  color=[BAND_COLORS[b] for b in counts.index],
                  edgecolor='#0a0e1a', linewidth=1.2, width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(val)), ha='center', va='bottom', fontsize=8,
                color='#94a3b8', fontfamily='monospace')
    ax.set_ylabel("Borrowers")
    ax.set_xticklabels(counts.index, rotation=20, ha='right', fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.markdown("#### PD Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.hist(filtered["PD"], bins=40, color='#38bdf8', edgecolor='#0a0e1a',
             alpha=0.85, linewidth=0.5)
    ax2.axvline(avg_pd, color='#f87171', linestyle='--', linewidth=1.5,
                label=f'Mean PD: {avg_pd:.2%}')
    ax2.set_xlabel("Probability of Default")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig2)

with col_c:
    st.markdown("#### Credit Score Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    ax3.hist(filtered["Credit_Score"], bins=40, color='#818cf8',
             edgecolor='#0a0e1a', alpha=0.85, linewidth=0.5)
    ax3.axvline(avg_score, color='#34d399', linestyle='--', linewidth=1.5,
                label=f'Mean: {avg_score:.0f}')
    ax3.set_xlabel("Credit Score")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig3)

st.divider()

# ══════════════════════ EXPECTED LOSS ════════════════════════════
st.markdown("## 💰 Expected Loss Analysis")

col_d, col_e = st.columns(2)

with col_d:
    st.markdown("#### Expected Loss by Risk Band")
    loss_band = filtered.groupby("Risk_Band")["Expected_Loss"].sum().reindex(BAND_ORDER).dropna()
    fig4, ax4 = plt.subplots(figsize=(5.5, 3.8))
    bars4 = ax4.barh(loss_band.index, loss_band.values,
                     color=[BAND_COLORS[b] for b in loss_band.index],
                     edgecolor='#0a0e1a', linewidth=1, height=0.55)
    for bar, val in zip(bars4, loss_band.values):
        ax4.text(bar.get_width() + loss_band.max()*0.01,
                 bar.get_y() + bar.get_height()/2,
                 f"${val/1e6:.2f}M" if val >= 1e6 else f"${val:,.0f}",
                 va='center', fontsize=7.5, color='#94a3b8')
    ax4.set_xlabel("Expected Loss ($)")
    plt.tight_layout()
    st.pyplot(fig4)

with col_e:
    st.markdown("#### Exposure vs Expected Loss (Bubble)")
    fig5, ax5 = plt.subplots(figsize=(5.5, 3.8))
    band_agg = filtered.groupby("Risk_Band").agg(
        Exposure=("loan_amnt", "sum"),
        EL=("Expected_Loss", "sum"),
        AvgPD=("PD", "mean"),
        Count=("PD", "count")
    ).reindex(BAND_ORDER).dropna()
    for band, row in band_agg.iterrows():
        ax5.scatter(row["Exposure"]/1e6, row["EL"]/1e6,
                    s=row["Count"]/port_size*3000,
                    color=BAND_COLORS[band], alpha=0.75,
                    edgecolors='#0a0e1a', linewidth=1, label=band)
    ax5.set_xlabel("Total Exposure ($M)")
    ax5.set_ylabel("Expected Loss ($M)")
    ax5.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    st.pyplot(fig5)

st.divider()

# ══════════════════════ TOP RISKY BORROWERS ═══════════════════════
st.markdown("## 🚨 Top 20 Riskiest Borrowers")

risky = filtered.sort_values("PD", ascending=False).head(20)

display_cols = [c for c in ["loan_amnt","int_rate","annual_inc","PD","Credit_Score","Expected_Loss","Risk_Band"]
                if c in risky.columns]

styled = risky[display_cols].style\
    .format({
        "PD": "{:.2%}",
        "Credit_Score": "{:.0f}",
        "Expected_Loss": "${:,.0f}",
        "loan_amnt": "${:,.0f}",
        "int_rate": "{:.2f}%",
        "annual_inc": "${:,.0f}",
    })\
    .background_gradient(subset=["PD"], cmap="Reds")\
    .background_gradient(subset=["Expected_Loss"], cmap="YlOrRd")

st.dataframe(styled, use_container_width=True, height=420)

st.divider()

# ══════════════════════ STRESS TEST ══════════════════════════════
st.markdown("## 🔴 Stress Test Scenario")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

stressed_pd   = np.minimum(filtered["PD"] * stress_mult, 1.0)
stressed_loss = (stressed_pd * LGD * filtered["loan_amnt"]).sum()
base_loss     = filtered["Expected_Loss"].sum()
delta_loss    = stressed_loss - base_loss
vhigh_stress  = (stressed_pd >= 0.30).mean() * 100

col_s1.metric("Stressed Expected Loss",  f"${stressed_loss:,.0f}",
              delta=f"+${delta_loss:,.0f}", delta_color="inverse")
col_s2.metric("PD Stress Multiplier",    f"{stress_mult:.1f}×")
col_s3.metric("LGD Assumption",          f"{lgd_override}%")
col_s4.metric("Stressed Very High Risk", f"{vhigh_stress:.1f}%",
              delta=f"{vhigh_stress - vhigh_pct:+.1f}pp", delta_color="inverse")

# Waterfall chart: base EL vs stressed EL by band
st.markdown("#### Stress Loss Waterfall by Risk Band")
filtered2 = filtered.copy()
filtered2["Stressed_EL"] = stressed_pd.values * LGD * filtered2["loan_amnt"]
band_stress = filtered2.groupby("Risk_Band")[["Expected_Loss","Stressed_EL"]].sum().reindex(BAND_ORDER).dropna()

fig_st, ax_st = plt.subplots(figsize=(9, 3.5))
x = np.arange(len(band_stress))
w = 0.35
ax_st.bar(x - w/2, band_stress["Expected_Loss"]/1e6, w,
          label="Base EL", color='#38bdf8', edgecolor='#0a0e1a', linewidth=0.8)
ax_st.bar(x + w/2, band_stress["Stressed_EL"]/1e6, w,
          label="Stressed EL", color='#f87171', edgecolor='#0a0e1a', linewidth=0.8)
ax_st.set_xticks(x)
ax_st.set_xticklabels(band_stress.index, fontsize=8.5)
ax_st.set_ylabel("Expected Loss ($M)")
ax_st.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig_st)

st.divider()

# ══════════════════════ SHAP EXPLAINABILITY ═══════════════════════
st.markdown("## 🧠 AI Risk Explainability (SHAP)")

feat_cols = [c for c in filtered.columns
             if c not in ["PD","Risk_Band","Credit_Score","Expected_Loss"]]
X_shap = filtered[feat_cols].copy()

SHAP_SAMPLE = min(300, len(X_shap))

with st.spinner("Computing SHAP values..."):
    try:
        explainer   = shap.Explainer(model)
        shap_values = explainer(X_shap.sample(SHAP_SAMPLE, random_state=42)
                                if len(X_shap) > SHAP_SAMPLE else X_shap)

        col_sh1, col_sh2 = st.columns(2)

        with col_sh1:
            st.markdown("#### Global Feature Importance")
            fig_sh, ax_sh = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, X_shap.sample(SHAP_SAMPLE, random_state=42)
                              if len(X_shap) > SHAP_SAMPLE else X_shap,
                              plot_type="bar", show=False, color='#38bdf8')
            plt.tight_layout()
            st.pyplot(fig_sh)

        with col_sh2:
            st.markdown("#### SHAP Beeswarm Plot")
            fig_sw, ax_sw = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, X_shap.sample(SHAP_SAMPLE, random_state=42)
                              if len(X_shap) > SHAP_SAMPLE else X_shap,
                              show=False)
            plt.tight_layout()
            st.pyplot(fig_sw)

        st.markdown("#### Individual Borrower Explanation")
        row_index = st.slider("Select Borrower Index", 0, len(shap_values)-1, 0)
        fig_wf, ax_wf = plt.subplots(figsize=(10, 3.5))
        shap.plots.waterfall(shap_values[row_index], show=False)
        plt.tight_layout()
        st.pyplot(fig_wf)

    except Exception as e:
        st.warning(f"SHAP computation skipped: {e}")

st.divider()

# ══════════════════════ WHAT-IF ANALYSIS ══════════════════════════
st.markdown("## 🎛️ What-If Scenario Analysis")

col_wi1, col_wi2 = st.columns([1, 1])

with col_wi1:
    borrower_id = st.selectbox("Select Borrower for Simulation", filtered.index[:200])
    borrower_row = filtered.loc[[borrower_id]].copy()
    st.markdown("**Original Borrower Profile**")
    st.dataframe(borrower_row[feat_cols], use_container_width=True)

with col_wi2:
    st.markdown("**Adjust Scenario Parameters**")
    income_change = st.slider("Annual Income Change (%)", -50, 100, 0, 5,
                              help="Simulate income shock or improvement")
    rate_change   = st.slider("Interest Rate Change (pp)", -5, 10, 0, 1,
                              help="Rate hike or cut scenario")
    loan_change   = st.slider("Loan Amount Change (%)", -50, 100, 0, 5,
                              help="Refinancing or top-up scenario")

scenario = borrower_row[feat_cols].copy()
if "annual_inc" in scenario.columns:
    scenario["annual_inc"] *= (1 + income_change / 100)
if "int_rate" in scenario.columns:
    scenario["int_rate"] += rate_change
if "loan_amnt" in scenario.columns:
    scenario["loan_amnt"] *= (1 + loan_change / 100)

orig_pd = model.predict_proba(borrower_row[feat_cols])[:, 1][0]
new_pd  = model.predict_proba(scenario)[:, 1][0]
pd_delta = new_pd - orig_pd

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original PD",        f"{orig_pd:.2%}")
m2.metric("Scenario PD",        f"{new_pd:.2%}",
          delta=f"{pd_delta:+.2%}", delta_color="inverse")
m3.metric("Original Credit Score", str(int(850 - orig_pd * 550)))
m4.metric("Scenario Credit Score", str(int(850 - new_pd  * 550)),
          delta=str(int((850-new_pd*550) - (850-orig_pd*550))))

st.divider()

# ══════════════════════ PORTFOLIO HEATMAP ════════════════════════
st.markdown("## 🗺️ Portfolio Risk Heatmap")

# Correlation of numeric risk features
numeric_cols = [c for c in ["loan_amnt","int_rate","annual_inc","dti","PD","Credit_Score","Expected_Loss"]
                if c in filtered.columns]
if len(numeric_cols) >= 3:
    corr = filtered[numeric_cols].corr()
    fig_hm, ax_hm = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.RdYlGn_r
    im = ax_hm.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax_hm.set_xticks(range(len(corr.columns)))
    ax_hm.set_yticks(range(len(corr.columns)))
    ax_hm.set_xticklabels(corr.columns, rotation=35, ha='right', fontsize=8, color='#94a3b8')
    ax_hm.set_yticklabels(corr.columns, fontsize=8, color='#94a3b8')
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax_hm.text(j, i, f"{val:.2f}", ha='center', va='center',
                       fontsize=7.5, color='white' if abs(val) > 0.5 else '#e2e8f0')
    cbar = fig_hm.colorbar(im, ax=ax_hm, shrink=0.7)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=7)
    ax_hm.set_facecolor('#0d1526')
    fig_hm.patch.set_facecolor('#0a0e1a')
    ax_hm.set_title("Feature Correlation Matrix", color='#e2e8f0', fontsize=11, pad=12)
    plt.tight_layout()
    st.pyplot(fig_hm)
else:
    st.info("Not enough numeric columns to build heatmap.")

st.divider()

# ══════════════════════ DOWNLOAD ════════════════════════════════
st.markdown("## ⬇️ Export Results")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    csv_full = filtered.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Full Portfolio (CSV)",
        csv_full,
        "credit_risk_results.csv",
        "text/csv",
        use_container_width=True
    )

with col_dl2:
    risky_csv = risky.to_csv(index=False).encode()
    st.download_button(
        "🚨 Download Top 20 Risky Borrowers",
        risky_csv,
        "top_risky_borrowers.csv",
        "text/csv",
        use_container_width=True
    )

with col_dl3:
    stress_df = filtered[["loan_amnt","PD","Expected_Loss"]].copy()
    stress_df["Stressed_PD"] = np.minimum(filtered["PD"] * stress_mult, 1.0)
    stress_df["Stressed_EL"] = stress_df["Stressed_PD"] * LGD * stress_df["loan_amnt"]
    stress_csv = stress_df.to_csv(index=False).encode()
    st.download_button(
        "⚠️ Download Stress Test Results",
        stress_csv,
        "stress_test_results.csv",
        "text/csv",
        use_container_width=True
    )

st.divider()
st.markdown(
    "<div style='text-align:center;color:#334155;font-family:IBM Plex Mono,monospace;"
    "font-size:0.72rem;padding:12px 0;'>"
    "Banking Risk Analytics Platform · Powered by ML + SHAP · Basel III Aligned"
    "</div>",
    unsafe_allow_html=True
)
