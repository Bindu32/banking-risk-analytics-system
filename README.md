# Banking Risk Analytics System: Loan Default Prediction & Portfolio Risk Intelligence

## 1. Project Title & Executive Summary

**Banking Risk Analytics System for Loan Default Prediction, Portfolio Monitoring, and Expected Loss Intelligence**

This project is an end-to-end credit risk analytics platform built to help financial institutions make faster, more reliable, and more transparent lending decisions. It predicts the probability of borrower default, assigns risk bands, estimates expected credit loss, and enables scenario-based portfolio stress analysis.  
The system combines machine learning, explainable AI, and interactive business dashboards to support both borrower-level underwriting and portfolio-level risk management.  
It is designed for credit risk managers, lending teams, and bank leadership who need decision-ready insights rather than raw model outputs.  
By converting historical borrower data into actionable risk intelligence, the platform helps reduce exposure to high-risk loans, improve approval quality, and strengthen portfolio oversight.

**Live Application:** [Streamlit App](https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/)  
**Dataset Source:** [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## 2. Business Problem Statement

Lending institutions operate in a high-stakes environment where growth must be balanced against credit quality, regulatory expectations, and loss containment. In many real-world settings, borrower evaluation still depends on fragmented spreadsheets, static rule-based checks, and disconnected reporting layers. This creates delays in underwriting, inconsistent approval logic, and limited visibility into portfolio concentration risk.  

The business challenge is not only identifying which borrowers are likely to default, but also understanding how that risk translates into expected loss, capital exposure, and portfolio decision-making. Without a unified risk analytics solution, banks struggle to prioritize low-risk approvals, flag vulnerable borrowers early, and assess the downstream financial impact of lending decisions.  

This project addresses that gap by delivering an integrated analytics system that transforms raw borrower data into predictive risk scores, expected loss estimates, explainable decisions, and executive-ready portfolio insights.

---

## 3. Objectives

- Predict borrower-level **Probability of Default (PD)** using financial and behavioral features.
- Classify borrowers into **actionable risk bands** for underwriting decisions.
- Estimate **Expected Credit Loss (ECL / EL)** using exposure, PD, and LGD assumptions.
- Provide **portfolio-level risk analytics** for monitoring concentration and loss exposure.
- Support **scenario-based simulations** to assess policy or borrower changes.
- Improve transparency through **SHAP-based explainability**.
- Deliver a recruiter-grade project with strong alignment between **business problem, model design, and decision usability**.

---

## 4. Dataset Overview

| Attribute | Description |
|---|---|
| **Source** | [LendingClub Loan Dataset (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| **Dataset Type** | Historical consumer lending / credit performance data |
| **Scale** | Large multi-column loan dataset; project pipeline uses cleaned and transformed subsets for modeling and dashboarding |
| **Time Period** | Historical multi-period loan records |
| **Primary Use** | Default prediction, borrower segmentation, expected loss estimation, portfolio analytics |
| **Key Features** | Loan amount, annual income, DTI, interest rate, grade, sub-grade, term, employment length, credit history, revolving utilization, verification status |
| **Data Issues** | Missing values, skewed financial variables, categorical complexity, noisy text fields, class imbalance, and high-dimensional schema |

The LendingClub dataset is highly suitable for credit risk analytics because it contains both borrower characteristics and loan performance indicators that closely resemble real underwriting signals. It provides a practical foundation for building default models, risk tiers, and business-facing dashboards.  

Like most large lending datasets, it required substantial cleaning before use. The main challenges included high missingness in some fields, mixed-format categorical variables, and the need to engineer interpretable credit features from raw columns.

---

## 5. Solution Architecture

The solution follows a layered analytics pipeline that converts borrower data into decision-support outputs.

1. **Data Ingestion**  
   Raw borrower records are loaded from source files for transformation and modeling.

2. **Data Cleaning & Preparation**  
   High-missing-value columns are removed, missing values are imputed, and irrelevant identifiers are dropped.

3. **Feature Engineering**  
   Credit-relevant features such as term length, employment duration, credit history years, and prior default flags are created.

4. **Machine Learning Inference**  
   A classification model predicts the borrower’s probability of default.

5. **Risk Metrics Engine**  
   PD is converted into business metrics such as credit score, expected loss, and risk band.

6. **Analytics Delivery Layer**  
   Outputs are delivered through a Streamlit application for interactive borrower and scenario analysis, and through Power BI for portfolio-level visualization.

![Solution Architecture](path/to/architecture_diagram.png)

**Add this image as the architecture diagram showing the full workflow: borrower dataset → preprocessing → ML model → risk metrics engine → Streamlit app / Power BI dashboard.**

---

## 6. Tech Stack

| Layer | Tools | Purpose |
|---|---|---|
| **Data Processing** | Python, Pandas, NumPy | Cleaning, transformation, feature engineering |
| **Analysis** | Jupyter Notebook, SQL | EDA, KPI calculations, analytical validation |
| **Modeling** | Scikit-learn, SHAP | Default prediction and model explainability |
| **Visualization** | Streamlit, Power BI, Matplotlib, Seaborn | Interactive analytics and business reporting |
| **Deployment** | Streamlit Cloud | Live app hosting |
| **Data Warehouse** | SQL, star schema design | Risk analytics storage and BI-friendly modeling |

---

## 7. Methodology

### Data preprocessing

The raw loan data was first prepared for analytical reliability and modeling stability. Columns with extremely high missingness were removed to avoid noise-heavy predictors, text-heavy non-analytical fields were excluded, numeric features were median-imputed, and categorical fields were filled with `Unknown`.  

This approach was chosen because lending data often contains skewed financial values and inconsistent population coverage across columns. Median imputation is more robust than mean imputation for outlier-heavy variables such as income, utilization, and loan size.

### Feature engineering

Several raw variables were transformed into more analytically useful features:

- `term` → `term_months`
- `emp_length` → `emp_length_num`
- `earliest_cr_line` → `credit_history_years`
- `prior_default_flag` → historical adverse behavior indicator
- Label encoding for grade, sub-grade, home ownership, verification status, purpose, and state

These transformations were selected to improve both model compatibility and business interpretability. Rather than using raw text-heavy attributes, the engineered features create a more consistent analytical representation of borrower risk.

### KPI definitions

The project tracks business-critical credit risk KPIs:

- **Default Rate** = percentage of borrowers expected or observed to default
- **Probability of Default (PD)** = borrower-level default likelihood predicted by the model
- **Expected Loss (EL)** = PD × LGD × Exposure
- **Risk Band Distribution** = segmentation of borrowers into risk classes
- **Approval Action by Risk Band** = recommended lending action per borrower category

These KPIs were chosen because they align closely with underwriting decisions, loss forecasting, and executive reporting needs.

### Analytical techniques

The analytics workflow includes univariate distributions, bivariate default analysis, feature correlation analysis, grade-based default patterns, and exposure-risk relationships. This helps validate whether the dataset behaves like a realistic industrial credit dataset before model outputs are trusted.  

Visual analysis also supports dashboard storytelling by linking statistical patterns to business questions, such as whether higher loan amounts or higher utilization rates are associated with elevated risk.

### ML modeling and explainability

A machine learning classification model was used to predict borrower default probability. SHAP explainability was integrated to identify both global feature importance and local borrower-level decision drivers.  

This combination was selected because a predictive model alone is not enough in a regulated or decision-sensitive banking context. The explainability layer improves stakeholder trust and makes outputs more usable for analysts, managers, and reviewers.

### Alternatives considered

Alternative approaches could include pure rule-based underwriting, logistic regression scorecards, or more complex ensemble methods with stronger optimization. However, the chosen design balances predictive power, interpretability, and deployment simplicity, which is appropriate for a portfolio-grade analytics application.

---

## 8. Key Insights & Findings

| Insight | Business Impact |
|---|---|
| Higher interest rates were strongly associated with elevated default probability. | Supports risk-based pricing and tighter review for costly credit segments. |
| Expected loss was concentrated heavily in higher-risk borrower bands. | Helps lending teams prioritize monitoring and reduce capital exposure. |
| Credit score moved inversely with probability of default. | Reinforces the usefulness of borrower quality indicators in approval logic. |
| Higher-risk segments contributed disproportionately to portfolio loss concentration. | Enables targeted intervention rather than broad policy tightening. |
| Scenario-based changes in income, rate, and exposure produced visible changes in PD and EL. | Useful for pricing simulations, underwriting strategy, and stress testing. |

The project’s visual outputs also suggest that risk is not evenly distributed across the borrower population. A relatively smaller portion of high-risk borrowers can account for a materially larger portion of expected loss, which is exactly the kind of pattern credit teams need to identify early.  

The SHAP analysis strengthens these findings by showing which variables most consistently move model predictions upward or downward, making the system more trustworthy in practical decision settings.

---

## 9. Business Impact

This solution is designed to create measurable value across underwriting, portfolio management, and executive reporting.

| Impact Area | Expected Benefit |
|---|---|
| **Risk Reduction** | Reduces exposure to likely defaulters through early borrower risk identification |
| **Loss Control** | Quantifies expected loss before loan approval, supporting better capital protection |
| **Operational Efficiency** | Reduces manual borrower review time through automated scoring and segmentation |
| **Decision Consistency** | Standardizes approval / rejection recommendations across applicants |
| **Portfolio Visibility** | Improves visibility into risk concentration, exposure, and stress vulnerability |
| **Strategic Lending** | Helps prioritize low-risk approvals and escalate high-risk cases for review |

From a business standpoint, the platform helps shift lending from reactive review to proactive risk intelligence. Instead of only identifying defaults after deterioration occurs, the system enables earlier intervention, more selective approvals, and better expected loss planning.  

For banks and NBFC-style lending environments, this can contribute to lower non-performing asset build-up, more consistent underwriting quality, and stronger control over portfolio-level downside risk.

---

## 10. Dashboard / Application Highlights

**Deployed Application:** [Open Live Streamlit App](https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/)

![Dashboard Overview](path/to/dashboard1.png)

**Add this screenshot as the main dashboard homepage showing portfolio KPIs such as portfolio size, average PD, total exposure, total expected loss, and very high-risk percentage.**

![Detailed View](path/to/dashboard2.png)

**Add this screenshot as a detailed analytics view showing scenario analysis, SHAP explainability, risk distribution, or expected loss visuals.**

### What the application provides

- **Portfolio summary KPIs** for risk monitoring
- **Risk band distribution** across borrowers
- **PD distribution and credit score distribution**
- **Expected loss analysis by risk segment**
- **Stress test views** for adverse portfolio conditions
- **What-if scenario analysis** at borrower level
- **SHAP-based explainability** for model transparency

### Business usability

The application is built for users who need insight quickly, not just raw model outputs. A credit analyst can upload borrower data and immediately see predicted default risk, expected loss, and approval guidance. A risk manager can evaluate how portfolio risk shifts under stressed assumptions.  

This makes the platform useful across multiple decision layers, from individual loan review to executive portfolio oversight.

---

## 11. Challenges & Solutions

| Challenge | Solution |
|---|---|
| High-dimensional dataset with many missing fields | Dropped extremely sparse columns and used robust imputation strategies |
| Raw categorical data not suitable for modeling | Encoded key borrower attributes and engineered domain-specific numeric features |
| Need to balance predictive power with business transparency | Integrated SHAP for both global and local explanation |
| Borrower-level and portfolio-level users had different needs | Delivered insights through both Streamlit and Power BI layers |
| Risk analytics needed to be more than just a prediction | Added expected loss, risk bands, credit score logic, and scenario simulations |

---

## 12. Future Enhancements

- Add **model monitoring and drift detection** for production reliability
- Implement **automated retraining pipelines**
- Integrate **real-time borrower ingestion APIs**
- Add **macro-economic stress testing scenarios**
- Expand to **regulatory risk metrics** and provisioning views
- Introduce **role-based access controls** for enterprise deployment
- Build **alerting workflows** for high-risk borrower clusters and portfolio thresholds

---

## 13. Project Outcome Summary

This project demonstrates how machine learning, analytics engineering, and business intelligence can be combined into a practical credit risk decision system. It does not stop at model prediction; it translates borrower behavior into lending actions, expected loss estimates, portfolio signals, and explainable business insights.  

From a recruiter and hiring-manager perspective, this project shows strong business framing, technical depth, deployment maturity, and the ability to build analytics solutions that are usable by decision-makers. It is a portfolio-ready example of applied data science and business analytics in a high-value financial domain.

---

## Repository Links

- **ML + Streamlit Project:** [banking-risk-analytics-system](https://github.com/Bindu32/banking-risk-analytics-system)
- **Data Warehouse Project:** [banking-risk-analytics-data-warehouse](https://github.com/Bindu32/banking-risk-analytics-data-warehouse)

---

## How to Run the Project

```bash
git clone https://github.com/Bindu32/banking-risk-analytics-system.git
cd banking-risk-analytics-system
pip install -r requirements.txt
streamlit run app.py
```

Then open the local application in your browser, or use the deployed version here:  
**[https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/](https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/)**

---

## Suggested Image Placement Guide

Use the following screenshots / visuals in the repository:

| Image Placeholder | What it should show | Suggested Location |
|---|---|---|
| `![Solution Architecture](path/to/architecture_diagram.png)` | End-to-end system workflow | `assets/architecture_diagram.png` |
| `![Dashboard Overview](path/to/dashboard1.png)` | Main portfolio summary dashboard | `assets/dashboard_overview.png` |
| `![Detailed View](path/to/dashboard2.png)` | Scenario analysis / explainability / risk detail | `assets/detailed_view.png` |
| `![Expected Loss Analysis](path/to/el_analysis.png)` | EL by risk band / exposure vs EL | `assets/expected_loss_analysis.png` |
| `![Model Explainability](path/to/shap_view.png)` | SHAP summary or local explanation | `assets/shap_explainability.png` |

---

## Author

**Bindu Sri Majji**  
Final Year Computer Science Student  
Aspiring Data Analyst / Data Scientist
