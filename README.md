# Banking Risk Analytics System – Loan Default Prediction & Risk Insights
## Problem Definition & Business Context

**Project Title:** Banking Risk Analytics System – Loan Default Prediction & Risk Insights

## Live Demo

Try the deployed application:

Streamlit App  
[https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/]

The platform allows users to upload borrower datasets and generate:

- Probability of Default predictions
- Portfolio risk analytics
- Expected credit loss calculations
- Scenario-based risk simulations

**Business Problem:**  
This project aims to build an end-to-end system that predicts loan defaults and recommends approvals or rejections based on borrower risk. The system will provide actionable insights for credit decision-makers and risk managers.

**KPIs:**
- Default Rate (%)
- Expected Loss (EL = Probability of Default × Exposure × Loss Given Default)
- Risk Tiers: Low, Medium, High
- Approval Rate per risk tier

**Target Audience:**  
Bank executives, credit risk managers, and lending decision-makers.

**Expected Business Impact:**
- Reduce non-performing assets
- Prioritize approvals for low-risk borrowers
- Flag high-risk borrowers for review
- Improve overall portfolio risk management

## System Architecture
```
Borrower Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Machine Learning Model
(Default Prediction)
      │
      ▼
Risk Metrics Engine
(PD, Credit Score, Expected Loss)
      │
      ├── Streamlit Risk Platform
      │
      └── Power BI Portfolio Dashboard
```

### Dataset
[https://www.kaggle.com/datasets/wordsforthewise/lending-club]


### Step 1: Data Cleaning

1. **Objective:** Prepare a clean, reliable dataset for analysis and modeling.
2. **Actions performed:**
   - Dropped columns with extremely high missing values (>90%) to avoid unreliable features.
   - Removed irrelevant text/ID fields not needed for modeling.
   - Imputed numeric columns using median for robustness against skewed distributions.
   - Imputed categorical columns with 'Unknown' for missing values.
3. **Outcome:** Dataset reduced from 151 columns to 108, ready for feature engineering and modeling.

### Step 2: Feature Engineering

1. **Objective:** Transform dataset into numeric form with derived features for modeling and reporting.
2. **Actions performed:**
   - `term` → `term_months`
   - `emp_length` → `emp_length_num` (numeric)
   - `earliest_cr_line` → `credit_history_years`
   - `prior_default_flag` → indicates if borrower had prior default/charged off
   - Label encoded categorical variables: `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`
   - Dropped original object columns after conversion
3. **Outcome:** Fully numeric dataset with engineered features, maintaining integrity for dashboards and predictive modeling.

# Step 3: EDA & Visual Analysis

## Objectives
- Validate data realism for industrial-level insights.
- Identify outliers, distributions, and potential feature-target relationships.
- Generate actionable visuals for dashboards and reporting.

## Visual Analysis Performed
1. **Distribution of Loan Amount (`loan_amnt`)**: 
   - Checks if most loans are within realistic ranges.
2. **Interest Rate (`int_rate`) Distribution**:
   - Identifies high-interest loans or potential data issues.
3. **Credit History (`credit_history_years`)**:
   - Ensures borrowers have realistic credit experience.
4. **Prior Default Flag (`prior_default_flag`)**:
   - Validates target variable distribution for modeling.
5. **Loan Amount vs Default**:
   - Highlights trends of loan sizes for defaulters vs non-defaulters.
6. **Correlation Heatmap**:
   - Numeric feature correlations, useful for feature selection.
7. **Grade vs Default Rate**:
   - Shows risk by grade, useful for dashboard KPI.
8. **Revolving Utilization vs Default**:
   - Indicates whether high utilization correlates with default risk.
  

# Key Features
## Credit Risk Prediction
The model predicts the probability that a borrower will default using financial and behavioral attributes.

## Output:
- Probability of Default (PD)
- Credit Score
- Risk Band classification
- Risk Segmentation
  
## Risk Segmentation

Borrowers are grouped into portfolio risk tiers:

| Risk Band       | PD Range   | Lending Action |
|-----------------|-----------|----------------|
| Low Risk        | < 5%      | Approve loan |
| Medium Risk     | 5% – 15%  | Approve with monitoring |
| High Risk       | 15% – 30% | Higher interest / collateral |
| Very High Risk  | > 30%     | Reject or strict review |

## Expected Credit Loss Calculation

- The project estimates potential financial loss using the formula:
- Expected Loss = PD × LGD × Exposure

Where:
- PD = Probability of Default
- LGD = Loss Given Default
- Exposure = Loan Amount

## Portfolio Risk Analytics
Portfolio level insights include:
- Total expected loss
- Risk band distribution
- Default probability distribution
- Top high-risk borrowers
- Exposure vs default risk

## Model Explainability
The model integrates SHAP explainability to understand:
- Which features drive default risk
- Feature importance across the portfolio
- Local explanations for individual borrowers
This improves transparency for financial decision making.

## Scenario Analysis
The system supports what-if simulations where users can modify borrower attributes such as:
- income
- interest rate
- loan exposure
The model recalculates the updated default probability, helping evaluate credit policy changes.

## Power BI Dashboard

The Power BI dashboard provides an interactive view of the credit risk portfolio.

### Portfolio KPIs

- Total Borrowers
- Average Default Probability
- Total Expected Credit Loss
- Percentage of Very High Risk Borrowers

### Risk Distribution

The dashboard includes visualizations that show:

- Distribution of Probability of Default (PD)
- Breakdown of borrowers by Risk Band
- Expected Loss across different risk segments

### Exposure vs Risk Analysis

A scatter plot is used to analyze the relationship between exposure and risk:

- X-axis: Loan Amount
- Y-axis: Probability of Default
- Bubble Size: Expected Loss
- Color: Risk Band

### High Risk Borrowers

A ranked table highlights borrowers with the highest probability of default and largest potential financial exposure.

---

## Streamlit Risk Platform

The interactive analytics platform is built using Streamlit.

### Portfolio Dashboard

Displays portfolio-level metrics such as:

- Borrower distribution across risk bands
- Average default probability
- Total expected portfolio loss
- Overall risk segmentation

  <img width="1393" height="788" alt="Screenshot 2026-03-11 064940" src="https://github.com/user-attachments/assets/9055fe60-687d-4972-a7aa-71d7730e7db7" />


### Model Explainability

Uses SHAP explainability techniques to identify:

- Key features influencing default risk
- Feature importance for the prediction model
- Individual borrower risk explanations

### Scenario Analysis

Allows users to perform **What-If Analysis** by adjusting borrower attributes such as:

- Income
- Loan exposure
- Interest rate

The system recalculates the **Probability of Default** to show how these changes impact credit risk.

## Dataset
The dataset contains borrower financial attributes used for credit risk modeling.

Features:
- loan amount
- annual income
- debt-to-income ratio
- credit history length
- interest rate

Derived analytics columns include:
- PD
- Risk Band
- Credit Score
- Expected Loss

How to Run the Project
1. Clone the Repository
```
git clone https://github.com/yourusername/banking-risk-analytics-system.git
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Run the Streamlit Application
```
streamlit run app.py
```

4. Open the Dashboard
- The application will open in your browser.
- Upload a borrower dataset to generate predictions and portfolio analytics. - sample credt input

## Applications
This project demonstrates how machine learning can support:
- credit underwriting
- loan portfolio monitoring
- risk segmentation
- financial loss estimation
- explainable AI in banking

## Future Improvements
Potential extensions include:
- model monitoring and drift detection
- real-time data pipelines
- automated retraining
- regulatory risk metrics
- advanced stress testing
  
## Skills Demonstrated
- Machine Learning for Credit Risk Modeling
- Probability of Default Prediction
- Financial Risk Analytics
- Explainable AI with SHAP
- Interactive Dashboard Development
- Model Deployment with Streamlit
- Portfolio Risk Monitoring

## License
This project is for educational and portfolio purposes.

## Author
Bindu Sri

