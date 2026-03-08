# Banking Risk Analytics System – Loan Default Prediction & Risk Insights
## Problem Definition & Business Context

**Project Title:** Banking Risk Analytics System – Loan Default Prediction & Risk Insights

**Deployment link** https://banking-risk-analytics-system-ku5gdmzncge8mjihtfr28z.streamlit.app/

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
