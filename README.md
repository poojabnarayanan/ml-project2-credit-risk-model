# Credit Risk Prediction using Machine Learning

## 1. Overview

This project builds a Machine Learning–based Credit Risk Model for an NBFC to automate borrower risk assessment and improve default detection.

The objective is to identify high-risk applicants prior to loan sanctioning using financial and behavioral indicators.

## 2. Dataset

50,000 customer records consolidated from:

◉ **Customer Data** – Demographics & residence history

◉ **Loan Data** – Loan characteristics & repayment behavior

◉ **Bureau Data** – Historical credit performance

All datasets were merged into a unified borrower profile.

## 3. Methodology

 -- 75/25 Train–Test split to prevent data leakage   
 -- Missing value treatment and financial logic validation    
 -- Feature scaling using MinMaxScaler    
 -- Feature engineering (Loan-to-Income, Delinquency Ratio)   
 -- Multicollinearity removal using VIF    
 -- Feature selection using Information Value (IV > 0.02)   
 -- Class imbalance handled using SMOTE Tomek      
 -- Hyperparameter tuning with Optuna

## 4. Model Selection

Two models were evaluated:

◉ Logistic Regression   
◉ XGBoost

**Final Model**: Logistic Regression
Selected for superior defaulter detection (**Recall = 94%**), aligning with risk containment objectives.

## 5. Model Performance

**Recall**: 94%

**AUC**: 0.98

**Gini**: 0.96

**KS Statistic**: 85.9

#### Top 20% highest-risk borrowers contain ~98% of total defaults, demonstrating strong rank-ordering capability.

## 6. Key Insights

1. Financial behavior indicators significantly outperform demographic variables.

2. Loan-to-Income ratio is the strongest predictor of default.

3. The model effectively isolates high-risk applicants for safer automated lending decisions.

## 7. Tech Stack

`Python` | `Pandas` | `NumPy` | `Scikit-learn` | `XGBoost` | `Optuna` | `Imbalanced-learn`

## 8. Outcome

The project demonstrates how ML can enhance credit underwriting through improved default detection and strong risk segmentation.

## 9. Setup Installation 

## Installation

### 1. **Clone the repository:**

```bash
git https://github.com/poojabnarayanan/ml-project2-credit-risk-model.git
cd ml-project2-credit-risk-model
````
### Step 2: Install Required Packages

Install all required Python packages using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

Start the Streamlit application with:
```bash
streamlit run main.py
```

Note: This project does not include the actual data files