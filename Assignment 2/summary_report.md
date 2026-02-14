# Summary Report
**Date:** February 14, 2026  
**To:** Data Science Leadership & Senior Management  
**From:** Jiaqi Pan, Irene Xu  
**Subject:** Comparative Analysis of Machine Learning Models for Identifying High-Potential Enterprises  

---

## 1. Executive Summary

This report summarizes the development and evaluation of predictive models designed to identify high-growth firms using Bisnode panel data (2010–2015). Three distinct modeling approaches were evaluated: Logistic Regression (Logit), Lasso Regularized Logistic Regression, and Random Forest.

**Key Findings:**
* **Best Performing Model:** The Logit Model (M4) demonstrated superior performance in minimizing expected business loss, outperforming the Random Forest model despite similar statistical accuracy.
* **Strategic Implication:** The optimal decision threshold is extremely low (approximately 0.9%). This indicates that the most effective strategy is to flag a broad range of candidates to minimize the risk of missing high-value targets, given the high cost of false negatives.
* **Industry Dynamics:** The model exhibited higher predictive stability in the Services sector compared to the Manufacturing sector, where growth patterns appear more volatile.

---

## 2. Key Decision Points

The following strategic decisions guided the modeling process and sample design:

### 2.1 Target Variable Definition
* **Metric Selection:** We defined "Fast Growth" as a Compound Annual Growth Rate (CAGR) of Total Assets exceeding 20% over a two-year period (2012–2014).
* **Justification:** Total Assets were selected over Revenue or Employment because asset accumulation represents a deliberate structural expansion and capital deployment. A two-year horizon was chosen to filter out short-term accounting noise and capture sustained scaling.

### 2.2 Feature Engineering Strategy
* **Data Preparation:** The dataset was filtered to exclude micro-firms and records with incomplete financial statements to ensure data quality.
* **Predictor Variables:** The feature set included core financial ratios, management characteristics (e.g., foreign management), and non-linear transformations (quadratic terms) to capture life-cycle effects.

### 2.3 Loss Function Design
* **Asymmetric Cost Structure:** We defined a custom loss function where the cost of a False Negative (missed opportunity) is set at 10 units, while the cost of a False Positive (incorrect flag) is set at 1 unit.
* **Business Logic:** This 10:1 ratio reflects a Venture Capital or B2B sales environment where missing a high-growth firm is significantly more detrimental to the portfolio than the administrative cost of screening a non-growth firm.

---

## 3. Results

The models were evaluated using 5-fold cross-validation, assessing both statistical accuracy (RMSE) and business utility (Expected Loss).

### 3.1 Model Performance Comparison
* **Statistical Accuracy (RMSE):** The Logit M4 model (RMSE 0.4253) and the Random Forest model (RMSE 0.4261) performed almost identically in terms of probability estimation accuracy.
* **Business Utility (Expected Loss):** The Logit M4 model achieved a lower average expected loss of 0.715 per firm, compared to 0.744 for the Random Forest model.

### 3.2 Threshold Optimization
* **Logit Threshold:** The optimal probability threshold for the Logit model was identified at 0.0091 (0.91%).
* **Random Forest Threshold:** The optimal threshold for the Random Forest model was higher, at approximately 0.07 (7%).
* **Holdout Performance:** On the holdout set, the Logit model successfully identified 99.9% of the actual high-growth firms, although this came at the expense of a high false positive rate.

---

## 4. Interpretation

### 4.1 Why Logit Outperformed Random Forest
* **Calibration at Extremes:** The Logistic Regression model utilizes a sigmoid function that provides smooth, continuous probability estimates near zero. This feature allowed for precise calibration at the ultra-low threshold (0.9%) required by our 10:1 loss function. The Random Forest model, being a tree-based ensemble, struggled to provide granular probability differentiation in this specific lower tail of the distribution.

### 4.2 Industry Sub-Analysis
* **Services Sector:** The Services sector yielded a lower RMSE (0.4780), indicating that growth in this sector is more strongly correlated with standard financial predictors.
* **Manufacturing Sector:** The Manufacturing sector showed a higher RMSE (0.5526), suggesting that asset growth in this industry is driven by lumpy capital expenditures or external factors not fully captured by the current feature set.

---

## 5. Decision and Recommendations

Based on the analysis, we propose the following actions:

**Primary Decision:**
* **Model Deployment:** We recommend deploying the Logit (M4) model for the upcoming targeting cycle. Its superior performance on the expected loss metric makes it the most economically viable option.

**Operational Recommendations:**
* **Threshold Strategy:** Implement the aggressive classification threshold of 0.9%. The team must be prepared to process a high volume of leads, as the model effectively prioritizes recall (capturing all potential winners) over precision.
* **Sector-Specific Workflows:** For the Manufacturing sector, we recommend supplementing the model output with manual domain expertise or alternative data sources (e.g., supply chain signals) to mitigate the higher prediction error observed in this group.
