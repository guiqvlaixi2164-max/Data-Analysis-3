# Summary Report
**Date:** February 14, 2026  
**To:** Data Science Leadership & Senior Management  
**From:** Jiaqi Pan, Irene Xu  
**Subject:** Comparative Analysis of Machine Learning Models for Identifying High-Potential Enterprises  

---

## 1. Executive Summary

This report summarizes the development and evaluation of predictive models designed to identify high-growth firms using Bisnode panel data (2012–2013). Three distinct modeling approaches were evaluated: Logistic Regression (Logit), Lasso Regularized Logistic Regression, and Random Forest.

**Key Findings:**
* **Recommended Model:** The Logit Model (M4) is recommended for deployment due to its interpretability, highest cross-validation AUC (0.718), and lowest CV expected loss (0.669). While the Random Forest achieves a slightly lower holdout expected loss (0.712 vs. 0.733), the gap is marginal and M4's transparent coefficients allow analysts to understand why a firm is flagged.
* **Strategic Implication:** The optimal decision threshold is approximately 10%. This results in flagging roughly 89% of candidates to minimize the risk of missing high-value targets, given the high cost of false negatives.
* **Industry Dynamics:** The model exhibited higher predictive stability in the Services sector compared to the Manufacturing sector, where growth patterns appear more volatile.

---

## 2. Key Decision Points

The following strategic decisions guided the modeling process and sample design:

### 2.1 Target Variable Definition
* **Metric Selection:** We defined "Fast Growth" as ≥20% year-over-year growth in total assets from 2012 to 2013.
* **Justification:** Total Assets were selected over Revenue or Employment because asset accumulation represents a deliberate structural expansion and capital deployment, is comparable across industries, and is a mandatory balance sheet item with standardized accounting treatment.

### 2.2 Feature Engineering Strategy
* **Data Preparation:** The dataset was filtered to exclude micro-firms and records with incomplete financial statements to ensure data quality.
* **Predictor Variables:** The feature set included core financial ratios, management characteristics (e.g., foreign management), and non-linear transformations (quadratic terms) to capture life-cycle effects.

### 2.3 Loss Function Design
* **Asymmetric Cost Structure:** We defined a custom loss function where the cost of a False Negative (missed opportunity) is set at 10 units, while the cost of a False Positive (incorrect flag) is set at 1 unit.
* **Business Logic:** This 10:1 ratio reflects a Venture Capital or B2B sales environment where missing a high-growth firm is significantly more detrimental to the portfolio than the administrative cost of screening a non-growth firm.

Table 1: Loss Function Configuration
| Outcome Type | Scenario | Assigned Cost | Business Rationale |
| :--- | :--- | :--- | :--- |
| False Negative (FN) | Missed a High-Growth Firm | 10 | High Opportunity Cost (Lost Revenue/Equity) |
| False Positive (FP) | Incorrectly Flagged as High-Growth | 1 | Low Administrative Cost (Screening Effort) |
---

## 3. Results

The models were evaluated using 5-fold cross-validation, assessing both statistical accuracy (RMSE) and business utility (Expected Loss).

### 3.1 Model Performance Comparison
* **Statistical Accuracy (RMSE):** The Logit M4 model (CV RMSE 0.425) and the Random Forest model (CV RMSE 0.426) performed almost identically in terms of probability estimation accuracy.
* **Business Utility (Expected Loss):** On cross-validation, the Logit M4 model achieved the lowest average expected loss of 0.669 per firm, compared to 0.670 for the Random Forest. On the holdout set, RF achieved a slightly lower expected loss (0.712 vs. 0.733), though the gap is marginal.

### 3.2 Threshold Optimization
* **Logit Threshold:** The optimal probability threshold for the Logit model was identified at approximately 0.10 (10%).
* **Random Forest Threshold:** The optimal threshold for the Random Forest model was similar, at approximately 0.096 (9.6%).
* **Holdout Performance:** On the holdout set, the Logit model successfully identified 96% of the actual high-growth firms (959 out of 998), although this came at the expense of a high false positive rate (~87%).

---

## 4. Interpretation

### 4.1 Why Logit Is Preferred Despite RF's Holdout Advantage
* **Interpretability:** Logistic regression coefficients allow analysts to understand which financial characteristics drive each prediction — essential in a VC context where flagged firms undergo human review. Random Forest, as an ensemble of 100 trees, does not offer this transparency.
* **Stronger Discrimination:** M4 achieves the highest CV AUC (0.718 vs. RF's 0.714), meaning it ranks fast-growth firms more reliably — a threshold-independent advantage that persists if cost assumptions change.
* **CV Consistency:** M4 achieves the lowest expected loss across all 5 CV folds (0.669 vs. RF's 0.670). RF's holdout advantage (0.712 vs. 0.733) likely reflects sampling variation in a single train-test split rather than systematic superiority.

### 4.2 Industry Sub-Analysis
* **Services Sector:** The Services sector yielded a lower logit RMSE (0.4782), indicating that growth in this sector is more strongly correlated with standard financial predictors.
* **Manufacturing Sector:** The Manufacturing sector showed a higher logit RMSE (0.5434), suggesting that asset growth in this industry is driven by lumpy capital expenditures or external factors not fully captured by the current feature set. Random Forest outperformed Logit in both sectors on expected loss, consistent with RF's strength in small-sample, non-linear settings.

Table 2: Sector-Specific Performance (RMSE)
| Industry Sector | Logit RMSE | Prediction Difficulty | Key Takeaway |
| :--- | :--- | :--- | :--- |
| Services | 0.4782 | Moderate | Growth is predictable via standard financials. |
| Manufacturing| 0.5434 | High | Growth is volatile; likely driven by external factors. |
---

## 5. Decision and Recommendations

Based on the analysis, we propose the following actions:

**Primary Decision:**
* **Model Deployment:** We recommend deploying the Logit (M4) model for the upcoming targeting cycle. Although the Random Forest achieves a marginally lower holdout expected loss, M4's interpretability, highest CV AUC, and lowest CV expected loss make it the more practical and auditable choice.

**Operational Recommendations:**
* **Threshold Strategy:** Implement the classification threshold of approximately 10%. The team must be prepared to process a high volume of leads (~89% of candidates flagged), as the model effectively prioritizes recall (capturing 96% of potential winners) over precision.
* **Sector-Specific Workflows:** For the Manufacturing sector, we recommend supplementing the model output with manual domain expertise or alternative data sources (e.g., supply chain signals) to mitigate the higher prediction error observed in this group.
