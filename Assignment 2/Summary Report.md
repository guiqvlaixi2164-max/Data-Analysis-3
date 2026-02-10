# REPORT: Predictive Modeling of High-Growth Firms

**Date:** February 9, 2026  
**To:** Data Science Leadership & Senior Management  
**From:** Jiaqi Pan, Irene Xu  
**Subject:** Comparative Analysis of Machine Learning Models for Identifying High-Potential Enterprises  

---

## 1. Executive Summary
The objective of this analysis was to develop a predictive model to identify "High-Growth" companies using historical Bisnode data (2012–2014). The goal is to deploy this model to prioritize targets for investment or B2B acquisition in the upcoming fiscal year.

We evaluated three modeling approaches: **Logistic Regression (OLS)**, **Lasso Regularized Logistic Regression**, and **Random Forest**.

**Key Findings:**
* **Best Model:** The **Logit (M4)** model outperformed Random Forest in terms of business value (Expected Loss).
* While Random Forest had similar statistical accuracy (RMSE), the Logit model's ability to calibrate at extremely low probability thresholds made it superior for our specific cost structure.
* **Strategic Implication:** Due to the high cost of missing a "winner" (False Negative), the optimal strategy is highly aggressive. The model suggests flagging nearly all viable firms (Threshold 0.9%) to ensure no high-growth opportunity is missed.
* **Industry Insight:** **Services** firms were easier to predict (lower RMSE) compared to **Manufacturing** firms, suggesting that asset-based growth in manufacturing is more volatile or driven by external factors not captured in standard financial statements.

---

## 2. Problem Formulation & Target Variable Definition

### 2.1 Defining "High Growth"
We defined a "High Growth" firm as one achieving **20% year-over-year growth in Total Assets** (2012 vs. 2013).

### 2.2 Justification
We selected **Total Assets** over Sales or Employment for the following reasons:
* **Investment Signal:** Rapid asset accumulation indicates a deliberate strategy of expansion and capital deployment, whereas sales can fluctuate due to short-term pricing changes.
* **Data Reliability:** Balance sheet asset data is mandatory and standardized, whereas employment counts in SME data often contain missing values or inconsistencies.
* **Threshold (20%):** This cutoff isolates firms significantly outperforming the organic market growth rate (typically 3-5%), capturing the top ~25% of performers.

*Alternative metrics considered included **Employee Growth** (rejected due to lagging nature) and **Sales Growth** (rejected due to volatility).*

---

## 3. Data Management & Feature Engineering

**Data Source:** Bisnode Panel Data (2010–2015).

**Sample Design:**
* **Predictor Year:** 2012.
* **Outcome Window:** Growth calculated from 2012 to 2013.
* **Filtering:** We excluded micro-firms and firms with incomplete financial records to reduce noise.

**Feature Sets:**
The models utilized a rich set of financial predictors including:
* **Core Financials:** Sales (log), Profit/Loss, Liquidity Ratios.
* **Management:** `foreign_management` (indicator of foreign ownership).
* **Derived Features:** Quadratic terms (e.g., $CEO\_age^2$) and interaction terms to capture non-linear life-cycle effects.

---

## 4. Part I: Probability Prediction & Model Selection

We trained three models to predict the probability $P(Growth=1)$. Performance was evaluated using 5-fold Cross-Validation.

**Models Evaluated:**
1.  **Logit (M4):** A carefully specified logistic regression with domain-selected features.
2.  **Lasso Logit:** Logistic regression with L1 regularization for automated feature selection.
3.  **Random Forest:** An ensemble method to capture complex non-linear interactions.

**Statistical Performance (RMSE):**
* **Logit (M4):** RMSE $\approx$ 0.4253
* **Random Forest:** RMSE $\approx$ 0.4261

**Result:** The models were statistically tied. Random Forest did not provide a significant accuracy lift over the well-specified linear model.

---

## 5. Part II: Classification & Business Decision

To translate probabilities into decisions, we defined a **Loss Function** reflecting the asymmetric nature of Venture Capital/B2B Sales.

### 5.1 The Loss Function
* **False Positive (FP):** Cost = **$1** (Cost of administrative review/marketing).
* **False Negative (FN):** Cost = **$10** (Opportunity cost of missing a high-growth winner).
* **Ratio:** 1:10.

### 5.2 Optimal Decision Threshold
Because missing a winner is **10x more costly** than investigating a false lead, the optimal strategy requires a very low classification threshold.

* **Optimal Threshold (Logit M4):** **0.0091 (0.91%)**.
* **Interpretation:** If the model predicts even a **0.9% probability** of growth, we classify the firm as a target.

### 5.3 Model Selection based on Loss
The **Logit (M4)** model yielded the lowest expected loss per firm:
* **Logit M4 Expected Loss:** **$0.715** per firm.
* **Random Forest Expected Loss:** **$0.744** per firm.

**Why Logit Won:** The Logistic Regression's sigmoid function is continuous and well-calibrated at the extreme tails (near 0). Random Forest, being a tree-based method, often struggles to predict probabilities close to zero with high granularity. Since our optimal threshold was extremely low (0.9%), Logit provided more precise sorting of low-probability candidates.

---

## 6. Task 2: Industry Sub-Analysis (Manufacturing vs. Services)

We split the dataset into **Manufacturing** (Industry codes 20-30) and **Services** (Industry codes 40-60) to test model robustness.

### 6.1 Performance Comparison
We applied the same Loss Function (FP=$1, FN=$10) to both sectors.

| Metric | Manufacturing | Services |
| :--- | :--- | :--- |
| **Model RMSE (Logit)** | 0.5526 (High Error) | 0.4780 (Lower Error) |
| **Difficulty** | Harder to Predict | Easier to Predict |
| **Optimal Threshold** | 0.2% | 0.9% |

**Findings:**
* **Services:** The model performed significantly better (lower RMSE). Service firms likely have financial structures (cash flow, shorter asset cycles) that are more transparent in the dataset.
* **Manufacturing:** Predicting asset growth in manufacturing proved difficult (high RMSE). This suggests that manufacturing growth is "lumpy"—driven by large, irregular capital expenditures that are hard to predict from previous year's financials alone.

### 6.2 Strategic Implication
* **For Services:** Deploy the Logit model with confidence; the signals are clear.
* **For Manufacturing:** The model is less reliable. We recommend supplementing the model with external data (e.g., patent filings, news regarding factory expansion) rather than relying solely on the financial model.

---

## 7. Conclusion & Recommendations

The **Logit (M4)** model is the robust choice for this specific business problem. While Random Forest is a powerful tool, the specific requirement to minimize False Negatives (at a 10:1 cost ratio) favors the smooth calibration of the Logit model.

**Recommendations for Management:**
1.  **Adopt the "Dragnet" Strategy:** Given the low cost of False Positives ($1), the team should use the recommended **0.9% probability threshold**. This means verifying a large number of leads to ensure no "Unicorns" are missed.
2.  **Sector-Specific Expectations:** Expect higher conversion rates in the **Services** sector. Be prepared for lower precision in **Manufacturing**, and consider allocating more senior analysts to review manufacturing leads to offset the model's lower predictive power in that sector.
3.  **Future Improvement:** To improve the Manufacturing model, we must move beyond annual financial statements. Incorporating quarterly order book data or supply chain signals is necessary to reduce the high RMSE in that sector.

---

## Appendix: Confusion Matrix - Holdout Set
**Model:** Logit M4 @ Threshold 0.0091

| | Predicted: No Growth | Predicted: High Growth |
| :--- | :--- | :--- |
| **Actual: No Growth** | ~4 (0.1%) | ~2,852 (99.9%) |
| **Actual: High Growth** | ~1 (0.1%) | ~1,143 (99.9%) |

**Note:** The model effectively classifies almost everyone as "High Growth" to avoid missing the 1,144 actual winners. This is the mathematically optimal solution given the $1 vs $10 cost structure.
