<div style="font-size: 12pt;">

# Technical Report: Predicting Firm Fast Growth

Code: [firm-fast-growth-prediction-new.ipynb](https://github.com/guiqvlaixi2164-max/Data-Analysis-3/blob/main/Assignment%202/code/firm-fast-growth-prediction-new.ipynb)

## 1. Project Overview

**Objective:**
Build a binary classification model to predict whether a firm will experience fast growth, defined as ≥20% year-over-year growth in total assets from 2012 to 2013. The project proceeds in three stages: (I) probability prediction with cross-validated model comparison, (II) classification with a custom business loss function, and (III) discussion of practical applicability. An optional industry-level analysis compares performance across manufacturing and services sectors.

**Dataset:**
The bisnode-firms panel dataset (2010–2015), filtered to a 2012 cross-section of alive firms with revenues between €1,000 and €10 million. After cleaning, the modeling sample contains 19,996 observations with 118 features. The fast growth base rate is 28.86% (5,770 positive cases).

**Definition of Fast Growth:**
Total assets were chosen over employee counts or revenue because they capture the full scale of resource deployment (property, equipment, working capital, intangibles), are comparable across capital-intensive and knowledge-intensive industries, and are mandatory balance sheet items with standardized accounting treatment. The 20% threshold falls roughly at the 75th–80th percentile of SME growth rates, filtering out organic growth noise while capturing firms making substantive strategic investments. Growth is measured from 2012 to 2013 to avoid contamination from the 2008–2009 crisis.

---

## 2. Data Preparation

### 2.1 Panel Balancing and Initial Cleaning

The raw panel was balanced by filling missing year–firm combinations, and five columns with excessive missingness were dropped (`COGS`, `finished_prod`, `net_dom_sales`, `net_exp_sales`, `wages`). Year 2016 was excluded. A `status_alive` indicator was created for firms with positive sales.

### 2.2 Label Construction

Total assets were computed as the sum of three balance sheet components (intangible assets, current assets, and fixed assets), with negative values floored to zero and flagged. The asset growth rate was then calculated as the year-over-year percentage change in total assets from 2012 to 2013. Firms with growth rates ≥ 20% were labeled as fast-growth (1); all others as non-fast-growth (0). Observations with missing or invalid total assets in either year were excluded.

### 2.3 Sample Design

The analysis uses a cross-section of firms that were alive in 2012 with annual revenues between €1,000 and €10 million. This revenue filter focuses on small and medium enterprises (SMEs) while excluding micro-entities and larger firms whose growth dynamics differ substantially.

### 2.4 Feature Engineering

**Sales transformations:**
Log sales, sales in millions, and first-differenced log sales (a growth rate proxy) were created. Negative sales were replaced with 1. For new firms (age ≤ 1 or incomplete balance sheet year), the sales growth variable was set to 0.

**Financial ratios:**
Eight profit & loss items were normalized by sales (e.g., `extra_exp / sales`), and eight balance sheet items were normalized by total assets (e.g., `curr_assets / total_assets`). Infinite or missing values resulting from zero denominators were replaced with 0.

**Winsorization and flags:**
Ratios bounded at [0, 1] or [−1, 1] were winsorized, with corresponding flag indicators created for values that exceeded bounds or were erroneous. Quadratic terms were added for variables that can take any sign (e.g., income before tax) to allow U-shaped relationships. Zero-variance flag columns were dropped.

**Growth-relevant features.** Eight additional features were engineered:

| Feature | Formula | Notes |
|---|---|---|
| `asset_tangibility` | `tang_assets / total_assets` | Physical capital intensity |
| `liquidity_ratio` | `curr_assets / curr_liab` | Winsorized at 10 |
| `roa` | `profit_loss_year / total_assets` | Clipped to [−1, 1] |
| `labor_intensity` | `personnel_exp / sales` | From P&L ratio |
| `leverage` | `(total_assets − share_eq) / total_assets` | Clipped to [0, 1] |
| `ln_total_assets` | `log(total_assets)` | Firm size |
| `age_cat` | Binned age | young/medium/mature/old |
| `young_high_roa` | `(age ≤ 5) & (roa > 0.1)` | High-potential startup interaction |

**Industry consolidation.** The two-digit industry code was collapsed into broader categories: codes <26 → 20, 31 → 30, 36–54 → 40, >56 → 60, missing → 99.

---

## 3. Model Specification

### 3.1 Variable Sets

Models were built from modular variable blocks of increasing richness:

- **Raw financials** (16 vars): balance sheet and P&L items such as current assets, current liabilities, extra expenses/income, etc.
- **Engineered ratios** (12 vars): balance sheet and P&L items normalized by total assets or sales (e.g., `liq_assets_bs`, `curr_assets_bs`)
- **Quadratic terms** (3 vars): squared versions of signed ratios to capture non-linear effects
- **Flag indicators** (~20 vars): dummy variables marking winsorized or erroneous observations
- **Sales growth** (1 var): winsorized first-differenced log sales
- **HR and firm characteristics** (7 vars): CEO gender, foreign management, firm age (linear and squared), new firm indicator, industry category, and regional indicator
- **Growth features** (6 vars): asset tangibility, liquidity ratio, ROA, leverage, log total assets, and young × high-ROA interaction

### 3.2 Five Logit Specifications

| Model | Description | # Coefficients |
|---|---|---|
| M1 | Log sales + sales growth + profitability ratio | 4 |
| M2 | M1 + balance sheet ratios + age + foreign management | 9 |
| M3 | Log sales + all engineered ratios + sales growth + HR/firm characteristics + growth features | 30 |
| M4 | M3 + quadratic terms + flag indicators + quality variables | 61 |
| M5 | M4 + industry × age/sales/gender interactions + size interactions | 162 |

Each successive model adds more complexity. The goal is to determine whether additional variables improve predictive performance or lead to overfitting.

### 3.3 LASSO Logit

Uses the M5-level variable set (the most complex specification) with an additional squared log-sales term. Features were standardized prior to fitting. Regularization strength was searched over 10 values of λ spaced evenly on the log scale between 10⁻¹ and 10⁻⁴. The best λ was selected by minimum mean CV RMSE (converted from negative Brier score). The optimal LASSO retained 74 non-zero coefficients out of the original 162+, automatically discarding redundant or uninformative features.

### 3.4 Random Forest

The Random Forest used 100 trees with Gini splitting. Hyperparameters were tuned via 5-fold cross-validated grid search over `max_features` ∈ {5, 6, 7} and `min_samples_split` ∈ {11, 16}. Best parameters: `max_features=6`, `min_samples_split=16`.

---

## 4. Part I: Probability Prediction Results

### 4.1 Cross-Validation Setup

An 80/20 train-holdout split was applied (`random_state=42`), yielding 15,996 training and 4,000 holdout observations. All CV used 5-fold cross-validation with shuffling.

### 4.2 Summary of CV Performance

| Model | # Coefficients | CV RMSE | CV AUC | Training Time |
|---|---|---|---|---|
| M1 | 4 | 0.452 | 0.564 | — |
| M2 | 9 | 0.436 | 0.672 | — |
| M3 | 30 | 0.425 | 0.716 | — |
| **M4** | **61** | **0.425** | **0.718** | — |
| M5 | 162 | 0.426 | 0.715 | — |
| LASSO | 74 | 0.425 | 0.718 | 192s |
| RF | n.a. | 0.426 | 0.714 | 391s |

All logit models were trained in ~417s total.

RMSE measures how close the predicted probabilities are to the actual outcomes (lower is better). AUC measures how well the model ranks fast-growth firms above non-growth firms (higher is better; 0.5 = random, 1.0 = perfect). An AUC of ~0.72 means that if we pick one random fast-growth firm and one random non-growth firm, the model assigns a higher probability to the fast-growth firm about 72% of the time.

### 4.3 Model Selection Rationale

M3, M4, LASSO, and RF achieve nearly identical CV RMSE (~0.425–0.426). **M4** was selected as the preferred logit model because it achieves the highest AUC (0.718), indicating the best discrimination ability — i.e., it is the best at ranking fast-growth firms above non-growth firms. M4 is also simpler than M5 (61 vs. 162 coefficients) and does not require the feature standardization that LASSO needs. Adding M5's interaction terms did not improve AUC (0.715), suggesting those interactions are not informative. RF shows comparable RMSE but slightly lower AUC (0.714).

### 4.4 Holdout Evaluation

M4 holdout RMSE: **0.429**. RF holdout RMSE: **0.429**, AUC: **0.696**.

The slight increase from CV RMSE (0.425) to holdout RMSE (0.429) indicates minimal overfitting — the model generalizes well to unseen data.

![Calibration Plot — Logit M4 on Holdout](graphs/calibration_logit_holdout.png)

The calibration plot compares predicted probabilities against actual fast-growth rates. Points close to the diagonal line indicate well-calibrated predictions. Deviations reveal where the model over- or under-estimates the probability of fast growth.

![ROC Curve — Logit M4 on Holdout](graphs/roc_logit_holdout.png)

The ROC curve shows the trade-off between catching fast-growth firms (sensitivity, y-axis) and incorrectly flagging non-growth firms (false positive rate, x-axis) at every possible threshold. The shaded area under the curve represents the AUC. A curve hugging the top-left corner would indicate near-perfect discrimination; our curve shows moderate but meaningful discriminative ability.

---

## 5. Part II: Classification

### 5.1 Business Problem

The model serves as a venture capital screening tool that flags firms likely to experience fast growth from a large candidate pool. Analysts then prioritize flagged firms for due diligence.

### 5.2 Loss Function

Asymmetric costs reflect VC economics:

- **FP = $1** — wasted analyst time reviewing a non-growth firm
- **FN = $10** — opportunity cost of missing a fast-growing firm

The 10:1 ratio captures the VC principle that missing a winner is far more costly than evaluating a non-winner. This asymmetry will push the optimal classification threshold well below the conventional 0.5, because it is 10 times more important to avoid missing a fast-grower than to avoid a false alarm.

### 5.3 Optimal Threshold Search

For each model and CV fold, the classification threshold was chosen to minimize the total expected loss. The search iterates over all possible thresholds along the ROC curve, computing at each point:

> Expected loss = (FP count × $1 + FN count × $10) / N

The threshold that produces the lowest expected loss is selected as the optimal threshold for that fold. Results are averaged across the 5 folds.

### 5.4 Classification Results

| Model | Avg Optimal Threshold | Avg Expected Loss |
|---|---|---|
| M1 | 0.1489 | 0.7092 |
| M2 | 0.1195 | 0.6927 |
| M3 | 0.1030 | 0.6701 |
| **M4** | **0.1007** | **0.6694** |
| M5 | 0.0886 | 0.6783 |
| LASSO | 0.1157 | 0.6705 |
| RF | 0.0958 | 0.6704 |

M4 achieves the lowest average expected loss (0.6694), closely followed by M3 (0.6701), RF (0.6704), and LASSO (0.6705). Optimal thresholds range from ~0.09 (M5) to ~0.15 (M1), all well below 0.5 due to the 10:1 cost asymmetry. Models with better probability predictions (lower CV RMSE) tend to achieve lower expected loss, confirming that accurate probability estimation translates into better business outcomes. M4 was selected for holdout evaluation as it achieves the best expected loss alongside the highest AUC (0.718).

The following plots illustrate the loss function and ROC curve for M4 and Random Forest on Fold 5 of cross-validation.

![Loss Plot — Logit M4, Fold 5](graphs/loss_logit_m4_fold5.png)

The loss curve shows expected loss (y-axis) at each possible threshold (x-axis). The dashed vertical line marks the optimal threshold that minimizes expected loss. The curve is steep near zero and flattens at higher thresholds, reflecting that the cost of missing fast-growers (FN) dominates when the threshold is set too high.

![ROC Curve with Optimal Threshold — Logit M4, Fold 5](graphs/roc_optimal_logit_m4_fold5.png)

The ROC curve with the optimal threshold marked as a black dot. The dot's position near the top-right of the curve shows that at the optimal threshold, the model achieves very high sensitivity (catching most fast-growers) at the cost of a high false positive rate (flagging most non-growers as well).

![Loss Plot — Random Forest, Fold 5](graphs/loss_rf_fold5.png)

The Random Forest loss curve shows a similar shape but with a higher optimal threshold than M4. RF's discrete tree structure produces a less smooth loss curve, and its optimal threshold tends to be higher because the ensemble's probability predictions are less extreme at the tails.

![ROC Curve with Optimal Threshold — Random Forest, Fold 5](graphs/roc_optimal_rf_fold5.png)

The RF ROC curve with its optimal threshold marked. Compared to M4, the optimal point sits slightly further from the top-right corner, indicating that RF misses more fast-growth firms at its optimal threshold.

### 5.5 Holdout Expected Loss

M4 holdout expected loss: 0.733. 
RF holdout expected loss: 0.712.

RF outperforms M4 by 0.021 per firm (2.9% lower loss) on the holdout set. Despite this, **M4 remains our recommended model** for the following reasons:

1. **Interpretability.** Logistic regression coefficients are directly interpretable: analysts can understand which financial characteristics drive the prediction and by how much. This transparency is critical in a VC screening context where flagged firms undergo human review — analysts need to understand *why* a firm was flagged, not just *that* it was flagged. Random Forest, as a black-box ensemble of 100 trees, does not offer this.

2. **Discrimination ability.** M4 achieves the highest CV AUC (0.718 vs. RF's 0.714), meaning it ranks fast-growth firms above non-growth firms more reliably. AUC is a threshold-independent metric: if the loss function or cost assumptions change in deployment, a model with better AUC will adapt more gracefully to different threshold choices.

3. **CV consistency.** M4 achieves the lowest average expected loss across all 5 CV folds (0.6694 vs. RF's 0.6704). The RF's holdout advantage may reflect favorable sampling variation in a single 80/20 split rather than a systematic superiority.

4. **Practical gap is small.** The 0.021 per-firm difference translates to ~21 dollars per 1,000 firms screened — a marginal gap that does not outweigh the interpretability and auditability benefits of a parametric model.

---

## 6. Part III: Discussion of Results

### 6.1 Confusion Matrix (M4, Holdout)

|  | Predicted No Growth | Predicted Fast Growth |
|---|---|---|
| **Actual No Growth** | 335 | 2,203 |
| **Actual Fast Growth** | 39 | 959 |

Note: the holdout set contains 4,000 observations, but 464 were dropped due to missing values in model variables (primarily `age`), leaving 3,536 observations for evaluation.

With threshold ~0.10, M4 achieves 96.09% sensitivity (catches 959 of 998 fast-growth firms), 13.20% specificity, and 30.33% precision. The model flags roughly 89% of evaluated firms as potential fast-growers — a consequence of the 10:1 FN/FP cost asymmetry pushing the threshold well below 0.5.

The expected loss breaks down as:
- False Positive cost: 2,203 × 1 = 2,203
- False Negative cost: 39 × 10 = 390
- Total: 2,593 for 3,536 firms (0.733 per firm)

### 6.2 Comparison with Universal Flagging

The naive strategy of flagging all firms as fast-growth would yield an expected loss of ~0.718 per firm (= proportion of non-growth firms × 1). M4's holdout expected loss (0.733) is slightly worse than this baseline, suggesting some overfitting of the threshold or distributional shift in the holdout set. In contrast, the Random Forest achieves $0.712 on the holdout, slightly outperforming the universal-flagging baseline.

This does not mean M4 is useless. The predicted probabilities still provide a meaningful ranking — firms with higher predicted probabilities are more likely to be true fast-growers. In practice, analysts could use the probability ranking to prioritize which flagged firms to review first. While RF slightly outperforms the baseline on holdout, M4's interpretability and superior AUC make it the preferred choice for deployment.

### 6.3 Practical Assessment

The predictions are actionable as a first-pass screening tool: analysts can prioritize flagged firms for deeper due diligence by ranking them by predicted probability, rather than treating the binary flag as a final investment decision.

Key limitations include:

- **Single-year definition:** Fast growth is measured over 2012–2013 only. A firm that grew rapidly in that specific year may not sustain growth, and the model may not generalize to other time periods.
- **Macroeconomic sensitivity:** The model cannot capture economy-wide shocks or industry disruptions that affect growth patterns.
- **High flagging rate:** With the current loss function, the model flags ~89% of firms. Its practical value lies in the probability ranking, not the binary classification.
- **No external validation:** The model has not been tested on out-of-sample time periods, which would strengthen confidence before deployment.

---

## 7. Industry Analysis

### 7.1 Setup

Two subsamples were defined: Manufacturing (`ind2_cat` ∈ {20, 30}) and Services (`ind2_cat` ∈ {40, 60}). Both use the same loss function (FP=$1, FN=$10). Industry-specific logit (M4-style, without the industry category variable) and RF models were trained with 5-fold CV.

### 7.2 Results

| Metric | Manufacturing | Services |
|---|---|---|
| Train size | 175 | 340 |
| Fast growth rate (train) | 26.86% | 27.65% |
| Holdout size | 38 | 88 |
| Logit CV RMSE | 0.5434 | 0.4782 |
| Logit Avg Expected Loss | 0.6312 | 0.6919 |
| RF CV RMSE | **0.4427** | **0.4227** |
| RF Avg Expected Loss | **0.6115** | **0.5689** |
| **Recommended** | **RF** | **RF** |

### 7.3 Key Takeaways

**Sample size concerns.** Both subsamples are very small (175 and 340 training observations). Manufacturing is particularly problematic: its holdout fast-growth rate is 42.11%, far higher than the 26.86% training rate. This large discrepancy suggests that with only 38 holdout firms, the split is not representative, and results should be treated as exploratory rather than reliable.

**RF dominates on both metrics.** In both industries, Random Forest achieves better probability predictions (lower RMSE) and lower expected loss than logit. Logit produces very low thresholds (0.0056 for manufacturing, 0.0198 for services), effectively flagging nearly all firms, but its poorer probability estimates lead to higher overall loss. RF's thresholds are higher (0.1359 and 0.1367), yet its superior probability predictions more than compensate — a pattern consistent with the holdout results from the pooled analysis, where RF also outperforms M4 on expected loss. The likely explanation is that with small industry-specific samples (175–340 firms), logit's many parameters overfit, while RF's built-in regularization (ensemble averaging, `min_samples_split`) provides better generalization.

**Recommendation.** The full-sample pooled model with ~16,000 training observations provides more stable and trustworthy predictions than either industry-specific model. Industry effects are better incorporated as features within the unified model rather than by splitting the data into subsamples too small for reliable modeling.

---

</div>
