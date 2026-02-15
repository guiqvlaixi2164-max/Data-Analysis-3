# Predicting Firm Fast Growth

Binary classification project that predicts whether a firm will achieve fast growth (>=20% year-over-year total asset growth) using financial and demographic data from the Bisnode firms dataset.

## Folder Structure

```
Assignment 2/
├── code/
│   ├── firm-fast-growth-data-prep-new.ipynb    # Data cleaning & feature engineering (use this)
│   ├── firm-fast-growth-prediction-new.ipynb   # Model training & evaluation (use this)
│   ├── firm-fast-growth-data-prep.ipynb        # Earlier draft (kept for reference)
│   └── firm-fast-growth-prediction.ipynb       # Earlier draft (kept for reference)
├── data/
│   ├── cs_bisnode_panel.csv                    # Raw panel data (input)
│   ├── work5.csv                               # Intermediate dataset
│   └── bisnode_firms_clean.csv                 # Final cleaned dataset
├── graphs/                                     # Model evaluation plots (ROC, calibration, loss)
├── summary_report.md                           # Non-technical summary
├── technical_report.md                         # Detailed methodology & results
└── requirements.txt                            # Python dependencies
```

## Environment Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

**Note:** `code/firm-fast-growth-data-prep.ipynb` and `code/firm-fast-growth-prediction.ipynb` are earlier drafts kept for reference. Please use the two `-new` notebooks for all review and reproduction.

## Reproducing Full Pipeline

1. **Data Preparation** -- Open and run all cells in `code/firm-fast-growth-data-prep-new.ipynb`. This reads `data/cs_bisnode_panel.csv`, performs label engineering, sample filtering, and feature engineering, then writes `data/bisnode_firms_clean.csv`.

2. **Prediction** -- Open and run all cells in `code/firm-fast-growth-prediction-new.ipynb`. This reads `data/bisnode_firms_clean.csv`, trains Logistic Regression, LASSO, Random Forest, and Decision Tree models with cross-validation, and saves evaluation plots to `graphs/`.
