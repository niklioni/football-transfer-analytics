# ⚽ Football Transfer Market Analysis

A data science assessment analysing 15+ years of professional football transfers: cleaning a messy real-world dataset, exploring how transfer spending has evolved across Europe's top leagues, and building machine learning models to predict transfer fees.

Everything lives in a single, self-contained notebook: **[`analysis.ipynb`](analysis.ipynb)**.

## Overview

The notebook walks through a complete data science workflow on football transfer data:

1. **Data Loading & Merging** — combine transfers, clubs, and competitions into one dataset
2. **Data Exploration** — assess completeness and understand the raw distributions
3. **Data Cleaning & Preparation** — filter to a reliable time window, handle missing values, drop invalid rows
4. **Outlier Analysis** — identify extreme transfer fees using the IQR method
5. **Exploratory Data Analysis** — visualise global spending trends and the dominance of the Premier League
6. **Machine Learning** — predict transfer fees with Linear Regression and Random Forest, evaluated via train/test split and 5-fold cross-validation

## Dataset

The notebook expects three CSV files in a `data/` folder (not included in this repo):

| File | Description |
|---|---|
| `data/transfers.csv` | Individual player transfers (player, clubs, date, fee, market value) |
| `data/clubs.csv` | Club metadata, including which domestic competition each club plays in |
| `data/competitions.csv` | Competition metadata, including the country each competition belongs to |

Place these files under `data/` in the project root before running the notebook.

## Key Findings

- **Data quality**: of the ~100k raw transfer records, only ~41k survive cleaning — most drop-off comes from transfers whose club couldn't be mapped to a domestic competition/country (~56%), plus a small tranche of unreliable pre-2010 records.
- **Free transfers dominate**: roughly two-thirds of all transfers involve no fee (loans, free agents, youth moves); the analysis separates these out and focuses on the ~8,800 *paid* transfers for spending and modelling questions.
- **England leads spending**: the Premier League consistently outspends Spain, Italy, Germany, and France, and its share of total European transfer spending has grown over time.
- **Market value predicts fees well**: a Random Forest model reaches **R² ≈ 0.78 / MAE ≈ €2.8M** on held-out data, only marginally ahead of plain Linear Regression (R² ≈ 0.76) — meaning most of the predictive power comes from a player's market value rather than nonlinear effects, and cross-validation (R² ≈ 0.777–0.778 for both) confirms the results generalise rather than overfitting to one split.

## Tech Stack

- **pandas** / **numpy** — data loading, merging, cleaning
- **matplotlib** — visualisation
- **scikit-learn** — Linear Regression, Random Forest, train/test split, cross-validation, evaluation metrics

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/niklioni/DataScienceAssessment.git
cd DataScienceAssessment

# 2. Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter

# 3. Add the data/ folder (see Dataset section above)

# 4. Launch the notebook
jupyter notebook analysis.ipynb
```

## Project Structure

```
DataScienceAssessment/
├── analysis.ipynb    # Full analysis: cleaning, EDA, and ML
├── data/              # transfers.csv, clubs.csv, competitions.csv (not tracked)
└── README.md
```
