# NYC Airbnb Price Prediction

End-to-end data preparation, modelling, feature selection, and SHAP explainability pipeline using ten Inside Airbnb NYC listing snapshots from August 2025 to June 2026. May 2026 is not present in the supplied files.

## Project structure

```text
airbnb-price-prediction/
├── data/
│   ├── raw/                 # Original monthly .csv.gz snapshots; never modified
│   └── processed/           # Master, cleaned, and feature datasets
├── code/
│   ├── 01_build_master_dataset.py
│   ├── 02_data_audit.py
│   ├── 03_clean_data.py
│   ├── 04_feature_engineering.py
│   ├── 05_eda.py
│   ├── 06_modeling.py
│   ├── 07_lasso.py
│   ├── 08_shap_analysis.py
│   ├── config.py
│   └── utils.py
├── output/
│   ├── audit/
│   ├── figures/
│   ├── models/
│   └── tables/
├── report/
├── requirements.txt
└── run_pipeline.py
```

## Setup

From the repository root:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the pipeline

Run every stage:

```bash
python run_pipeline.py
```

Run one stage:

```bash
python run_pipeline.py --step clean
python run_pipeline.py --step modeling
python run_pipeline.py --step shap
```

Run from a stage through the end:

```bash
python run_pipeline.py --from-step features
```

Available stage names are `build`, `audit`, `clean`, `features`, `eda`, `modeling`, `lasso`, and `shap`.

## Pipeline stages

1. **Build:** combines all snapshots and adds snapshot metadata.
2. **Audit:** creates missingness, numerical, price, duplicate, and monthly summaries.
3. **Clean:** parses prices/percentages/booleans/dates, removes invalid records, and removes prices above the 99.5th percentile.
4. **Features:** creates temporal, host-tenure, review-recency, availability, capacity-ratio, and host-scale features.
5. **EDA:** produces monthly, borough, and room-type tables and figures.
6. **Modelling:** uses June 2026 as a time-based holdout and uses Linear Regression as the formal baseline, followed by cross-validated Lasso regression and Random Forest on log price. A median predictor is retained internally only as a development sanity check.
7. **Lasso:** exports non-zero coefficients and the 30 strongest selected features.
8. **SHAP:** exports global Random Forest feature importance based on Shapley values.

## Evaluation design

The latest snapshot, June 2026, is held out for testing. This is preferable to a random row split because the same Airbnb listing can appear in multiple snapshots. A random split could expose earlier or later observations of the same property to both training and testing.

By default, the modelling stage uses all historical observations before June 2026 (approximately 126,588 rows) and evaluates on the complete June 2026 holdout (approximately 21,477 rows). This ensures the training set is substantially larger than the test set. For quicker development runs, set `USE_FULL_TRAINING = False` in `code/config.py`; `TRAIN_SAMPLE_SIZE` then controls the reproducible sample cap. LassoCV selects its regularisation strength using five-fold cross-validation.

## Main outputs

- `data/processed/master_listings.csv.gz`
- `data/processed/cleaned_listings.csv.gz`
- `data/processed/model_features.csv.gz`
- `output/tables/model_comparison.csv`
- `output/tables/lasso_top_30_features.csv`
- `output/tables/shap_feature_importance.csv`
- `output/figures/shap_global_importance.png`
- `output/models/linear_regression_pipeline.joblib`
- `output/models/lasso_pipeline.joblib`
- `output/models/random_forest_pipeline.joblib`
- `output/tables/holdout_predictions.csv`

## Current tested result

Previously generated sample-run results showed Random Forest performing best, followed by LassoCV. The updated default configuration uses the full historical training set and adds ordinary Linear Regression as the formal regression baseline, so rerunning `python run_pipeline.py --from-step modeling` will regenerate all comparison metrics. Log-target predictions are constrained to the central observed training-price range before inverse transformation to prevent unrealistic exponential extrapolation. Results may change if the cleaning thresholds, features, sample size, or model parameters are changed.

## Final report generation

Run the entire project, including report generation:

```bash
python run_pipeline.py
```

After changing only model settings, rerun modelling through the final report:

```bash
python run_pipeline.py --from-step modeling
```

Generate only the final tables, diagnostic figures, and Markdown report from existing predictions:

```bash
python run_pipeline.py --step report
```

The final report is written to `report/model_evaluation_report.md`. The report includes MAE, median absolute error, RMSE, R-squared, MAPE, and prediction-within-tolerance rates for ±5%, ±10%, ±20%, and ±30%. These tolerance rates are regression measures and should not be described as classification accuracy.


Lasso outputs are supplied at two levels: encoded category coefficients and aggregated original-feature importance. The Lasso-versus-SHAP comparison uses the aggregated level so both methods are compared consistently.

## Final report outputs

The final reporting stage intentionally keeps the main analysis focused. It reports only:

- MAE (USD)
- RMSE (USD)
- R²
- MAPE (%)
- Percentage of predictions within ±20% of the observed price

Detailed model outputs remain in `output/tables/` for traceability, but the written report uses one concise model table and five core figures. Related predictors are aggregated into business-facing feature families, including Availability and Review Scores, before comparing Lasso and Random Forest SHAP importance.
