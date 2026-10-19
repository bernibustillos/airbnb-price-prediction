# NYC Airbnb Price Prediction

Predicting and explaining nightly prices for **new** NYC Airbnb listings — ones with no booking history — from property, host and location features alone.

**Team:** Bernardo Bustillos, Amit Kumar, Shuvechchha Kunwar · Georgia Tech machine learning team project, Summer 2026
*Forked from [kumaramit003/airbnb-price-prediction](https://github.com/kumaramit003/airbnb-price-prediction).*

## Results

Evaluated on a time-based holdout (June 2026, 21,477 listings), trained on all earlier snapshots (126,588 rows).

| Model | MAE (USD) | RMSE (USD) | R² | MAPE | Within ±20% |
| --- | --- | --- | --- | --- | --- |
| Naive median predictor | $162.68 | — | −0.11 | — | — |
| Linear Regression | $108.90 | $255.79 | 0.38 | 39.8% | 32.9% |
| LassoCV | $108.79 | $255.81 | 0.38 | 39.8% | 33.1% |
| **Random Forest** | **$71.81** | **$181.09** | **0.69** | **23.3%** | **57.2%** |

**Key findings**

- **Location, capacity & space, and property type** drive over half of the Random Forest's importance (SHAP).
- **Booking constraints** (e.g. minimum nights) matter far more to the Random Forest than to Lasso, evidence that booking rules interact with property type and borough rather than shifting price by a flat amount.
- **Prices are nonlinear:** Lasso matched plain linear regression, so the Random Forest's ~30-point R² gain comes from interactions, not from removing collinear features.
- The model is most reliable under ~$250/night, where most of the market sits, and under-predicts rare luxury listings.

## Data

Ten monthly [Inside Airbnb](https://insideairbnb.com/get-the-data/) NYC snapshots, August 2025 – June 2026 (355,307 records). Three winter snapshots had no prices and were dropped, leaving 148,065 observations across seven months.

## Method

1. **Clean:** parse prices, drop missing/zero/implausible prices, impute remaining gaps (median / most frequent), model price on a log scale.
2. **Engineer features** in 10 families: location (borough, lat/long), capacity & space (incl. per-guest ratios), property type, room type, booking constraints, availability, host characteristics (tenure, multi-listing flag), review scores, review activity (recency, rate), and cyclical seasonality.
3. **Prevent leakage:** the same listing appears in many monthly snapshots, so the newest month is held out instead of a random split; near-duplicate features (|r| > 0.95) are pruned on training data only.
4. **Model:** Linear Regression (baseline), LassoCV (5-fold CV, also used for feature selection), Random Forest.
5. **Explain:** SHAP values for the Random Forest and Lasso coefficients, aggregated to feature families for a like-for-like comparison.

## Project structure

```
airbnb-price-prediction/
├── data/
│   ├── raw/          # Original monthly snapshots (never modified)
│   └── processed/    # Master, cleaned and feature datasets
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
├── output/           # Audit tables, figures, models, predictions
├── report/
├── requirements.txt
└── run_pipeline.py
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py                      # every stage
python run_pipeline.py --step modeling      # one stage
python run_pipeline.py --from-step features # from a stage to the end
```

Stages: `build`, `audit`, `clean`, `features`, `eda`, `modeling`, `lasso`, `shap`, `report`.

## Limitations

- Availability fields mix unbooked dates with dates hosts blocked, so they are a rough demand signal.
- Seven non-contiguous months of advertised (not transacted) prices understate seasonality.
- Findings are specific to NYC's market and regulations.
