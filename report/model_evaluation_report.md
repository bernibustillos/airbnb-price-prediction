# NYC Airbnb Price Prediction — Final Model Report

## Data preparation

The raw monthly snapshots contained 355,307 listing observations. Duplicate listing-snapshot records were removed, listings with missing or non-positive prices were excluded, the extreme upper 0.5% of prices was removed, and basic capacity and NYC coordinate plausibility checks were applied. The resulting modelling dataset contained 148,065 observations. Missing predictor values were handled inside each modelling pipeline using median or most-frequent-value imputation, avoiding unnecessary row deletion.

## Evaluation design

The final evaluation used the **2026-06 snapshot** as an unseen holdout test set containing **21,477 listings**, while all earlier monthly snapshots were used for training. This time-based split reduces leakage from repeated listings across snapshots.

## Model performance

| Model | MAE (USD) | RMSE (USD) | R² | MAPE (%) | Within ±20% (%) |
|---|---|---|---|---|---|
| Linear Regression (Baseline) | 108.90 | 255.79 | 0.38 | 39.81 | 32.92 |
| LassoCV | 108.79 | 255.81 | 0.38 | 39.75 | 33.06 |
| Random Forest | 71.81 | 181.09 | 0.69 | 23.33 | 57.16 |

Linear Regression served as the baseline predictive model because of its simplicity and interpretability.

LassoCV achieved almost identical predictive performance to Linear Regression (R² = **0.376** and **0.376**, respectively). This indicates that L1 regularisation did not materially improve predictive accuracy, although Lasso remained valuable for feature selection and interpretation.

**Random Forest** achieved the best performance with an MAE of **$71.81**, RMSE of **$181.09**, and R² of **0.687**. It placed **57.2%** of predictions within ±20% of the observed nightly price. Its advantage suggests that Airbnb prices depend on nonlinear relationships and interactions among location, property characteristics, capacity, availability, and host characteristics.

## Main pricing drivers

To improve interpretability, related variables were grouped into business-oriented feature families. **Capacity & Space** combines accommodates, bedrooms, bathrooms, beds, and per-guest space measures; **Availability** combines 30-, 60-, 90-, and 365-day availability; and **Review Scores** combines the individual rating dimensions. This aggregation enables a direct comparison between Lasso coefficients and Random Forest SHAP importance. Both importance measures were normalised separately to percentages before comparison.

| rank | Feature family | Lasso importance (%) | RF SHAP importance (%) |
|---|---|---|---|
| 1 | Location | 26.26 | 17.60 |
| 2 | Capacity & Space | 17.75 | 19.72 |
| 3 | Property Type | 15.16 | 19.38 |
| 4 | Host Characteristics | 15.08 | 8.01 |
| 5 | Booking Constraints | 2.89 | 15.21 |
| 6 | Room Type | 9.83 | 5.64 |
| 7 | Review Scores | 4.76 | 4.75 |
| 8 | Availability | 1.94 | 5.46 |

Both Lasso and Random Forest consistently identified Location, Capacity & Space, and Property Type as major drivers of Airbnb prices. This agreement strengthens confidence in the findings despite the models’ different assumptions. Feature families ranked substantially higher by Random Forest SHAP, such as Booking Constraints, indicate nonlinear relationships or interactions that are not fully captured by the linear Lasso model.

## Final figures

The written submission should use only the following core figures:

1. `actual_vs_predicted_random_forest.png`
2. `residuals_vs_predicted_random_forest.png`
3. `shap_summary_beeswarm.png`
4. `lasso_feature_family_importance.png`
5. `lasso_vs_shap_feature_families.png`

## Interpretation caution

The target represents the advertised nightly listing price at each snapshot date rather than a completed booking price or realised revenue. The results therefore provide evidence-based initial pricing guidance, not a guarantee of occupancy or income.