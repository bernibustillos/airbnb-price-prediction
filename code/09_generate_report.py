"""Generate the concise final evaluation package for the written submission."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import TABLES_DIR, FIGURES_DIR, MODELS_DIR, PROJECT_ROOT
from utils import save_csv

PREDICTIONS_FILE = TABLES_DIR / "holdout_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLUMNS = {
    "Linear Regression (Baseline)": "linear_regression_prediction",
    "LassoCV": "lasso_prediction",
    "Random Forest": "random_forest_prediction",
}


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.20) -> float:
    mask = y_true > 0
    relative_error = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(relative_error <= tolerance) * 100)


def make_random_forest_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Create only the two diagnostic plots needed in the final report."""
    residual = y_true - y_pred
    display_max = float(np.quantile(np.concatenate([y_true, y_pred]), 0.99))
    mask = (y_true <= display_max) & (y_pred <= display_max)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.18, s=10)
    plt.plot([0, display_max], [0, display_max], linestyle="--", linewidth=1.5)
    plt.xlabel("Actual nightly price (USD)")
    plt.ylabel("Predicted nightly price (USD)")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted_random_forest.png", dpi=300)
    plt.close()

    pred_limit = float(np.quantile(y_pred, 0.99))
    resid_limit = float(np.quantile(np.abs(residual), 0.99))
    residual_mask = (y_pred <= pred_limit) & (np.abs(residual) <= resid_limit)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred[residual_mask], residual[residual_mask], alpha=0.18, s=10)
    plt.axhline(0, linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted nightly price (USD)")
    plt.ylabel("Residual: actual − predicted (USD)")
    plt.title("Random Forest: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_vs_predicted_random_forest.png", dpi=300)
    plt.close()


def build_family_comparison() -> pd.DataFrame:
    lasso_path = TABLES_DIR / "lasso_feature_family_importance.csv"
    shap_path = TABLES_DIR / "shap_feature_family_importance.csv"
    if not lasso_path.exists() or not shap_path.exists():
        return pd.DataFrame()

    lasso = pd.read_csv(lasso_path)[["feature_family", "lasso_importance_pct"]]
    shap = pd.read_csv(shap_path)[["feature_family", "shap_importance_pct"]]
    comparison = lasso.merge(shap, on="feature_family", how="outer").fillna(0)
    comparison["combined_importance_pct"] = (
        comparison["lasso_importance_pct"] + comparison["shap_importance_pct"]
    ) / 2
    comparison = comparison.sort_values("combined_importance_pct", ascending=False).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    save_csv(comparison, TABLES_DIR / "lasso_shap_feature_family_comparison.csv")

    top = comparison.head(10).sort_values("combined_importance_pct")
    y = np.arange(len(top))
    height = 0.36
    plt.figure(figsize=(10, 7))
    plt.barh(y - height / 2, top["lasso_importance_pct"], height=height, label="Lasso")
    plt.barh(y + height / 2, top["shap_importance_pct"], height=height, label="Random Forest SHAP")
    plt.yticks(y, top["feature_family"])
    plt.xlabel("Share of model importance (%)")
    plt.title("Main Pricing Drivers: Lasso vs Random Forest SHAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lasso_vs_shap_feature_families.png", dpi=300)
    plt.close()
    return comparison


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for c in view.select_dtypes(include=["float"]).columns:
        view[c] = view[c].map(lambda x: f"{x:.2f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator] + rows)


def cleaning_paragraph() -> str:
    path = TABLES_DIR / "cleaning_summary.json"
    if not path.exists():
        return "The data were cleaned using transparent target, duplicate, capacity, and geographic validity rules."
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return (
        f"The raw monthly snapshots contained {summary['initial_rows']:,} listing observations. "
        "Duplicate listing-snapshot records were removed, listings with missing or non-positive prices were excluded, "
        "the extreme upper 0.5% of prices was removed, and basic capacity and NYC coordinate plausibility checks were applied. "
        f"The resulting modelling dataset contained {summary['clean_rows']:,} observations. Missing predictor values were "
        "handled inside each modelling pipeline using median or most-frequent-value imputation, avoiding unnecessary row deletion."
    )


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError("Run 06_modeling.py before 09_generate_report.py.")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    y_true = predictions["actual_price"].to_numpy(dtype=float)


    rows = []
    for model_name, column in MODEL_COLUMNS.items():
        y_pred = predictions[column].to_numpy(dtype=float)
        rows.append({
            "Model": model_name,
            "MAE (USD)": mean_absolute_error(y_true, y_pred),
            "RMSE (USD)": mean_squared_error(y_true, y_pred) ** 0.5,
            "R²": r2_score(y_true, y_pred),
            "MAPE (%)": safe_mape(y_true, y_pred),
            "Within ±20% (%)": within_tolerance(y_true, y_pred),
        })

    evaluation = pd.DataFrame(rows)
    save_csv(evaluation, TABLES_DIR / "final_model_evaluation.csv")

    rf_pred = predictions["random_forest_prediction"].to_numpy(dtype=float)
    make_random_forest_diagnostics(y_true, rf_pred)
    family_comparison = build_family_comparison()

    best = evaluation.loc[evaluation["MAE (USD)"].idxmin()]
    linear = evaluation[evaluation["Model"] == "Linear Regression (Baseline)"].iloc[0]
    lasso = evaluation[evaluation["Model"] == "LassoCV"].iloc[0]

    report = [
        "# NYC Airbnb Price Prediction — Final Model Report",
        "",
        "## Data preparation",
        "",
        cleaning_paragraph(),
        "",
        "## Evaluation design",
        "",
        f"The final evaluation used the **{predictions['snapshot_month'].iloc[0]} snapshot** as an unseen holdout test set containing **{len(predictions):,} listings**, while all earlier monthly snapshots were used for training. This time-based split reduces leakage from repeated listings across snapshots.",
        "",
        "## Model performance",
        "",
        markdown_table(evaluation, ["Model", "MAE (USD)", "RMSE (USD)", "R²", "MAPE (%)", "Within ±20% (%)"]),
        "",
        "Linear Regression served as the baseline predictive model because of its simplicity and interpretability.",
        "",
        f"LassoCV achieved almost identical predictive performance to Linear Regression (R² = **{lasso['R²']:.3f}** and **{linear['R²']:.3f}**, respectively). This indicates that L1 regularisation did not materially improve predictive accuracy, although Lasso remained valuable for feature selection and interpretation.",
        "",
        f"**Random Forest** achieved the best performance with an MAE of **${best['MAE (USD)']:.2f}**, RMSE of **${best['RMSE (USD)']:.2f}**, and R² of **{best['R²']:.3f}**. It placed **{best['Within ±20% (%)']:.1f}%** of predictions within ±20% of the observed nightly price. Its advantage suggests that Airbnb prices depend on nonlinear relationships and interactions among location, property characteristics, capacity, availability, and host characteristics.",
        "",
        "## Main pricing drivers",
        "",
        "To improve interpretability, related variables were grouped into business-oriented feature families. **Capacity & Space** combines accommodates, bedrooms, bathrooms, beds, and per-guest space measures; **Availability** combines 30-, 60-, 90-, and 365-day availability; and **Review Scores** combines the individual rating dimensions. This aggregation enables a direct comparison between Lasso coefficients and Random Forest SHAP importance. Both importance measures were normalised separately to percentages before comparison.",
    ]

    if not family_comparison.empty:
        report.extend([
            "",
            markdown_table(
                family_comparison.head(8).rename(columns={
                    "feature_family": "Feature family",
                    "lasso_importance_pct": "Lasso importance (%)",
                    "shap_importance_pct": "RF SHAP importance (%)",
                }),
                ["rank", "Feature family", "Lasso importance (%)", "RF SHAP importance (%)"],
            ),
            "",
            "Both Lasso and Random Forest consistently identified Location, Capacity & Space, and Property Type as major drivers of Airbnb prices. This agreement strengthens confidence in the findings despite the models’ different assumptions. Feature families ranked substantially higher by Random Forest SHAP, such as Booking Constraints, indicate nonlinear relationships or interactions that are not fully captured by the linear Lasso model.",
        ])

    report.extend([
        "",
        "## Final figures",
        "",
        "The written submission should use only the following core figures:",
        "",
        "1. `actual_vs_predicted_random_forest.png`",
        "2. `residuals_vs_predicted_random_forest.png`",
        "3. `shap_summary_beeswarm.png`",
        "4. `lasso_feature_family_importance.png`",
        "5. `lasso_vs_shap_feature_families.png`",
        "",
        "## Interpretation caution",
        "",
        "The target represents the advertised nightly listing price at each snapshot date rather than a completed booking price or realised revenue. The results therefore provide evidence-based initial pricing guidance, not a guarantee of occupancy or income.",
    ])

    report_path = REPORT_DIR / "model_evaluation_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Final report created: {report_path}")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
