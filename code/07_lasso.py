"""Export detailed Lasso coefficients and concise feature-family importance."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MODELS_DIR, TABLES_DIR, FIGURES_DIR
from utils import save_csv, save_json
from feature_groups import feature_family

model_path = MODELS_DIR / "lasso_pipeline.joblib"
if not model_path.exists():
    raise FileNotFoundError("Run 06_modeling.py before 07_lasso.py.")

pipe = joblib.load(model_path)
pre = pipe.named_steps["preprocessor"]
names = np.asarray(pre.get_feature_names_out(), dtype=str)
coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float)
alpha = float(pipe.named_steps["model"].alpha_)
cat_columns = list(pre.transformers_[1][2])
num_columns = list(pre.transformers_[0][2])


def base_feature_name(encoded_name: str) -> str:
    name = encoded_name.split("__", 1)[-1]
    if name in num_columns:
        return name
    for column in sorted(cat_columns, key=len, reverse=True):
        if name == column or name.startswith(column + "_"):
            return column
    return name


detail = pd.DataFrame({
    "encoded_feature": names,
    "base_feature": [base_feature_name(name) for name in names],
    "coefficient_log_price": coef,
})
detail["abs_coefficient"] = detail["coefficient_log_price"].abs()
detail = detail[detail["coefficient_log_price"] != 0].sort_values("abs_coefficient", ascending=False)
save_csv(detail, TABLES_DIR / "lasso_coefficients.csv")

# Original-feature importance: L2 norm across one-hot levels.
base = (
    detail.groupby("base_feature", as_index=False)
    .agg(
        lasso_l2_importance=("coefficient_log_price", lambda s: float(np.sqrt(np.square(s).sum()))),
        nonzero_encoded_terms=("coefficient_log_price", "size"),
    )
    .sort_values("lasso_l2_importance", ascending=False)
    .reset_index(drop=True)
)
base.insert(0, "rank", np.arange(1, len(base) + 1))
save_csv(base, TABLES_DIR / "lasso_base_feature_importance.csv")

# Business-facing family importance. Summing original-feature L2 magnitudes
# avoids cancellation while preserving the relative linear signal.
base["feature_family"] = base["base_feature"].map(feature_family)
family = (
    base.groupby("feature_family", as_index=False)["lasso_l2_importance"].sum()
    .sort_values("lasso_l2_importance", ascending=False)
    .reset_index(drop=True)
)
family["lasso_importance_pct"] = 100 * family["lasso_l2_importance"] / family["lasso_l2_importance"].sum()
family.insert(0, "rank", np.arange(1, len(family) + 1))
save_csv(family, TABLES_DIR / "lasso_feature_family_importance.csv")

save_json({
    "alpha_selected_by_cv": alpha,
    "nonzero_encoded_features": int(len(detail)),
    "nonzero_base_features": int(len(base)),
    "target": "log1p(price_numeric)",
    "final_report_ranking": "normalized feature-family importance",
}, TABLES_DIR / "lasso_summary.json")

plot_data = family.head(10).sort_values("lasso_importance_pct")
plt.figure(figsize=(9, 6))
plt.barh(plot_data["feature_family"], plot_data["lasso_importance_pct"])
plt.xlabel("Share of aggregated Lasso importance (%)")
plt.title("Lasso: Main Pricing-Driver Families")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lasso_feature_family_importance.png", dpi=300)
plt.close()

print(f"Lasso exports completed; alpha={alpha:.6g}.")
