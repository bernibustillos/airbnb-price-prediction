"""Generate concise SHAP explanations for the Random Forest model."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from config import FEATURE_FILE, MODELS_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_STATE
from utils import read_csv, save_csv
from feature_groups import feature_family

SAMPLE_SIZE = 500

_df = read_csv(FEATURE_FILE)
latest = sorted(_df["snapshot_month"].unique())[-1]
test = _df[_df["snapshot_month"] == latest].copy()
exclude = {"id", "host_id", "snapshot_month", "snapshot_date", "price_numeric"}
X = test[[c for c in _df.columns if c not in exclude]]
if len(X) > SAMPLE_SIZE:
    X = X.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

model_path = MODELS_DIR / "random_forest_pipeline.joblib"
if not model_path.exists():
    raise FileNotFoundError("Run 06_modeling.py before 08_shap_analysis.py.")

pipe = joblib.load(model_path)
pre = pipe.named_steps["preprocessor"]
model = pipe.named_steps["model"]
Xt = pre.transform(X)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
feature_names = np.asarray(pre.get_feature_names_out(), dtype=str)

explainer = shap.TreeExplainer(model)
shap_values = np.asarray(explainer.shap_values(Xt, check_additivity=False))
if shap_values.ndim == 3:
    shap_values = shap_values[..., 0]

importance = np.abs(shap_values).mean(axis=0)
out = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap_log_price": importance,
}).sort_values("mean_abs_shap_log_price", ascending=False)
save_csv(out, TABLES_DIR / "shap_feature_importance.csv")

out["feature_family"] = out["feature"].map(feature_family)
family = (
    out.groupby("feature_family", as_index=False)["mean_abs_shap_log_price"].sum()
    .sort_values("mean_abs_shap_log_price", ascending=False)
    .reset_index(drop=True)
)
family["shap_importance_pct"] = 100 * family["mean_abs_shap_log_price"] / family["mean_abs_shap_log_price"].sum()
family.insert(0, "rank", np.arange(1, len(family) + 1))
save_csv(family, TABLES_DIR / "shap_feature_family_importance.csv")

# Keep one detailed SHAP figure to show effect direction and dispersion.
shap.summary_plot(shap_values, Xt, feature_names=feature_names, max_display=15, show=False)
plt.title("Random Forest SHAP Summary")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"SHAP analysis completed using {len(X):,} holdout observations.")
