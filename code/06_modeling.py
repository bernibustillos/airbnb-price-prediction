"""Train Linear Regression, LassoCV, and Random Forest using a latest-snapshot holdout."""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import (
    FEATURE_FILE, TABLES_DIR, MODELS_DIR, RANDOM_STATE,
    USE_FULL_TRAINING, TRAIN_SAMPLE_SIZE, CORRELATION_THRESHOLD,
)
from utils import read_csv, save_csv, save_json

_df = read_csv(FEATURE_FILE)
print(f"Loaded feature data: {_df.shape}", flush=True)
latest = sorted(_df["snapshot_month"].unique())[-1]
train = _df[_df["snapshot_month"] != latest].copy()
test = _df[_df["snapshot_month"] == latest].copy()
full_train_rows = len(train)
if not USE_FULL_TRAINING and TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)

exclude = {"id", "host_id", "snapshot_month", "snapshot_date", "price_numeric"}
X_train = train[[c for c in _df.columns if c not in exclude]]
X_test = test[X_train.columns]
y_train, y_test = train["price_numeric"], test["price_numeric"]
y_train_log = np.log1p(y_train)
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Remove near-duplicate numeric predictors using training data only, avoiding
# leakage from the June holdout. This mainly reduces redundant availability and
# host-count variables and improves coefficient stability.
numeric_for_corr = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
corr = numeric_for_corr.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
correlated_to_drop = [column for column in upper.columns if (upper[column] > CORRELATION_THRESHOLD).any()]
if correlated_to_drop:
    X_train = X_train.drop(columns=correlated_to_drop)
    X_test = X_test.drop(columns=correlated_to_drop)
    num_cols = [c for c in num_cols if c not in correlated_to_drop]

pairs = []
for column in correlated_to_drop:
    related = upper.index[upper[column] > CORRELATION_THRESHOLD].tolist()
    for retained in related:
        pairs.append({
            "dropped_feature": column,
            "correlated_with": retained,
            "absolute_correlation": float(corr.loc[retained, column]),
        })
save_csv(pd.DataFrame(pairs, columns=["dropped_feature", "correlated_with", "absolute_correlation"]),
         TABLES_DIR / "correlation_pruning.csv")

# Constrain log predictions to the central observed training range before returning to the dollar scale.
log_lower, log_upper = np.quantile(y_train_log, [0.001, 0.999])

def inverse_log_prediction(pred_log):
    return np.expm1(np.clip(pred_log, log_lower, log_upper)).clip(0)

results = []
def record(name, pred, seconds):
    results.append({
        "model": name,
        "test_snapshot": latest,
        "train_rows": len(train),
        "test_rows": len(test),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
        "fit_seconds": seconds,
        "full_available_train_rows": full_train_rows,
    })

baseline = np.full(len(y_test), y_train.median())
record("Median baseline", baseline, 0.0)

linear_pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ]), cat_cols),
])

mode = "full historical training set" if USE_FULL_TRAINING else "sampled training set"
print(f"Using {mode}: {len(train):,} rows; test rows: {len(test):,}", flush=True)

linear = Pipeline([
    ("preprocessor", linear_pre),
    ("model", LinearRegression()),
])
t0 = time.time()
linear.fit(X_train, y_train_log)
linear_pred = inverse_log_prediction(linear.predict(X_test))
record("Linear Regression (log target)", linear_pred, time.time() - t0)
print("Linear Regression completed", flush=True)
joblib.dump(linear, MODELS_DIR / "linear_regression_pipeline.joblib")

lasso = Pipeline([
    ("preprocessor", linear_pre),
    ("model", LassoCV(
        alphas=np.logspace(-4, -1, 30),
        cv=5,
        max_iter=10000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )),
])
t0 = time.time()
lasso.fit(X_train, y_train_log)
lasso_pred = inverse_log_prediction(lasso.predict(X_test))
record("LassoCV (log target)", lasso_pred, time.time() - t0)
print("LassoCV completed", flush=True)
joblib.dump(lasso, MODELS_DIR / "lasso_pipeline.joblib")

predictions = pd.DataFrame({
    "id": test["id"].values,
    "snapshot_month": test["snapshot_month"].values,
    "actual_price": y_test.values,
    "median_baseline_prediction": baseline,
    "linear_regression_prediction": linear_pred,
    "lasso_prediction": lasso_pred,
})

tree_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
], verbose_feature_names_out=False)
forest = Pipeline([
    ("preprocessor", tree_pre),
    ("model", RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )),
])
t0 = time.time()
forest.fit(X_train, y_train_log)
forest_pred = inverse_log_prediction(forest.predict(X_test))
record("Random Forest (log target)", forest_pred, time.time() - t0)
print("Random Forest completed", flush=True)
joblib.dump(forest, MODELS_DIR / "random_forest_pipeline.joblib")
predictions["random_forest_prediction"] = forest_pred
save_csv(predictions, TABLES_DIR / "holdout_predictions.csv")

metrics = pd.DataFrame(results).sort_values("MAE")
save_csv(metrics, TABLES_DIR / "model_comparison.csv")
best = metrics.iloc[0]["model"]
save_json({
    "holdout_snapshot": latest,
    "best_model": best,
    "feature_columns": X_train.columns.tolist(),
    "categorical_columns": cat_cols,
    "numeric_columns": num_cols,
    "correlation_threshold": float(CORRELATION_THRESHOLD),
    "correlated_numeric_features_dropped": correlated_to_drop,
    "log_prediction_bounds": [float(log_lower), float(log_upper)],
    "price_prediction_bounds": [float(np.expm1(log_lower)), float(np.expm1(log_upper))],
    "lasso_alpha": float(lasso.named_steps["model"].alpha_),
    "training_price_median": float(y_train.median()),
    "use_full_training": bool(USE_FULL_TRAINING),
    "training_rows_used": int(len(train)),
    "full_available_train_rows": int(full_train_rows),
}, MODELS_DIR / "model_metadata.json")
print(metrics.to_string(index=False))
