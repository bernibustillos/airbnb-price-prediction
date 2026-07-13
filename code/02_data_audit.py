"""Create non-destructive data-quality and target-variable audit outputs."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MASTER_FILE, AUDIT_DIR, FIGURES_DIR
from utils import read_csv, save_csv, save_json, parse_price

_df = read_csv(MASTER_FILE)
_df["price_numeric"] = parse_price(_df["price"])

summary = {
    "rows": len(_df), "columns": _df.shape[1], "unique_listings": int(_df["id"].nunique()),
    "snapshots": int(_df["snapshot_month"].nunique()),
    "duplicate_rows": int(_df.duplicated().sum()),
    "duplicate_listing_snapshot_pairs": int(_df.duplicated(["id", "snapshot_month"]).sum()),
    "missing_price": int(_df["price_numeric"].isna().sum()),
    "nonpositive_price": int((_df["price_numeric"] <= 0).sum()),
}
save_json(summary, AUDIT_DIR / "dataset_summary.json")

missing = pd.DataFrame({
    "column": _df.columns,
    "dtype": _df.dtypes.astype(str).values,
    "missing_count": _df.isna().sum().values,
    "missing_pct": (_df.isna().mean() * 100).round(2).values,
    "unique_values": [_df[c].nunique(dropna=True) for c in _df.columns],
}).sort_values("missing_pct", ascending=False)
save_csv(missing, AUDIT_DIR / "column_audit.csv")

numeric = _df.select_dtypes(include=np.number)
numeric_summary = numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
numeric_summary["skew"] = numeric.skew(numeric_only=True)
save_csv(numeric_summary.reset_index().rename(columns={"index": "column"}), AUDIT_DIR / "numeric_summary.csv")

price_summary = _df["price_numeric"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99, .995]).to_frame("value")
save_csv(price_summary.reset_index().rename(columns={"index": "statistic"}), AUDIT_DIR / "price_summary.csv")

monthly = _df.groupby("snapshot_month").agg(
    rows=("id", "size"), unique_listings=("id", "nunique"),
    valid_prices=("price_numeric", "count"), mean_price=("price_numeric", "mean"),
    median_price=("price_numeric", "median")
).reset_index()
save_csv(monthly, AUDIT_DIR / "monthly_summary.csv")

valid_price = _df.loc[_df["price_numeric"].gt(0), "price_numeric"].dropna()
plt.figure(figsize=(8, 5)); plt.hist(valid_price.clip(upper=valid_price.quantile(.99)), bins=60)
plt.xlabel("Nightly price (USD; capped at 99th percentile for display)"); plt.ylabel("Listings")
plt.title("NYC Airbnb price distribution"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "audit_price_distribution.png", dpi=200); plt.close()

plt.figure(figsize=(8, 5)); plt.hist(np.log1p(valid_price), bins=60)
plt.xlabel("log(1 + nightly price)"); plt.ylabel("Listings"); plt.title("Log-transformed price distribution")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "audit_log_price_distribution.png", dpi=200); plt.close()
print("Data audit completed.")
