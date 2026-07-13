"""Generate report-ready descriptive tables and figures."""
import pandas as pd
import matplotlib.pyplot as plt
from config import FEATURE_FILE, FIGURES_DIR, TABLES_DIR
from utils import read_csv, save_csv

_df = read_csv(FEATURE_FILE)
monthly = _df.groupby("snapshot_month").agg(listings=("id", "size"), median_price=("price_numeric", "median"), mean_price=("price_numeric", "mean")).reset_index()
save_csv(monthly, TABLES_DIR / "eda_monthly_price.csv")

borough = _df.groupby("neighbourhood_group_cleansed").agg(listings=("id", "size"), median_price=("price_numeric", "median"), mean_price=("price_numeric", "mean")).sort_values("median_price", ascending=False).reset_index()
save_csv(borough, TABLES_DIR / "eda_borough_price.csv")

room = _df.groupby("room_type").agg(listings=("id", "size"), median_price=("price_numeric", "median"), mean_price=("price_numeric", "mean")).sort_values("median_price", ascending=False).reset_index()
save_csv(room, TABLES_DIR / "eda_room_type_price.csv")

plt.figure(figsize=(9, 5)); plt.plot(monthly["snapshot_month"], monthly["median_price"], marker="o")
plt.xticks(rotation=45, ha="right"); plt.ylabel("Median nightly price (USD)"); plt.xlabel("Snapshot month"); plt.title("Median NYC Airbnb price by snapshot")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "monthly_median_price.png", dpi=200); plt.close()

plt.figure(figsize=(8, 5)); plt.bar(borough["neighbourhood_group_cleansed"], borough["median_price"])
plt.xticks(rotation=30, ha="right"); plt.ylabel("Median nightly price (USD)"); plt.title("Median price by borough")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "borough_median_price.png", dpi=200); plt.close()

plt.figure(figsize=(8, 5)); plt.bar(room["room_type"], room["median_price"])
plt.xticks(rotation=25, ha="right"); plt.ylabel("Median nightly price (USD)"); plt.title("Median price by room type")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "room_type_median_price.png", dpi=200); plt.close()
print("EDA outputs completed.")
