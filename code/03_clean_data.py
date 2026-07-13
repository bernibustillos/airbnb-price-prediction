"""Clean the target and essential structured fields using transparent rules."""
import pandas as pd
from config import MASTER_FILE, CLEAN_FILE, TABLES_DIR
from utils import read_csv, save_csv, save_json, parse_price, parse_percent

_df = read_csv(MASTER_FILE)
initial_rows = len(_df)
_df["price_numeric"] = parse_price(_df["price"])

for column in ["host_response_rate", "host_acceptance_rate"]:
    if column in _df:
        _df[column] = parse_percent(_df[column])

for column in [
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "has_availability", "instant_bookable",
]:
    if column in _df:
        _df[column] = (
            _df[column].astype("string").str.lower()
            .map({"t": 1, "f": 0, "true": 1, "false": 0})
        )

for column in ["snapshot_date", "host_since", "first_review", "last_review"]:
    if column in _df:
        _df[column] = pd.to_datetime(_df[column], errors="coerce")

# Recover a numeric bathroom count from bathrooms_text where necessary.
if "bathrooms_text" in _df:
    parsed = pd.to_numeric(
        _df["bathrooms_text"].astype("string")
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0],
        errors="coerce",
    )
    _df["bathrooms"] = pd.to_numeric(_df.get("bathrooms"), errors="coerce").fillna(parsed)

removal_steps = []

def record_step(rule: str, before: int, after: int) -> None:
    removal_steps.append({"rule": rule, "rows_removed": int(before - after), "rows_remaining": int(after)})

before = len(_df)
_df = _df.drop_duplicates(subset=["id", "snapshot_month"], keep="last")
record_step("Duplicate listing-snapshot rows", before, len(_df))

before = len(_df)
_df = _df[_df["price_numeric"].notna() & _df["price_numeric"].gt(0)].copy()
record_step("Missing or non-positive price", before, len(_df))

# Remove only the extreme upper tail; the log target handles remaining skew.
upper_price = float(_df["price_numeric"].quantile(0.995))
before = len(_df)
_df = _df[_df["price_numeric"].le(upper_price)].copy()
record_step("Price above the 99.5th percentile", before, len(_df))

# Minimal plausibility checks for core capacity and NYC coordinates.
before = len(_df)
_df = _df[_df["accommodates"].fillna(0).between(1, 20)]
record_step("Invalid accommodates value", before, len(_df))

before = len(_df)
_df = _df[
    _df["latitude"].between(40.45, 41.05)
    & _df["longitude"].between(-74.30, -73.65)
]
record_step("Coordinates outside NYC plausibility bounds", before, len(_df))

save_csv(_df, CLEAN_FILE)
summary = {
    "initial_rows": int(initial_rows),
    "clean_rows": int(len(_df)),
    "rows_removed": int(initial_rows - len(_df)),
    "price_upper_99_5_percentile": upper_price,
    "cleaning_rules": removal_steps,
    "note": "Model-specific missing-value imputation is performed inside the training pipelines.",
}
save_json(summary, TABLES_DIR / "cleaning_summary.json")
save_csv(pd.DataFrame(removal_steps), TABLES_DIR / "cleaning_row_summary.csv")
print(f"Saved cleaned data: {CLEAN_FILE} ({len(_df):,} rows)")
