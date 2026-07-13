"""Create interpretable, model-ready features while retaining unencoded values."""
import numpy as np
import pandas as pd
from config import CLEAN_FILE, FEATURE_FILE, TABLES_DIR
from utils import read_csv, save_csv, save_json


def simplify_property_type(value: object) -> str:
    """Collapse detailed Inside Airbnb property labels into stable broad groups."""
    text = str(value).strip().lower()
    if not text or text == "nan" or text == "unknown":
        return "Unknown"
    if "hotel" in text or "boutique" in text or "resort" in text or "hostel" in text:
        return "Hotel or serviced stay"
    if "shared room" in text:
        return "Shared room"
    if "private room" in text or text.startswith("room in"):
        return "Private room"
    if text.startswith("entire "):
        return "Entire home"
    unique_terms = (
        "boat", "camper", "rv", "tiny home", "treehouse", "houseboat",
        "cave", "castle", "barn", "yurt", "tent", "dome", "earthen home",
    )
    if any(term in text for term in unique_terms):
        return "Unique stay"
    return "Other"


_df = read_csv(CLEAN_FILE)
for column in ["snapshot_date", "host_since", "first_review", "last_review"]:
    if column in _df.columns:
        _df[column] = pd.to_datetime(_df[column], errors="coerce")

_df["snapshot_year"] = _df["snapshot_date"].dt.year
_df["snapshot_month_num"] = _df["snapshot_date"].dt.month
_df["snapshot_month_sin"] = np.sin(2 * np.pi * _df["snapshot_month_num"] / 12)
_df["snapshot_month_cos"] = np.cos(2 * np.pi * _df["snapshot_month_num"] / 12)
_df["host_tenure_days"] = (_df["snapshot_date"] - _df["host_since"]).dt.days.clip(lower=0)
_df["days_since_last_review"] = (_df["snapshot_date"] - _df["last_review"]).dt.days.clip(lower=0)
active_years = ((_df["snapshot_date"] - _df["first_review"]).dt.days / 365.25).clip(lower=0.25)
_df["reviews_per_active_year"] = _df["number_of_reviews"] / active_years
_df["availability_ratio"] = _df["availability_365"] / 365
safe_guests = _df["accommodates"].replace(0, np.nan)
_df["bedrooms_per_guest"] = _df["bedrooms"] / safe_guests
_df["beds_per_guest"] = _df["beds"] / safe_guests
_df["bathrooms_per_guest"] = _df["bathrooms"] / safe_guests
_df["is_entire_home"] = (_df["room_type"] == "Entire home/apt").astype(int)
_df["is_professional_host"] = (_df["calculated_host_listings_count"].fillna(1) >= 5).astype(int)
_df["property_type_grouped"] = _df["property_type"].map(simplify_property_type)

#  Location is represented by borough plus latitude and longitude.
categorical = [
    "neighbourhood_group_cleansed",
    "room_type",
    "property_type_grouped",
    "host_response_time",
]
numeric = [
    "latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "maximum_nights", "availability_30", "availability_60",
    "availability_90", "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "reviews_per_month", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "host_response_rate", "host_acceptance_rate", "host_listings_count",
    "host_total_listings_count", "calculated_host_listings_count", "host_tenure_days",
    "days_since_last_review", "reviews_per_active_year", "availability_ratio",
    "bedrooms_per_guest", "beds_per_guest", "bathrooms_per_guest",
    "snapshot_month_sin", "snapshot_month_cos", "host_is_superhost",
    "host_identity_verified", "instant_bookable", "is_entire_home", "is_professional_host",
]
identifier = ["id", "host_id", "snapshot_month", "snapshot_date"]
columns = [c for c in identifier + ["price_numeric"] + categorical + numeric if c in _df.columns]
model_df = _df[columns].copy()
for column in categorical:
    if column in model_df:
        model_df[column] = model_df[column].fillna("Unknown").astype(str)

save_csv(model_df, FEATURE_FILE)
save_json({
    "categorical_features": [c for c in categorical if c in model_df],
    "numeric_features": [c for c in numeric if c in model_df],
    "excluded_high_cardinality_features": ["neighbourhood_cleansed", "property_type"],
    "property_type_replacement": "property_type_grouped",
    "location_representation": ["neighbourhood_group_cleansed", "latitude", "longitude"],
    "rows": len(model_df),
    "columns": model_df.shape[1],
}, TABLES_DIR / "feature_manifest.json")
print(f"Saved engineered features: {FEATURE_FILE} ({model_df.shape[0]:,} x {model_df.shape[1]})")
