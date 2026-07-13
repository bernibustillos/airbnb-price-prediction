"""Shared business-facing feature-family mappings for model interpretation."""

FEATURE_FAMILIES = {
    # Location
    "neighbourhood_group_cleansed": "Location",
    "latitude": "Location",
    "longitude": "Location",
    # Listing type
    "property_type_grouped": "Property Type",
    "room_type": "Room Type",
    "is_entire_home": "Room Type",
    # Capacity and physical attributes
    "accommodates": "Capacity & Space",
    "bathrooms": "Capacity & Space",
    "bedrooms": "Capacity & Space",
    "beds": "Capacity & Space",
    "bathrooms_per_guest": "Capacity & Space",
    "bedrooms_per_guest": "Capacity & Space",
    "beds_per_guest": "Capacity & Space",
    # Stay rules
    "minimum_nights": "Booking Constraints",
    "maximum_nights": "Booking Constraints",
    # Availability
    "availability_30": "Availability",
    "availability_60": "Availability",
    "availability_90": "Availability",
    "availability_365": "Availability",
    "availability_ratio": "Availability",
    # Review score dimensions
    "review_scores_rating": "Review Scores",
    "review_scores_accuracy": "Review Scores",
    "review_scores_cleanliness": "Review Scores",
    "review_scores_checkin": "Review Scores",
    "review_scores_communication": "Review Scores",
    "review_scores_location": "Review Scores",
    "review_scores_value": "Review Scores",
    # Review activity
    "number_of_reviews": "Review Activity",
    "number_of_reviews_ltm": "Review Activity",
    "number_of_reviews_l30d": "Review Activity",
    "reviews_per_month": "Review Activity",
    "reviews_per_active_year": "Review Activity",
    "days_since_last_review": "Review Activity",
    # Host attributes
    "host_response_time": "Host Characteristics",
    "host_response_rate": "Host Characteristics",
    "host_acceptance_rate": "Host Characteristics",
    "host_listings_count": "Host Characteristics",
    "host_total_listings_count": "Host Characteristics",
    "calculated_host_listings_count": "Host Characteristics",
    "host_tenure_days": "Host Characteristics",
    "host_is_superhost": "Host Characteristics",
    "host_identity_verified": "Host Characteristics",
    "is_professional_host": "Host Characteristics",
    # Booking and time controls
    "instant_bookable": "Booking Settings",
    "snapshot_month_sin": "Seasonality",
    "snapshot_month_cos": "Seasonality",
}


def feature_family(feature_name: str) -> str:
    """Return a reader-friendly family for an original model feature."""
    return FEATURE_FAMILIES.get(feature_name, "Other")
