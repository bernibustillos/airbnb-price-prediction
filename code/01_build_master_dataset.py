"""Build one listing-snapshot master table from all raw Inside Airbnb files.

The builder processes each raw file in chunks.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from config import RAW_DIR, MASTER_FILE, PROCESSED_DIR
from utils import save_csv, save_json

KEEP_COLUMNS = [
    "id", "host_id", "host_since", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_is_superhost", "host_listings_count",
    "host_total_listings_count", "host_has_profile_pic", "host_identity_verified",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms", "bathrooms_text",
    "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
    "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
    "maximum_maximum_nights", "has_availability", "availability_30", "availability_60",
    "availability_90", "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "first_review", "last_review", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "instant_bookable", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms", "reviews_per_month",
]

CHUNK_SIZE = 25_000


def snapshot_month_from_path(path: Path) -> str:
    return path.name.removeprefix("listings_").removesuffix(".csv.gz").replace("_", "-")


def main() -> None:
    files = sorted(RAW_DIR.glob("listings_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")

    # Read headers first and create one stable union schema for every chunk.
    file_columns: dict[Path, list[str]] = {}
    union_columns: list[str] = []
    for path in files:
        header = pd.read_csv(path, nrows=0, compression="gzip")
        header.columns = header.columns.str.strip().str.lower().str.replace(" ", "_")
        available = [column for column in KEEP_COLUMNS if column in header.columns]
        file_columns[path] = available
        for column in available:
            if column not in union_columns:
                union_columns.append(column)

    output_columns = ["snapshot_month", "snapshot_date", "source_file", *union_columns]
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MASTER_FILE.exists():
        MASTER_FILE.unlink()

    summaries: list[dict] = []
    missing_counts = {column: 0 for column in output_columns}
    total_rows = 0
    unique_listing_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    first_write = True

    # gzip.open keeps one valid gzip stream while pandas appends CSV chunks.
    with gzip.open(MASTER_FILE, mode="wt", encoding="utf-8", newline="") as output:
        for path in files:
            snapshot_month = snapshot_month_from_path(path)
            snapshot_date = pd.Timestamp(f"{snapshot_month}-01")
            available = file_columns[path]
            snapshot_rows = 0
            snapshot_ids: set[str] = set()

            for raw in pd.read_csv(
                path,
                usecols=available,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                compression="gzip",
            ):
                raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
                frame = raw.reindex(columns=union_columns)
                frame.insert(0, "source_file", path.name)
                frame.insert(0, "snapshot_date", snapshot_date)
                frame.insert(0, "snapshot_month", snapshot_month)
                frame = frame.reindex(columns=output_columns)

                frame.to_csv(output, index=False, header=first_write)
                first_write = False

                rows = len(frame)
                snapshot_rows += rows
                total_rows += rows
                chunk_missing = frame.isna().sum()
                for column, count in chunk_missing.items():
                    missing_counts[column] += int(count)

                if "id" in frame.columns:
                    ids = frame["id"].dropna().astype(str)
                    snapshot_ids.update(ids)
                    unique_listing_ids.update(ids)
                    for listing_id in ids:
                        pair = (listing_id, snapshot_month)
                        if pair in seen_pairs:
                            duplicate_pairs += 1
                        else:
                            seen_pairs.add(pair)

            summaries.append({
                "snapshot_month": snapshot_month,
                "rows": snapshot_rows,
                "raw_columns": len(pd.read_csv(path, nrows=0, compression="gzip").columns),
                "kept_columns": len(available),
                "unique_listing_ids": len(snapshot_ids),
            })
            print(f"Processed {path.name}: {snapshot_rows:,} rows", flush=True)

    summary = pd.DataFrame(summaries)
    save_csv(summary, PROCESSED_DIR / "snapshot_summary.csv")

    missing = pd.DataFrame({
        "column": output_columns,
        "missing_pct": [
            (missing_counts[column] / total_rows * 100) if total_rows else 0.0
            for column in output_columns
        ],
    }).sort_values("missing_pct", ascending=False)
    save_csv(missing, PROCESSED_DIR / "missing_values.csv")

    save_json({
        "rows": total_rows,
        "columns": len(output_columns),
        "unique_listings": len(unique_listing_ids),
        "snapshots": len(files),
        "first_snapshot": summary["snapshot_month"].min(),
        "last_snapshot": summary["snapshot_month"].max(),
        "duplicate_listing_snapshot_pairs": duplicate_pairs,
        "builder": "chunked streaming CSV",
        "chunk_size": CHUNK_SIZE,
    }, PROCESSED_DIR / "master_overview.json")

    print(
        f"Saved master dataset: {MASTER_FILE} "
        f"({total_rows:,} x {len(output_columns)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
