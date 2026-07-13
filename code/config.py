from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIT_DIR = OUTPUT_DIR / "audit"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42
TARGET = "price_numeric"
MASTER_FILE = PROCESSED_DIR / "master_listings.csv.gz"
CLEAN_FILE = PROCESSED_DIR / "cleaned_listings.csv.gz"
FEATURE_FILE = PROCESSED_DIR / "model_features.csv.gz"

# Final-project setting: use all historical rows before the latest snapshot.
# For quicker development runs, set USE_FULL_TRAINING=False and choose a cap.
USE_FULL_TRAINING = True
TRAIN_SAMPLE_SIZE = 10_000

# Numeric features above this absolute training-set correlation are pruned.
CORRELATION_THRESHOLD = 0.95

for directory in [PROCESSED_DIR, AUDIT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
