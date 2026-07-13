"""Run the complete Airbnb price-prediction pipeline or selected stages."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = {
    "build": "01_build_master_dataset.py",
    "audit": "02_data_audit.py",
    "clean": "03_clean_data.py",
    "features": "04_feature_engineering.py",
    "eda": "05_eda.py",
    "modeling": "06_modeling.py",
    "lasso": "07_lasso.py",
    "shap": "08_shap_analysis.py",
    "report": "09_generate_report.py",
}

def main():
    parser = argparse.ArgumentParser(description="NYC Airbnb price-prediction pipeline")
    parser.add_argument("--step", choices=STEPS, help="Run only one stage")
    parser.add_argument("--from-step", choices=STEPS, help="Run from this stage through final report generation")
    args = parser.parse_args()
    names = list(STEPS)
    selected = [args.step] if args.step else names[names.index(args.from_step):] if args.from_step else names
    for name in selected:
        script = ROOT / "code" / STEPS[name]
        print(f"\n{'='*70}\nRunning {name}: {script.name}\n{'='*70}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
