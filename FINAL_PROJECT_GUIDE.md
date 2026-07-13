# Final project run guide

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the complete pipeline:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

To rerun only the final modelling and reporting stages after the processed feature file already exists:

```bash
python run_pipeline.py --from-step modeling
```

## Principal final outputs

- `report/model_evaluation_report.md`
- `output/tables/final_model_evaluation.csv`
- `output/tables/cleaning_row_summary.csv`
- `output/tables/lasso_shap_feature_family_comparison.csv`
- `output/figures/actual_vs_predicted_random_forest.png`
- `output/figures/residuals_vs_predicted_random_forest.png`
- `output/figures/shap_summary_beeswarm.png`
- `output/figures/lasso_feature_family_importance.png`
- `output/figures/lasso_vs_shap_feature_families.png`

The detailed coefficient and SHAP tables are retained for reproducibility but do not need to appear in the main report.
