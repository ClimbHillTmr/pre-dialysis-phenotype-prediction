# HD Phenotype Prediction

**Pre-Dialysis Prediction of Hemodynamic Phenotypes Using Machine Learning: A Dual-Center Validation Study**

## Overview

This repository contains the code and analysis for predicting hemodynamic phenotypes before dialysis sessions using LightGBM with SHAP interpretability, validated across two independent dialysis centers.

## Research Questions

1. Can we predict which hemodynamic phenotype a patient will belong to using only pre-dialysis features?
2. How well does the model generalize across two independent dialysis centers?
3. What features drive the prediction of high-risk phenotypes (P0)?

## Key Findings

- **Model performance**: P0 AUROC 0.858 (Shenyi), 0.839 (Fuding)
- **AUROC vs AUPRC discrepancy**: Explained by P0 phenotype prevalence differences (18.9% vs 30.0%)
- **Top predictors**: Prescribed UFR, pre-dialysis SBP, age
- **Clinical utility**: 8 pre-dialysis features sufficient for risk stratification

## Repository Structure

```
├── src/
│   ├── Project_Landscape_V2.py      # Main prediction pipeline
│   └── analyze_auroc_auprc.py       # AUROC/AUPRC discrepancy analysis
├── figures/
│   ├── Figure3_Phenotype_Prediction_Full.png
│   ├── Figure3_ROC_Comparison.png
│   ├── Figure3_SHAP_Comparison.png
│   ├── Figure3_SHAP_Summary_P0.png
│   ├── Figure3_SHAP_Summary_P0_Fuding.png
│   └── Figure3_SHAP_UFR_Special.png
├── data/
│   └── (processed data files)
├── models/
│   └── (trained LightGBM models)
└── docs/
    └── (additional documentation)
```

## Methods

### Feature Engineering

- **8 pre-dialysis features**: age, pre_weight, dry_weight, pre_sbp, pre_dbp, prescribed_ufr, idwg, sex
- **Missing value handling**: KNN imputation (k=5)
- **Feature space unification**: Inferred pre_sbp/pre_dbp from time_sbp[0]/time_dbp[0] for Fuding center

### LightGBM Multi-Class Classification

- **Task**: Predict phenotype (P0, P1, P2, P3) from pre-dialysis features
- **Training**: Separate models for each center (no data leakage)
- **Evaluation**: AUROC, AUPRC per class, overall accuracy

### SHAP Interpretability

- **Method**: SHAP Summary Plot for P0 class
- **Features**: Direction and magnitude of feature contributions
- **Cross-center comparison**: SHAP patterns between Shenyi and Fuding

## Requirements

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn
- lightgbm
- shap
- matplotlib, seaborn

## Usage

```bash
cd src
python Project_Landscape_V2.py
```

## Citation

If you use this code, please cite:

> [Your paper citation here]

## License

[License type]
