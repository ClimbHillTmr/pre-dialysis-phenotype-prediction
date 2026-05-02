# HemoPredict

**Pre-Dialysis Prediction of Hemodynamic Phenotypes Using Machine Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HemoPredict predicts a patient's hemodynamic phenotype **before dialysis begins** using pre-dialysis clinical features. By training center-specific LightGBM models, the system enables proactive risk stratification and personalized treatment planning.

**Key capabilities:**
- **Pre-Dialysis Prediction**: Multi-class classification of hemodynamic phenotypes using pre-dialysis features
- **Center-Specific Models**: Separate models trained for each dialysis center to account for practice variations
- **Cross-Center Validation**: Assessment of model generalizability across centers
- **Interpretable AI**: SHAP analysis for feature attribution and clinical insights

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py
```

Results will be saved to `results/figures/` and `results/tables/`.

## Project Structure

```
pre-dialysis-phenotype-prediction/
├── config/
│   └── settings.py          # Configuration parameters
├── src/
│   ├── main.py              # Main pipeline entry point
│   ├── data/
│   │   └── loader.py        # Feature extraction
│   ├── model/
│   │   └── trainer.py       # Model training and evaluation
│   └── experiments/
│       └── ablation.py      # Ablation studies
├── results/
│   ├── figures/             # Generated visualizations
│   └── tables/              # Statistical results (CSV)
├── requirements.txt
└── README.md
```

## Methods

### Feature Engineering

**Pre-dialysis features:**
- **Demographics**: Age, sex
- **Baseline Vitals**: Pre-dialysis SBP, DBP
- **Treatment Parameters**: Prescribed UFR, dialysis duration
- **Fluid Status**: IDWG (interdialytic weight gain), dry weight, pre-dialysis weight

### Model Training

- **Algorithm**: LightGBM with class weight balancing
- **Validation**: Train/test split with stratification (80/20)
- **Missing Values**: KNN imputation (k=5)
- **Hyperparameters**: n_estimators=200, learning_rate=0.05, max_depth=6

### Evaluation Metrics

- **AUROC**: One-vs-rest macro average for multi-class classification
- **AUPRC**: Average precision for imbalanced classes
- **Calibration**: Calibration curves for probability reliability
- **Decision Curve**: Clinical utility assessment

## Output

### Visualizations
- **Feature Importance**: Top predictive factors for phenotype assignment
- **Calibration Curves**: Model probability calibration by class
- **SHAP Summary**: Feature attribution analysis

### Statistical Tables
- **Performance Metrics**: AUROC, AUPRC, accuracy per center
- **Ablation Results**: Feature importance and model comparison
- **Cross-Center Validation**: Generalizability assessment

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hemoPredict2026,
  title={Pre-Dialysis Prediction of Hemodynamic Phenotypes in Hemodialysis: A Machine Learning Approach},
  author={Your Name},
  journal={Under Review},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
