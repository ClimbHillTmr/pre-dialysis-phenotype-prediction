# Pre-Dialysis Phenotype Prediction

## Overview

This repository contains the prediction pipeline for identifying hemodynamic phenotypes in hemodialysis patients **before dialysis begins**. The system uses pre-dialysis clinical features to predict which trajectory phenotype a patient will follow during their dialysis session.

## Project Structure

```
pre-dialysis-phenotype-prediction/
├── config/
│   └── settings.py          # Configuration parameters
├── src/
│   ├── data/
│   │   └── loader.py        # Data loading and feature extraction
│   ├── model/
│   │   └── trainer.py       # Model training and evaluation
│   ├── experiments/
│   │   └── ablation.py      # Ablation studies
│   └── main.py              # Main pipeline
├── results/
│   ├── figures/             # Generated visualizations
│   └── tables/              # Statistical results
├── requirements.txt         # Dependencies
└── README.md
```

## Key Features

- **Center-Specific Modeling**: Separate LightGBM models trained for each center (Shenyi, Fuding)
- **Comprehensive Evaluation**: AUROC, AUPRC, calibration curves, decision curve analysis
- **Ablation Studies**: Feature importance analysis and model comparison
- **Cross-Center Validation**: Generalizability assessment across centers
- **SHAP Interpretability**: Feature attribution analysis for clinical insights

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Dependencies

- Python 3.8+
- lightgbm
- scikit-learn
- shap
- matplotlib
- numpy
- pandas

## Citation

If you use this code in your research, please cite our paper (to be added).

## License

MIT License
