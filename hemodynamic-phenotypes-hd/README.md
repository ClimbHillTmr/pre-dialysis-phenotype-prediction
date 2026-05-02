# HD Phenotype Landscape

**Unsupervised Discovery of Hemodynamic Phenotypes in Hemodialysis: A Dual-Center Study of 226,800 Sessions**

## Overview

This repository contains the code and analysis for discovering hemodynamic phenotypes in hemodialysis patients using Soft-DTW K-Means clustering, and analyzing cross-center differences between two dialysis centers.

## Research Questions

1. Can we identify distinct hemodynamic trajectory phenotypes from 240-minute dialysis sessions?
2. Do the phenotype distributions differ significantly between two dialysis centers?
3. What clinical factors (UFR, IDWG) explain the cross-center phenotype distribution differences?

## Key Findings

- **4 phenotypes identified**: P0 (Severe Drop), P1 (Moderate Drop), P2 (Stable), P3 (Mild Drop)
- **Cross-center difference**: Fuding has significantly higher P0 prevalence (22.9%) vs Shenyi (18.9%)
- **Root cause**: Higher prescribed UFR in Fuding center drives P0 phenotype prevalence
- **Clinical validation**: P0 phenotype has highest IDH rate, confirming clinical relevance

## Repository Structure

```
├── src/
│   ├── Project_Landscape_V2.py      # Main analysis pipeline
│   ├── analyze_p0_prevalence.py     # P0 prevalence analysis
│   └── check_p0_prevalence.py       # P0 prevalence checker
├── figures/
│   ├── Figure1_Phenotype_Landscape_Full.png
│   ├── Figure2_Root_Cause_Analysis.png
│   ├── phase2_center_landscape.png
│   └── phenotype_landscape_4panel.png
├── data/
│   └── (processed data files)
├── models/
│   └── (clustering centroids)
└── docs/
    └── Project_Landscape_Report_V2.md
```

## Methods

### Soft-DTW K-Means Clustering

- **Input**: ΔSBP trajectories (240 minutes, 12 time points)
- **Distance metric**: Dynamic Time Warping (DTW)
- **K**: 4 phenotypes
- **Two-stage approach**: 
  1. Cluster 50,000 stratified samples with 50 iterations
  2. Assign remaining 176,800 samples via 1-NN (FastDTW)

### Cross-Center Statistical Testing

- **Chi-square test**: Phenotype distribution differences
- **T-test**: UFR and IDWG differences between centers
- **Effect size**: Cohen's d for continuous variables

## Requirements

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn
- fastdtw
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
