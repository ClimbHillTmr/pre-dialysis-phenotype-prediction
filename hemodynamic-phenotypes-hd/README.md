# HemoDynamics

**Unsupervised Discovery of Hemodynamic Phenotypes in Hemodialysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HemoDynamics discovers distinct hemodynamic response patterns during hemodialysis sessions using time-series clustering. By analyzing 240-minute systolic blood pressure trajectories, the system identifies clinically meaningful phenotypes that correlate with intradialytic hypotension (IDH) risk.

**Key capabilities:**
- **Phenotype Discovery**: Soft-DTW K-Means clustering identifies 4 distinct hemodynamic trajectory patterns
- **Clinical Validation**: Phenotypes show significant correlation with IDH incidence rates
- **Cross-Center Analysis**: Statistical comparison of phenotype distributions across dialysis centers
- **Large-Scale Processing**: Two-stage clustering handles 200,000+ dialysis sessions efficiently

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
hemodynamic-phenotypes-hd/
├── config/
│   └── settings.py          # Configuration parameters
├── src/
│   ├── main.py              # Main pipeline entry point
│   ├── data/
│   │   └── loader.py        # Data loading and preprocessing
│   ├── clustering/
│   │   └── soft_dtw_kmeans.py  # Soft-DTW K-Means algorithm
│   ├── analysis/
│   │   └── phenotype_stats.py  # Statistical analysis
│   ├── visualization/
│   │   └── plotter.py       # Publication-quality figures
│   └── experiments/
│       └── ablation.py      # Ablation studies
├── results/
│   ├── figures/             # Generated visualizations
│   └── tables/              # Statistical results (CSV)
├── requirements.txt
└── README.md
```

## Methods

### Soft-DTW K-Means Clustering

The system uses a two-stage clustering approach:

1. **Stage 1**: Soft-DTW K-Means on 50,000 stratified samples (50 iterations)
2. **Stage 2**: 1-NN assignment of remaining samples to centroids

**Key design choices:**
- **Input**: ΔSBP trajectories (relative to pre-dialysis baseline)
- **Distance**: Dynamic Time Warping (DTW) via `fastdtw`
- **Parallelization**: Multi-core DTW distance computation via `joblib`

### Clinical IDH Definition

Intradialytic hypotension is defined using the clinical standard:
- **Threshold**: Minimum intradialytic SBP < 90 mmHg
- **Excludes**: Relative drop definitions to avoid circular reasoning

## Output

### Visualizations
- **Mean Trajectories**: Average SBP trajectory per phenotype with confidence bands
- **Center Distribution**: Phenotype prevalence comparison between centers
- **IDH Rates**: Intradialytic hypotension incidence by phenotype

### Statistical Tables
- **Phenotype Statistics**: N, IDH rate, mean SBP metrics per phenotype
- **Ablation Results**: K-value selection and distance metric comparison

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hemoDynamics2026,
  title={Unsupervised Discovery of Hemodynamic Phenotypes in Hemodialysis: A Dual-Center Study},
  author={Your Name},
  journal={Under Review},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
