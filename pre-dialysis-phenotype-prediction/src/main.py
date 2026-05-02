"""
Main pipeline for pre-dialysis phenotype prediction.
Orchestrates data loading, model training, evaluation, and ablation studies.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SHENYI_DATA_DIR,
    FUDING_DATA_DIR,
    FUDING_PREPOST_CSV,
    FUDING_MID_CSV,
)
from src.data.loader import (
    parse_shenyi_features,
    parse_fuding_features,
    extract_features,
)
from src.model.trainer import (
    train_lightgbm,
    compute_metrics,
    plot_feature_importance,
    plot_calibration_curve,
)
from src.experiments.ablation import (
    feature_ablation,
    model_comparison,
    cross_center_validation,
)


def load_and_parse_data():
    """Load and parse data from both centers."""
    print("\n" + "=" * 80)
    print("Phase 1: Data Loading and Feature Extraction")
    print("=" * 80)

    all_sessions = []

    # Load Shenyi data
    print("\n[Shenyi] Loading data...")
    shenyi_files = list(Path(SHENYI_DATA_DIR).glob("*.csv"))
    print(f"  Found {len(shenyi_files)} files")

    for f in shenyi_files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            session = parse_shenyi_features(row)
            if session:
                all_sessions.append(session)

    print(f"  Parsed {len(all_sessions)} valid sessions")
    n_shenyi = len(all_sessions)

    # Load Fuding data
    print("\n[Fuding] Loading data...")
    pre_post_df = pd.read_csv(os.path.join(FUDING_DATA_DIR, FUDING_PREPOST_CSV))
    mid_df = pd.read_csv(os.path.join(FUDING_DATA_DIR, FUDING_MID_CSV))

    n_fuding = 0
    for _, row in mid_df.iterrows():
        session = parse_fuding_features(row, pre_post_df)
        if session:
            all_sessions.append(session)
            n_fuding += 1

    print(f"  Parsed {n_fuding} valid sessions")
    print(f"\nTotal sessions: {len(all_sessions)}")

    return all_sessions


def train_center_models(all_sessions, phenotype_labels):
    """Train separate models for each center."""
    print("\n" + "=" * 80)
    print("Phase 2: Center-Specific Model Training")
    print("=" * 80)

    results = {}

    for center_name in ["shenyi", "fuding"]:
        center_sessions = [s for s in all_sessions if s["center"] == center_name]
        X, y, feature_names = extract_features(center_sessions, phenotype_labels)

        # Filter out sessions without labels
        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 100:
            print(
                f"\n  {center_name}: Too few labeled samples ({len(X_valid)}), skipping"
            )
            continue

        model, X_train, X_test, y_train, y_test, y_prob = train_lightgbm(
            X_valid, y_valid, center_name=center_name.title()
        )

        metrics = compute_metrics(y_test, y_prob, model.predict(X_test))

        results[center_name] = {
            "model": model,
            "feature_names": feature_names,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_prob": y_prob,
            "metrics": metrics,
        }

        print(f"\n  {center_name} AUROC: {metrics['auroc_macro']:.4f}")
        print(f"  {center_name} AUPRC: {metrics['auprc_macro']:.4f}")

    return results


def run_ablation_studies(all_sessions, phenotype_labels, output_dir="results"):
    """Run ablation studies."""
    print("\n" + "=" * 80)
    print("Phase 3: Ablation Studies")
    print("=" * 80)

    ablation_results = {}

    for center_name in ["shenyi", "fuding"]:
        center_sessions = [s for s in all_sessions if s["center"] == center_name]
        X, y, feature_names = extract_features(center_sessions, phenotype_labels)

        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 100:
            continue

        # Feature ablation
        feat_results = feature_ablation(
            X_valid, y_valid, feature_names, center_name.title()
        )

        # Model comparison
        model_results = model_comparison(X_valid, y_valid, center_name.title())

        ablation_results[center_name] = {
            "feature_ablation": feat_results,
            "model_comparison": model_results,
        }

    # Save ablation results
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    for center_name, results in ablation_results.items():
        # Feature ablation
        feat_df = pd.DataFrame(
            [
                {
                    "Feature": k,
                    "AUROC": v.get("auroc", v),
                    "Drop": v.get("drop", np.nan),
                }
                for k, v in results["feature_ablation"].items()
            ]
        )
        feat_df.to_csv(
            os.path.join(output_dir, "tables", f"ablation_features_{center_name}.csv"),
            index=False,
        )

        # Model comparison
        model_df = pd.DataFrame(
            [
                {"Model": k, "AUROC": v["auroc"]}
                for k, v in results["model_comparison"].items()
            ]
        )
        model_df.to_csv(
            os.path.join(output_dir, "tables", f"ablation_models_{center_name}.csv"),
            index=False,
        )

    return ablation_results


def run_cross_center_evaluation(results, all_sessions, phenotype_labels):
    """Run cross-center validation."""
    print("\n" + "=" * 80)
    print("Phase 4: Cross-Center Validation")
    print("=" * 80)

    cross_results = {}

    if "shenyi" in results and "fuding" in results:
        X_shenyi, y_shenyi, _ = extract_features(
            [s for s in all_sessions if s["center"] == "shenyi"], phenotype_labels
        )
        X_fuding, y_fuding, _ = extract_features(
            [s for s in all_sessions if s["center"] == "fuding"], phenotype_labels
        )

        valid_shenyi = y_shenyi >= 0
        valid_fuding = y_fuding >= 0

        # Shenyi -> Fuding
        cross_results["shenyi_to_fuding"] = cross_center_validation(
            X_shenyi[valid_shenyi],
            y_shenyi[valid_shenyi],
            X_fuding[valid_fuding],
            y_fuding[valid_fuding],
            "Shenyi",
            "Fuding",
        )

        # Fuding -> Shenyi
        cross_results["fuding_to_shenyi"] = cross_center_validation(
            X_fuding[valid_fuding],
            y_fuding[valid_fuding],
            X_shenyi[valid_shenyi],
            y_shenyi[valid_shenyi],
            "Fuding",
            "Shenyi",
        )

    return cross_results


def save_results(results, ablation_results, cross_results, output_dir="results"):
    """Save all results to files."""
    print("\n" + "=" * 80)
    print("Phase 5: Saving Results")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    for center_name, res in results.items():
        # Save metrics
        metrics_df = pd.DataFrame(
            {
                "Metric": ["AUROC (macro)", "AUPRC (macro)", "Accuracy"],
                "Value": [
                    res["metrics"]["auroc_macro"],
                    res["metrics"]["auprc_macro"],
                    res["metrics"]["accuracy"],
                ],
            }
        )
        metrics_df.to_csv(
            os.path.join(table_dir, f"metrics_{center_name}.csv"), index=False
        )

        # Plot feature importance
        plot_feature_importance(
            res["model"],
            res["feature_names"],
            os.path.join(fig_dir, f"feature_importance_{center_name}.pdf"),
        )

        # Plot calibration curve
        plot_calibration_curve(
            res["y_test"],
            res["y_prob"],
            output_path=os.path.join(fig_dir, f"calibration_{center_name}.pdf"),
        )

    # Save cross-center results
    if cross_results:
        cross_df = pd.DataFrame(
            [{"Direction": k, "AUROC": v["auroc"]} for k, v in cross_results.items()]
        )
        cross_df.to_csv(
            os.path.join(table_dir, "cross_center_validation.csv"), index=False
        )

    print(f"\n  Results saved to: {output_dir}/")
    print(f"    - figures/: Visualization plots")
    print(f"    - tables/: Statistical results")


def load_phenotype_labels():
    """Load phenotype labels from Repository 1 output."""
    import os
    from pathlib import Path

    label_file = Path(__file__).parent.parent / "data" / "phenotype_labels.csv"

    if not label_file.exists():
        print(f"\n  Warning: Phenotype labels file not found at {label_file}")
        print("  Please run Repository 1 (HemoDynamics) first to generate labels.")
        return {}

    labels_df = pd.read_csv(label_file)
    labels = {}
    for _, row in labels_df.iterrows():
        labels[row["session_id"]] = int(row["phenotype_id"])

    print(f"  Loaded {len(labels)} phenotype labels")
    return labels


def main():
    """Main pipeline."""
    print("=" * 80)
    print("Pre-Dialysis Phenotype Prediction Pipeline")
    print("=" * 80)

    # Load and parse
    all_sessions = load_and_parse_data()

    # Load phenotype labels from Repository 1
    phenotype_labels = load_phenotype_labels()

    if not phenotype_labels:
        print("\n  ERROR: No phenotype labels available. Exiting.")
        print("  Please run Repository 1 (HemoDynamics) first.")
        sys.exit(1)

    # Train center-specific models
    results = train_center_models(all_sessions, phenotype_labels)

    # Run ablation studies
    ablation_results = run_ablation_studies(all_sessions, phenotype_labels)

    # Cross-center validation
    cross_results = run_cross_center_evaluation(results, all_sessions, phenotype_labels)

    # Save all results
    save_results(results, ablation_results, cross_results)

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nResults saved to: results/")
    print(f"  - figures/: Visualization plots")
    print(f"  - tables/: Statistical results")


if __name__ == "__main__":
    main()
