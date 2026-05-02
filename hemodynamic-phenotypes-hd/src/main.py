"""
Main pipeline for hemodynamic phenotype discovery.
Orchestrates data loading, clustering, analysis, and visualization.
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
    N_CLUSTERS,
    SAMPLE_SIZE,
    RANDOM_STATE,
    MAX_SEQ_LEN,
)
from src.data.loader import parse_shenyi, parse_fuding, extract_idh_label
from src.clustering.soft_dtw_kmeans import (
    soft_dtw_kmeans,
    assign_to_centroids,
    characterize_phenotype,
)
from src.analysis.phenotype_stats import (
    compute_phenotype_statistics,
    compute_center_distribution,
    chi_square_test,
)
from src.visualization.plotter import (
    plot_mean_trajectories,
    plot_center_distribution,
    plot_idh_rate_by_phenotype,
)
from src.experiments.ablation import run_k_ablation, run_distance_ablation


def load_and_parse_data():
    """Load and parse data from both centers."""
    print("\n" + "=" * 80)
    print("Phase 1: Data Loading and Preprocessing")
    print("=" * 80)

    all_sessions = []

    # Load Shenyi data
    print("\n[Shenyi] Loading data...")
    shenyi_files = list(Path(SHENYI_DATA_DIR).glob("*.csv"))
    print(f"  Found {len(shenyi_files)} files")

    for f in shenyi_files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            session = parse_shenyi(row)
            if session:
                all_sessions.append(session)

    print(f"  Parsed {len(all_sessions)} valid sessions")

    # Load Fuding data
    print("\n[Fuding] Loading data...")
    pre_post_df = pd.read_csv(os.path.join(FUDING_DATA_DIR, FUDING_PREPOST_CSV))
    mid_df = pd.read_csv(os.path.join(FUDING_DATA_DIR, FUDING_MID_CSV))

    n_fuding = 0
    for _, row in mid_df.iterrows():
        session = parse_fuding(row, pre_post_df)
        if session:
            all_sessions.append(session)
            n_fuding += 1

    print(f"  Parsed {n_fuding} valid sessions")
    print(f"\nTotal sessions: {len(all_sessions)}")

    return all_sessions


def normalize_trajectories(all_sessions, max_len=MAX_SEQ_LEN):
    """Normalize trajectories using delta-SBP method."""
    print("\n" + "=" * 80)
    print("Phase 2: Trajectory Normalization")
    print("=" * 80)

    normalized = []
    for s in all_sessions:
        pre_sbp = s["time_sbp"][0]
        delta_sbp = [x - pre_sbp for x in s["time_sbp"]]
        if len(delta_sbp) > max_len:
            delta_sbp = delta_sbp[:max_len]
        s["delta_sbp"] = delta_sbp
        normalized.append(s)

    print(f"  Normalized {len(normalized)} trajectories")
    return normalized


def run_clustering(normalized_sessions, sample_size=SAMPLE_SIZE):
    """Run two-stage clustering."""
    print("\n" + "=" * 80)
    print("Phase 3: Two-Stage Clustering")
    print("=" * 80)

    # Sample for clustering
    np.random.seed(RANDOM_STATE)
    sample = np.random.choice(
        normalized_sessions, min(sample_size, len(normalized_sessions)), replace=False
    )
    X_sample = np.array([s["delta_sbp"] for s in sample])

    # Stage 1: Soft-DTW K-Means on sample
    labels_sample, centroids = soft_dtw_kmeans(X_sample)

    # Stage 2: Assign remaining to centroids
    X_all = np.array([s["delta_sbp"] for s in normalized_sessions])
    labels_all = assign_to_centroids(X_all, centroids)

    # Assign phenotype IDs
    for i, s in enumerate(normalized_sessions):
        s["phenotype_id"] = int(labels_all[i])
        s["idh_label"] = extract_idh_label(s)

    print(f"\nFinal phenotype distribution:")
    for pid in range(N_CLUSTERS):
        count = sum(1 for s in normalized_sessions if s["phenotype_id"] == pid)
        print(f"  P{pid}: {count} ({count/len(normalized_sessions)*100:.1f}%)")

    return normalized_sessions, centroids


def run_analysis(labeled_sessions, output_dir="results"):
    """Run comprehensive analysis."""
    print("\n" + "=" * 80)
    print("Phase 4: Analysis and Visualization")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # Phenotype statistics
    print("\n[Stats] Computing phenotype statistics...")
    stats = compute_phenotype_statistics(labeled_sessions)

    # Center distributions
    dist_shenyi = compute_center_distribution(labeled_sessions, "shenyi")
    dist_fuding = compute_center_distribution(labeled_sessions, "fuding")

    # Chi-square test
    n_shenyi = sum(1 for s in labeled_sessions if s["center"] == "shenyi")
    n_fuding = sum(1 for s in labeled_sessions if s["center"] == "fuding")
    chi2, p_value, dof = chi_square_test(dist_shenyi, dist_fuding, n_shenyi, n_fuding)
    print(f"\n  Chi-square test: chi2={chi2:.2f}, p={p_value:.2e}, dof={dof}")

    # Visualizations
    print("\n[Visualization] Generating plots...")
    plot_mean_trajectories(
        labeled_sessions,
        os.path.join(fig_dir, "mean_trajectories.pdf"),
        "Mean SBP Trajectories by Phenotype",
    )
    plot_center_distribution(
        dist_shenyi, dist_fuding, os.path.join(fig_dir, "center_distribution.pdf")
    )

    # IDH rates
    idh_rates = {}
    for pid in range(N_CLUSTERS):
        sessions = [s for s in labeled_sessions if s["phenotype_id"] == pid]
        idh_count = sum(1 for s in sessions if s["idh_label"] == "IDH")
        idh_rates[pid] = idh_count / len(sessions) if sessions else 0
        print(f"  P{pid} IDH rate: {idh_rates[pid]*100:.1f}%")

    plot_idh_rate_by_phenotype(
        idh_rates, os.path.join(fig_dir, "idh_rate_by_phenotype.pdf")
    )

    # Save results table
    results_df = pd.DataFrame(
        [
            {
                "Phenotype": f"P{pid}",
                "N": stats[pid]["n"],
                "IDH Rate (%)": stats[pid]["idh_rate"] * 100,
                "Mean Final SBP": stats[pid]["mean_final_sbp"],
                "Mean Min SBP": stats[pid]["mean_min_sbp"],
                "Mean SBP Drop": stats[pid]["mean_sbp_drop"],
            }
            for pid in stats.keys()
        ]
    )
    results_df.to_csv(os.path.join(table_dir, "phenotype_statistics.csv"), index=False)
    print(f"\n  Saved: {os.path.join(table_dir, 'phenotype_statistics.csv')}")

    # Save phenotype labels for Repository 2
    labels_df = pd.DataFrame(
        [
            {
                "session_id": s["session_id"],
                "center": s["center"],
                "phenotype_id": s["phenotype_id"],
                "idh_label": s["idh_label"],
            }
            for s in labeled_sessions
        ]
    )
    labels_df.to_csv(os.path.join(table_dir, "phenotype_labels.csv"), index=False)
    print(f"  Saved: {os.path.join(table_dir, 'phenotype_labels.csv')}")

    return {
        "chi2": chi2,
        "p_value": p_value,
        "idh_rates": idh_rates,
        "stats": stats,
    }


def run_ablation_studies(normalized_sessions, output_dir="results"):
    """Run ablation studies."""
    print("\n" + "=" * 80)
    print("Phase 5: Ablation Studies")
    print("=" * 80)

    # Sample for ablation
    np.random.seed(RANDOM_STATE)
    sample_size = min(5000, len(normalized_sessions))
    sample = np.random.choice(normalized_sessions, sample_size, replace=False)
    X_sample = np.array([s["delta_sbp"] for s in sample])

    # K-value ablation
    k_results, best_k = run_k_ablation(X_sample)

    # Distance metric ablation
    dist_results = run_distance_ablation(X_sample)

    # Save ablation results
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    k_df = pd.DataFrame(
        [
            {
                "K": k,
                "Silhouette": v["silhouette"],
                "Calinski-Harabasz": v.get("calinski_harabasz", np.nan),
            }
            for k, v in k_results.items()
        ]
    )
    k_df.to_csv(
        os.path.join(output_dir, "tables", "ablation_k_values.csv"), index=False
    )

    return {"best_k": best_k, "k_results": k_results, "dist_results": dist_results}


def main():
    """Main pipeline."""
    print("=" * 80)
    print("Hemodynamic Phenotype Discovery Pipeline")
    print("=" * 80)

    # Load and parse
    all_sessions = load_and_parse_data()

    # Normalize
    normalized = normalize_trajectories(all_sessions)

    # Cluster
    labeled, centroids = run_clustering(normalized)

    # Analyze
    analysis_results = run_analysis(labeled)

    # Ablation
    ablation_results = run_ablation_studies(labeled)

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nResults saved to: results/")
    print(f"  - figures/: Visualization plots")
    print(f"  - tables/: Statistical results")

    return labeled, analysis_results, ablation_results


if __name__ == "__main__":
    main()
