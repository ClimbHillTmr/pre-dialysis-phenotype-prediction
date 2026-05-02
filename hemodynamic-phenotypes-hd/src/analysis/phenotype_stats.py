"""Analysis functions for hemodynamic phenotype characterization."""

import numpy as np
import pandas as pd
from scipy import stats


def compute_phenotype_statistics(labeled_sessions):
    """Compute comprehensive statistics for each phenotype."""
    phenotypes = {}
    for s in labeled_sessions:
        pid = s["phenotype_id"]
        if pid not in phenotypes:
            phenotypes[pid] = {"sbp_trajectories": [], "idh_count": 0, "total": 0}
        phenotypes[pid]["sbp_trajectories"].append(s["time_sbp"])
        phenotypes[pid]["total"] += 1
        if s.get("idh_label") == "IDH":
            phenotypes[pid]["idh_count"] += 1

    results = {}
    for pid, data in phenotypes.items():
        n = data["total"]
        idh_rate = data["idh_count"] / n if n > 0 else 0
        trajectories = data["sbp_trajectories"]

        # Compute mean trajectory
        max_len = max(len(t) for t in trajectories)
        padded = [
            np.pad(t, (0, max_len - len(t)), constant_values=np.nan)
            for t in trajectories
        ]
        traj_array = np.array(padded)
        mean_traj = np.nanmean(traj_array, axis=0)

        # Compute statistics
        final_sbp = np.nanmean([t[-1] for t in trajectories])
        min_sbp = np.nanmean([np.min(t) for t in trajectories])
        sbp_drop = final_sbp - np.nanmean([t[0] for t in trajectories])

        results[pid] = {
            "n": n,
            "idh_rate": idh_rate,
            "mean_final_sbp": final_sbp,
            "mean_min_sbp": min_sbp,
            "mean_sbp_drop": sbp_drop,
            "mean_trajectory": mean_traj,
        }

    return results


def compute_center_distribution(labeled_sessions, center):
    """Compute phenotype distribution for a specific center."""
    center_sessions = [s for s in labeled_sessions if s["center"] == center]
    n_total = len(center_sessions)
    if n_total == 0:
        return {}

    counts = {}
    for s in center_sessions:
        pid = s["phenotype_id"]
        counts[pid] = counts.get(pid, 0) + 1

    return {pid: count / n_total for pid, count in counts.items()}


def chi_square_test(dist1, dist2, n1, n2):
    """Chi-square test for distribution difference."""
    all_pids = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    observed = np.array(
        [[dist1.get(pid, 0) * n1, dist2.get(pid, 0) * n2] for pid in all_pids]
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    return chi2, p_value, dof


def compute_ufr_statistics(labeled_sessions):
    """Compute UFR statistics by phenotype."""
    results = {}
    for pid in range(4):
        sessions = [s for s in labeled_sessions if s["phenotype_id"] == pid]
        ufr_values = [
            s["prescribed_ufr"]
            for s in sessions
            if not np.isnan(s.get("prescribed_ufr", np.nan))
        ]
        if ufr_values:
            results[pid] = {
                "mean_ufr": np.mean(ufr_values),
                "std_ufr": np.std(ufr_values),
                "median_ufr": np.median(ufr_values),
                "n": len(ufr_values),
            }
    return results


def compute_idwg_statistics(labeled_sessions):
    """Compute IDWG statistics by phenotype."""
    results = {}
    for pid in range(4):
        sessions = [s for s in labeled_sessions if s["phenotype_id"] == pid]
        idwg_values = [
            s["idwg"] for s in sessions if not np.isnan(s.get("idwg", np.nan))
        ]
        if idwg_values:
            results[pid] = {
                "mean_idwg": np.mean(idwg_values),
                "std_idwg": np.std(idwg_values),
                "median_idwg": np.median(idwg_values),
                "n": len(idwg_values),
            }
    return results
