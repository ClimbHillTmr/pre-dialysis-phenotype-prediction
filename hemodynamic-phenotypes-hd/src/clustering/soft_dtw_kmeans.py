"""Soft-DTW K-Means clustering for hemodynamic trajectory phenotyping."""

import numpy as np
from fastdtw import fastdtw
from joblib import Parallel, delayed
import multiprocessing

from config.settings import N_CLUSTERS, RANDOM_STATE, KMEANS_MAX_ITER, N_JOBS


def euclidean_dist(x, y):
    """Euclidean distance between two arrays."""
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))


def dtw_distance(x, y):
    """DTW distance using fastdtw."""
    d, _ = fastdtw(x, y, dist=euclidean_dist)
    return d


def compute_distances_single(x, centers):
    """Compute DTW distances for a single sample to all centers."""
    distances = np.zeros(len(centers))
    for j, c in enumerate(centers):
        distances[j] = dtw_distance(x, c)
    return distances


def compute_distances_parallel(X, centers, n_jobs=N_JOBS):
    """Parallel DTW distance computation."""
    distances = Parallel(n_jobs=n_jobs)(
        delayed(compute_distances_single)(x, centers) for x in X
    )
    return np.array(distances)


def compute_centroids(X, labels, n_clusters=N_CLUSTERS):
    """Compute new centroids as mean of assigned samples."""
    new_centers = []
    for k in range(n_clusters):
        cluster_pts = [X[i] for i in range(len(X)) if labels[i] == k]
        if not cluster_pts:
            new_centers.append(X[0])
            continue
        min_len = min(len(p) for p in cluster_pts)
        truncated = [p[:min_len] for p in cluster_pts]
        new_centers.append(np.mean(truncated, axis=0))
    return np.array(new_centers)


def soft_dtw_kmeans(
    X_sample, max_iter=KMEANS_MAX_ITER, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
):
    """
    Soft-DTW K-Means clustering.

    Parameters
    ----------
    X_sample : np.ndarray
        Sampled trajectories for clustering.
    max_iter : int
        Maximum number of iterations.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster assignments.
    centroids : np.ndarray
        Final cluster centroids.
    """
    print(f"\n[Clustering] Soft-DTW K-Means on {len(X_sample)} samples...")
    print(f"  Using {multiprocessing.cpu_count() - 2} parallel cores")

    np.random.seed(random_state)
    idx = np.random.choice(len(X_sample), n_clusters, replace=False)
    centers = X_sample[idx].copy()
    labels = np.zeros(len(X_sample), dtype=int)

    for it in range(max_iter):
        print(f"  Iter {it}: Computing DTW distance matrix...", flush=True)
        distances = compute_distances_parallel(X_sample, centers)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            print(f"  Converged at iteration {it}")
            break
        labels = new_labels
        centers = compute_centroids(X_sample, labels, n_clusters)
        print(f"  Iter {it}: Cluster distribution {np.bincount(labels)}", flush=True)

    print(f"  Final centroids shape: {centers.shape}")
    return labels, centers


def assign_to_centroids(X_remaining, centroids):
    """Assign remaining samples to nearest centroid using 1-NN."""
    print(
        f"\n[Assignment] 1-NN assignment for {len(X_remaining)} samples...", flush=True
    )
    distances = compute_distances_parallel(X_remaining, centroids)
    labels_remaining = np.argmin(distances, axis=1)
    print(f"  Assignment complete: {np.bincount(labels_remaining)}", flush=True)
    return labels_remaining


def characterize_phenotype(centroid):
    """Characterize phenotype based on centroid trajectory."""
    final = centroid[-1]
    min_val = np.min(centroid)
    if final < -30 and min_val < -35:
        return "P0: Severe Drop"
    elif final < -15:
        return "P1: Moderate Drop"
    elif final >= -15:
        return "P2: Stable"
    else:
        return "P3: Mild Drop"
